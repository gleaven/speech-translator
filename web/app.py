import asyncio
import json
import os
import socket
import logging

import httpx
from fastapi import FastAPI, File, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response

SERVICEROUTER_URL = os.environ.get("SERVICEROUTER_URL", "http://demo-servicerouter:8080")
BROADCAST_API_URL = os.environ.get("BROADCAST_API_URL", "http://demo-speech-translator:8088")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PIPELINE_HOST = os.environ.get("PIPELINE_HOST", "demo-speech-translator")
SEND_PORT = int(os.environ.get("SEND_PORT", "12345"))
RECV_PORT = int(os.environ.get("RECV_PORT", "12346"))
CHUNK_SIZE = 1024  # bytes (512 int16 samples)
TRANSCRIPT_FILE = "/shared/transcripts.jsonl"
LANGUAGE_HINT_FILE = "/shared/language_hint"

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.get("/health")
async def health():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2)
        s.connect((PIPELINE_HOST, SEND_PORT))
        s.close()
        return {"status": "ok", "pipeline": "reachable"}
    except Exception as e:
        return {"status": "error", "pipeline": "unreachable", "detail": str(e)}


@app.get("/api/system/stats")
async def api_system_stats():
    """Proxy system stats from service router for GPU utilization display."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{SERVICEROUTER_URL}/api/system/stats")
            return r.json()
    except Exception as e:
        return {"gpu_percent": None, "error": str(e)}


@app.get("/api/language-hint")
async def get_language_hint():
    """Get the current source language hint."""
    try:
        with open(LANGUAGE_HINT_FILE, "r") as f:
            hint = f.read().strip()
        return {"language": hint or "auto"}
    except FileNotFoundError:
        return {"language": "auto"}


@app.post("/api/language-hint")
async def set_language_hint(request: Request):
    """Set the source language hint for Whisper STT."""
    body = await request.json()
    lang = body.get("language", "auto").strip()
    with open(LANGUAGE_HINT_FILE, "w") as f:
        f.write(lang)
    logger.info(f"Language hint set to: {lang}")
    return {"language": lang}


@app.get("/broadcast")
async def broadcast_page():
    return FileResponse("static/broadcast.html")


# -- Broadcast API proxies --------------------------------------------------

@app.get("/api/broadcast/languages")
async def proxy_broadcast_languages():
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(f"{BROADCAST_API_URL}/api/broadcast/languages")
    return Response(content=r.content, status_code=r.status_code, media_type="application/json")


@app.post("/api/broadcast/transcribe")
async def proxy_broadcast_transcribe(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    async with httpx.AsyncClient(timeout=30.0) as client:
        files = {"file": (file.filename or "audio.webm", audio_bytes, file.content_type or "audio/webm")}
        r = await client.post(f"{BROADCAST_API_URL}/api/broadcast/transcribe", files=files)
    return Response(content=r.content, status_code=r.status_code, media_type="application/json")


@app.post("/api/broadcast/translate")
async def proxy_broadcast_translate(request: Request):
    body = await request.body()
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(
            f"{BROADCAST_API_URL}/api/broadcast/translate",
            content=body, headers={"content-type": "application/json"},
        )
    return Response(content=r.content, status_code=r.status_code, media_type="application/json")


@app.post("/api/broadcast/tts")
async def proxy_broadcast_tts(request: Request):
    body = await request.body()
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(
            f"{BROADCAST_API_URL}/api/broadcast/tts",
            content=body, headers={"content-type": "application/json"},
        )
    return Response(
        content=r.content, status_code=r.status_code,
        media_type=r.headers.get("content-type", "audio/wav"),
    )


@app.post("/api/broadcast/process")
async def proxy_broadcast_process(request: Request):
    body = await request.body()
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(
            f"{BROADCAST_API_URL}/api/broadcast/process",
            content=body, headers={"content-type": "application/json"},
        )
    return Response(content=r.content, status_code=r.status_code, media_type="application/json")


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    logger.info("WebSocket client connected")

    send_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    recv_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    send_sock.settimeout(5)
    recv_sock.settimeout(5)

    try:
        logger.info(f"Connecting to pipeline send socket {PIPELINE_HOST}:{SEND_PORT}...")
        send_sock.connect((PIPELINE_HOST, SEND_PORT))
        logger.info("Send socket connected")

        logger.info(f"Connecting to pipeline recv socket {PIPELINE_HOST}:{RECV_PORT}...")
        recv_sock.connect((PIPELINE_HOST, RECV_PORT))
        logger.info("Recv socket connected")

        send_sock.settimeout(None)
        recv_sock.settimeout(None)

    except Exception as e:
        logger.error(f"Failed to connect to pipeline: {type(e).__name__}: {e}")
        try:
            await ws.send_json({"type": "error", "message": f"Pipeline not ready: {type(e).__name__}"})
        except Exception:
            pass
        send_sock.close()
        recv_sock.close()
        await ws.close()
        return

    logger.info("Both pipeline sockets connected, starting audio bridge")
    stop_event = asyncio.Event()
    loop = asyncio.get_event_loop()

    # Track transcript file position to send only new entries
    transcript_pos = _get_file_size(TRANSCRIPT_FILE)

    async def recv_from_pipeline():
        bytes_forwarded = 0
        try:
            while not stop_event.is_set():
                data = await loop.run_in_executor(None, _recv_chunk, recv_sock, CHUNK_SIZE)
                if data is None:
                    logger.info("Pipeline recv socket closed (no data)")
                    break
                await ws.send_bytes(data)
                bytes_forwarded += len(data)
        except Exception as e:
            if not stop_event.is_set():
                logger.error(f"recv_from_pipeline error: {type(e).__name__}: {e}")
        finally:
            logger.info(f"recv_from_pipeline done, total bytes: {bytes_forwarded}")
            stop_event.set()

    async def poll_transcripts():
        """Poll the shared transcript file and send new lines to the browser."""
        nonlocal transcript_pos
        try:
            while not stop_event.is_set():
                await asyncio.sleep(0.5)
                new_lines = _read_new_lines(TRANSCRIPT_FILE, transcript_pos)
                if new_lines:
                    for line, new_pos in new_lines:
                        transcript_pos = new_pos
                        try:
                            data = json.loads(line)
                            await ws.send_json({
                                "type": "transcript",
                                "source": data.get("type", "user"),
                                "text": data.get("text", ""),
                                "lang": data.get("lang", ""),
                            })
                        except (json.JSONDecodeError, Exception):
                            pass
        except Exception as e:
            if not stop_event.is_set():
                logger.error(f"poll_transcripts error: {type(e).__name__}: {e}")

    recv_task = asyncio.create_task(recv_from_pipeline())
    transcript_task = asyncio.create_task(poll_transcripts())
    bytes_sent = 0

    try:
        while True:
            data = await ws.receive_bytes()
            send_sock.sendall(data)
            bytes_sent += len(data)
    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected (sent {bytes_sent} bytes total)")
    except Exception as e:
        logger.error(f"WebSocket send loop error: {type(e).__name__}: {e}")
    finally:
        stop_event.set()
        try:
            recv_sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        recv_task.cancel()
        transcript_task.cancel()
        send_sock.close()
        recv_sock.close()
        logger.info("Pipeline sockets closed, session ended")


def _recv_chunk(sock: socket.socket, chunk_size: int) -> bytes | None:
    data = b""
    while len(data) < chunk_size:
        try:
            packet = sock.recv(chunk_size - len(data))
        except OSError:
            return None
        if not packet:
            return None
        data += packet
    return data


def _get_file_size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def _read_new_lines(path: str, pos: int) -> list[tuple[str, int]]:
    """Read new complete lines from file starting at pos."""
    try:
        with open(path, "r") as f:
            f.seek(pos)
            results = []
            for line in f:
                if line.endswith("\n"):
                    results.append((line.strip(), pos + len(line.encode())))
                    pos += len(line.encode())
            return results
    except OSError:
        return []
