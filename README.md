# Speech Translator — Real-Time Speech-to-Speech Translation + Command Broadcast

> Speak into a browser mic, hear the result back as natural speech.
> Whisper STT → LLM translate → Kokoro TTS, all local on a single GPU,
> plus a one-to-many "Command Broadcast" mode that fans English out to
> 80+ written languages and 7 spoken ones in parallel.

---

## What this demo is

Speech Translator is a packaged, browser-driven distribution of
HuggingFace's open-source
[`speech-to-speech`](https://github.com/huggingface/speech-to-speech)
pipeline (Apache 2.0; see `LICENSE`). The upstream project is a
modular STT → LLM → TTS cascade designed for command-line use; this
demo wraps it in a containerised stack with:

- An HTTPS web UI that streams microphone audio to the pipeline over a
  WebSocket and plays the spoken result back through the speakers.
- A bundled **Ollama** container for the translation LLM, so the demo
  is truly self-contained — no external API keys, no cloud calls.
- A second mode (**Command Broadcast**) implemented as a separate
  FastAPI server (`broadcast_server.py`) that adds NLLB-200
  many-language translation and per-language Kokoro TTS on top of the
  same image.
- Optional Caddy reverse proxy with auto-HTTPS.

The full cascade (VAD → Whisper → LLM-translate → Kokoro TTS) runs
locally on a single NVIDIA GPU. **No audio leaves the box.**

### Mode 1 — Translate

Open the root URL, pick a source language (or leave it on
**Auto-detect**), and click the mic button. The pipeline streams 16 kHz
PCM to the container; once the VAD detects a complete utterance,
Whisper Large v3 transcribes it, the LLM cleans/translates the text into
fluent English, and Kokoro speaks the English back. Both the source
transcription and the English rendering scroll into the on-screen
transcript panel as they're produced.

What you can adjust live in the browser:

- **Source language** (`auto`, or any of ~90 Whisper-supported codes) —
  written to a shared file the pipeline polls.
- **Mic on/off** — start/stop streaming without restarting anything.

### Mode 2 — Command Broadcast

Click **BROADCAST** in the top nav. Speak an English command into the
mic; faster-whisper (`small.en`, CPU/int8) transcribes it locally, then
**NLLB-200-distilled-600M** fan-out translates it into every selected
target language in parallel. Each card lights up with the translated
text; for the seven languages where Kokoro has a voice (Spanish,
French, Chinese, Italian, Portuguese, Japanese, Hindi — plus English),
a play button generates the WAV on demand. Hit **BROADCAST ALL** and
the system speaks every TTS-capable card aloud, one after another.

The Broadcast server ships defaults for **80+ target languages** across
European, South Asian, East/Central Asian, African, and Turkic
families (full list in `broadcast_server.py`).

---

## Capabilities (at a glance)

- Streaming **VAD → STT → LLM → TTS** cascade with sub-second
  response on a modern GPU.
- **Whisper Large v3** STT on GPU with auto language detection across
  ~90 languages (configurable via `STT_MODEL`).
- **LLM translation** through any OpenAI-compatible endpoint —
  defaults to bundled Ollama serving `gpt-oss:20b`; swap to anything
  by changing `LLM_MODEL` and `LLM_BASE_URL`.
- **Kokoro 82M** neural TTS with selectable voice (`KOKORO_VOICE`) and
  CPU/CUDA placement (`KOKORO_DEVICE`).
- **Command Broadcast** mode: faster-whisper STT + NLLB-200 fan-out
  translation to 80+ written languages + Kokoro TTS for 7 spoken
  languages, all served from a single FastAPI sidecar.
- **HTTPS web UI** out of the box (self-signed cert) — required for
  browser microphone access.
- **Bundled Ollama** with persistent model volume; or BYO
  OpenAI-compatible endpoint via `docker-compose.byo.yml`.
- Optional **Caddy reverse proxy** with Let's Encrypt for public
  hostnames.
- Per-utterance **transcript log** (`/shared/transcripts.jsonl`)
  shared between pipeline and web container.

---

## Reference build platform

This demo was built and tested on a **Dell Pro Max GB10** (NVIDIA Grace
Blackwell, **ARM / aarch64** architecture). Two Dockerfiles ship in
the repo and you should pick the one that matches your host:

| Dockerfile | Target | Base image |
|---|---|---|
| `Dockerfile` | **x86_64** + recent NVIDIA GPU | `nvidia/cuda:12.8.0-devel-ubuntu22.04`, PyTorch nightly cu128 (Blackwell / sm_120+ ready) |
| `Dockerfile.arm64` | **ARM64 / aarch64** Jetson-class hosts | `nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3` |

`docker-compose.yml` references `Dockerfile` by default. On
ARM/Jetson hosts override the `dockerfile:` field (or pass
`--build-arg`/`-f` as needed) — see step 6 below.

---

## Requirements

| Requirement | Minimum | Notes |
|---|---|---|
| OS | Linux | macOS / Windows lack the GPU pass-through this pipeline needs. |
| Docker | 24.x or newer | With Compose **v2** (`docker compose`, not `docker-compose`). |
| GPU | NVIDIA, ≥ 16 GB VRAM | Whisper Large v3 (~6 GB) + `gpt-oss:20b` (~14 GB) + Kokoro on CUDA. Drop to `whisper-medium` and `qwen2.5:7b` for ~10 GB GPUs. |
| GPU driver | Recent enough for your CUDA version | `nvidia-smi` must work on the host. |
| NVIDIA Container Toolkit | Installed and configured for Docker | Required to expose the GPU to the container. |
| Disk | ~30 GB | Pipeline image ~10 GB; Whisper Large v3 ~3 GB; NLLB-200 ~2.5 GB; Kokoro ~1 GB; `gpt-oss:20b` ~13 GB. |
| RAM | 16 GB recommended | The Broadcast server keeps Whisper + NLLB + multiple Kokoro pipelines resident. |
| Audio | Live mic via browser, speakers for playback | Quiet environment recommended. |
| API key | None | Everything runs locally. |

---

## Installation (step-by-step)

These instructions assume a fresh Linux box. If you already have
Docker + the NVIDIA Container Toolkit working, skip to step 4.

### 1. Verify your GPU is visible to the host

```bash
nvidia-smi
```

You should see a table with your GPU model, driver version, and CUDA
version. If this command fails, **fix your NVIDIA driver before going
further** — the rest will not work.

### 2. Install Docker Engine + Compose v2

Ubuntu / Debian:

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER"   # let your user run docker without sudo
newgrp docker                      # apply the new group in this shell
docker compose version             # should print "Docker Compose version v2.x.x"
```

If `docker compose version` reports "command not found", install the
plugin:

```bash
sudo apt install docker-compose-plugin
```

### 3. Install the NVIDIA Container Toolkit

Ubuntu / Debian:

```bash
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify it works inside Docker:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

You should see the same `nvidia-smi` table you saw on the host.

### 4. Clone the repo

```bash
git clone https://github.com/gleaven/speech-translator.git
cd speech-translator
```

### 5. Create the environment file

```bash
cp .env.example .env
```

The defaults are sensible. Edit `.env` only if you need to change
ports, change the LLM, or move Kokoro to GPU. See **Configuration**
below for the full list.

### 6. Build and start

**x86_64 host** (default — uses `Dockerfile`):

```bash
docker compose up -d --build
```

**ARM64 / Jetson host** — point the pipeline build at the ARM
Dockerfile. The simplest way is a one-line override in your shell:

```bash
DOCKER_BUILDKIT=1 docker compose build \
  --build-arg DOCKERFILE=Dockerfile.arm64 pipeline
# or override the dockerfile path in compose:
sed -i 's|dockerfile: Dockerfile$|dockerfile: Dockerfile.arm64|' docker-compose.yml
docker compose up -d --build
```

The first build takes **10–20 minutes** (CUDA base image, PyTorch
nightly, faster-whisper, Kokoro, MeloTTS, ParlerTTS, NLTK data,
unidic). Subsequent starts take ~10 seconds.

### 7. Pull the translation LLM

The bundled `demo-ollama` container starts empty:

```bash
docker exec demo-ollama ollama pull gpt-oss:20b   # ~13 GB
```

Smaller alternative for tighter GPUs:

```bash
docker exec demo-ollama ollama pull qwen2.5:7b    # ~4.5 GB
# then set LLM_MODEL=qwen2.5:7b in .env and `docker compose up -d`
```

Whisper Large v3, NLLB-200, and Kokoro models download automatically
on first request into the `speech-translator-cache` named volume.

### 8. Verify it's healthy

```bash
docker compose ps
# demo-ollama should be "healthy" within ~60 s
# demo-speech-translator and demo-speech-translator-web should be "running"

curl -sk https://localhost:${APP_PORT:-8092}/health
# {"status":"ok","pipeline":"reachable"}

# Broadcast warmup status (will flip to "ok" once Whisper/NLLB/Kokoro warm):
curl -s http://localhost:${BROADCAST_API_PORT:-8088}/health
# {"status":"ok","service":"broadcast","models_ready":true}
```

### 9. Open the UIs

- **Translate:** <https://localhost:8092/>
- **Command Broadcast:** <https://localhost:8092/broadcast>

The UI is **HTTPS-only** (browsers refuse mic access over plain HTTP);
the bundled web container generates a self-signed cert at build time.
Accept the warning, grant microphone permission, and start speaking.

### 10. (Optional) Tail the logs

```bash
docker compose logs -f pipeline web ollama
```

Useful lines to look for:

```
====== Broadcast model warmup starting ======
Loading Whisper small.en on CPU...
Loading NLLB-200-distilled-600M on cuda (cache)...
Loading Kokoro pipeline for English (a)...
====== Broadcast model warmup complete in 12.3s ======
Receiver waiting to be connected...
WebSocket client connected
```

---

## Configuration

Set in `.env` or export in your shell.

| Variable | Default | What it controls |
|---|---|---|
| `APP_PORT` | `8092` | Browser-facing HTTPS port for both UIs. |
| `LLM_MODEL` | `gpt-oss:20b` | Translation LLM (any model the LLM endpoint serves). |
| `LLM_BASE_URL` | `http://demo-ollama:11434/v1` | OpenAI-compatible endpoint for the LLM. |
| `LLM_API_KEY` | `unused` | Required by the OpenAI client; any non-empty string works for Ollama. |
| `STT_MODEL` | `openai/whisper-large-v3` | HF model id for streaming STT (`openai/whisper-medium`, `distil-whisper/distil-large-v3`, etc.). |
| `TRANSLATE_LANGUAGE` | `auto` | Source-language hint for Whisper (`auto`, `en`, `es`, `fr`, …). Live-overridable via the UI. |
| `KOKORO_VOICE` | `af_heart` | Kokoro voice id for the streaming TTS (see Kokoro VOICES.md). |
| `KOKORO_LANG_CODE` | `a` | Kokoro language code (`a` = American English, `b` = British, `j` = Japanese, …). |
| `KOKORO_DEVICE` | `cpu` | `cpu` or `cuda`. CUDA needs ~1 GB more VRAM but eliminates audio under-runs on long lines. |
| `HF_TOKEN` | _(empty)_ | HuggingFace token, only needed for gated models. |
| `TRANSFORMERS_OFFLINE` | `0` | Set to `1` to forbid network downloads. |
| `HF_HUB_OFFLINE` | `0` | Same, for `huggingface_hub`. |
| `PIPELINE_SEND_PORT` | `12345` | Host TCP port the pipeline accepts mic audio on. |
| `PIPELINE_RECV_PORT` | `12346` | Host TCP port the pipeline streams TTS audio out on. |
| `BROADCAST_API_PORT` | `8088` | Host port for the Broadcast FastAPI server. |
| `OLLAMA_HOST_PORT` | `11434` | Host port for the bundled Ollama. |
| `DEMO_HOSTNAME` | `localhost` | Hostname Caddy serves under (proxy profile only). |
| `HTTP_PORT` | `8081` | Caddy HTTP port. |
| `HTTPS_PORT` | `8443` | Caddy HTTPS port. |

### Picking a smaller LLM

```bash
docker exec demo-ollama ollama pull qwen2.5:7b
# .env:
LLM_MODEL=qwen2.5:7b
docker compose up -d
```

### Moving Kokoro to GPU

```bash
# .env:
KOKORO_DEVICE=cuda
docker compose up -d
```

---

## Live controls (in the browser)

**Translate page** (`/`):

- **Source language dropdown** — `Auto-detect` plus ~90 Whisper
  language codes; selection is persisted to a shared hint file the
  pipeline reads.
- **Mic button** — start/stop streaming. The level meter shows live
  input amplitude; the four pipeline-stage dots (VAD → Whisper STT →
  LLM Cleanup → Kokoro TTS) animate as each stage activates.
- **Transcript pane** — both source-language transcription and
  English rendering append in real time.
- **GPU utilisation** badge — pulls live percentage from the bundled
  service router (if reachable; otherwise stays at 0%).

**Broadcast page** (`/broadcast`):

- **Record button** — captures English speech, transcribes via
  faster-whisper, then auto-fan-outs translation to all selected
  language cards.
- **Per-card play button** — synthesises and plays Kokoro TTS for the
  card's translation (TTS-capable languages only — see grid below).
- **BROADCAST ALL** — speaks every TTS-capable card sequentially.

Kokoro currently has voices for: **Spanish, French, Chinese, Italian,
Portuguese, Japanese, Hindi, English** (8 total). All other languages
in the broadcast list are translation-only (text appears, no audio).

---

## External services (BYO)

To skip the bundled Ollama and route translation at any
OpenAI-compatible endpoint (your own Ollama, LiteLLM proxy, vLLM,
OpenAI itself, etc.):

```bash
# .env:
LLM_BASE_URL=http://my-litellm:4000/v1
LLM_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini

docker compose -f docker-compose.yml -f docker-compose.byo.yml up -d
```

`docker-compose.byo.yml` sets `replicas: 0` on the bundled `ollama`
service and clears the pipeline's `depends_on`, so only `pipeline` and
`web` start.

| Variable | Example |
|---|---|
| `LLM_BASE_URL` | `https://api.openai.com/v1` |
| `LLM_API_KEY` | `sk-...` |
| `LLM_MODEL` | `gpt-4o-mini` |

---

## Optional HTTPS reverse proxy

The web container already serves HTTPS via a self-signed cert. To
front it with Caddy + Let's Encrypt instead (e.g. for a real
hostname):

```bash
DEMO_HOSTNAME=speech.example.com docker compose --profile proxy up -d
```

`Caddyfile` is intentionally minimal:

```
{$DEMO_HOSTNAME:localhost} {
    reverse_proxy web:8089
}
```

For local testing keep `DEMO_HOSTNAME=localhost` and Caddy issues a
self-signed cert.

---

## Authentication

Speech Translator runs **without authentication** by default. Live
microphone streams and the translation LLM are sensitive — keep the
dashboard off the public internet unless you've added auth in front:

- **Caddy basic auth** — add a `basic_auth` block to the Caddyfile.
- **oauth2-proxy in front of Caddy** — for SSO-style auth.
- **Cloudflare Tunnel + Access policies** — easiest if you're already
  on Cloudflare.

---

## Architecture (file map)

| Path | Purpose |
|---|---|
| `s2s_pipeline.py` | Upstream HuggingFace entry point. Wires the `VAD → STT → LLM → TTS` thread chain together based on CLI args. |
| `baseHandler.py` | Base class for every pipeline stage: queue-in / queue-out, lifecycle, error recovery. |
| `broadcast_server.py` | Standalone FastAPI server on port 8088 powering Command Broadcast (faster-whisper + NLLB-200 + Kokoro fan-out). |
| `listen_and_play.py` | CLI client (sounddevice) for testing the pipeline outside the browser. |
| `arguments_classes/` | Per-handler dataclasses parsed by `HfArgumentParser` (one file per STT/LLM/TTS variant). |
| `connections/` | TCP socket receiver/sender + a local-audio streamer used in dev. |
| `VAD/` | Silero-based voice activity detector that gates Whisper. |
| `STT/` | Whisper, faster-whisper, lightning-whisper-mlx, moonshine, paraformer handler implementations. |
| `LLM/` | Local Transformers, MLX, OpenAI-compatible (`openai_api_language_model.py`) and chat utilities. |
| `TTS/` | Kokoro (default), MeloTTS, ParlerTTS, ChatTTS, FacebookMMS handlers. |
| `web/app.py` | FastAPI bridge: serves the static UIs, opens the WebSocket → TCP audio bridge to the pipeline, proxies Broadcast API calls, streams transcript updates. |
| `web/static/index.html` | Translate UI (cyber-themed; vanilla JS). |
| `web/static/broadcast.html` | Command Broadcast UI. |
| `Dockerfile` | x86_64 image: CUDA 12.8.0-devel + PyTorch nightly cu128 + all pipeline deps + NLTK + unidic data. |
| `Dockerfile.arm64` | ARM64 / Jetson image based on `l4t-pytorch:r35.2.1-pth2.0-py3`. |
| `docker-compose.yml` | Pipeline + Web + Ollama + (opt) Caddy. |
| `docker-compose.byo.yml` | Override that disables bundled Ollama. |
| `Caddyfile` | Trivial reverse proxy with auto-HTTPS. |

The pipeline container starts **two processes** at once: the FastAPI
broadcast server in the background (`python3 broadcast_server.py &`)
and the streaming `s2s_pipeline.py` in the foreground. Both share the
GPU.

---

## Troubleshooting

- **Browser refuses microphone access** — the page must be served over
  HTTPS (`localhost` is also accepted in some browsers). The bundled
  web container generates a self-signed cert; accept the exception, or
  put Caddy in front for a trusted cert.
- **Translate mode never speaks** — confirm the LLM model is pulled
  (`docker exec demo-ollama ollama list`) and that the pipeline log
  shows the OpenAI client warmup line. If you see "Connection refused
  to demo-ollama:11434", the Ollama healthcheck hasn't passed yet —
  give it ~60 s on first start.
- **Pipeline OOMs at startup** — Whisper Large v3 needs ~6 GB,
  `gpt-oss:20b` needs ~14 GB, Kokoro on CUDA needs ~1 GB. On a
  single-GPU box you may need to switch to a smaller LLM
  (`LLM_MODEL=qwen2.5:7b`) and/or smaller STT
  (`STT_MODEL=openai/whisper-medium`).
- **Audio is choppy / words clip** — Kokoro on CPU can underrun on
  long utterances. Set `KOKORO_DEVICE=cuda` and bounce the pipeline.
- **Broadcast page shows cards but no audio** — only 7 languages
  (Spanish/French/Chinese/Italian/Portuguese/Japanese/Hindi) are
  TTS-capable; the rest are translation-only by design.
- **Broadcast `/health` returns `warming_up` for a long time** — first
  use downloads NLLB-200 (~2.5 GB) and several Kokoro pipelines into
  `speech-translator-cache`. Watch `docker logs -f
  demo-speech-translator | grep -E "Loading|loaded"`.
- **GPU not visible to containers** — verify with `docker run --rm
  --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi`. If that
  fails, fix the NVIDIA Container Toolkit before continuing.
- **First request hangs for minutes** — Whisper Large v3 + Kokoro
  pipelines are downloading on demand. Subsequent requests are
  instant.
- **`docker compose down -v` lost my models** — `-v` removes the named
  volumes (`speech-translator-cache`, `speech-translator-ollama-data`).
  Drop the `-v` to keep them; you'll save ~20 GB of re-downloads.
- **ARM64 build fails on x86 wheel** — make sure you're using
  `Dockerfile.arm64`, not the default `Dockerfile`. Some upstream
  packages (`parler_tts`, `melotts`) build from source on ARM and can
  take 5+ extra minutes.

---

## FAQ

**Q: Can I use a CPU?** No. Whisper Large v3 streaming and the LLM
both need GPU; on CPU latency would be tens of seconds per utterance.

**Q: Does the Translate mode have to translate to English?** As
shipped, yes — the system prompt in `docker-compose.yml` instructs the
LLM to "translate it into natural, fluent English." Edit the
`--open_api_init_chat_prompt` argument to change the target language
(or to switch from translation to summarisation, dictation cleanup,
etc.).

**Q: How is Broadcast different from Translate?** Translate is a
real-time speech-in / speech-out cascade for one target language at a
time, driven by the streaming `s2s_pipeline.py`. Broadcast is a
request/response REST API that takes one English utterance and
fan-outs translation (and optionally speech) to many languages at
once.

**Q: Can I add more Kokoro voices to Broadcast?** Yes — extend
`KOKORO_LANG_CONFIG` in `broadcast_server.py` with the appropriate
`lang_code` and `voice` from Kokoro's `VOICES.md`.

**Q: Is anything sent to the cloud?** No, by default. With the BYO
override and an OpenAI / Anthropic / etc. endpoint, only the
translated text leaves — never raw audio.

---

## Credits

Built by Andrew Meinecke.

Based on the open-source
[`speech-to-speech`](https://github.com/huggingface/speech-to-speech)
pipeline by The HuggingFace Inc. team, distributed under the Apache
License 2.0 (see `LICENSE`).

## Components & Licensing

This demo is released under Apache License 2.0 (see `LICENSE`). It
bundles or wraps the following third-party components, each retaining
its own license:

**Code dependencies (compiled into the container image):**

| Component | License | Use in this demo |
|---|---|---|
| HuggingFace [`speech-to-speech`](https://github.com/huggingface/speech-to-speech) | Apache 2.0 | Pipeline scaffold (VAD → STT → LLM → TTS handlers) |
| [PyTorch nightly + cu128](https://github.com/pytorch/pytorch) | BSD-3 | GPU autodiff (Blackwell `sm_120+`) |
| HuggingFace [`transformers`](https://github.com/huggingface/transformers) (incl. NLLB tokenizer / sentencepiece) | Apache 2.0 | Model loading |
| [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper) | MIT | CTranslate2-accelerated Whisper inference |
| [`parler-tts`](https://github.com/huggingface/parler-tts) | Apache 2.0 | Description-conditioned TTS handler |
| [MeloTTS](https://github.com/myshell-ai/MeloTTS) (`andimarafioti` fork pinned in `requirements.txt`) | MIT | Multi-lingual TTS handler |
| [**ChatTTS**](https://github.com/2noise/ChatTTS) | **AGPL-3.0+** (code) / **CC BY-NC 4.0** (model weights) | Optional TTS handler |
| [Moonshine](https://github.com/usefulsensors/moonshine) (`andimarafioti` fork pinned in `requirements.txt`) | MIT | Optional small-footprint STT |
| [Kokoro](https://github.com/hexgrad/kokoro) (`kokoro` PyPI package) | Apache 2.0 | Default TTS handler in Broadcast mode |
| [FunASR](https://github.com/modelscope/FunASR) | MIT | Optional Paraformer STT |
| [ModelScope](https://github.com/modelscope/modelscope) | Apache 2.0 | Model registry / loader for FunASR weights |
| [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) | MIT / Apache 2.0 (dual) | Real-time noise suppression |
| [Silero VAD](https://github.com/snakers4/silero-vad) | MIT | Voice activity detection |
| [NLTK](https://github.com/nltk/nltk) | Apache 2.0 | Tokenizer assets (`punkt_tab`, etc.) |
| [eSpeak NG](https://github.com/espeak-ng/espeak-ng) (apt package) | GPL-3.0 | Phonemizer binary called by some TTS pipelines |
| [FFmpeg](https://github.com/FFmpeg/FFmpeg) (apt package) | LGPL-2.1+ / GPL-2+ (config-dependent) | Audio decode |
| [FastAPI](https://github.com/fastapi/fastapi) | MIT | Broadcast-mode API + WebSocket |
| [Ollama](https://github.com/ollama/ollama) (bundled service in `docker-compose.yml`) | MIT | Local translation LLM |
| [NVIDIA CUDA base image](https://hub.docker.com/r/nvidia/cuda) | NVIDIA Deep Learning Container Software License | GPU runtime |
| [Caddy](https://github.com/caddyserver/caddy) (optional `--profile proxy`) | Apache 2.0 | HTTPS termination |

**Model weights (downloaded from HuggingFace at runtime by the
container, not redistributed in this repo):**

| Model | License | Notes |
|---|---|---|
| [OpenAI Whisper](https://huggingface.co/openai/whisper-large-v3) (`STT_MODEL` default) | MIT | Permissive |
| [Kokoro 82M](https://huggingface.co/hexgrad/Kokoro-82M) (Broadcast TTS default) | Apache 2.0 | Permissive |
| [**NLLB-200 distilled-600M**](https://huggingface.co/facebook/nllb-200-distilled-600M) (Broadcast translation) | **CC BY-NC 4.0** | Research / non-commercial; Meta states the model is "not released for production deployment" |
| LLM served by the bundled Ollama (`gpt-oss:20b` by default) | per chosen model | Override via `LLM_MODEL` |

### License notes

- The **default Translate mode** (Whisper + Ollama LLM + Kokoro) uses
  only permissive licenses.
- **Broadcast mode** loads
  [NLLB-200](https://huggingface.co/facebook/nllb-200-distilled-600M)
  for many-language translation. NLLB-200 is **CC BY-NC 4.0** and is
  designated by Meta as a research-only model not authorised for
  production deployment.
- **ChatTTS** is listed in `requirements.txt` because the upstream
  `speech-to-speech` package wires it up as an optional TTS handler.
  It is **AGPL-3.0+** code and **CC BY-NC 4.0** model weights. The
  default pipeline does not load it; if you switch to it
  (`--tts chat_tts`), both restrictions apply.
- **eSpeak NG** (GPL-3.0) and the **FFmpeg** apt build (LGPL/GPL,
  depending on configure flags) are invoked as separate processes;
  redistributing the built container image redistributes their
  binaries — read each project's source-availability requirements
  before doing so.
