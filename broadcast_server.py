"""
Broadcast API Server — English to Many Languages

FastAPI server running on port 8088 alongside the streaming pipeline.
Provides REST endpoints for the Command Broadcast feature:
  - Transcribe English speech (Whisper base.en on GPU)
  - Translate to target languages (NLLB-200-distilled-600M on CPU)
  - Generate TTS audio per language (Kokoro 82M on CPU)
"""

import io
import logging
import os
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Language configuration
# ---------------------------------------------------------------------------

NLLB_LANG_MAP = {
    # Kokoro TTS languages (have voice output)
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "zh": "zho_Hans",
    "it": "ita_Latn",
    "pt": "por_Latn",
    "ja": "jpn_Jpan",
    "hi": "hin_Deva",
    "en": "eng_Latn",
    # Translation-only — major world languages
    "ar": "arb_Arab",
    "bn": "ben_Beng",
    "ru": "rus_Cyrl",
    "de": "deu_Latn",
    "ko": "kor_Hang",
    "tr": "tur_Latn",
    "vi": "vie_Latn",
    "th": "tha_Thai",
    "id": "ind_Latn",
    "ur": "urd_Arab",
    "pl": "pol_Latn",
    "uk": "ukr_Cyrl",
    "nl": "nld_Latn",
    "el": "ell_Grek",
    "sv": "swe_Latn",
    "ro": "ron_Latn",
    "tl": "tgl_Latn",
    "sw": "swh_Latn",
    "fa": "pes_Arab",
    "ms": "zsm_Latn",
    # South Asian
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "mr": "mar_Deva",
    "gu": "guj_Gujr",
    "kn": "kan_Knda",
    "ml": "mal_Mlym",
    "pa": "pan_Guru",
    "ne": "npi_Deva",
    "si": "sin_Sinh",
    # European
    "cs": "ces_Latn",
    "hu": "hun_Latn",
    "fi": "fin_Latn",
    "da": "dan_Latn",
    "no": "nob_Latn",
    "bg": "bul_Cyrl",
    "hr": "hrv_Latn",
    "sr": "srp_Cyrl",
    "sk": "slk_Latn",
    "lt": "lit_Latn",
    "lv": "lvs_Latn",
    "et": "est_Latn",
    "sl": "slv_Latn",
    "sq": "als_Latn",
    "mk": "mkd_Cyrl",
    "bs": "bos_Latn",
    "is": "isl_Latn",
    "ca": "cat_Latn",
    "gl": "glg_Latn",
    "cy": "cym_Latn",
    "ga": "gle_Latn",
    "mt": "mlt_Latn",
    "he": "heb_Hebr",
    # African
    "am": "amh_Ethi",
    "ha": "hau_Latn",
    "yo": "yor_Latn",
    "ig": "ibo_Latn",
    "zu": "zul_Latn",
    "xh": "xho_Latn",
    "af": "afr_Latn",
    "so": "som_Latn",
    # East/Central Asian & Turkic
    "my": "mya_Mymr",
    "km": "khm_Khmr",
    "lo": "lao_Laoo",
    "mn": "khk_Cyrl",
    "ka": "kat_Geor",
    "hy": "hye_Armn",
    "az": "azj_Latn",
    "uz": "uzn_Latn",
    "kk": "kaz_Cyrl",
    "ps": "pbt_Arab",
    "tg": "tgk_Cyrl",
}

KOKORO_LANG_CONFIG = {
    "es": {"lang_code": "e", "voice": "ef_dora"},
    "fr": {"lang_code": "f", "voice": "ff_siwis"},
    "zh": {"lang_code": "z", "voice": "zf_xiaobei"},
    "it": {"lang_code": "i", "voice": "if_sara"},
    "pt": {"lang_code": "p", "voice": "pf_dora"},
    "ja": {"lang_code": "j", "voice": "jf_alpha"},
    "hi": {"lang_code": "h", "voice": "hf_alpha"},
    "en": {"lang_code": "a", "voice": "af_heart"},
}

TTS_LANGS = set(KOKORO_LANG_CONFIG.keys())

LANG_NAMES = {
    "es": "Spanish", "fr": "French", "zh": "Chinese",
    "it": "Italian", "pt": "Portuguese", "ja": "Japanese",
    "hi": "Hindi", "en": "English",
    # Major world
    "ar": "Arabic", "bn": "Bengali", "ru": "Russian",
    "de": "German", "ko": "Korean", "tr": "Turkish",
    "vi": "Vietnamese", "th": "Thai", "id": "Indonesian",
    "ur": "Urdu", "pl": "Polish", "uk": "Ukrainian",
    "nl": "Dutch", "el": "Greek", "sv": "Swedish",
    "ro": "Romanian", "tl": "Filipino", "sw": "Swahili",
    "fa": "Persian", "ms": "Malay",
    # South Asian
    "ta": "Tamil", "te": "Telugu", "mr": "Marathi",
    "gu": "Gujarati", "kn": "Kannada", "ml": "Malayalam",
    "pa": "Punjabi", "ne": "Nepali", "si": "Sinhala",
    # European
    "cs": "Czech", "hu": "Hungarian", "fi": "Finnish",
    "da": "Danish", "no": "Norwegian", "bg": "Bulgarian",
    "hr": "Croatian", "sr": "Serbian", "sk": "Slovak",
    "lt": "Lithuanian", "lv": "Latvian", "et": "Estonian",
    "sl": "Slovenian", "sq": "Albanian", "mk": "Macedonian",
    "bs": "Bosnian", "is": "Icelandic", "ca": "Catalan",
    "gl": "Galician", "cy": "Welsh", "ga": "Irish",
    "mt": "Maltese", "he": "Hebrew",
    # African
    "am": "Amharic", "ha": "Hausa", "yo": "Yoruba",
    "ig": "Igbo", "zu": "Zulu", "xh": "Xhosa",
    "af": "Afrikaans", "so": "Somali",
    # East/Central Asian & Turkic
    "my": "Burmese", "km": "Khmer", "lo": "Lao",
    "mn": "Mongolian", "ka": "Georgian", "hy": "Armenian",
    "az": "Azerbaijani", "uz": "Uzbek", "kk": "Kazakh",
    "ps": "Pashto", "tg": "Tajik",
}

# TTS languages first, then translation-only grouped by region
DEFAULT_TARGET_LANGS = [
    # TTS (Kokoro)
    "es", "fr", "zh", "it", "pt", "ja", "hi",
    # Major world languages
    "ar", "bn", "de", "ru", "ko", "tr", "id",
    "vi", "th", "ur", "fa", "ms", "tl", "sw",
    # South Asian
    "ta", "te", "mr", "gu", "kn", "ml", "pa", "ne", "si",
    # European
    "pl", "uk", "nl", "el", "sv", "ro", "cs", "hu", "fi",
    "da", "no", "bg", "hr", "sr", "sk", "lt", "lv", "et",
    "sl", "sq", "mk", "bs", "is", "ca", "gl", "cy", "ga",
    "mt", "he", "af",
    # African
    "am", "ha", "yo", "ig", "zu", "xh", "so",
    # East/Central Asian & Turkic
    "my", "km", "lo", "mn", "ka", "hy",
    "az", "uz", "kk", "ps", "tg",
]

SUPPORTED_LANGS = list(NLLB_LANG_MAP.keys())

# ---------------------------------------------------------------------------
# Model manager with lazy loading and thread safety
# ---------------------------------------------------------------------------

class BroadcastModels:
    def __init__(self):
        self._whisper = None
        self._nllb_model = None
        self._nllb_tokenizer = None
        self._nllb_device = "cpu"
        self._kokoro_pipelines: dict = {}  # kokoro_lang_code -> KPipeline
        self._init_lock = threading.Lock()
        self._nllb_lock = threading.Lock()
        self._kokoro_locks: dict = {}  # kokoro_lang_code -> Lock
        self._ready = False

    @property
    def is_ready(self):
        return self._ready

    def warmup(self):
        """Eagerly load all models at startup so the first request is fast."""
        import time
        t0 = time.time()
        logger.info("====== Broadcast model warmup starting ======")

        try:
            self.get_whisper()
        except Exception as e:
            logger.error(f"Warmup: Whisper failed: {type(e).__name__}: {e}")

        try:
            self.get_nllb()
        except Exception as e:
            logger.error(f"Warmup: NLLB failed: {type(e).__name__}: {e}")

        try:
            self.get_kokoro("en")
        except Exception as e:
            logger.error(f"Warmup: Kokoro English failed: {type(e).__name__}: {e}")

        self._ready = True
        elapsed = time.time() - t0
        logger.info(f"====== Broadcast model warmup complete in {elapsed:.1f}s ======")

    def get_whisper(self):
        if self._whisper is None:
            with self._init_lock:
                if self._whisper is None:
                    from faster_whisper import WhisperModel
                    # CTranslate2 in this image is CPU-only; use small.en for best accuracy
                    logger.info("Loading Whisper small.en on CPU...")
                    self._whisper = WhisperModel(
                        "small.en", device="cpu", compute_type="int8"
                    )
                    logger.info("Whisper small.en loaded")
        return self._whisper

    def _load_nllb(self):
        """Load NLLB model and tokenizer, trying local cache first."""
        from transformers import M2M100ForConditionalGeneration, NllbTokenizer
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = "facebook/nllb-200-distilled-600M"

        # Try loading from local cache first (fast, no network)
        for offline in (True, False):
            try:
                mode = "cache" if offline else "network"
                logger.info(f"Loading NLLB-200-distilled-600M on {device} ({mode})...")
                self._nllb_tokenizer = NllbTokenizer.from_pretrained(
                    model_name, local_files_only=offline
                )
                self._nllb_model = M2M100ForConditionalGeneration.from_pretrained(
                    model_name, local_files_only=offline
                )
                self._nllb_model.to(device)
                self._nllb_model.eval()
                self._nllb_device = device
                logger.info(f"NLLB loaded on {device} ({mode})")
                return
            except Exception as e:
                if offline:
                    logger.info(f"NLLB not in cache, trying network download...")
                else:
                    raise

    def get_nllb(self):
        if self._nllb_model is None:
            with self._init_lock:
                if self._nllb_model is None:
                    self._load_nllb()
        return self._nllb_model, self._nllb_tokenizer

    def get_kokoro(self, lang: str):
        config = KOKORO_LANG_CONFIG[lang]
        lang_code = config["lang_code"]
        if lang_code not in self._kokoro_pipelines:
            with self._init_lock:
                if lang_code not in self._kokoro_pipelines:
                    from kokoro import KPipeline
                    logger.info(f"Loading Kokoro pipeline for {LANG_NAMES.get(lang, lang)} ({lang_code})...")
                    pipe = KPipeline(lang_code=lang_code, device="cpu")
                    # Warmup
                    for _ in pipe("test", voice=config["voice"], speed=1.0):
                        pass
                    self._kokoro_pipelines[lang_code] = pipe
                    self._kokoro_locks[lang_code] = threading.Lock()
                    logger.info(f"Kokoro {lang_code} loaded")
        return self._kokoro_pipelines[lang_code], self._kokoro_locks[lang_code]


models = BroadcastModels()
executor = ThreadPoolExecutor(max_workers=8)

# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def transcribe_audio(audio_path: str) -> str:
    """Transcribe English audio using Whisper base.en."""
    whisper = models.get_whisper()
    segments, info = whisper.transcribe(
        audio_path, language="en", without_timestamps=True
    )
    text = " ".join(seg.text.strip() for seg in segments).strip()
    return text


def translate_text(text: str, target_lang: str) -> str:
    """Translate English text to target language using NLLB-200."""
    import torch

    if target_lang not in NLLB_LANG_MAP:
        raise ValueError(f"Unsupported language: {target_lang}")

    model, tokenizer = models.get_nllb()
    src_lang = "eng_Latn"
    tgt_lang = NLLB_LANG_MAP[target_lang]

    tokenizer.src_lang = src_lang
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(models._nllb_device) for k, v in inputs.items()}

    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)

    last_error = None
    for attempt in range(3):
        try:
            with models._nllb_lock:
                with torch.no_grad():
                    generated = model.generate(
                        **inputs,
                        forced_bos_token_id=forced_bos_token_id,
                        max_length=512,
                        num_beams=5,
                        early_stopping=True,
                        no_repeat_ngram_size=4,
                        repetition_penalty=1.2,
                    )
            translated = tokenizer.decode(generated[0], skip_special_tokens=True)
            return translated
        except Exception as e:
            last_error = e
            logger.warning(
                f"Translation failed ({target_lang}, attempt {attempt + 1}/3): "
                f"{type(e).__name__}: {e}"
            )
            if attempt < 2:
                import time
                time.sleep(1.0 * (2 ** attempt))

    raise last_error


def synthesize_speech(text: str, lang: str) -> bytes:
    """Generate TTS audio as WAV bytes using Kokoro."""
    from scipy.io import wavfile as wavfile_io

    if lang not in KOKORO_LANG_CONFIG:
        raise ValueError(f"Unsupported TTS language: {lang}")

    config = KOKORO_LANG_CONFIG[lang]
    pipeline, lock = models.get_kokoro(lang)

    segments = []
    try:
        with lock:
            for _gs, _ps, audio in pipeline(text, voice=config["voice"], speed=1.0):
                if audio is not None:
                    arr = np.array(audio, dtype=np.float32) if not isinstance(audio, np.ndarray) else audio.astype(np.float32)
                    segments.append(arr)
    except Exception as e:
        logger.error(f"TTS synthesis failed for {lang}: {type(e).__name__}: {e}")
        raise ValueError(f"TTS synthesis failed: {e}")

    if not segments:
        raise ValueError(f"TTS produced no audio for {lang}: {text[:80]!r}")

    full_audio = np.concatenate(segments)
    np.clip(full_audio, -1.0, 1.0, out=full_audio)
    audio_int16 = (full_audio * 32767).astype(np.int16)

    buf = io.BytesIO()
    wavfile_io.write(buf, 24000, audio_int16)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Broadcast API")


@app.on_event("startup")
async def startup_warmup():
    """Load all models eagerly so the first request doesn't pay cold-start cost."""
    import asyncio
    loop = asyncio.get_event_loop()
    loop.run_in_executor(executor, models.warmup)


class TranslateRequest(BaseModel):
    text: str
    target_lang: str


class TTSRequest(BaseModel):
    text: str
    lang: str


class ProcessRequest(BaseModel):
    text: str
    target_langs: list[str] | None = None


@app.get("/health")
async def health():
    return {
        "status": "ok" if models.is_ready else "warming_up",
        "service": "broadcast",
        "models_ready": models.is_ready,
    }


@app.get("/api/broadcast/languages")
async def get_languages():
    """Return supported languages and defaults."""
    langs = []
    for code in DEFAULT_TARGET_LANGS:
        if code == "en":
            continue
        langs.append({
            "code": code,
            "name": LANG_NAMES.get(code, code),
            "default": True,
            "has_tts": code in TTS_LANGS,
        })
    return {"languages": langs, "defaults": DEFAULT_TARGET_LANGS, "tts_langs": list(TTS_LANGS - {"en"})}


@app.post("/api/broadcast/transcribe")
async def api_transcribe(file: UploadFile = File(...)):
    """Transcribe uploaded audio to English text."""
    import asyncio

    audio_bytes = await file.read()
    if len(audio_bytes) == 0:
        raise HTTPException(400, "Empty audio file")

    # Write to temp file for faster-whisper
    suffix = os.path.splitext(file.filename or "audio.webm")[1] or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(executor, transcribe_audio, tmp_path)
    finally:
        os.unlink(tmp_path)

    if not text:
        raise HTTPException(400, "No speech detected")

    return {"text": text}


@app.post("/api/broadcast/translate")
async def api_translate(req: TranslateRequest):
    """Translate English text to a target language."""
    import asyncio

    if req.target_lang not in NLLB_LANG_MAP:
        raise HTTPException(400, f"Unsupported language: {req.target_lang}. Supported: {list(NLLB_LANG_MAP.keys())}")
    if not req.text.strip():
        raise HTTPException(400, "Empty text")

    loop = asyncio.get_event_loop()
    translated = await loop.run_in_executor(executor, translate_text, req.text, req.target_lang)
    return {"text": translated, "lang": req.target_lang}


@app.post("/api/broadcast/tts")
async def api_tts(req: TTSRequest):
    """Generate TTS audio for text in a given language."""
    import asyncio

    if req.lang not in KOKORO_LANG_CONFIG:
        raise HTTPException(400, f"Unsupported TTS language: {req.lang}")
    if not req.text.strip():
        raise HTTPException(400, "Empty text")

    loop = asyncio.get_event_loop()
    wav_bytes = await loop.run_in_executor(executor, synthesize_speech, req.text, req.lang)
    return Response(content=wav_bytes, media_type="audio/wav")


@app.post("/api/broadcast/process")
async def api_process(req: ProcessRequest):
    """Batch translate English text to multiple languages."""
    import asyncio

    if not req.text.strip():
        raise HTTPException(400, "Empty text")

    target_langs = req.target_langs or DEFAULT_TARGET_LANGS
    for lang in target_langs:
        if lang not in NLLB_LANG_MAP:
            raise HTTPException(400, f"Unsupported language: {lang}")

    loop = asyncio.get_event_loop()

    async def do_translate(lang):
        try:
            text = await loop.run_in_executor(executor, translate_text, req.text, lang)
            return {"lang": lang, "name": LANG_NAMES.get(lang, lang), "text": text}
        except Exception as e:
            return {"lang": lang, "name": LANG_NAMES.get(lang, lang), "error": str(e)}

    results = await asyncio.gather(*[do_translate(lang) for lang in target_langs])
    return {"source_text": req.text, "translations": list(results)}


if __name__ == "__main__":
    port = int(os.environ.get("BROADCAST_PORT", "8088"))
    logger.info(f"Starting Broadcast API server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
