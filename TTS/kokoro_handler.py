import logging
import numpy as np
from baseHandler import BaseHandler
from rich.console import Console

logger = logging.getLogger(__name__)
console = Console()

# Language code mapping from Whisper language codes to Kokoro lang codes
WHISPER_LANGUAGE_TO_KOKORO_LANG = {
    "en": "a",  # American English (to match af_heart voice)
    "ja": "j",  # Japanese
    "zh": "z",  # Chinese
    "fr": "f",  # French
    "es": "e",  # Spanish
    "it": "i",  # Italian
    "pt": "p",  # Portuguese
    "hi": "h",  # Hindi
}


class KokoroTTSHandler(BaseHandler):
    """
    Kokoro TTS handler for CUDA/CPU devices.
    Uses the native kokoro library for inference.
    """

    def setup(
        self,
        should_listen,
        model_name="hexgrad/Kokoro-82M",
        device="cuda",
        voice="bm_fable",
        lang_code="b",
        speed=1.0,
        blocksize=512,
        gen_kwargs=None,  # Unused, but passed by the pipeline
    ):
        self.should_listen = should_listen
        self.model_name = model_name
        self.device = device
        self.voice = voice
        self.lang_code = lang_code
        self.speed = speed
        self.blocksize = blocksize

        # Import kokoro library
        try:
            from kokoro import KPipeline
            self._KPipeline = KPipeline
        except ImportError:
            raise ImportError(
                "kokoro is required for Kokoro TTS. Install with: pip install kokoro>=0.9.2 soundfile\n"
                "Also ensure espeak-ng is installed: apt-get install espeak-ng (Linux) or brew install espeak-ng (macOS)"
            )

        logger.info(f"Loading Kokoro model with lang_code: {lang_code}, device: {device}")
        self.pipeline = self._KPipeline(lang_code=lang_code, device=device)
        logger.info(f"Kokoro pipeline loaded successfully")

        self.warmup()

    def warmup(self):
        """Warm up the model with a dummy inference."""
        logger.info(f"Warming up {self.__class__.__name__}")

        # Run a short dummy inference to warm up the model
        for _ in self.pipeline("Hello", voice=self.voice, speed=self.speed):
            pass

        logger.info(f"{self.__class__.__name__} warmed up")

    def _switch_language(self, new_lang_code):
        """Safely switch Kokoro pipeline language, falling back on failure."""
        if new_lang_code == self.lang_code:
            return
        try:
            logger.info(f"Switching Kokoro language: {self.lang_code} → {new_lang_code}")
            self.pipeline = self._KPipeline(lang_code=new_lang_code, device=self.device)
            self.lang_code = new_lang_code
            logger.info(f"Kokoro language switched to {new_lang_code}")
        except Exception as e:
            logger.error(
                f"Failed to switch Kokoro to lang_code={new_lang_code}: "
                f"{type(e).__name__}: {e}  — keeping {self.lang_code}"
            )

    def process(self, llm_sentence):
        """
        Process text input and generate audio output.

        Args:
            llm_sentence: Either a string or tuple of (text, language_code)

        Yields:
            Audio chunks as numpy int16 arrays
        """
        from scipy.signal import resample_poly

        language_code = None
        if isinstance(llm_sentence, tuple):
            llm_sentence, language_code = llm_sentence
            # Map Whisper language code to Kokoro language code
            new_lang_code = WHISPER_LANGUAGE_TO_KOKORO_LANG.get(
                language_code, self.lang_code
            )
            self._switch_language(new_lang_code)

        # Skip empty text
        if not llm_sentence or not llm_sentence.strip():
            logger.debug("TTS: empty text, skipping")
            self.should_listen.set()
            return

        console.print(f"[green]ASSISTANT: {llm_sentence}")

        # Accumulate all audio segments before chunking to avoid
        # clicks from zero-padding between segments
        all_segments = []
        try:
            for gs, ps, audio in self.pipeline(
                llm_sentence, voice=self.voice, speed=self.speed
            ):
                if audio is None:
                    continue

                # Ensure audio is numpy array
                if not isinstance(audio, np.ndarray):
                    audio = np.array(audio, dtype=np.float32)
                else:
                    audio = audio.astype(np.float32)

                all_segments.append(audio)
        except Exception as e:
            logger.error(f"TTS synthesis failed: {type(e).__name__}: {e}")

        if not all_segments:
            logger.warning(f"TTS produced no audio for: {llm_sentence[:80]!r}")
            self.should_listen.set()
            return

        # Concatenate all segments into one continuous stream
        full_audio = np.concatenate(all_segments)

        # Resample from 24kHz to 16kHz in a single pass (no boundary artifacts)
        full_audio = resample_poly(full_audio, up=2, down=3)

        # Clip to [-1, 1] to prevent int16 overflow distortion
        np.clip(full_audio, -1.0, 1.0, out=full_audio)

        # Convert to int16 format (use 32767 to avoid overflow at +1.0)
        full_audio = (full_audio * 32767).astype(np.int16)

        # Yield in fixed-size chunks
        for i in range(0, len(full_audio), self.blocksize):
            chunk = full_audio[i : i + self.blocksize]
            if len(chunk) < self.blocksize:
                # Apply fade-out on the final partial chunk to avoid a click
                padded = np.zeros(self.blocksize, dtype=np.int16)
                fade_len = min(64, len(chunk))
                faded = chunk.astype(np.float32)
                faded[-fade_len:] *= np.linspace(1.0, 0.0, fade_len)
                padded[: len(chunk)] = faded.astype(np.int16)
                chunk = padded
            yield chunk

        self.should_listen.set()
