import torchaudio
from VAD.vad_iterator import VADIterator
from baseHandler import BaseHandler
import numpy as np
import torch
from rich.console import Console

from utils.utils import int2float
import logging

logger = logging.getLogger(__name__)

console = Console()


class VADHandler(BaseHandler):
    """
    Handles voice activity detection. When voice activity is detected, audio will be accumulated until the end of speech is detected and then passed
    to the following part.
    """

    def setup(
        self,
        should_listen,
        thresh=0.3,
        sample_rate=16000,
        min_silence_ms=1000,
        min_speech_ms=500,
        max_speech_ms=float("inf"),
        speech_pad_ms=30,
        audio_enhancement=False,
    ):
        self.should_listen = should_listen
        self.sample_rate = sample_rate
        self.min_silence_ms = min_silence_ms
        self.min_speech_ms = min_speech_ms
        self.max_speech_ms = max_speech_ms
        self.model, _ = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
        self.iterator = VADIterator(
            self.model,
            threshold=thresh,
            sampling_rate=sample_rate,
            min_silence_duration_ms=min_silence_ms,
            speech_pad_ms=speech_pad_ms,
        )
        self.audio_enhancement = audio_enhancement
        if audio_enhancement:
            from df.enhance import init_df
            self.enhanced_model, self.df_state, _ = init_df()
        self._chunk_count = 0

    def process(self, audio_chunk):
        audio_int16 = np.frombuffer(audio_chunk, dtype=np.int16)
        audio_float32 = int2float(audio_int16)

        # Log audio levels periodically for diagnostics
        self._chunk_count += 1
        if self._chunk_count % 100 == 1:
            rms = np.sqrt(np.mean(audio_float32 ** 2))
            peak = np.max(np.abs(audio_float32))
            logger.info(f"VAD chunk #{self._chunk_count}: rms={rms:.6f}, peak={peak:.6f}, samples={len(audio_float32)}")

        vad_output = self.iterator(torch.from_numpy(audio_float32))
        if vad_output is not None and len(vad_output) != 0:
            logger.info("VAD: end of speech detected")
            array = torch.cat(vad_output).cpu().numpy()
            duration_ms = len(array) / self.sample_rate * 1000
            if duration_ms < self.min_speech_ms or duration_ms > self.max_speech_ms:
                logger.info(
                    f"VAD: audio input of duration: {len(array) / self.sample_rate:.2f}s, skipping (min={self.min_speech_ms}ms)"
                )
            else:
                logger.info(f"VAD: speech segment {duration_ms:.0f}ms, forwarding to STT")
                self.should_listen.clear()
                if self.audio_enhancement:
                    from df.enhance import enhance
                    if self.sample_rate != self.df_state.sr():
                        audio_float32 = torchaudio.functional.resample(
                            torch.from_numpy(array),
                            orig_freq=self.sample_rate,
                            new_freq=self.df_state.sr(),
                        )
                        enhanced = enhance(
                            self.enhanced_model,
                            self.df_state,
                            audio_float32.unsqueeze(0),
                        )
                        enhanced = torchaudio.functional.resample(
                            enhanced,
                            orig_freq=self.df_state.sr(),
                            new_freq=self.sample_rate,
                        )
                    else:
                        enhanced = enhance(
                            self.enhanced_model, self.df_state, audio_float32
                        )
                    array = enhanced.numpy().squeeze()
                yield array

    @property
    def min_time_to_debug(self):
        return 0.00001
