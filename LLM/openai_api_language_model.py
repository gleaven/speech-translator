import logging
import time

from nltk import sent_tokenize
from rich.console import Console
from openai import OpenAI

from baseHandler import BaseHandler
from LLM.chat import Chat

logger = logging.getLogger(__name__)

console = Console()

WHISPER_LANGUAGE_TO_LLM_LANGUAGE = {
    "en": "english",
    "fr": "french",
    "es": "spanish",
    "zh": "chinese",
    "ja": "japanese",
    "ko": "korean",
    "de": "german",
    "pt": "portuguese",
    "it": "italian",
    "nl": "dutch",
    "pl": "polish",
    "hi": "hindi",
    "ru": "russian",
    "ar": "arabic",
    "tr": "turkish",
    "sv": "swedish",
    "da": "danish",
    "no": "norwegian",
    "fi": "finnish",
    "el": "greek",
    "cs": "czech",
    "ro": "romanian",
    "hu": "hungarian",
    "uk": "ukrainian",
    "vi": "vietnamese",
    "th": "thai",
    "id": "indonesian",
    "ms": "malay",
    "he": "hebrew",
    "fa": "persian",
}

# Retry configuration
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0  # seconds, doubles each attempt


class OpenApiModelHandler(BaseHandler):
    """
    Handles the language model part.
    """
    def setup(
        self,
        model_name="deepseek-chat",
        device="cuda",
        gen_kwargs={},
        base_url =None,
        api_key=None,
        stream=False,
        user_role="user",
        chat_size=1,
        init_chat_role="system",
        init_chat_prompt="You are a helpful AI assistant.",
    ):
        self.model_name = model_name
        self.stream = stream
        self.chat = Chat(chat_size)
        if init_chat_role:
            if not init_chat_prompt:
                raise ValueError(
                    "An initial promt needs to be specified when setting init_chat_role."
                )
            self.chat.init_chat({"role": init_chat_role, "content": init_chat_prompt})
        self.user_role = user_role
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.warmup()

    def warmup(self):
        logger.info(f"Warming up {self.__class__.__name__}")
        start = time.time()
        response = self._call_with_retry(
            [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Hello"},
            ]
        )
        end = time.time()
        logger.info(
            f"{self.__class__.__name__}:  warmed up! time: {(end - start):.3f} s"
        )

    def _call_with_retry(self, messages):
        """Call Ollama/OpenAI with retry + exponential backoff."""
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                return self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    stream=self.stream,
                )
            except Exception as e:
                last_error = e
                wait = RETRY_BACKOFF * (2 ** attempt)
                logger.warning(
                    f"LLM call failed (attempt {attempt + 1}/{MAX_RETRIES}): "
                    f"{type(e).__name__}: {e}"
                )
                if attempt < MAX_RETRIES - 1:
                    logger.info(f"Retrying in {wait:.0f}s...")
                    time.sleep(wait)
        raise last_error

    def process(self, prompt):
            logger.debug("call api language model...")

            language_code = None
            if isinstance(prompt, tuple):
                prompt, language_code = prompt
                if language_code[-5:] == "-auto":
                    language_code = language_code[:-5]

            self.chat.append({"role": self.user_role, "content": prompt})

            response = self._call_with_retry(self.chat.to_list())

            if self.stream:
                generated_text, printable_text = "", ""
                for chunk in response:
                    new_text = chunk.choices[0].delta.content or ""
                    generated_text += new_text
                    printable_text += new_text
                    sentences = sent_tokenize(printable_text)
                    if len(sentences) > 1:
                        yield sentences[0], language_code
                        printable_text = new_text
                self.chat.append({"role": "assistant", "content": generated_text})
                # don't forget last sentence
                yield printable_text, language_code
            else:
                generated_text = response.choices[0].message.content
                self.chat.append({"role": "assistant", "content": generated_text})
                # Write transcript for web UI
                try:
                    import json, time as _time
                    with open("/shared/transcripts.jsonl", "a") as f:
                        f.write(json.dumps({"type": "assistant", "text": generated_text, "ts": _time.time()}) + "\n")
                        f.flush()
                except Exception:
                    pass
                yield generated_text, language_code
