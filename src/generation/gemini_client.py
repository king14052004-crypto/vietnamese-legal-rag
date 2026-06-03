import itertools
import os
import time


class BatchGeminiClient:
    def __init__(
        self,
        api_keys: list[str] | None = None,
        model: str | None = None,
        sleep_seconds: float = 0.2,
    ):
        keys = api_keys or _load_api_keys_from_env()
        if not keys:
            raise ValueError("Set GEMINI_API_KEY or comma-separated GEMINI_API_KEYS")
        self.api_keys = keys
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
        self.sleep_seconds = sleep_seconds
        self._key_cycle = itertools.cycle(self.api_keys)

    def generate(self, prompt: str) -> str:
        from google import genai

        last_error = None
        for _ in range(len(self.api_keys)):
            api_key = next(self._key_cycle)
            try:
                client = genai.Client(api_key=api_key)
                response = client.models.generate_content(model=self.model, contents=prompt)
                time.sleep(self.sleep_seconds)
                return response.text or ""
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"Gemini generation failed for all keys: {last_error}")

    def generate_batch(self, prompts: list[str]) -> list[str]:
        return [self.generate(prompt) for prompt in prompts]


def _load_api_keys_from_env() -> list[str]:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))
    keys = os.getenv("GEMINI_API_KEYS") or os.getenv("GEMINI_API_KEY") or ""
    return [key.strip() for key in keys.split(",") if key.strip()]
