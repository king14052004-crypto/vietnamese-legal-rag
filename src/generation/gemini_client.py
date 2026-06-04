import itertools
import os
import time


class BatchGeminiClient:
    def __init__(
        self,
        api_keys: list[str] | None = None,
        model: str | None = None,
        sleep_seconds: float = 0.2,
        rpm_limit: int | None = None,
    ):
        keys = api_keys or _load_api_keys_from_env()
        if not keys:
            raise ValueError("Set GEMINI_API_KEY or comma-separated GEMINI_API_KEYS")
        self.api_keys = keys
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
        rpm = rpm_limit or int(os.getenv("GEMINI_RPM_LIMIT", "12"))
        self.sleep_seconds = max(sleep_seconds, 60 / max(1, rpm))
        self._key_cycle = itertools.cycle(self.api_keys)
        self._last_request_at = {api_key: 0.0 for api_key in self.api_keys}

    def generate(self, prompt: str) -> str:
        from google import genai

        last_error = None
        for _ in range(len(self.api_keys)):
            api_key = next(self._key_cycle)
            try:
                self._wait_for_rate_limit(api_key)
                client = genai.Client(api_key=api_key)
                response = client.models.generate_content(model=self.model, contents=prompt)
                return response.text or ""
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"Gemini generation failed for all keys: {last_error}")

    def generate_batch(self, prompts: list[str]) -> list[str]:
        return [self.generate(prompt) for prompt in prompts]

    def _wait_for_rate_limit(self, api_key: str) -> None:
        elapsed = time.monotonic() - self._last_request_at[api_key]
        wait_seconds = self.sleep_seconds - elapsed
        if wait_seconds > 0:
            time.sleep(wait_seconds)
        self._last_request_at[api_key] = time.monotonic()


def _load_api_keys_from_env() -> list[str]:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))
    raw_values = [
        os.getenv("GOOGLE_API_KEY") or "",
        os.getenv("GEMINI_API_KEY") or "",
        os.getenv("GEMINI_API_KEYS") or "",
    ]
    keys = []
    seen = set()
    for raw_value in raw_values:
        for key in raw_value.split(","):
            key = key.strip()
            if key and key not in seen:
                keys.append(key)
                seen.add(key)
    return keys
