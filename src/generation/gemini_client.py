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
        global_rpm_limit: int | None = None,
        max_cycles: int = 4,
        backoff_seconds: float = 20.0,
    ):
        keys = api_keys or _load_api_keys_from_env()
        if not keys:
            raise ValueError("Set GEMINI_API_KEY or comma-separated GEMINI_API_KEYS")
        self.api_keys = keys
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
        rpm = rpm_limit or int(os.getenv("GEMINI_RPM_LIMIT", "12"))
        self.sleep_seconds = max(sleep_seconds, 60 / max(1, rpm))
        global_rpm = global_rpm_limit or int(os.getenv("GEMINI_GLOBAL_RPM_LIMIT", "60"))
        self.global_sleep_seconds = 60 / max(1, global_rpm)
        self._last_global_request_at = 0.0
        self.max_cycles = max_cycles
        self.backoff_seconds = backoff_seconds
        self._key_cycle = itertools.cycle(self.api_keys)
        self._last_request_at = {api_key: 0.0 for api_key in self.api_keys}

    def generate(self, prompt: str) -> str:
        from google import genai

        last_error = None
        for cycle in range(self.max_cycles):
            if cycle > 0:
                time.sleep(self.backoff_seconds * cycle)
            for _ in range(len(self.api_keys)):
                api_key = next(self._key_cycle)
                try:
                    self._wait_for_rate_limit(api_key)
                    client = genai.Client(api_key=api_key)
                    response = client.models.generate_content(model=self.model, contents=prompt)
                    return response.text or ""
                except Exception as exc:
                    if not _is_retryable_error(exc):
                        raise
                    last_error = exc
        raise RuntimeError(f"Gemini generation failed for all keys: {last_error}")

    def generate_batch(self, prompts: list[str]) -> list[str]:
        return [self.generate(prompt) for prompt in prompts]

    def _wait_for_rate_limit(self, api_key: str) -> None:
        now = time.monotonic()
        wait_seconds = max(
            self.sleep_seconds - (now - self._last_request_at[api_key]),
            self.global_sleep_seconds - (now - self._last_global_request_at),
        )
        if wait_seconds > 0:
            time.sleep(wait_seconds)
        now = time.monotonic()
        self._last_request_at[api_key] = now
        self._last_global_request_at = now


def _is_retryable_error(error: Exception) -> bool:
    message = str(error).lower()
    return any(marker in message for marker in ("429", "resource_exhausted", "quota", "rate", "503", "unavailable", "500", "timeout"))


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
