import os

from dotenv import load_dotenv

from src.evaluation.config import PROJECT_ROOT


def load_gemini_env() -> None:
    load_dotenv(PROJECT_ROOT / ".env")


def gemini_keys() -> list[str]:
    load_gemini_env()
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


def has_gemini_key() -> bool:
    return bool(gemini_keys())


def first_gemini_key() -> str | None:
    return next(iter(gemini_keys()), None)


def ensure_google_api_key() -> str:
    api_key = first_gemini_key()
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY, GEMINI_API_KEYS, or GOOGLE_API_KEY")
    os.environ.setdefault("GOOGLE_API_KEY", api_key)
    return api_key
