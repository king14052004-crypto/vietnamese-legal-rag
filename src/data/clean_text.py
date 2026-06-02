import html
import re


def normalize_text(text: str | None) -> str:
    text = html.unescape(text or "")
    text = re.sub(r"<script.*?</script>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_for_match(text: str | None) -> str:
    return normalize_text(text).lower()


def tokenize_vi(text: str | None) -> list[str]:
    return re.findall(r"[\wÀ-ỹ]+", normalize_for_match(text))
