import re

from src.data.clean_text import normalize_text
from src.data.schema import LegalChunk, LegalDocument


ARTICLE_PATTERN = re.compile(r"(?=(?:Điều|ĐIỀU)\s+\d+[a-zA-Z]?\.)")


def split_by_article_or_window(text: str, max_words: int = 260, overlap: int = 40) -> list[tuple[str | None, str]]:
    text = normalize_text(text)
    if not text:
        return []

    parts = [part.strip() for part in ARTICLE_PATTERN.split(text) if part.strip()]
    if len(parts) > 1:
        chunks = []
        for part in parts:
            article_match = re.match(r"(Điều\s+\d+[a-zA-Z]?)\.", part, flags=re.IGNORECASE)
            chunks.append((article_match.group(1) if article_match else None, part))
        return chunks

    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunk_text = " ".join(words[start:end])
        chunks.append((None, chunk_text))
        if end == len(words):
            break
        start = max(0, end - overlap)
    return chunks


def chunk_document(document: LegalDocument, max_words: int = 260, overlap: int = 40) -> list[LegalChunk]:
    source_text = document.content_text or document.content_html or ""
    chunks = []
    for idx, (article, text) in enumerate(split_by_article_or_window(source_text, max_words, overlap)):
        if len(text.split()) < 30:
            continue
        chunks.append(
            LegalChunk(
                chunk_id=f"{document.id}_{idx}",
                doc_id=document.id,
                title=document.title or "",
                text=text,
                article=article,
                source_url=document.source_url,
                doc_type=document.loai_van_ban,
                effective_date=document.ngay_co_hieu_luc,
                metadata={
                    "so_ky_hieu": document.so_ky_hieu,
                    "agency": document.co_quan_ban_hanh,
                    "status": document.tinh_trang_hieu_luc,
                },
            )
        )
    return chunks


def chunk_documents(documents: list[LegalDocument], max_words: int = 260, overlap: int = 40) -> list[LegalChunk]:
    chunks = []
    for document in documents:
        chunks.extend(chunk_document(document, max_words, overlap))
    return chunks
