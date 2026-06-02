from datasets import load_dataset

from src.data.schema import LegalDocument

DATASET_NAME = "th1nhng0/vietnamese-legal-documents"


def load_vietnamese_legal_metadata(limit: int | None = None) -> list[dict]:
    meta = load_dataset(DATASET_NAME, "metadata", split="data")
    if limit:
        meta = meta.select(list(range(min(limit, len(meta)))))
    return [dict(row) for row in meta]


def load_vietnamese_legal_content(limit: int | None = None) -> list[dict]:
    content = load_dataset(DATASET_NAME, "content", split="data")
    if limit:
        content = content.select(list(range(min(limit, len(content)))))
    return [dict(row) for row in content]


def merge_metadata_content(metadata_rows, content_rows) -> list[LegalDocument]:
    content_map = {str(row["id"]): row.get("content_html", " ") for row in content_rows}
    legal_documents = []
    for meta in metadata_rows:
        doc_id = str(meta["id"])
        if doc_id in content_map:
            doc_data = {**meta, "content_html": content_map[doc_id], "raw_metadata": dict(meta)}
            doc_data["id"] = doc_id
            legal_documents.append(LegalDocument(**doc_data))
    return legal_documents


def load_documents_from_jsonl(path: str) -> list[LegalDocument]:
    import json

    documents = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                documents.append(LegalDocument(**row))
    return documents