"""Build a small labor-law sample corpus that is committed to Git.

The full Gemini-filtered corpus (~14k docs) is too large for Git, so this script
creates `data/processed/labor_corpus_sample.jsonl`: a few hundred labor-law
documents that let anyone clone the repo and run the demo immediately.

Selection is intentionally simple and transparent:
1. Keep documents whose `linh_vuc`/`nganh` metadata mentions labor topics.
2. Prefer substantive document types (Luat, Nghi dinh, Thong tu) and documents
   that still have legal effect.
3. Keep documents with at least 3 "Dieu ..." article headings so chunking by
   article works well.

Usage:
    python scripts/make_sample_corpus.py
"""

import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.clean_text import normalize_text  # noqa: E402
from src.data.chunking import ARTICLE_PATTERN  # noqa: E402
from src.data.load_dataset import DATASET_NAME  # noqa: E402
from src.data.schema import LegalDocument  # noqa: E402

OUTPUT_PATH = PROJECT_ROOT / "data/processed/labor_corpus_sample.jsonl"
MAX_DOCUMENTS = 300
MIN_ARTICLE_HEADINGS = 3
MIN_WORDS = 200
MAX_WORDS = 6000

LABOR_TOPICS = ("lao động", "việc làm", "tiền lương", "bảo hiểm xã hội")
PREFERRED_DOC_TYPES = ("Luật", "Pháp lệnh", "Nghị định", "Thông tư", "Thông tư liên tịch")


def is_labor_document(meta: dict) -> bool:
    fields = f"{meta.get('linh_vuc') or ''} {meta.get('nganh') or ''} {meta.get('title') or ''}"
    return any(topic in fields.lower() for topic in LABOR_TOPICS)


def selection_priority(meta: dict) -> tuple:
    doc_type = meta.get("loai_van_ban") or ""
    in_effect = meta.get("tinh_trang_hieu_luc") == "Còn hiệu lực"
    type_rank = PREFERRED_DOC_TYPES.index(doc_type) if doc_type in PREFERRED_DOC_TYPES else len(PREFERRED_DOC_TYPES)
    return (0 if in_effect else 1, type_rank, meta.get("id") or "")


def main() -> None:
    metadata_path = hf_hub_download(DATASET_NAME, "data/metadata.parquet", repo_type="dataset")
    content_path = hf_hub_download(DATASET_NAME, "data/content.parquet", repo_type="dataset")

    metadata = pq.read_table(metadata_path).to_pylist()
    candidates = {str(meta["id"]): meta for meta in metadata if is_labor_document(meta)}
    print(f"Labor-law candidates from metadata: {len(candidates)}")

    documents = []
    content_file = pq.ParquetFile(content_path)
    for batch in content_file.iter_batches(batch_size=2000, columns=["id", "content_html"]):
        for doc_id, content_html in zip(batch["id"].to_pylist(), batch["content_html"].to_pylist()):
            meta = candidates.get(str(doc_id))
            if meta is None:
                continue
            text = normalize_text(content_html)
            word_count = len(text.split())
            article_headings = len(ARTICLE_PATTERN.findall(text))
            if word_count < MIN_WORDS or word_count > MAX_WORDS:
                continue
            if article_headings < MIN_ARTICLE_HEADINGS:
                continue
            documents.append((meta, text))

    documents.sort(key=lambda item: selection_priority(item[0]))
    documents = documents[:MAX_DOCUMENTS]
    print(f"Selected documents: {len(documents)}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for meta, text in documents:
            field_names = {field for field in LegalDocument.__dataclass_fields__ if field != "raw_metadata"}
            row = {key: (None if value == "None" else value) for key, value in meta.items() if key in field_names}
            row["id"] = str(meta["id"])
            row["content_text"] = text
            row["source_url"] = f"https://vbpl.vn/TW/Pages/vbpq-toanvan.aspx?ItemID={meta['id']}"
            document = LegalDocument(**row)
            f.write(json.dumps(document.to_dict(), ensure_ascii=False) + "\n")

    size_mb = OUTPUT_PATH.stat().st_size / 1_000_000
    print(f"Wrote {OUTPUT_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
