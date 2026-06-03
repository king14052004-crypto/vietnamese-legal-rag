import json
import sys
from pathlib import Path
from typing import Iterator

from tqdm import tqdm

from src.data.clean_text import normalize_text
from src.data.filter_labor import is_labor_related
from src.data.load_dataset import DATASET_NAME
from src.data.schema import LegalDocument

DEFAULT_OUTPUT = "data/processed/labor_corpus.jsonl"
BATCH_SIZE = 64


def _download_dataset_file(filename: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=DATASET_NAME, filename=filename, repo_type="dataset")


def _iter_parquet_rows(parquet_file, columns: list[str] | None = None) -> Iterator[dict]:
    for batch in parquet_file.iter_batches(batch_size=BATCH_SIZE, columns=columns):
        yield from batch.to_pylist()


def _compact_document(document: LegalDocument) -> dict:
    return {
        "id": document.id,
        "title": document.title,
        "so_ky_hieu": document.so_ky_hieu,
        "ngay_ban_hanh": document.ngay_ban_hanh,
        "loai_van_ban": document.loai_van_ban,
        "ngay_co_hieu_luc": document.ngay_co_hieu_luc,
        "ngay_het_hieu_luc": document.ngay_het_hieu_luc,
        "nganh": document.nganh,
        "linh_vuc": document.linh_vuc,
        "co_quan_ban_hanh": document.co_quan_ban_hanh,
        "pham_vi": document.pham_vi,
        "tinh_trang_hieu_luc": document.tinh_trang_hieu_luc,
        "content_text": document.content_text,
        "source_url": document.source_url,
    }


def build_labor_corpus(output_path: str = DEFAULT_OUTPUT) -> int:
    import pyarrow.parquet as pq

    metadata = pq.ParquetFile(_download_dataset_file("data/metadata.parquet"))
    content = pq.ParquetFile(_download_dataset_file("data/content.parquet"))
    metadata_by_id = {str(row["id"]): row for row in _iter_parquet_rows(metadata)}

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    seen_ids = set()
    written = 0
    with path.open("w", encoding="utf-8") as f:
        content_rows = _iter_parquet_rows(content, columns=["id", "content_html"])
        for content_row in tqdm(content_rows, desc="filter labor docs", total=content.metadata.num_rows):
            doc_id = str(content_row["id"])
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)

            metadata_row = metadata_by_id.get(doc_id)
            if metadata_row is None:
                continue

            document_data = dict(metadata_row)
            document_data["id"] = doc_id
            document_data["content_text"] = normalize_text(content_row.get("content_html", ""))
            document = LegalDocument(**document_data)
            if is_labor_related(document):
                f.write(json.dumps(_compact_document(document), ensure_ascii=False) + "\n")
                written += 1
    return written


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    count = build_labor_corpus()
    print(f"wrote {count} labor documents to {DEFAULT_OUTPUT}")


if __name__ == "__main__":
    main()
