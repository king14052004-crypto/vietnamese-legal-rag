import argparse
import json
from pathlib import Path

from tqdm import tqdm

from src.data.clean_text import normalize_text
from src.data.filter_labor import is_labor_related
from src.data.load_dataset import DATASET_NAME
from src.data.schema import LegalDocument


def _score_document(document: LegalDocument) -> int:
    text = " ".join(
        [
            document.title or "",
            document.nganh or "",
            document.linh_vuc or "",
            normalize_text(document.content_html or "")[:4000],
        ]
    ).lower()
    phrases = [
        "bộ luật lao động",
        "hợp đồng lao động",
        "người lao động",
        "người sử dụng lao động",
        "bảo hiểm thất nghiệp",
        "trợ cấp thôi việc",
        "trợ cấp mất việc",
        "kỷ luật lao động",
        "tranh chấp lao động",
        "an toàn vệ sinh lao động",
        "lương tối thiểu",
        "giấy phép lao động",
    ]
    return sum(3 if phrase in (document.title or "").lower() else 1 for phrase in phrases if phrase in text)


def build_labor_sample(output_path: str, max_docs: int = 500, scan_limit: int = 30000) -> list[LegalDocument]:
    from datasets import load_dataset

    metadata = load_dataset(DATASET_NAME, "metadata", split=f"data[:{scan_limit}]")
    content = load_dataset(DATASET_NAME, "content", split=f"data[:{scan_limit}]")
    content_map = {str(row["id"]): row.get("content_html", "") for row in content}

    candidates = []
    for row in tqdm(metadata, desc="filter labor docs"):
        doc_id = str(row["id"])
        document = LegalDocument(
            **dict(row),
            id=doc_id,
            content_html=content_map.get(doc_id, ""),
            content_text=normalize_text(content_map.get(doc_id, "")),
            raw_metadata=dict(row),
        )
        if is_labor_related(document):
            candidates.append((_score_document(document), document))

    candidates.sort(key=lambda item: item[0], reverse=True)
    documents = [document for score, document in candidates[:max_docs] if score > 0]

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for document in documents:
            f.write(json.dumps(document.to_dict(), ensure_ascii=False) + "\n")
    return documents


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/processed/labor_corpus_sample.jsonl")
    parser.add_argument("--max-docs", type=int, default=500)
    parser.add_argument("--scan-limit", type=int, default=30000)
    args = parser.parse_args()
    documents = build_labor_sample(args.output, args.max_docs, args.scan_limit)
    print(f"wrote {len(documents)} labor documents to {args.output}")
    for document in documents[:5]:
        print("-", document.title)


if __name__ == "__main__":
    main()
