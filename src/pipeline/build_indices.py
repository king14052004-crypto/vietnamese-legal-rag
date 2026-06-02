import argparse
import json
from pathlib import Path

from src.data.chunking import chunk_documents
from src.data.load_dataset import load_documents_from_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/processed/labor_corpus_sample.jsonl")
    parser.add_argument("--chunks", default="data/processed/labor_chunks.jsonl")
    args = parser.parse_args()

    documents = load_documents_from_jsonl(args.corpus)
    chunks = chunk_documents(documents)
    output = Path(args.chunks)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
    print(f"wrote {len(chunks)} chunks to {args.chunks}")


if __name__ == "__main__":
    main()
