"""Evaluate the 6 retrieval methods on the gold-labeled benchmark.

Metrics (all exact, based on the gold chunk each question was generated from):
- recall@5      : 1 if the gold chunk is in the top 5 results.
- doc_recall@5  : 1 if any chunk of the gold document is in the top 5.
- mrr           : 1 / rank of the gold chunk (0 if absent).
- ndcg@5        : binary-gain nDCG of the gold chunk position.

Writes `reports/retrieval_evaluation.json` and `reports/retrieval_evaluation.md`.

Usage:
    python scripts/run_retrieval_eval.py
"""

import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.chunking import chunk_documents  # noqa: E402
from src.data.load_dataset import load_documents_from_jsonl  # noqa: E402
from src.retrieval.pipeline import RETRIEVAL_METHODS, RetrievalPipeline  # noqa: E402

CORPUS_PATH = PROJECT_ROOT / "data/processed/labor_corpus_sample.jsonl"
BENCHMARK_PATH = PROJECT_ROOT / "reports/benchmark_questions.json"
OUTPUT_JSON = PROJECT_ROOT / "reports/retrieval_evaluation.json"
OUTPUT_MD = PROJECT_ROOT / "reports/retrieval_evaluation.md"
CACHE_DIR = PROJECT_ROOT / ".cache"
TOP_K = 5


def question_metrics(results, gold_chunk_id: str, gold_doc_id: str) -> dict:
    chunk_ids = [result.chunk.chunk_id for result in results]
    doc_ids = [result.chunk.doc_id for result in results]
    rank = chunk_ids.index(gold_chunk_id) + 1 if gold_chunk_id in chunk_ids else None
    return {
        "recall@5": 1.0 if rank else 0.0,
        "doc_recall@5": 1.0 if gold_doc_id in doc_ids else 0.0,
        "mrr": 1.0 / rank if rank else 0.0,
        "ndcg@5": 1.0 / math.log2(rank + 1) if rank else 0.0,
    }


def retrieval_score(row: dict) -> float:
    return 0.4 * row["recall@5"] + 0.3 * row["mrr"] + 0.2 * row["ndcg@5"] + 0.1 * row["doc_recall@5"]


def main() -> None:
    benchmark = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    items = benchmark["items"]
    documents = load_documents_from_jsonl(CORPUS_PATH)
    chunks = chunk_documents(documents)
    pipeline = RetrievalPipeline(chunks, cache_dir=CACHE_DIR)
    print(f"{len(items)} questions, {len(chunks)} chunks")

    details = []
    summary = []
    for method in RETRIEVAL_METHODS:
        rows = []
        for item in items:
            results = pipeline.retrieve(item["question"], top_k=TOP_K, method=method)
            metrics = question_metrics(results, item["gold_chunk_id"], item["gold_doc_id"])
            rows.append(metrics)
            details.append({
                "question_id": item["id"],
                "method": method,
                **metrics,
                "retrieved_chunk_ids": [result.chunk.chunk_id for result in results],
            })
        aggregate = {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}
        aggregate["method"] = method
        aggregate["retrieval_score"] = retrieval_score(aggregate)
        summary.append(aggregate)
        print(f"{method:35s} recall@5={aggregate['recall@5']:.3f} mrr={aggregate['mrr']:.3f} score={aggregate['retrieval_score']:.3f}")

    summary.sort(key=lambda row: row["retrieval_score"], reverse=True)
    payload = {
        "benchmark": BENCHMARK_PATH.name,
        "questions": len(items),
        "chunks": len(chunks),
        "top_k": TOP_K,
        "score_formula": "0.4*recall@5 + 0.3*mrr + 0.2*ndcg@5 + 0.1*doc_recall@5",
        "summary": summary,
        "details": details,
    }
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Retrieval Evaluation",
        "",
        f"- Benchmark: `{BENCHMARK_PATH.name}` ({len(items)} paraphrased questions with gold chunk labels)",
        f"- Corpus: `{CORPUS_PATH.name}` ({len(chunks)} chunks)",
        f"- Metrics are exact: a hit means the gold chunk (the chunk the question was generated from) appears in the top {TOP_K}.",
        f"- Ranking score: `{payload['score_formula']}`",
        "",
        "| Method | Recall@5 | MRR | nDCG@5 | Doc Recall@5 | Score |",
        "|---|---|---|---|---|---|",
    ]
    for row in summary:
        lines.append(
            f"| {row['method']} | {row['recall@5']:.3f} | {row['mrr']:.3f} | {row['ndcg@5']:.3f} "
            f"| {row['doc_recall@5']:.3f} | {row['retrieval_score']:.3f} |"
        )
    lines += [
        "",
        f"Best method by retrieval score: **{summary[0]['method']}**.",
        "",
        "Note: questions are paraphrased away from the legal wording, so keyword-only",
        "retrieval is expected to miss some questions and the metrics can separate methods.",
    ]
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_JSON} and {OUTPUT_MD}")


if __name__ == "__main__":
    main()
