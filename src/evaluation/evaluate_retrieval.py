import argparse
import json
from pathlib import Path

from src.data.chunking import chunk_documents
from src.data.load_dataset import load_documents_from_jsonl
from src.evaluation.metrics import citation_coverage, ndcg_at_k, recall_at_k, reciprocal_rank
from src.evaluation.test_queries import LABOR_TEST_QUERIES
from src.retrieval.pipeline import RetrievalPipeline


def evaluate_methods(
    corpus_path: str,
    methods: list[str] | None = None,
    top_k: int = 8,
    use_tfidf_fallback: bool = False,
) -> dict:
    methods = methods or ["bm25", "vector", "hybrid", "hybrid_rrf", "hybrid_rrf_mmr"]
    documents = load_documents_from_jsonl(corpus_path)
    chunks = chunk_documents(documents)
    retriever = RetrievalPipeline(chunks, use_tfidf_fallback=use_tfidf_fallback)

    report = {"num_documents": len(documents), "num_chunks": len(chunks), "methods": {}}
    for method in methods:
        rows = []
        for query in LABOR_TEST_QUERIES:
            results = retriever.retrieve(query["question"], method=method, top_k=top_k)
            rows.append(
                {
                    "question": query["question"],
                    "recall@5": recall_at_k(results, query["relevant_terms"], min(5, top_k)),
                    "mrr": reciprocal_rank(results, query["relevant_terms"]),
                    "ndcg@5": ndcg_at_k(results, query["relevant_terms"], min(5, top_k)),
                    "citation_coverage": citation_coverage(results),
                    "top_title": results[0].chunk.title if results else None,
                }
            )
        report["methods"][method] = {
            "recall@5": sum(row["recall@5"] for row in rows) / len(rows),
            "mrr": sum(row["mrr"] for row in rows) / len(rows),
            "ndcg@5": sum(row["ndcg@5"] for row in rows) / len(rows),
            "citation_coverage": sum(row["citation_coverage"] for row in rows) / len(rows),
            "details": rows,
        }
    return report


def write_markdown_report(report: dict, output_path: str) -> None:
    lines = [
        "# Retrieval Evaluation",
        "",
        f"- Documents: {report['num_documents']}",
        f"- Chunks: {report['num_chunks']}",
        "",
        "| Method | Recall@5 | MRR | nDCG@5 | Citation coverage |",
        "|---|---:|---:|---:|---:|",
    ]
    best_method = None
    best_score = -1.0
    for method, metrics in report["methods"].items():
        balance_score = 0.45 * metrics["recall@5"] + 0.35 * metrics["mrr"] + 0.20 * metrics["ndcg@5"]
        if balance_score > best_score:
            best_method = method
            best_score = balance_score
        lines.append(
            f"| {method} | {metrics['recall@5']:.3f} | {metrics['mrr']:.3f} | "
            f"{metrics['ndcg@5']:.3f} | {metrics['citation_coverage']:.3f} |"
        )
    lines.extend(["", f"Recommended balanced method: `{best_method}`.", ""])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/processed/labor_corpus_sample.jsonl")
    parser.add_argument("--output-json", default="reports/retrieval_evaluation.json")
    parser.add_argument("--output-md", default="reports/evaluation.md")
    parser.add_argument("--tfidf-fallback", action="store_true")
    args = parser.parse_args()
    report = evaluate_methods(args.corpus, use_tfidf_fallback=args.tfidf_fallback)
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown_report(report, args.output_md)
    print(json.dumps({k: v for k, v in report.items() if k != "methods"}, ensure_ascii=False))
    for method, metrics in report["methods"].items():
        print(method, {k: round(v, 3) for k, v in metrics.items() if k != "details"})


if __name__ == "__main__":
    main()
