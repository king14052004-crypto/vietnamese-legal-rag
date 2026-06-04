import json
from collections import defaultdict
from math import isfinite, log2

import faiss
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data.chunking import chunk_documents
from src.data.clean_text import normalize_for_match, tokenize_vi
from src.data.load_dataset import load_documents_from_jsonl
from src.data.schema import LegalChunk, SearchResult
from src.evaluation.config import BENCHMARK_MAX_CHUNKS, CORPUS_PATH, FIGURES_DIR, RETRIEVAL_JSON, RETRIEVAL_MD, TOP_K
from src.evaluation.ragas_runtime import retrieval_metrics
from src.evaluation.testset import ensure_testset

METHODS = ["bm25", "vector", "hybrid", "hybrid_rrf", "hybrid_rrf_mmr"]
STOPWORDS = {
    "cua",
    "cho",
    "cac",
    "mot",
    "nhung",
    "theo",
    "quy",
    "dinh",
    "duoc",
    "khong",
    "trong",
    "voi",
    "khi",
    "thi",
    "la",
    "ve",
    "va",
}


def chunk_text(chunk: LegalChunk) -> str:
    return f"{chunk.title} {chunk.text}"


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
    return np.asarray(matrix / norms, dtype="float32")


def build_benchmark_chunks(testset: list[dict], max_chunks: int = BENCHMARK_MAX_CHUNKS) -> list[LegalChunk]:
    documents = load_documents_from_jsonl(str(CORPUS_PATH))
    all_chunks = chunk_documents(documents)
    terms = sorted({term for item in testset for term in item.get("expected_terms", []) if term})
    selected: list[LegalChunk] = []
    seen_ids = set()

    for chunk in all_chunks:
        text = normalize_for_match(chunk_text(chunk))
        if terms and not any(normalize_for_match(term) in text for term in terms):
            continue
        if chunk.chunk_id in seen_ids:
            continue
        seen_ids.add(chunk.chunk_id)
        selected.append(chunk)
        if len(selected) >= max_chunks:
            break

    if len(selected) < 200:
        for chunk in all_chunks:
            if chunk.chunk_id in seen_ids:
                continue
            seen_ids.add(chunk.chunk_id)
            selected.append(chunk)
            if len(selected) >= min(max_chunks, 1000):
                break
    return selected


class BenchmarkRetriever:
    def __init__(self, chunks: list[LegalChunk]):
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize_vi(chunk_text(chunk)) for chunk in chunks])
        passages = [f"passage: {chunk_text(chunk)}" for chunk in chunks]
        self.vectorizer = TfidfVectorizer(max_features=384, ngram_range=(1, 2))
        self.embeddings = normalize_rows(self.vectorizer.fit_transform(passages).toarray())
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def retrieve(self, query: str, method: str, top_k: int = TOP_K) -> list[SearchResult]:
        sparse = self.retrieve_bm25(query, max(30, top_k * 5))
        vector = self.retrieve_vector(query, max(30, top_k * 5))
        if method == "bm25":
            return sparse[:top_k]
        if method == "vector":
            return vector[:top_k]
        if method == "hybrid":
            return self.weighted_hybrid(sparse, vector, top_k)
        if method == "hybrid_rrf":
            return self.rrf([sparse, vector], top_k)
        if method == "hybrid_rrf_mmr":
            return self.mmr(self.rrf([sparse, vector], max(20, top_k * 4)), top_k)
        raise ValueError(f"Unknown retrieval method: {method}")

    def retrieve_bm25(self, query: str, top_k: int) -> list[SearchResult]:
        scores = self.bm25.get_scores(tokenize_vi(query))
        indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:top_k]
        return [
            SearchResult(self.chunks[idx], float(scores[idx]), "bm25", rank)
            for rank, idx in enumerate(indices, start=1)
        ]

    def retrieve_vector(self, query: str, top_k: int) -> list[SearchResult]:
        query_vector = normalize_rows(self.vectorizer.transform([query]).toarray())
        scores, indices = self.index.search(query_vector, top_k)
        return [
            SearchResult(self.chunks[int(idx)], float(score), "vector", rank)
            for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1)
            if idx >= 0
        ]

    @staticmethod
    def _minmax(results: list[SearchResult]) -> dict[str, float]:
        scores = [result.score for result in results]
        low, high = min(scores), max(scores)
        if high == low:
            return {result.chunk.chunk_id: 1.0 for result in results}
        return {result.chunk.chunk_id: (result.score - low) / (high - low) for result in results}

    def weighted_hybrid(self, sparse: list[SearchResult], vector: list[SearchResult], top_k: int) -> list[SearchResult]:
        sparse_scores = self._minmax(sparse)
        vector_scores = self._minmax(vector)
        chunks = {result.chunk.chunk_id: result.chunk for result in sparse + vector}
        scores = defaultdict(float)
        for chunk_id, score in sparse_scores.items():
            scores[chunk_id] += 0.55 * score
        for chunk_id, score in vector_scores.items():
            scores[chunk_id] += 0.45 * score
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        return [SearchResult(chunks[chunk_id], score, "hybrid", rank) for rank, (chunk_id, score) in enumerate(ordered, start=1)]

    @staticmethod
    def rrf(result_lists: list[list[SearchResult]], top_k: int) -> list[SearchResult]:
        scores = defaultdict(float)
        chunks = {}
        for results in result_lists:
            for rank, result in enumerate(results, start=1):
                chunk_id = result.chunk.chunk_id
                chunks[chunk_id] = result.chunk
                scores[chunk_id] += 1.0 / (60 + rank)
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        return [SearchResult(chunks[chunk_id], score, "hybrid_rrf", rank) for rank, (chunk_id, score) in enumerate(ordered, start=1)]

    @staticmethod
    def mmr(results: list[SearchResult], top_k: int) -> list[SearchResult]:
        selected: list[SearchResult] = []
        seen_docs = set()
        for result in results:
            if result.chunk.doc_id in seen_docs and len(selected) < max(1, top_k - 1):
                continue
            selected.append(result)
            seen_docs.add(result.chunk.doc_id)
            if len(selected) == top_k:
                break
        if len(selected) < top_k:
            selected.extend(result for result in results if result not in selected)
        return [
            SearchResult(result.chunk, result.score, "hybrid_rrf_mmr", rank)
            for rank, result in enumerate(selected[:top_k], start=1)
        ]


def content_tokens(text: str) -> set[str]:
    return {token for token in tokenize_vi(normalize_for_match(text)) if len(token) > 2 and token not in STOPWORDS}


def reference_overlap(result: SearchResult, item: dict) -> float:
    expected = content_tokens(f"{item.get('reference', '')} {item.get('question', '')}")
    if not expected:
        return 0.0
    observed = content_tokens(chunk_text(result.chunk))
    return len(expected & observed) / len(expected)


def is_relevant(result: SearchResult, item: dict) -> bool:
    terms = item.get("expected_terms", [])
    text = normalize_for_match(chunk_text(result.chunk))
    if terms:
        return any(normalize_for_match(term) in text for term in terms)
    return reference_overlap(result, item) >= 0.12


def term_metrics(results: list[SearchResult], item: dict) -> dict[str, float]:
    terms = item.get("expected_terms", [])
    if terms:
        gains = [1.0 if is_relevant(result, item) else 0.0 for result in results[:TOP_K]]
    else:
        gains = [reference_overlap(result, item) for result in results[:TOP_K]]
    dcg = sum(gain / log2(idx + 2) for idx, gain in enumerate(gains))
    ideal_dcg = sum(gain / log2(idx + 2) for idx, gain in enumerate(sorted(gains, reverse=True)))
    first_relevant = next((idx for idx, result in enumerate(results, start=1) if is_relevant(result, item)), None)
    combined = normalize_for_match(" ".join(chunk_text(result.chunk) for result in results[:TOP_K]))
    if terms:
        context_recall = sum(1 for term in terms if normalize_for_match(term) in combined) / max(len(terms), 1)
    else:
        expected = content_tokens(f"{item.get('reference', '')} {item.get('question', '')}")
        observed = content_tokens(combined)
        context_recall = len(expected & observed) / max(len(expected), 1)
    return {
        "recall@5": 1.0 if any(gains) else 0.0,
        "mrr": 1.0 / first_relevant if first_relevant else 0.0,
        "ndcg@5": dcg / ideal_dcg if ideal_dcg else 0.0,
        "term_context_precision": sum(gains) / max(len(gains), 1),
        "term_context_recall": context_recall,
        "citation_coverage": sum(bool(result.chunk.title and result.chunk.doc_id) for result in results) / max(len(results), 1),
    }


def run_ragas_retrieval_metrics(rows: list[dict]) -> dict[str, dict[str, float]]:
    try:
        from datasets import Dataset
        from ragas import evaluate

        metrics = retrieval_metrics()
        output = {}
        for method in METHODS:
            method_rows = [row for row in rows if row["method"] == method]
            dataset = Dataset.from_list(
                [
                    {
                        "user_input": row["question"],
                        "retrieved_contexts": row["contexts"],
                        "reference_contexts": [row["reference"]],
                        "reference": row["reference"],
                    }
                    for row in method_rows
                ]
            )
            result = evaluate(dataset=dataset, metrics=metrics, show_progress=False)
            frame = result.to_pandas()
            precision_col = "non_llm_context_precision_with_reference"
            recall_col = "non_llm_context_recall"
            precision = float(frame[precision_col].mean())
            recall = float(frame[recall_col].mean())
            output[method] = {
                "ragas_context_precision": precision if isfinite(precision) else None,
                "ragas_context_recall": recall if isfinite(recall) else None,
            }
        return output
    except Exception as exc:
        print(f"RAGAS retrieval metrics unavailable: {exc}")
        return {}


def selection_score(row: dict) -> float:
    context_precision = row.get("ragas_context_precision")
    context_recall = row.get("ragas_context_recall")
    if context_precision is None or not isfinite(context_precision):
        context_precision = row["term_context_precision"]
    if context_recall is None or not isfinite(context_recall):
        context_recall = row["term_context_recall"]
    return 0.30 * row["mrr"] + 0.25 * row["ndcg@5"] + 0.25 * context_precision + 0.20 * context_recall


def plot_reports(summary: pd.DataFrame, details: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ordered = summary.sort_values("selection_score", ascending=False)

    plt.figure(figsize=(9, 4.5))
    plt.bar(ordered["method"], ordered["selection_score"], color="#2563eb")
    plt.ylabel("Selection score")
    plt.title("Retrieval method comparison")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "retrieval_selection_score.png", dpi=160)
    plt.close()

    metric_cols = ["mrr", "ndcg@5", "term_context_precision", "term_context_recall"]
    x = np.arange(len(summary["method"]))
    width = 0.18
    plt.figure(figsize=(10, 5))
    for idx, metric in enumerate(metric_cols):
        plt.bar(x + (idx - 1.5) * width, summary[metric], width=width, label=metric)
    plt.xticks(x, summary["method"], rotation=25, ha="right")
    plt.ylim(0, 1.05)
    plt.title("Retrieval metric breakdown")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "retrieval_metric_comparison.png", dpi=160)
    plt.close()

    pivot = details.pivot_table(index="question_id", columns="method", values="ndcg@5", aggfunc="mean")
    plt.figure(figsize=(9, 7))
    plt.imshow(pivot.values, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(label="nDCG@5")
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=25, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.title("Per-question retrieval quality")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "retrieval_query_heatmap.png", dpi=160)
    plt.close()


def write_reports(report: dict) -> None:
    RETRIEVAL_JSON.parent.mkdir(parents=True, exist_ok=True)
    RETRIEVAL_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = pd.DataFrame(report["summary"])
    details = pd.DataFrame(report["details"])
    plot_reports(summary, details)

    selected = report["selected_method"]
    lines = [
        "# Retrieval Evaluation",
        "",
        f"- Testset size: `{report['testset_size']}`",
        f"- Benchmark chunks: `{report['num_chunks']}`",
        f"- RAGAS retrieval metrics backend: `{report['ragas_backend']}`",
        f"- Selected method: `{selected}`",
        "",
        "| Method | MRR | nDCG@5 | Lexical/Term CP | Lexical/Term CR | RAGAS CP | RAGAS CR | Selection score |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.sort_values("selection_score", ascending=False).to_dict("records"):
        ragas_cp = row.get("ragas_context_precision")
        ragas_cr = row.get("ragas_context_recall")
        ragas_cp_text = f"{ragas_cp:.3f}" if ragas_cp is not None else "n/a"
        ragas_cr_text = f"{ragas_cr:.3f}" if ragas_cr is not None else "n/a"
        lines.append(
            f"| {row['method']} | {row['mrr']:.3f} | {row['ndcg@5']:.3f} | "
            f"{row['term_context_precision']:.3f} | {row['term_context_recall']:.3f} | "
            f"{ragas_cp_text} | {ragas_cr_text} | {row['selection_score']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Figures",
            "",
            "- `reports/figures/retrieval_selection_score.png`",
            "- `reports/figures/retrieval_metric_comparison.png`",
            "- `reports/figures/retrieval_query_heatmap.png`",
            "",
        ]
    )
    RETRIEVAL_MD.write_text("\n".join(lines), encoding="utf-8")


def run_benchmark() -> dict:
    testset = ensure_testset()
    chunks = build_benchmark_chunks(testset)
    retriever = BenchmarkRetriever(chunks)

    details = []
    ragas_rows = []
    for method in METHODS:
        for item in testset:
            results = retriever.retrieve(item["question"], method, top_k=TOP_K)
            metrics = term_metrics(results, item)
            row = {
                "question_id": item["id"],
                "question": item["question"],
                "reference": item["reference"],
                "method": method,
                "top_title": results[0].chunk.title if results else None,
                "contexts": [result.chunk.text for result in results],
                **metrics,
            }
            details.append({key: value for key, value in row.items() if key != "contexts"})
            ragas_rows.append(row)

    ragas_metrics = run_ragas_retrieval_metrics(ragas_rows)
    summary = []
    for method in METHODS:
        method_rows = [row for row in details if row["method"] == method]
        aggregate = {
            "method": method,
            **{
                metric: sum(row[metric] for row in method_rows) / len(method_rows)
                for metric in ["recall@5", "mrr", "ndcg@5", "term_context_precision", "term_context_recall", "citation_coverage"]
            },
            "ragas_context_precision": ragas_metrics.get(method, {}).get("ragas_context_precision"),
            "ragas_context_recall": ragas_metrics.get(method, {}).get("ragas_context_recall"),
        }
        aggregate["selection_score"] = selection_score(aggregate)
        summary.append(aggregate)

    selected = sorted(summary, key=lambda row: (row["selection_score"], row["mrr"], row["ndcg@5"]), reverse=True)[0]["method"]
    report = {
        "testset_size": len(testset),
        "num_chunks": len(chunks),
        "methods": METHODS,
        "ragas_backend": "ragas_evaluate_non_llm" if ragas_metrics else "lexical_fallback_ragas_error",
        "selected_method": selected,
        "summary": summary,
        "details": details,
    }
    write_reports(report)
    return report


def main() -> None:
    report = run_benchmark()
    print(f"selected={report['selected_method']} chunks={report['num_chunks']} ragas_backend={report['ragas_backend']}")


if __name__ == "__main__":
    main()
