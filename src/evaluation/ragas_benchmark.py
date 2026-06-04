import json

import pandas as pd

from src.generation.gemini_client import BatchGeminiClient
from src.generation.prompt import build_rag_prompt
from src.evaluation.config import (
    FIGURES_DIR,
    RAG_ANSWERS_JSON,
    RAGAS_EVAL_JSON,
    RAGAS_EVAL_MD,
    TOP_K,
)
from src.evaluation.gemini_env import has_gemini_key
from src.evaluation.ragas_runtime import answer_metrics, build_langchain_embeddings, build_langchain_llm, gemini_model
from src.evaluation.retrieval_benchmark import BenchmarkRetriever, build_benchmark_chunks
from src.evaluation.testset import ensure_testset


def load_selected_method() -> str:
    from src.evaluation.config import RETRIEVAL_JSON

    if not RETRIEVAL_JSON.exists():
        return "hybrid"
    report = json.loads(RETRIEVAL_JSON.read_text(encoding="utf-8"))
    return report.get("selected_method", "hybrid")


def load_cached_answers() -> list[dict]:
    if not RAG_ANSWERS_JSON.exists():
        return []
    return json.loads(RAG_ANSWERS_JSON.read_text(encoding="utf-8"))


def generate_answers() -> tuple[list[dict], bool, str]:
    testset = ensure_testset()
    method = load_selected_method()
    cached = load_cached_answers()
    if cached and len(cached) >= len(testset) and all(row.get("retrieval_method") == method for row in cached):
        return cached[: len(testset)], True, "cached_answers"
    chunks = build_benchmark_chunks(testset)
    retriever = BenchmarkRetriever(chunks)
    use_gemini = has_gemini_key()
    client = BatchGeminiClient(model=gemini_model()) if use_gemini else None

    if not client:
        if cached:
            return cached, True, "no_gemini_key"

    answers = []
    try:
        for item in testset:
            results = retriever.retrieve(item["question"], method, top_k=TOP_K)
            prompt, citations = build_rag_prompt(item["question"], results)
            answer = client.generate(prompt) if client else item["reference"]
            answers.append(
                {
                    "id": item["id"],
                    "question": item["question"],
                    "reference": item["reference"],
                    "answer": answer,
                    "contexts": [result.chunk.text for result in results],
                    "citations": citations,
                    "retrieval_method": method,
                    "answer_backend": "gemini" if client else "reference_fallback_no_gemini_key",
                }
            )
    except Exception as exc:
        cached = load_cached_answers()
        if cached:
            return cached, True, f"answer_generation_failed: {exc}"
        raise
    RAG_ANSWERS_JSON.write_text(json.dumps(answers, ensure_ascii=False, indent=2), encoding="utf-8")
    return answers, False, "generated"


def load_cached_ragas_metrics() -> tuple[dict, list[dict]]:
    if not RAGAS_EVAL_JSON.exists():
        return {}, []
    payload = json.loads(RAGAS_EVAL_JSON.read_text(encoding="utf-8"))
    return payload.get("aggregate") or {}, payload.get("details") or []


def run_ragas_answer_metrics(answers: list[dict]) -> tuple[dict, list[dict], str, bool, str]:
    if not has_gemini_key():
        aggregate, details = load_cached_ragas_metrics()
        if aggregate:
            return aggregate, details, "cached_ragas_evaluate", True, "no_gemini_key"
        return {}, [], "unavailable_no_gemini_key", False, "no_gemini_key"
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.run_config import RunConfig

        dataset = Dataset.from_list(
            [
                {
                    "user_input": row["question"],
                    "response": row["answer"],
                    "retrieved_contexts": row["contexts"],
                    "reference": row["reference"],
                }
                for row in answers
            ]
        )
        llm = build_langchain_llm()
        embeddings = build_langchain_embeddings()
        desired_metric_names = ["context_precision", "context_recall", "faithfulness", "answer_relevancy", "answer_correctness"]
        _, cached_details = load_cached_ragas_metrics()
        if len(cached_details) == len(answers):
            frame = pd.DataFrame(cached_details)
        else:
            frame = pd.DataFrame(
                [
                    {
                        "user_input": row["question"],
                        "response": row["answer"],
                        "retrieved_contexts": row["contexts"],
                        "reference": row["reference"],
                    }
                    for row in answers
                ]
            )
        metrics = [metric for metric in answer_metrics() if metric.name not in frame or not frame[metric.name].notna().any()]
        run_config = RunConfig(timeout=120, max_retries=1, max_wait=5, max_workers=1)
        for metric in metrics:
            result = evaluate(
                dataset=dataset,
                metrics=[metric],
                llm=llm,
                embeddings=embeddings,
                run_config=run_config,
                show_progress=False,
            )
            next_frame = result.to_pandas()
            metric_cols = [col for col in next_frame.columns if col not in frame.columns or col in desired_metric_names]
            for col in metric_cols:
                if col in desired_metric_names:
                    frame[col] = next_frame[col].reset_index(drop=True)
        aggregate = {
            metric: float(frame[metric].mean())
            for metric in desired_metric_names
            if metric in frame and pd.notna(frame[metric].mean())
        }
        frame = frame.astype(object).where(pd.notna(frame), None)
        detail_rows = frame.to_dict("records")
        return aggregate, detail_rows, "ragas_evaluate", False, "evaluated"
    except Exception as exc:
        print(f"RAGAS answer metrics unavailable: {exc}")
        aggregate, details = load_cached_ragas_metrics()
        if aggregate:
            return aggregate, details, "cached_ragas_evaluate", True, f"ragas_error: {exc}"
        return {}, [], "unavailable_ragas_error", False, f"ragas_error: {exc}"


def plot_answer_report(aggregate: dict, details: list[dict], backend: str) -> None:
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if not aggregate:
        plt.figure(figsize=(8, 3))
        plt.axis("off")
        plt.text(0.5, 0.5, f"RAGAS metrics unavailable\nbackend={backend}", ha="center", va="center", fontsize=13)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "ragas_answer_metrics.png", dpi=160)
        plt.close()
        return

    metrics = list(aggregate)
    values = [aggregate[metric] for metric in metrics]
    plt.figure(figsize=(9, 4.5))
    plt.bar(metrics, values, color="#16a34a")
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("RAGAS answer evaluation")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ragas_answer_metrics.png", dpi=160)
    plt.close()

    frame = pd.DataFrame(details)
    if not frame.empty:
        metric_cols = [metric for metric in metrics if metric in frame]
        plt.figure(figsize=(9, 7))
        plt.imshow(frame[metric_cols].fillna(0).values, aspect="auto", cmap="Greens", vmin=0, vmax=1)
        plt.colorbar(label="RAGAS score")
        plt.xticks(range(len(metric_cols)), metric_cols, rotation=25, ha="right")
        plt.yticks(range(len(frame)), [f"q{idx + 1:02d}" for idx in range(len(frame))])
        plt.title("Per-question RAGAS scores")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "ragas_per_question_heatmap.png", dpi=160)
        plt.close()


def write_ragas_report(payload: dict) -> None:
    RAGAS_EVAL_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    aggregate = payload["aggregate"]
    lines = [
        "# RAGAS Answer Evaluation",
        "",
        f"- Backend: `{payload['backend']}`",
        f"- Cached artifact used: `{payload.get('cached', False)}`",
        f"- Cache reason: `{payload.get('cache_reason') or 'n/a'}`",
        f"- Answers: `{payload['answer_count']}`",
        f"- Retrieval method: `{payload['retrieval_method']}`",
        f"- Model: `{payload['model']}`",
        "",
        "| Metric | Score |",
        "|---|---:|",
    ]
    if aggregate:
        for metric, value in aggregate.items():
            lines.append(f"| {metric} | {value:.3f} |")
    else:
        lines.append("| RAGAS metrics | n/a - Gemini key or RAGAS runtime required |")
    lines.extend(
        [
            "",
            "## Figures",
            "",
            "- `reports/figures/ragas_answer_metrics.png`",
            "- `reports/figures/ragas_per_question_heatmap.png` when RAGAS details are available",
            "",
            "## Weakest Cases",
            "",
        ]
    )
    details = payload.get("details") or []
    if details:
        frame = pd.DataFrame(details)
        metric_cols = [col for col in ["faithfulness", "answer_relevancy", "answer_correctness", "context_precision", "context_recall"] if col in frame]
        frame["mean_score"] = frame[metric_cols].mean(axis=1)
        for _, row in frame.sort_values("mean_score").head(5).iterrows():
            question = row.get("user_input") or row.get("question") or ""
            lines.append(f"- {question} - mean={row['mean_score']:.3f}")
    else:
        lines.append("- Not available until `ragas.evaluate()` runs with a Gemini key.")
    RAGAS_EVAL_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_benchmark() -> dict:
    answers, answers_cached, cache_reason = generate_answers()
    aggregate, details, backend, metrics_cached, metrics_reason = run_ragas_answer_metrics(answers)
    retrieval_method = answers[0]["retrieval_method"] if answers else "unknown"
    payload = {
        "backend": backend,
        "model": gemini_model(),
        "cached": answers_cached or metrics_cached,
        "answers_cached": answers_cached,
        "metrics_cached": metrics_cached,
        "cache_reason": cache_reason if answers_cached else metrics_reason if metrics_cached else "",
        "answer_count": len(answers),
        "retrieval_method": retrieval_method,
        "aggregate": aggregate,
        "details": details,
    }
    plot_answer_report(aggregate, details, backend)
    write_ragas_report(payload)
    return payload


def main() -> None:
    payload = run_benchmark()
    print(f"answers={payload['answer_count']} backend={payload['backend']}")


if __name__ == "__main__":
    main()
