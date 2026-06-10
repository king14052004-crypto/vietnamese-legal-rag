"""Evaluate answer quality of every retrieval method with RAGAS.

Flow (kept deliberately simple):
1. For each retrieval method, retrieve context and generate an answer with
   Gemini for all benchmark questions (answers are cached in `.cache/` so the
   script can resume).
2. Score each (question, answer) with RAGAS: faithfulness, answer_relevancy and
   answer_correctness, using Gemini as judge and a local sentence-transformers
   model for embeddings (no extra API quota).
3. Coverage fix: RAGAS returns NaN when an LLM call fails (usually quota).
   Rows with NaN are re-evaluated up to MAX_PASSES times with a different API
   key, so nearly all 30 questions end up scored instead of only a handful.

Writes `reports/ragas_evaluation.json` and `reports/ragas_evaluation.md`.

Usage (requires GEMINI_API_KEYS):
    python scripts/run_generation_eval.py
"""

import json
import math
import os
import sys
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.chunking import chunk_documents  # noqa: E402
from src.data.load_dataset import load_documents_from_jsonl  # noqa: E402
from src.generation.gemini_client import BatchGeminiClient, _load_api_keys_from_env  # noqa: E402
from src.generation.prompt import build_rag_prompt  # noqa: E402
from src.retrieval.pipeline import RETRIEVAL_METHODS, RetrievalPipeline  # noqa: E402

CORPUS_PATH = PROJECT_ROOT / "data/processed/labor_corpus_sample.jsonl"
BENCHMARK_PATH = PROJECT_ROOT / "reports/benchmark_questions.json"
ANSWERS_CACHE = PROJECT_ROOT / ".cache/eval_answers.json"
OUTPUT_JSON = PROJECT_ROOT / "reports/ragas_evaluation.json"
OUTPUT_MD = PROJECT_ROOT / "reports/ragas_evaluation.md"
CACHE_DIR = PROJECT_ROOT / ".cache"
TOP_K = 6
MAX_PASSES = 4
METRIC_NAMES = ["faithfulness", "answer_relevancy", "answer_correctness"]
SCORE_WEIGHTS = {"faithfulness": 0.4, "answer_relevancy": 0.3, "answer_correctness": 0.3}
LOCAL_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"


def generate_answers(items: list[dict], pipeline: RetrievalPipeline) -> dict:
    """Generate (and cache) one answer per (method, question)."""
    cache = json.loads(ANSWERS_CACHE.read_text(encoding="utf-8")) if ANSWERS_CACHE.exists() else {}
    client = BatchGeminiClient()
    for method in RETRIEVAL_METHODS:
        for item in items:
            key = f"{method}::{item['id']}"
            if key in cache:
                continue
            results = pipeline.retrieve(item["question"], top_k=TOP_K, method=method)
            prompt, _ = build_rag_prompt(item["question"], results)
            cache[key] = {
                "answer": client.generate(prompt),
                "contexts": [result.chunk.text[:1500] for result in results],
            }
            ANSWERS_CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        print(f"answers ready: {method}")
    return cache


def build_judge(api_key: str):
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_core.rate_limiters import InMemoryRateLimiter
    from langchain_google_genai import ChatGoogleGenerativeAI
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite"),
        google_api_key=api_key,
        temperature=0,
        timeout=120,
        max_retries=2,
        rate_limiter=InMemoryRateLimiter(requests_per_second=0.4, check_every_n_seconds=0.2),
    )
    embeddings = HuggingFaceEmbeddings(model_name=LOCAL_EMBEDDING_MODEL)
    return LangchainLLMWrapper(llm), LangchainEmbeddingsWrapper(embeddings)


def ragas_scores(rows: list[dict], api_key: str) -> list[dict]:
    """Run RAGAS on the given rows and return one score dict per row."""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.run_config import RunConfig

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from ragas.metrics import answer_correctness, answer_relevancy, faithfulness

    llm, embeddings = build_judge(api_key)
    dataset = Dataset.from_list(rows)
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, answer_correctness],
        llm=llm,
        embeddings=embeddings,
        run_config=RunConfig(max_workers=2, max_retries=5, timeout=180),
        show_progress=True,
    )
    frame = result.to_pandas()
    return [{name: float(frame.iloc[i][name]) for name in METRIC_NAMES} for i in range(len(rows))]


def is_missing(score: dict) -> bool:
    return any(score.get(name) is None or math.isnan(score[name]) for name in METRIC_NAMES)


def main() -> None:
    items = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))["items"]
    documents = load_documents_from_jsonl(CORPUS_PATH)
    chunks = chunk_documents(documents)
    pipeline = RetrievalPipeline(chunks, cache_dir=CACHE_DIR)
    answers = generate_answers(items, pipeline)
    api_keys = _load_api_keys_from_env()

    summary = []
    details = {}
    for method_index, method in enumerate(RETRIEVAL_METHODS):
        rows = [
            {
                "question": item["question"],
                "answer": answers[f"{method}::{item['id']}"]["answer"],
                "contexts": answers[f"{method}::{item['id']}"]["contexts"],
                "ground_truth": item["reference"],
            }
            for item in items
        ]
        scores: list[dict] = [{} for _ in rows]
        pending = list(range(len(rows)))
        for attempt in range(MAX_PASSES):
            if not pending:
                break
            api_key = api_keys[(method_index * MAX_PASSES + attempt) % len(api_keys)]
            try:
                new_scores = ragas_scores([rows[i] for i in pending], api_key)
            except Exception as error:
                print(f"{method} pass {attempt + 1} failed: {error}")
                continue
            for position, row_index in enumerate(pending):
                for name in METRIC_NAMES:
                    value = new_scores[position][name]
                    if not math.isnan(value):
                        scores[row_index][name] = value
            pending = [i for i in range(len(rows)) if is_missing(scores[i])]
            print(f"{method} pass {attempt + 1}: {len(rows) - len(pending)}/{len(rows)} fully scored")

        aggregate = {"method": method}
        for name in METRIC_NAMES:
            values = [score[name] for score in scores if name in score]
            aggregate[name] = sum(values) / len(values) if values else None
            aggregate[f"{name}_n"] = len(values)
        if all(aggregate[name] is not None for name in METRIC_NAMES):
            aggregate["generation_score"] = sum(SCORE_WEIGHTS[name] * aggregate[name] for name in METRIC_NAMES)
        else:
            aggregate["generation_score"] = None
        summary.append(aggregate)
        details[method] = [
            {"question_id": items[i]["id"], **scores[i]} for i in range(len(rows))
        ]
        print(f"{method}: {aggregate}")

    summary.sort(key=lambda row: row["generation_score"] or 0.0, reverse=True)
    payload = {
        "benchmark": BENCHMARK_PATH.name,
        "questions": len(items),
        "metrics": METRIC_NAMES,
        "score_formula": "0.4*faithfulness + 0.3*answer_relevancy + 0.3*answer_correctness",
        "summary": summary,
        "details": details,
    }
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# RAGAS Generation Evaluation",
        "",
        f"- Benchmark: `{BENCHMARK_PATH.name}` ({len(items)} questions)",
        "- Judge LLM: Gemini; embeddings: local sentence-transformers (no extra quota).",
        "- NaN rows are retried with different API keys, so coverage (n) is close to the full question set.",
        f"- Ranking score: `{payload['score_formula']}`",
        "",
        "| Method | Faithfulness (n) | Answer Relevancy (n) | Answer Correctness (n) | Score |",
        "|---|---|---|---|---|",
    ]
    for row in summary:
        cells = [
            f"{row[name]:.3f} ({row[f'{name}_n']}/{len(items)})" if row[name] is not None else "—"
            for name in METRIC_NAMES
        ]
        score = f"{row['generation_score']:.3f}" if row["generation_score"] is not None else "—"
        lines.append(f"| {row['method']} | {cells[0]} | {cells[1]} | {cells[2]} | {score} |")
    lines += ["", f"Best method by generation score: **{summary[0]['method']}**."]
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_JSON} and {OUTPUT_MD}")


if __name__ == "__main__":
    main()
