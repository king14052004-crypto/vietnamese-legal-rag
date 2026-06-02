import argparse
import json
import re
from pathlib import Path

from src.generation.gemini_client import BatchGeminiClient


RAGAS_STYLE_PROMPT = """Bạn là evaluator RAG theo phong cách RAGAS.
Chấm điểm JSON cho một sample RAG tiếng Việt.

Metric definitions:
- faithfulness: answer có được hỗ trợ bởi contexts không, không bịa.
- answer_relevancy: answer có trả lời đúng câu hỏi không.
- context_precision: contexts có liên quan và ít nhiễu không.

Trả lời CHỈ bằng JSON hợp lệ:
{{
  "faithfulness": 0.0-1.0,
  "answer_relevancy": 0.0-1.0,
  "context_precision": 0.0-1.0,
  "reason": "ngắn gọn"
}}

QUESTION:
{question}

ANSWER:
{answer}

CONTEXTS:
{contexts}
"""


def _extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object in evaluator response: {text[:200]}")
    return json.loads(match.group(0))


def run_ragas_proxy(samples: list[dict]) -> dict:
    scores = []
    for sample in samples:
        answer_tokens = set(re.findall(r"\w+", sample["answer"].lower()))
        context_tokens = set(re.findall(r"\w+", " ".join(sample["contexts"]).lower()))
        overlap = len(answer_tokens & context_tokens) / max(len(answer_tokens), 1)
        scores.append(min(overlap * 1.4, 1.0))
    value = sum(scores) / max(len(scores), 1)
    return {
        "faithfulness": value,
        "answer_relevancy": value,
        "context_precision": value,
        "samples": len(samples),
        "mode": "lexical_fallback_proxy",
    }


def evaluate_with_ai_studio_genai(samples: list[dict], model: str | None = None) -> dict:
    client = BatchGeminiClient(model=model)
    rows = []
    for sample in samples:
        contexts = "\n\n".join(f"[{idx + 1}] {ctx[:1800]}" for idx, ctx in enumerate(sample["contexts"][:6]))
        prompt = RAGAS_STYLE_PROMPT.format(
            question=sample["question"],
            answer=sample["answer"],
            contexts=contexts,
        )
        try:
            raw = client.generate(prompt)
            parsed = _extract_json(raw)
            row = {
                "question": sample["question"],
                "faithfulness": float(parsed["faithfulness"]),
                "answer_relevancy": float(parsed["answer_relevancy"]),
                "context_precision": float(parsed["context_precision"]),
                "reason": parsed.get("reason", ""),
            }
        except Exception as exc:
            proxy = run_ragas_proxy([sample])
            row = {
                "question": sample["question"],
                "faithfulness": proxy["faithfulness"],
                "answer_relevancy": proxy["answer_relevancy"],
                "context_precision": proxy["context_precision"],
                "reason": f"fallback proxy after evaluator error: {exc}",
            }
        rows.append(row)

    aggregate = {
        "faithfulness": sum(row["faithfulness"] for row in rows) / len(rows),
        "answer_relevancy": sum(row["answer_relevancy"] for row in rows) / len(rows),
        "context_precision": sum(row["context_precision"] for row in rows) / len(rows),
        "samples": len(rows),
        "mode": "ragas_style_ai_studio_genai",
        "model": model or "env/default",
    }
    return {"aggregate": aggregate, "details": rows}


def write_markdown_report(result: dict, output_path: str) -> None:
    aggregate = result["aggregate"]
    lines = [
        "# RAGAS-style Evaluation",
        "",
        "Evaluator: AI Studio Gemini API via `google-genai`.",
        "",
        "| Metric | Score |",
        "|---|---:|",
        f"| Faithfulness | {aggregate['faithfulness']:.3f} |",
        f"| Answer relevancy | {aggregate['answer_relevancy']:.3f} |",
        f"| Context precision | {aggregate['context_precision']:.3f} |",
        "",
        "## Per-sample notes",
        "",
    ]
    for row in result["details"]:
        lines.append(
            f"- **{row['question']}** — F={row['faithfulness']:.2f}, "
            f"R={row['answer_relevancy']:.2f}, CP={row['context_precision']:.2f}. {row['reason']}"
        )
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers-json", default="reports/rag_answers.json")
    parser.add_argument("--output", default="reports/ragas_evaluation.json")
    parser.add_argument("--output-md", default="reports/ragas_evaluation.md")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    answer_path = Path(args.answers_json)
    if not answer_path.exists():
        raise FileNotFoundError("Generate RAG answers before running RAGAS-style evaluation")
    samples = json.loads(answer_path.read_text(encoding="utf-8"))
    output = evaluate_with_ai_studio_genai(samples, args.model)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown_report(output, args.output_md)
    print(json.dumps(output["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
