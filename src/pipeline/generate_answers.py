import argparse
import json
from pathlib import Path

from src.data.chunking import chunk_documents
from src.data.load_dataset import load_documents_from_jsonl
from src.evaluation.test_queries import LABOR_TEST_QUERIES
from src.generation.gemini_client import BatchGeminiClient
from src.generation.prompt import build_rag_prompt
from src.retrieval.pipeline import RetrievalPipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/processed/labor_corpus_sample.jsonl")
    parser.add_argument("--method", default="hybrid_rrf")
    parser.add_argument("--output", default="reports/rag_answers.json")
    parser.add_argument("--tfidf-fallback", action="store_true")
    args = parser.parse_args()

    documents = load_documents_from_jsonl(args.corpus)
    chunks = chunk_documents(documents)
    retriever = RetrievalPipeline(chunks, use_tfidf_fallback=args.tfidf_fallback)
    client = BatchGeminiClient()

    samples = []
    for query in LABOR_TEST_QUERIES:
        results = retriever.retrieve(query["question"], method=args.method, top_k=6)
        prompt, citations = build_rag_prompt(query["question"], results)
        answer = client.generate(prompt)
        samples.append(
            {
                "question": query["question"],
                "answer": answer,
                "contexts": [result.chunk.text for result in results],
                "citations": citations,
                "ground_truth": "; ".join(query["relevant_terms"]),
            }
        )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(samples)} answers to {args.output}")


if __name__ == "__main__":
    main()
