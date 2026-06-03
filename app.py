import argparse
from pathlib import Path

from src.data.chunking import chunk_documents
from src.data.load_dataset import load_documents_from_jsonl
from src.generation.gemini_client import BatchGeminiClient
from src.generation.prompt import build_rag_prompt
from src.retrieval.pipeline import RetrievalPipeline


def main() -> None:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--corpus", default=str(project_root / "data/processed/labor_corpus_sample.jsonl"))
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--tfidf-fallback", action="store_true")
    parser.add_argument("--no-llm", action="store_true")
    args = parser.parse_args()

    documents = load_documents_from_jsonl(args.corpus)
    chunks = chunk_documents(documents)
    retriever = RetrievalPipeline(chunks, use_tfidf_fallback=args.tfidf_fallback)
    results = retriever.retrieve(args.question, top_k=args.top_k)

    if args.no_llm:
        for result in results:
            print(f"[{result.rank}] {result.chunk.title} ({result.score:.4f})")
            print(result.chunk.text[:600], "\n")
        return

    client = BatchGeminiClient()
    prompt, citations = build_rag_prompt(args.question, results)
    print(client.generate(prompt))
    print("\nSources:")
    for citation in citations:
        print(f"- [{citation['label']}] {citation['title']} / {citation['chunk_id']}")


if __name__ == "__main__":
    main()
