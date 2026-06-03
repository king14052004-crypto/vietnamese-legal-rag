import sys
from pathlib import Path

from src.data.chunking import chunk_documents
from src.data.load_dataset import load_documents_from_jsonl
from src.generation.gemini_client import BatchGeminiClient
from src.generation.prompt import build_rag_prompt
from src.retrieval.pipeline import RetrievalPipeline

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CORPUS = PROJECT_ROOT / "data/processed/labor_corpus.jsonl"
DEFAULT_TOP_K = 6


def read_question() -> str:
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        raise SystemExit('Usage: python app.py "<question>"')
    return question


def print_sources(results) -> None:
    print("Retrieved sources:")
    for result in results:
        print(f"\n[{result.rank}] {result.chunk.title} ({result.score:.4f})")
        print(result.chunk.text[:600])


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    question = read_question()

    documents = load_documents_from_jsonl(DEFAULT_CORPUS)
    chunks = chunk_documents(documents)
    retriever = RetrievalPipeline(chunks, use_tfidf_fallback=True)
    results = retriever.retrieve(question, top_k=DEFAULT_TOP_K)

    try:
        client = BatchGeminiClient()
    except ValueError:
        print("No Gemini key found, so the app is showing retrieval results only.")
        print_sources(results)
        return

    prompt, citations = build_rag_prompt(question, results)
    try:
        print(client.generate(prompt))
    except RuntimeError as exc:
        print(f"Gemini generation failed: {exc}")
        print_sources(results)
        return

    print("\nSources:")
    for citation in citations:
        print(f"- [{citation['label']}] {citation['title']} / {citation['chunk_id']}")


if __name__ == "__main__":
    main()
