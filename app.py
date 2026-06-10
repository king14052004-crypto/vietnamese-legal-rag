import sys
from pathlib import Path

from src.data.chunking import chunk_documents
from src.data.load_dataset import load_documents_from_jsonl
from src.generation.gemini_client import BatchGeminiClient
from src.generation.prompt import build_rag_prompt
from src.retrieval.pipeline import RetrievalPipeline

PROJECT_ROOT = Path(__file__).resolve().parent
FULL_CORPUS = PROJECT_ROOT / "data/processed/labor_corpus.jsonl"
SAMPLE_CORPUS = PROJECT_ROOT / "data/processed/labor_corpus_sample.jsonl"
CACHE_DIR = PROJECT_ROOT / ".cache"
DEFAULT_TOP_K = 6


def pick_corpus() -> Path:
    return FULL_CORPUS if FULL_CORPUS.exists() else SAMPLE_CORPUS


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

    corpus_path = pick_corpus()
    print(f"Corpus: {corpus_path.name}")
    documents = load_documents_from_jsonl(corpus_path)
    chunks = chunk_documents(documents)
    retriever = RetrievalPipeline(chunks, cache_dir=CACHE_DIR)
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
