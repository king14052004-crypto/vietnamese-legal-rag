from pathlib import Path

import streamlit as st

from src.data.chunking import chunk_documents
from src.data.load_dataset import load_documents_from_jsonl
from src.generation.gemini_client import BatchGeminiClient
from src.generation.prompt import build_rag_prompt
from src.retrieval.pipeline import RetrievalPipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = PROJECT_ROOT / "data/processed/labor_corpus.jsonl"
DEFAULT_QUESTION = "Người lao động đơn phương chấm dứt hợp đồng cần báo trước bao lâu?"
DEFAULT_TOP_K = 6


@st.cache_resource(show_spinner="Đang load corpus và build index...")
def load_retriever(corpus_path: str = str(DEFAULT_CORPUS)) -> RetrievalPipeline:
    documents = load_documents_from_jsonl(corpus_path)
    chunks = chunk_documents(documents)
    return RetrievalPipeline(chunks, use_tfidf_fallback=True)


def retrieve_contexts(
    question: str,
    corpus_path: str = str(DEFAULT_CORPUS),
):
    retriever = load_retriever(corpus_path)
    return retriever.retrieve(question, top_k=DEFAULT_TOP_K)


def generate_answer(question: str, results) -> tuple[str, list[dict]]:
    prompt, citations = build_rag_prompt(question, results)
    answer = BatchGeminiClient().generate(prompt)
    return answer, citations


def render_sources(results) -> None:
    for result in results:
        chunk = result.chunk
        with st.expander(f"[{result.rank}] {chunk.title} — score {result.score:.4f}"):
            st.write(chunk.text[:1400])
            st.caption(f"doc_id={chunk.doc_id} | chunk_id={chunk.chunk_id}")


def main() -> None:
    st.set_page_config(page_title="Vietnamese Legal RAG", layout="wide")
    st.title("Vietnamese Labor Legal RAG")
    st.caption("Demo tra cứu pháp luật lao động Việt Nam với retrieval pipeline + Gemini.")

    with st.sidebar:
        st.header("Demo settings")
        st.caption(f"Retrieved sources: {DEFAULT_TOP_K}")
        use_llm = st.checkbox("Generate Gemini answer", value=False)

    question = st.text_area("Câu hỏi", value=DEFAULT_QUESTION, height=90)
    if st.button("Ask", type="primary"):
        if not question.strip():
            st.warning("Nhập câu hỏi trước khi chạy.")
            return

        results = retrieve_contexts(question)

        if use_llm:
            try:
                answer, citations = generate_answer(question, results)
                st.subheader("Answer")
                st.write(answer)
                st.subheader("Citations")
                st.dataframe(citations, use_container_width=True)
            except Exception as exc:
                st.error(f"Gemini generation failed: {exc}")
                st.info("Tắt `Generate Gemini answer` để xem retrieval-only demo.")
        else:
            st.info("Retrieval-only mode. Bật `Generate Gemini answer` nếu đã set Gemini key.")

        st.subheader("Retrieved sources")
        render_sources(results)

    st.divider()
    st.caption("Demo chỉ dùng cho mục đích tra cứu thông tin, không thay thế tư vấn pháp lý.")


if __name__ == "__main__":
    main()
