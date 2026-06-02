from src.data.schema import SearchResult


SYSTEM_INSTRUCTION = """Bạn là trợ lý tra cứu pháp luật lao động Việt Nam.
Chỉ trả lời dựa trên CONTEXT. Nếu context không đủ, nói rõ là chưa đủ căn cứ.
Không đưa ra tư vấn pháp lý chắc chắn như luật sư. Luôn trích nguồn dạng [S1], [S2]."""


def build_context(results: list[SearchResult], max_chars: int = 9000) -> tuple[str, list[dict]]:
    blocks = []
    citations = []
    used_chars = 0
    for idx, result in enumerate(results, start=1):
        label = f"S{idx}"
        chunk = result.chunk
        text = chunk.text.strip()
        block = f"[{label}] {chunk.title}\n{chunk.article or ''}\n{text}"
        if used_chars + len(block) > max_chars:
            break
        used_chars += len(block)
        blocks.append(block)
        citations.append(
            {
                "label": label,
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "title": chunk.title,
                "article": chunk.article,
                "source_url": chunk.source_url,
                "score": result.score,
                "method": result.method,
            }
        )
    return "\n\n".join(blocks), citations


def build_rag_prompt(question: str, results: list[SearchResult]) -> tuple[str, list[dict]]:
    context, citations = build_context(results)
    prompt = f"""{SYSTEM_INSTRUCTION}

CONTEXT:
{context}

Câu hỏi: {question}

Trả lời ngắn gọn bằng tiếng Việt, có bullet nếu cần, và trích nguồn sau từng ý quan trọng."""
    return prompt, citations
