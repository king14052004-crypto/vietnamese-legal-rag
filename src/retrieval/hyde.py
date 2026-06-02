from collections.abc import Callable


def expand_query_hyde(query: str, generator: Callable[[str], str] | None = None) -> str:
    if generator is None:
        return query
    prompt = (
        "Viết một đoạn trả lời giả định ngắn bằng tiếng Việt cho câu hỏi pháp luật lao động sau. "
        "Không cần chắc chắn đúng, mục tiêu là mở rộng truy vấn retrieval.\n"
        f"Câu hỏi: {query}"
    )
    hypothetical_answer = generator(prompt)
    return f"{query}\n{hypothetical_answer}"
