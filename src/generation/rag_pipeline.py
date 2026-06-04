from collections.abc import Callable

from src.data.schema import RagAnswer
from src.generation.prompt import build_rag_prompt
from src.retrieval.pipeline import RetrievalPipeline


class LegalRagPipeline:
    def __init__(self, retriever: RetrievalPipeline, generator: Callable[[str], str]):
        self.retriever = retriever
        self.generator = generator

    def ask(self, question: str, top_k: int = 6) -> RagAnswer:
        results = self.retriever.retrieve(question, top_k=top_k)
        prompt, citations = build_rag_prompt(question, results)
        answer = self.generator(prompt)
        return RagAnswer(
            question=question,
            answer=answer,
            citations=citations,
            retrieval_method="hybrid",
            context_chunks=[result.chunk for result in results],
        )
