from rank_bm25 import BM25Okapi

from src.data.clean_text import tokenize_vi
from src.data.schema import LegalChunk, SearchResult


class SparseRetriever:
    def __init__(self, chunks: list[LegalChunk]):
        self.chunks = chunks
        self.tokenized = [tokenize_vi(f"{chunk.title} {chunk.text}") for chunk in chunks]
        self.index = BM25Okapi(self.tokenized)

    def retrieve(self, query: str, top_k: int = 10) -> list[SearchResult]:
        scores = self.index.get_scores(tokenize_vi(query))
        order = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:top_k]
        return [
            SearchResult(chunk=self.chunks[idx], score=float(scores[idx]), method="bm25", rank=rank + 1)
            for rank, idx in enumerate(order)
        ]
