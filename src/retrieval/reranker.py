from src.data.schema import SearchResult


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"):
        self.model_name = model_name
        self.model = None
        try:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(model_name)
        except Exception:
            self.model = None

    def rerank(self, query: str, candidates: list[SearchResult], top_k: int = 8) -> list[SearchResult]:
        if not candidates:
            return []
        if self.model is None:
            return candidates[:top_k]

        pairs = [[query, f"{candidate.chunk.title}\n{candidate.chunk.text}"] for candidate in candidates]
        scores = self.model.predict(pairs)
        reranked = [
            SearchResult(candidate.chunk, float(score), "cross_encoder", None)
            for candidate, score in zip(candidates, scores)
        ]
        reranked.sort(key=lambda result: result.score, reverse=True)
        for rank, result in enumerate(reranked[:top_k], start=1):
            result.rank = rank
        return reranked[:top_k]
