from src.data.schema import LegalChunk, SearchResult
from src.retrieval.fusion import reciprocal_rank_fusion, weighted_hybrid
from src.retrieval.mmr import apply_mmr
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.sparse import SparseRetriever
from src.retrieval.vector_store import FaissVectorRetriever


class RetrievalPipeline:
    def __init__(
        self,
        chunks: list[LegalChunk],
        embedding_model: str = "intfloat/multilingual-e5-small",
        use_tfidf_fallback: bool = False,
        enable_reranker: bool = False,
    ):
        self.sparse = SparseRetriever(chunks)
        self.vector = FaissVectorRetriever(chunks, embedding_model, use_tfidf_fallback)
        self.reranker = CrossEncoderReranker() if enable_reranker else None

    def retrieve(self, query: str, method: str = "hybrid_rrf", top_k: int = 8) -> list[SearchResult]:
        candidate_k = max(30, top_k * 5)
        sparse = self.sparse.retrieve(query, candidate_k)
        vector = self.vector.retrieve(query, candidate_k)

        if method == "bm25":
            results = sparse
        elif method == "vector":
            results = vector
        elif method == "hybrid":
            results = weighted_hybrid(sparse, vector, top_k=candidate_k)
        elif method in {"hybrid_rrf", "hybrid_rrf_mmr", "hybrid_rrf_rerank"}:
            results = reciprocal_rank_fusion([sparse, vector], top_k=candidate_k)
        else:
            raise ValueError(f"Unknown retrieval method: {method}")

        if method.endswith("_mmr"):
            return apply_mmr(results, top_k=top_k)
        if method.endswith("_rerank") and self.reranker is not None:
            return self.reranker.rerank(query, results, top_k=top_k)
        return results[:top_k]
