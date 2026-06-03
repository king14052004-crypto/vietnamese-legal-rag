import warnings
from collections import defaultdict

import numpy as np
from rank_bm25 import BM25Okapi

from src.data.clean_text import tokenize_vi
from src.data.schema import LegalChunk, SearchResult

DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"


class RetrievalPipeline:
    def __init__(
        self,
        chunks: list[LegalChunk],
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        use_tfidf_fallback: bool = False,
    ):
        if not chunks:
            raise ValueError("RetrievalPipeline requires at least one chunk")

        self.chunks = chunks
        self.embedding_model = embedding_model
        self._bm25 = BM25Okapi([tokenize_vi(self._chunk_text(chunk)) for chunk in chunks])
        self._encoder = None
        self._vectorizer = None
        self.vector_backend = "sentence_transformer"
        self._embeddings = self._encode_corpus(use_tfidf_fallback)
        self._index = self._build_faiss_index(self._embeddings)

    def retrieve(self, query: str, top_k: int = 8) -> list[SearchResult]:
        candidate_k = min(len(self.chunks), max(30, top_k * 5))
        sparse_results = self._retrieve_bm25(query, candidate_k)
        vector_results = self._retrieve_vector(query, candidate_k)
        return self._reciprocal_rank_fusion([sparse_results, vector_results], top_k)

    @staticmethod
    def _chunk_text(chunk: LegalChunk) -> str:
        return f"{chunk.title} {chunk.text}"

    def _encode_corpus(self, use_tfidf_fallback: bool) -> np.ndarray:
        texts = [f"passage: {self._chunk_text(chunk)}" for chunk in self.chunks]
        if use_tfidf_fallback:
            return self._encode_tfidf(texts)

        try:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self.embedding_model)
            embeddings = self._encoder.encode(texts, normalize_embeddings=True, show_progress_bar=True)
            return np.asarray(embeddings, dtype="float32")
        except Exception as exc:
            warnings.warn(
                f"Falling back to TF-IDF vectors because sentence embeddings failed: {exc}"
            )
            return self._encode_tfidf(texts)

    def _encode_tfidf(self, texts: list[str]) -> np.ndarray:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vector_backend = "tfidf"
        self._vectorizer = TfidfVectorizer(max_features=384, ngram_range=(1, 2))
        matrix = self._vectorizer.fit_transform(texts).astype("float32")
        embeddings = matrix.toarray()
        return self._normalize(embeddings)

    def _encode_query(self, query: str) -> np.ndarray:
        if self._encoder is not None:
            embedding = self._encoder.encode([f"query: {query}"], normalize_embeddings=True)
            return np.asarray(embedding, dtype="float32")
        if self._vectorizer is None:
            raise RuntimeError("Vector retriever was not initialized")
        embedding = self._vectorizer.transform([query]).astype("float32").toarray()
        return self._normalize(embedding)

    @staticmethod
    def _normalize(embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        return np.asarray(embeddings / norms, dtype="float32")

    @staticmethod
    def _build_faiss_index(embeddings: np.ndarray):
        import faiss

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        return index

    def _retrieve_bm25(self, query: str, top_k: int) -> list[SearchResult]:
        scores = self._bm25.get_scores(tokenize_vi(query))
        indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:top_k]
        return [
            SearchResult(self.chunks[idx], float(scores[idx]), "bm25", rank)
            for rank, idx in enumerate(indices, start=1)
        ]

    def _retrieve_vector(self, query: str, top_k: int) -> list[SearchResult]:
        scores, indices = self._index.search(self._encode_query(query), top_k)
        return [
            SearchResult(self.chunks[int(idx)], float(score), "faiss_vector", rank)
            for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1)
            if idx >= 0
        ]

    @staticmethod
    def _reciprocal_rank_fusion(result_lists: list[list[SearchResult]], top_k: int) -> list[SearchResult]:
        scores: dict[str, float] = defaultdict(float)
        chunks: dict[str, LegalChunk] = {}
        for results in result_lists:
            for rank, result in enumerate(results, start=1):
                chunk_id = result.chunk.chunk_id
                chunks[chunk_id] = result.chunk
                scores[chunk_id] += 1.0 / (60 + rank)

        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        return [
            SearchResult(chunks[chunk_id], float(score), "hybrid_rrf", rank)
            for rank, (chunk_id, score) in enumerate(ordered, start=1)
        ]
