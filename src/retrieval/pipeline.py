import warnings
from collections import defaultdict
from heapq import nlargest

import numpy as np
from rank_bm25 import BM25Okapi

from src.data.clean_text import tokenize_vi
from src.data.schema import LegalChunk, SearchResult

DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
DEFAULT_RETRIEVAL_METHOD = "hybrid_rrf_cross_encoder_mmr"
LARGE_CORPUS_CHUNK_THRESHOLD = 100_000


class RetrievalPipeline:
    def __init__(
        self,
        chunks: list[LegalChunk],
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        cross_encoder_model: str = DEFAULT_CROSS_ENCODER_MODEL,
        retrieval_method: str = DEFAULT_RETRIEVAL_METHOD,
        use_tfidf_fallback: bool = False,
    ):
        if not chunks:
            raise ValueError("RetrievalPipeline requires at least one chunk")

        self.chunks = chunks
        self.embedding_model = embedding_model
        self.cross_encoder_model = cross_encoder_model
        self.retrieval_method = retrieval_method
        self._encoder = None
        self._cross_encoder = None
        self._vectorizer = None
        self.vector_backend = "sentence_transformer"
        self._keyword_only = use_tfidf_fallback and len(chunks) > LARGE_CORPUS_CHUNK_THRESHOLD
        self._bm25 = (
            None
            if self._keyword_only
            else BM25Okapi([tokenize_vi(self._chunk_text(chunk)) for chunk in chunks])
        )
        if self._keyword_only:
            self.vector_backend = "keyword_scan"
            self._embeddings = None
            self._index = None
        else:
            self._embeddings = self._encode_corpus(use_tfidf_fallback)
            self._index = self._build_faiss_index(self._embeddings)

    def retrieve(self, query: str, top_k: int = 8) -> list[SearchResult]:
        candidate_k = min(len(self.chunks), max(30, top_k * 5))
        sparse_results = self._retrieve_bm25(query, candidate_k)
        vector_results = self._retrieve_vector(query, candidate_k)
        if self.retrieval_method == "hybrid":
            return self._weighted_hybrid(sparse_results, vector_results, top_k)
        if self.retrieval_method == "hybrid_rrf_cross_encoder_mmr":
            fused = self._rrf([sparse_results, vector_results], candidate_k)
            reranked = self._cross_encoder_rerank(query, fused[:candidate_k])
            return self._mmr(query, reranked, top_k)
        return self._weighted_hybrid(sparse_results, vector_results, top_k)

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
        if self._bm25 is None:
            return self._retrieve_keyword_scan(query, top_k)
        scores = self._bm25.get_scores(tokenize_vi(query))
        indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:top_k]
        return [
            SearchResult(self.chunks[idx], float(scores[idx]), "bm25", rank)
            for rank, idx in enumerate(indices, start=1)
        ]

    def _retrieve_keyword_scan(self, query: str, top_k: int) -> list[SearchResult]:
        terms = set(token for token in tokenize_vi(query) if len(token) > 1)
        if not terms:
            return []

        doc_stats = []
        doc_freqs: dict[str, int] = defaultdict(int)
        total_len = 0
        for idx, chunk in enumerate(self.chunks):
            counts: dict[str, int] = defaultdict(int)
            tokens = tokenize_vi(self._chunk_text(chunk))
            total_len += len(tokens)
            for token in tokens:
                if token in terms:
                    counts[token] += 1
            if counts:
                for token in counts:
                    doc_freqs[token] += 1
                doc_stats.append((idx, counts, len(tokens)))

        if not doc_stats:
            return []

        avg_len = total_len / len(self.chunks)
        scored = []
        for idx, counts, doc_len in doc_stats:
            score = 0.0
            for token, term_freq in counts.items():
                inverse_doc_freq = np.log(
                    1.0 + (len(self.chunks) - doc_freqs[token] + 0.5) / (doc_freqs[token] + 0.5)
                )
                denominator = term_freq + 1.5 * (0.25 + 0.75 * doc_len / avg_len)
                score += inverse_doc_freq * (term_freq * 2.5) / denominator
            scored.append((float(score), idx))

        top = nlargest(top_k, scored, key=lambda item: item[0])
        return [
            SearchResult(self.chunks[idx], score, "keyword_scan", rank)
            for rank, (score, idx) in enumerate(top, start=1)
        ]

    def _retrieve_vector(self, query: str, top_k: int) -> list[SearchResult]:
        if self._index is None:
            return []
        scores, indices = self._index.search(self._encode_query(query), top_k)
        return [
            SearchResult(self.chunks[int(idx)], float(score), "faiss_vector", rank)
            for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1)
            if idx >= 0
        ]

    def _rrf(self, result_lists: list[list[SearchResult]], top_k: int, k: int = 60) -> list[SearchResult]:
        scores: dict[str, float] = defaultdict(float)
        chunks: dict[str, LegalChunk] = {}
        for results in result_lists:
            for rank, result in enumerate(results, start=1):
                scores[result.chunk.chunk_id] += 1.0 / (k + rank)
                chunks[result.chunk.chunk_id] = result.chunk

        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        return [
            SearchResult(chunks[chunk_id], float(score), "hybrid_rrf", rank)
            for rank, (chunk_id, score) in enumerate(ordered, start=1)
        ]

    def _cross_encoder_rerank(self, query: str, candidates: list[SearchResult]) -> list[SearchResult]:
        if not candidates:
            return []
        try:
            if self._cross_encoder is None:
                from sentence_transformers import CrossEncoder

                self._cross_encoder = CrossEncoder(self.cross_encoder_model, max_length=512)
            pairs = [(query, self._chunk_text(result.chunk)) for result in candidates]
            scores = self._cross_encoder.predict(pairs, batch_size=16, show_progress_bar=False)
            ranked = sorted(zip(candidates, scores), key=lambda item: float(item[1]), reverse=True)
            return [
                SearchResult(result.chunk, float(score), "hybrid_rrf_cross_encoder", rank)
                for rank, (result, score) in enumerate(ranked, start=1)
            ]
        except Exception as exc:
            warnings.warn(f"Falling back to Hybrid RRF because Cross-Encoder reranking failed: {exc}")
            return candidates

    def _mmr(self, query: str, candidates: list[SearchResult], top_k: int, lambda_mult: float = 0.75) -> list[SearchResult]:
        relevance_scores = self._minmax(candidates)
        selected: list[SearchResult] = []
        remaining = candidates[:]
        while remaining and len(selected) < top_k:
            best_idx, best_score = 0, -float("inf")
            for idx, result in enumerate(remaining):
                relevance = max(
                    relevance_scores.get(result.chunk.chunk_id, 0.0),
                    self._query_overlap(query, result.chunk),
                )
                redundancy = max(
                    (self._chunk_overlap(result.chunk, chosen.chunk) for chosen in selected),
                    default=0.0,
                )
                score = lambda_mult * relevance - (1 - lambda_mult) * redundancy
                if score > best_score:
                    best_idx, best_score = idx, score
            chosen = remaining.pop(best_idx)
            selected.append(
                SearchResult(chosen.chunk, float(best_score), "hybrid_rrf_cross_encoder_mmr", len(selected) + 1)
            )
        return selected

    @staticmethod
    def _token_set(text: str) -> set[str]:
        return {token for token in tokenize_vi(text) if len(token) > 1}

    def _query_overlap(self, query: str, chunk: LegalChunk) -> float:
        query_tokens = self._token_set(query)
        chunk_tokens = self._token_set(self._chunk_text(chunk))
        if not query_tokens or not chunk_tokens:
            return 0.0
        return len(query_tokens & chunk_tokens) / len(query_tokens)

    def _chunk_overlap(self, left: LegalChunk, right: LegalChunk) -> float:
        left_tokens = self._token_set(self._chunk_text(left))
        right_tokens = self._token_set(self._chunk_text(right))
        if not left_tokens or not right_tokens:
            return 0.0
        return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)

    @staticmethod
    def _minmax(results: list[SearchResult]) -> dict[str, float]:
        if not results:
            return {}
        scores = [result.score for result in results]
        low, high = min(scores), max(scores)
        if high == low:
            return {result.chunk.chunk_id: 1.0 for result in results}
        return {result.chunk.chunk_id: (result.score - low) / (high - low) for result in results}

    def _weighted_hybrid(
        self,
        sparse_results: list[SearchResult],
        vector_results: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        if not vector_results:
            return [
                SearchResult(result.chunk, result.score, "hybrid", rank)
                for rank, result in enumerate(sparse_results[:top_k], start=1)
            ]

        sparse_scores = self._minmax(sparse_results)
        vector_scores = self._minmax(vector_results)
        chunks = {result.chunk.chunk_id: result.chunk for result in sparse_results + vector_results}
        scores: dict[str, float] = defaultdict(float)
        for chunk_id, score in sparse_scores.items():
            scores[chunk_id] += 0.55 * score
        for chunk_id, score in vector_scores.items():
            scores[chunk_id] += 0.45 * score

        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        return [
            SearchResult(chunks[chunk_id], float(score), "hybrid", rank)
            for rank, (chunk_id, score) in enumerate(ordered, start=1)
        ]
