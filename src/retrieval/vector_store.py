import numpy as np

from src.data.schema import LegalChunk, SearchResult


class FaissVectorRetriever:
    def __init__(
        self,
        chunks: list[LegalChunk],
        model_name: str = "intfloat/multilingual-e5-small",
        use_tfidf_fallback: bool = False,
    ):
        self.chunks = chunks
        self.model_name = model_name
        self.use_tfidf_fallback = use_tfidf_fallback
        self.model = None
        self.vectorizer = None
        self.index = None
        self.embeddings = self._encode_corpus()
        self._build_faiss()

    def _encode_corpus(self) -> np.ndarray:
        texts = [f"passage: {chunk.title}\n{chunk.text}" for chunk in self.chunks]
        if self.use_tfidf_fallback:
            return self._encode_tfidf(texts)

        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.model_name)
            embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
            return np.asarray(embeddings, dtype="float32")
        except Exception:
            return self._encode_tfidf(texts)

    def _encode_tfidf(self, texts: list[str]) -> np.ndarray:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.use_tfidf_fallback = True
        self.vectorizer = TfidfVectorizer(max_features=384, ngram_range=(1, 2))
        matrix = self.vectorizer.fit_transform(texts).astype("float32")
        embeddings = matrix.toarray()
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        return embeddings / norms

    def _encode_query(self, query: str) -> np.ndarray:
        if self.model is not None:
            embedding = self.model.encode([f"query: {query}"], normalize_embeddings=True)
            return np.asarray(embedding, dtype="float32")
        if self.vectorizer is None:
            raise RuntimeError("Vector retriever was not initialized")
        embedding = self.vectorizer.transform([query]).astype("float32").toarray()
        norm = np.linalg.norm(embedding, axis=1, keepdims=True) + 1e-12
        return embedding / norm

    def _build_faiss(self) -> None:
        import faiss

        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)

    def retrieve(self, query: str, top_k: int = 10) -> list[SearchResult]:
        query_embedding = self._encode_query(query)
        scores, indices = self.index.search(query_embedding, top_k)
        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            if idx < 0:
                continue
            results.append(SearchResult(self.chunks[int(idx)], float(score), "faiss_vector", rank))
        return results
