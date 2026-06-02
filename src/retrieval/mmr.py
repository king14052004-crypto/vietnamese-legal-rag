import numpy as np

from src.data.schema import SearchResult


def apply_mmr(results: list[SearchResult], top_k: int = 8, lambda_mult: float = 0.7) -> list[SearchResult]:
    if len(results) <= top_k:
        return results

    texts = [f"{result.chunk.title} {result.chunk.text}" for result in results]
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectors = TfidfVectorizer(max_features=512).fit_transform(texts).toarray()
    vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12)
    relevance = np.asarray([result.score for result in results], dtype="float32")
    relevance = (relevance - relevance.min()) / (relevance.max() - relevance.min() + 1e-12)

    selected = [int(np.argmax(relevance))]
    remaining = set(range(len(results))) - set(selected)
    while remaining and len(selected) < top_k:
        best_idx = None
        best_score = -float("inf")
        for idx in remaining:
            diversity_penalty = max(float(vectors[idx] @ vectors[chosen]) for chosen in selected)
            score = lambda_mult * float(relevance[idx]) - (1 - lambda_mult) * diversity_penalty
            if score > best_score:
                best_score = score
                best_idx = idx
        selected.append(best_idx)
        remaining.remove(best_idx)

    reranked = []
    for rank, idx in enumerate(selected, start=1):
        result = results[idx]
        reranked.append(SearchResult(result.chunk, result.score, f"{result.method}+mmr", rank))
    return reranked
