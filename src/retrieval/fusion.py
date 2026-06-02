from collections import defaultdict

from src.data.schema import LegalChunk, SearchResult


def minmax_normalize(results: list[SearchResult]) -> dict[str, float]:
    if not results:
        return {}
    scores = [result.score for result in results]
    low, high = min(scores), max(scores)
    if high == low:
        return {result.chunk.chunk_id: 1.0 for result in results}
    return {result.chunk.chunk_id: (result.score - low) / (high - low) for result in results}


def weighted_hybrid(
    sparse_results: list[SearchResult],
    vector_results: list[SearchResult],
    alpha: float = 0.55,
    top_k: int = 10,
) -> list[SearchResult]:
    chunks = {result.chunk.chunk_id: result.chunk for result in sparse_results + vector_results}
    sparse_scores = minmax_normalize(sparse_results)
    vector_scores = minmax_normalize(vector_results)
    combined = {}
    for chunk_id in chunks:
        combined[chunk_id] = (1 - alpha) * sparse_scores.get(chunk_id, 0.0) + alpha * vector_scores.get(chunk_id, 0.0)
    ordered = sorted(combined.items(), key=lambda item: item[1], reverse=True)[:top_k]
    return [
        SearchResult(chunks[chunk_id], float(score), "hybrid_weighted", rank + 1)
        for rank, (chunk_id, score) in enumerate(ordered)
    ]


def reciprocal_rank_fusion(result_lists: list[list[SearchResult]], top_k: int = 10, k: int = 60) -> list[SearchResult]:
    scores: dict[str, float] = defaultdict(float)
    chunks: dict[str, LegalChunk] = {}
    for results in result_lists:
        for rank, result in enumerate(results, start=1):
            chunks[result.chunk.chunk_id] = result.chunk
            scores[result.chunk.chunk_id] += 1.0 / (k + rank)
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
    return [
        SearchResult(chunks[chunk_id], float(score), "hybrid_rrf", rank + 1)
        for rank, (chunk_id, score) in enumerate(ordered)
    ]
