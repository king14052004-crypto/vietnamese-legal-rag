from math import log2

from src.data.clean_text import normalize_for_match
from src.data.schema import SearchResult


def is_relevant(result: SearchResult, relevant_terms: list[str]) -> bool:
    text = normalize_for_match(f"{result.chunk.title} {result.chunk.text}")
    return any(term.lower() in text for term in relevant_terms)


def recall_at_k(results: list[SearchResult], relevant_terms: list[str], k: int) -> float:
    return 1.0 if any(is_relevant(result, relevant_terms) for result in results[:k]) else 0.0


def reciprocal_rank(results: list[SearchResult], relevant_terms: list[str]) -> float:
    for idx, result in enumerate(results, start=1):
        if is_relevant(result, relevant_terms):
            return 1.0 / idx
    return 0.0


def ndcg_at_k(results: list[SearchResult], relevant_terms: list[str], k: int) -> float:
    gains = [1.0 if is_relevant(result, relevant_terms) else 0.0 for result in results[:k]]
    dcg = sum(gain / log2(idx + 2) for idx, gain in enumerate(gains))
    ideal_gains = sorted(gains, reverse=True)
    idcg = sum(gain / log2(idx + 2) for idx, gain in enumerate(ideal_gains))
    return dcg / idcg if idcg else 0.0


def citation_coverage(results: list[SearchResult]) -> float:
    return sum(1 for result in results if result.chunk.title and result.chunk.doc_id) / max(len(results), 1)
