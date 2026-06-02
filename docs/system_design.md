# Vietnamese Legal RAG System Design

## Pipeline

```text
Hugging Face legal corpus
  -> labor-law filter
  -> HTML/text normalization
  -> article-aware chunking
  -> BM25 index
  -> FAISS dense index
  -> hybrid fusion / RRF / MMR / optional rerank
  -> Gemini grounded answer with citations
  -> retrieval metrics + RAGAS-style AI Studio GenAI evaluation
```

## Retrieval methods compared

| Method | Purpose |
|---|---|
| BM25 / SparseRetrieval | Strong for exact legal phrases, article names, identifiers. |
| FAISS vector retrieval | Strong for natural-language questions and paraphrases. |
| Hybrid weighted | Blends normalized sparse and dense scores. |
| Hybrid + RRF | Rank-level fusion, less sensitive to incompatible score scales. |
| Hybrid + RRF + MMR | Keeps relevance while reducing duplicate context. |
| Cross-Encoder rerank | Optional quality boost for top candidates; slower. |
| HyDE | Optional query expansion before retrieval when a Gemini key is available. |

## Recommended default

Use `hybrid_rrf` for the portfolio demo because it balances:

- exact legal terminology recall,
- semantic matching,
- score-scale robustness,
- stable ranking across incompatible BM25/vector score scales.

Use MMR when retrieved contexts are too repetitive. Use Cross-Encoder reranking only when latency and model download size are acceptable.
