# Retrieval Evaluation

- Documents: 122
- Chunks: 1577
- Vector backend: `sentence_transformer`

| Method | Recall@5 | MRR | nDCG@5 | Citation coverage | Balanced score |
|---|---:|---:|---:|---:|---:|
| bm25 | 1.000 | 1.000 | 0.984 | 1.000 | 0.997 |
| vector | 1.000 | 0.938 | 0.933 | 1.000 | 0.965 |
| hybrid | 1.000 | 0.938 | 0.948 | 1.000 | 0.968 |
| hybrid_rrf | 1.000 | 1.000 | 0.984 | 1.000 | 0.997 |
| hybrid_rrf_mmr | 1.000 | 1.000 | 0.971 | 1.000 | 0.994 |

Recommended balanced method: `hybrid_rrf`.
