# Retrieval Evaluation

- Documents: 122
- Chunks: 1577

| Method | Recall@5 | MRR | nDCG@5 | Citation coverage |
|---|---:|---:|---:|---:|
| bm25 | 1.000 | 1.000 | 0.984 | 1.000 |
| vector | 1.000 | 0.938 | 0.964 | 1.000 |
| hybrid | 1.000 | 0.938 | 0.960 | 1.000 |
| hybrid_rrf | 1.000 | 1.000 | 0.993 | 1.000 |
| hybrid_rrf_mmr | 1.000 | 1.000 | 0.979 | 1.000 |

Recommended balanced method: `hybrid_rrf`.
