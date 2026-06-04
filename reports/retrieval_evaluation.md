# Retrieval Evaluation

- Testset size: `30`
- Benchmark chunks: `3500`
- RAGAS retrieval metrics backend: `ragas_evaluate_non_llm`
- Selected method: `hybrid`

| Method | MRR | nDCG@5 | Lexical CP | Lexical CR | RAGAS CP | RAGAS CR | Selection score |
|---|---:|---:|---:|---:|---:|---:|---:|
| hybrid | 1.000 | 0.959 | 0.570 | 0.888 | 0.031 | 0.100 | 0.567 |
| hybrid_rrf | 1.000 | 0.941 | 0.553 | 0.874 | 0.033 | 0.100 | 0.563 |
| vector | 1.000 | 0.932 | 0.473 | 0.788 | 0.050 | 0.067 | 0.559 |
| hybrid_rrf_mmr | 1.000 | 0.947 | 0.515 | 0.861 | 0.028 | 0.067 | 0.557 |
| bm25 | 1.000 | 0.976 | 0.589 | 0.885 | 0.008 | 0.033 | 0.553 |

## Figures

- `reports/figures/retrieval_selection_score.png`
- `reports/figures/retrieval_metric_comparison.png`
- `reports/figures/retrieval_query_heatmap.png`
