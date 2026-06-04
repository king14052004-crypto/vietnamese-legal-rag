# Retrieval Evaluation

- Question count: `30`
- Benchmark chunks: `128519`
- RAGAS retrieval metrics backend: `ragas_evaluate_non_llm`
- Selected method: `hybrid`

| Method | MRR | nDCG@5 | Lexical CP | Lexical CR | RAGAS CP | RAGAS CR | Selection score |
|---|---:|---:|---:|---:|---:|---:|---:|
| hybrid | 0.967 | 0.971 | 0.514 | 0.841 | 0.000 | 0.000 | 0.533 |
| bm25 | 0.975 | 0.958 | 0.579 | 0.845 | 0.000 | 0.000 | 0.532 |
| hybrid_rrf_mmr | 0.967 | 0.939 | 0.527 | 0.846 | 0.000 | 0.000 | 0.525 |
| hybrid_rrf | 0.967 | 0.938 | 0.531 | 0.845 | 0.000 | 0.000 | 0.524 |
| vector | 0.957 | 0.932 | 0.415 | 0.710 | 0.000 | 0.000 | 0.520 |

## Figures

- `reports/figures/retrieval_selection_score.png`
- `reports/figures/retrieval_metric_comparison.png`
- `reports/figures/retrieval_query_heatmap.png`
