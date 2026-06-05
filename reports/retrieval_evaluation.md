# Retrieval Evaluation

- Question count: `30`
- Benchmark chunks: `128519`
- RAGAS retrieval metrics backend: `ragas_evaluate_non_llm`
- Final method selected here: `False`
- Next step: notebook 03 combines these candidates with answer/RAGAS evaluation.

| Rank | Method | MRR | nDCG@5 | Lexical CP | Lexical CR | RAGAS CP | RAGAS CR | Retrieval candidate score |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | hybrid | 0.967 | 0.971 | 0.514 | 0.841 | 0.000 | 0.000 | 0.533 |
| 2 | bm25 | 0.975 | 0.958 | 0.579 | 0.845 | 0.000 | 0.000 | 0.532 |
| 3 | hybrid_rrf_mmr | 0.967 | 0.939 | 0.527 | 0.846 | 0.000 | 0.000 | 0.525 |
| 4 | hybrid_rrf | 0.967 | 0.938 | 0.531 | 0.845 | 0.000 | 0.000 | 0.524 |
| 5 | vector | 0.957 | 0.932 | 0.415 | 0.710 | 0.000 | 0.000 | 0.520 |

## Figures

- `reports/figures/retrieval_candidate_score.png`
- `reports/figures/retrieval_metric_comparison.png`
- `reports/figures/retrieval_query_heatmap.png`
