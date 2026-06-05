# Retrieval Evaluation

- Question count: `30`
- Benchmark chunks: `128519`
- Selection stage: `retrieval_diagnostics_only`
- Dense embedding model: `intfloat/multilingual-e5-small`
- Vector pool size: `300` BM25 candidates per query
- Cross-Encoder model: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`
- Retrieval score formula: `0.30*recall@5 + 0.25*mrr + 0.25*ndcg@5 + 0.10*lexical_context_precision + 0.10*lexical_context_recall`
- RAGAS context note: `Diagnostic only; not used in retrieval_score.`

| Rank | Method | Recall@5 | MRR | nDCG@5 | Lexical CP | Lexical CR | RAGAS CP | RAGAS CR | Retrieval score |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Hybrid + RRF + Cross-Encoder | 1.000 | 1.000 | 0.970 | 0.110 | 0.910 | N/A | N/A | 0.894 |
| 2 | Hybrid + RRF + Cross-Encoder + MMR | 1.000 | 1.000 | 0.971 | 0.106 | 0.908 | N/A | N/A | 0.894 |
| 3 | Hybrid | 1.000 | 1.000 | 0.973 | 0.109 | 0.888 | N/A | N/A | 0.893 |
| 4 | Hybrid + RRF | 1.000 | 1.000 | 0.957 | 0.109 | 0.898 | N/A | N/A | 0.890 |
| 5 | BM25 | 1.000 | 1.000 | 0.966 | 0.094 | 0.883 | N/A | N/A | 0.889 |
| 6 | Vector | 1.000 | 0.978 | 0.961 | 0.110 | 0.859 | N/A | N/A | 0.882 |

## Figures

- `reports/figures/retrieval_score.png`
- `reports/figures/retrieval_metric_comparison.png`
- `reports/figures/retrieval_query_heatmap.png`
