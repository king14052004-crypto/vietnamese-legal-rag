# Retrieval Evaluation

- Benchmark: `benchmark_questions.json` (30 paraphrased questions with gold chunk labels)
- Corpus: `labor_corpus_sample.jsonl` (2496 chunks)
- Metrics are exact: a hit means the gold chunk (the chunk the question was generated from) appears in the top 5.
- Ranking score: `0.4*recall@5 + 0.3*mrr + 0.2*ndcg@5 + 0.1*doc_recall@5`

| Method | Recall@5 | MRR | nDCG@5 | Doc Recall@5 | Score |
|---|---|---|---|---|---|
| hybrid_rrf_cross_encoder | 0.500 | 0.276 | 0.332 | 0.700 | 0.419 |
| hybrid_rrf_cross_encoder_mmr | 0.433 | 0.258 | 0.303 | 0.600 | 0.371 |
| vector | 0.400 | 0.228 | 0.271 | 0.667 | 0.349 |
| hybrid | 0.400 | 0.202 | 0.251 | 0.600 | 0.331 |
| hybrid_rrf | 0.300 | 0.158 | 0.194 | 0.567 | 0.263 |
| bm25 | 0.200 | 0.157 | 0.167 | 0.333 | 0.194 |

Best method by retrieval score: **hybrid_rrf_cross_encoder**.

Note: questions are paraphrased away from the legal wording, so keyword-only
retrieval is expected to miss some questions and the metrics can separate methods.
