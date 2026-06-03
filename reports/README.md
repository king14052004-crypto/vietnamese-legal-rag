# Reports

Notebook-generated evaluation artifacts live here:

- `evaluation.md`: retrieval comparison summary.
- `retrieval_evaluation.json`: detailed metrics and per-query top results.
- `rag_answers.json`: Gemini answers with retrieved contexts for the manual golden set.
- `ragas_evaluation.json`: RAGAS-style AI Studio GenAI smoke metrics for faithfulness, answer relevancy, and context precision on 8 manually curated questions.

Generate and analyze these files from `notebooks/02_retrieval_experiments.ipynb` and `notebooks/03_ragas_evaluation.ipynb`. Deploy-facing Python code does not generate benchmark reports.

Treat the current answer-evaluation numbers as regression/smoke-test evidence only. A portfolio-grade benchmark should add a reviewed 30-50 question synthetic set generated with `ragas.testset.TestsetGenerator` or an equivalent curated process.
