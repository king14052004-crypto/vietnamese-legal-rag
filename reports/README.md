# Reports

Notebook-generated evaluation artifacts live here:

- `evaluation.md`: retrieval comparison summary.
- `retrieval_evaluation.json`: detailed metrics and per-query top results.
- `rag_answers.json`: Gemini answers with retrieved contexts.
- `ragas_evaluation.json`: RAGAS-style AI Studio GenAI metrics for faithfulness, answer relevancy, and context precision.

Generate and analyze these files from `notebooks/02_retrieval_experiments.ipynb` and `notebooks/03_ragas_evaluation.ipynb`. Deploy-facing Python code does not generate benchmark reports.
