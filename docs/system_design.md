# Vietnamese Legal RAG System Design

## Notebook-First Workflow

```text
01_data_exploration.ipynb
  -> inspect raw data, plot article/length distributions, filter corpus with Gemini
02_retrieval_experiments.ipynb
  -> compare BM25 / vector / hybrid / Hybrid RRF / Cross-Encoder / MMR
  -> rank retrieval candidates only, without final selection
03_ragas_evaluation.ipynb
  -> generate answers for every retrieval candidate
  -> evaluate answer quality on the frozen RAGAS-generated benchmark
  -> select the final method using RAGAS generation metrics only
```

Experimental filtering, retrieval, and evaluation code stays in notebooks. Deploy-facing Python code only keeps data loading, chunking, the selected retrieval pipeline, Gemini generation, CLI, and Streamlit UI.

The evaluation workflow freezes RAGAS-generated synthetic benchmark questions in `reports/ragas_questions.json`, compares retrieval methods in notebook 02, then runs `ragas.evaluate()` answer metrics and final selection in notebook 03. Cached artifacts are marked explicitly when Gemini quota or keys are unavailable.

## Production Pipeline

```text
Gemini-filtered Vietnamese labor-law corpus
  -> legal-aware chunks
  -> BM25 index
  -> optional FAISS vector index for embedding experiments
  -> selected Hybrid + RRF + Cross-Encoder + MMR retrieval
  -> grounded Gemini answer with citations
  -> CLI / Streamlit UI
```

## Final Retrieval Method

The final notebook retriever is selected in notebook 03, after answer-level evaluation. The compared retrievers include:

- exact legal terminology recall from BM25,
- dense semantic reranking with FAISS,
- stable rank-level fusion without score-scale normalization,
- local Cross-Encoder reranking,
- MMR diversity.

Notebook 03 selected `Hybrid + RRF + Cross-Encoder + MMR` by `generation_score`. The local CLI/UI keeps that selected method while using memory-aware fallbacks on very large chunk sets so the full generated corpus can run on a laptop without hardcoded query terms or a prebuilt vector index.

## CLI / UI

The CLI and local UI default to a simple full-corpus retrieval flow that works on a laptop without a Gemini key. Gemini generation can be enabled after `GEMINI_API_KEY` or `GEMINI_API_KEYS` is configured.
