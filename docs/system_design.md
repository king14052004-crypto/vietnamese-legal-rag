# Vietnamese Legal RAG System Design

## Notebook-First Workflow

```text
01_data_exploration.ipynb
  -> explore corpus and filtering
02_retrieval_experiments.ipynb
  -> compare BM25 / vector / weighted hybrid / Hybrid RRF / MMR
  -> select Hybrid RRF from retrieval metrics
03_ragas_evaluation.ipynb
  -> generate answers with the selected pipeline
  -> evaluate answer quality on an 8-question manual golden-set smoke test
```

Experimental retrieval and evaluation code stays in notebooks. Deploy-facing Python code only keeps data preparation, the selected retrieval pipeline, Gemini generation, CLI, and Streamlit UI.

The answer-evaluation report is intentionally scoped as a smoke/regression signal. A broader benchmark should add a reviewed 30-50 question synthetic testset generated with `ragas.testset.TestsetGenerator` or an equivalent curated workflow.

## Production Pipeline

```text
Filtered Vietnamese labor-law corpus
  -> legal-aware chunks
  -> BM25 index
  -> FAISS vector index
  -> Reciprocal Rank Fusion
  -> grounded Gemini answer with citations
  -> CLI / Streamlit UI
```

## Selected Retrieval Method

Hybrid RRF is the production default because it combines:

- exact legal terminology recall from BM25,
- semantic matching from multilingual sentence embeddings and FAISS,
- stable rank-level fusion without score-scale normalization.

Weighted fusion and MMR remain notebook experiments, not production branches.

## UI

The local UI lives in `app/streamlit_app.py`. It defaults to retrieval-only mode with TF-IDF fallback so the demo works without a Gemini key. Gemini generation can be enabled after `GEMINI_API_KEY` or `GEMINI_API_KEYS` is configured.
