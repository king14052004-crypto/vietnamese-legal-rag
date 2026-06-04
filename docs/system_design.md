# Vietnamese Legal RAG System Design

## Notebook-First Workflow

```text
01_data_exploration.ipynb
  -> explore corpus and filtering
02_retrieval_experiments.ipynb
  -> compare BM25 / vector / weighted hybrid / Hybrid RRF / MMR
  -> select the retrieval method with a fixed score over retrieval and RAGAS context metrics
03_ragas_evaluation.ipynb
  -> generate answers with the selected pipeline
  -> evaluate answer quality on the frozen RAGAS-generated benchmark
```

Experimental retrieval and evaluation code stays in notebooks. Deploy-facing Python code only keeps data preparation, the selected retrieval pipeline, Gemini generation, CLI, and Streamlit UI.

The evaluation workflow freezes RAGAS-generated synthetic benchmark questions in `reports/ragas_questions.json`, compares retrieval methods in notebook 02, then runs `ragas.evaluate()` answer metrics in notebook 03. Cached artifacts are marked explicitly when Gemini quota or keys are unavailable.

## Production Pipeline

```text
Filtered Vietnamese labor-law corpus
  -> legal-aware chunks
  -> BM25 index
  -> optional FAISS vector index for embedding experiments
  -> selected hybrid retrieval
  -> grounded Gemini answer with citations
  -> CLI / Streamlit UI
```

## Selected Retrieval Method

The selected notebook retriever is the method with the highest fixed selection score. The compared retrievers combine:

- exact legal terminology recall from BM25,
- semantic matching from multilingual sentence embeddings and FAISS,
- stable rank-level fusion without score-scale normalization.

Weighted fusion, RRF, and MMR remain notebook experiments, not production branches. The local CLI/UI uses a memory-aware fallback on very large chunk sets so the full generated corpus can run on a laptop without hardcoded query terms or a prebuilt vector index.

## CLI / UI

The CLI and local UI default to a simple full-corpus retrieval flow that works on a laptop without a Gemini key. Gemini generation can be enabled after `GEMINI_API_KEY` or `GEMINI_API_KEYS` is configured.
