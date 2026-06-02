# Vietnamese Legal RAG

Portfolio project for Vietnamese labor-law Retrieval-Augmented Generation.

## What this project demonstrates

- Public Vietnamese legal corpus ingestion from Hugging Face/VBPL.
- Labor-law filtering and legal-aware chunking.
- Sparse retrieval with BM25.
- Dense retrieval with FAISS.
- Hybrid weighted fusion and Hybrid + RRF.
- MMR diversity filtering.
- Optional Cross-Encoder reranking.
- Gemini answer generation with round-robin batch API keys.
- Retrieval evaluation and RAGAS-style answer evaluation.

## Setup

```bash
python -m pip install -r requirements.txt
```

Optional Gemini config:

```bash
cp .env.example .env
export GEMINI_API_KEY=...
# or
export GEMINI_API_KEYS=key_1,key_2,key_3
export GEMINI_MODEL=gemini-3.1-flash-lite
```

The default uses AI Studio Gemini API (`google-genai`) with `gemini-3.1-flash-lite`.

## Data

Primary corpus:

- https://huggingface.co/datasets/th1nhng0/vietnamese-legal-documents

Regenerate the filtered labor-law corpus:

```bash
python -m src.data.build_corpus --max-docs 500 --scan-limit 30000
```

Build chunks:

```bash
python -m src.pipeline.build_indices
```

## Experiment notebooks

Run comparison and evaluation in notebooks:

```text
notebooks/02_retrieval_experiments.ipynb
notebooks/03_ragas_evaluation.ipynb
```

The Python modules under `src/` are intentionally kept simple so they can be reused by the CLI app or a future FastAPI/Streamlit deploy.

## Retrieval evaluation from CLI

Fast smoke evaluation without downloading sentence-transformer models:

```bash
python -m src.evaluation.evaluate_retrieval --tfidf-fallback
```

Full retrieval evaluation with multilingual sentence embeddings + FAISS:

```bash
python -m src.evaluation.evaluate_retrieval
```

Compared methods:

- `bm25`
- `vector`
- `hybrid`
- `hybrid_rrf`
- `hybrid_rrf_mmr`

## Ask questions

Without LLM, inspect retrieved sources:

```bash
python app.py "Người lao động đơn phương chấm dứt hợp đồng cần báo trước bao lâu?" --no-llm --tfidf-fallback
```

With Gemini:

```bash
python app.py "Người lao động đơn phương chấm dứt hợp đồng cần báo trước bao lâu?"
```

## Streamlit UI

Run a local demo UI:

```bash
streamlit run app/streamlit_app.py
```

The UI defaults to retrieval-only mode with TF-IDF fallback so it starts quickly.
Enable `Generate Gemini answer` after setting `GEMINI_API_KEYS`.

## Generate answers and evaluate RAGAS-style metrics

```bash
python -m src.pipeline.generate_answers --method hybrid_rrf
python -m src.evaluation.evaluate_ragas
```

`evaluate_ragas` uses AI Studio Gemini API via `google-genai` to judge RAGAS-style metrics:
faithfulness, answer relevancy, and context precision. If GenAI judging fails for a sample,
it falls back to a lexical proxy for that sample and records the reason.

## Selected method/model for future deploy

- Retrieval: `hybrid_rrf`
- Vector backend: FAISS
- LLM: Gemini 3.1 Flash Lite via AI Studio `google-genai`
- API key strategy: `BatchGeminiClient` round-robin over `GEMINI_API_KEYS`
- UI: Streamlit (`app/streamlit_app.py`)

Why this choice:
- BM25 handles exact legal terms.
- FAISS dense retrieval handles natural-language questions.
- RRF avoids fragile BM25/vector score normalization.
- MMR and Cross-Encoder reranking remain optional for quality/latency trade-offs.

## Recommended balanced retrieval method

Run `python -m src.evaluation.evaluate_retrieval --tfidf-fallback` to generate `reports/evaluation.md`.
The selected method is based on a weighted score of Recall@5, MRR, and nDCG@5.

## Legal disclaimer

This demo is for legal information retrieval only. It is not a substitute for professional legal advice.
