# Vietnamese Legal RAG

Portfolio project for Vietnamese labor-law Retrieval-Augmented Generation.

## Workflow

The project follows a notebook-first workflow:

1. Explore and filter the public Vietnamese legal corpus in `notebooks/01_data_exploration.ipynb`.
2. Compare retrieval methods in `notebooks/02_retrieval_experiments.ipynb`.
3. Select the balanced retrieval method from notebook metrics.
4. Evaluate generated answers in `notebooks/03_ragas_evaluation.ipynb`.
5. Keep only the selected pipeline in `src/` for the CLI and Streamlit demo.

The selected retrieval method is Hybrid RRF: BM25 + FAISS vector retrieval with Reciprocal Rank Fusion.

## Setup

```bash
python -m pip install -r requirements.txt
```

Optional Gemini config:

```bash
cp .env.example .env
# Set GEMINI_API_KEY or comma-separated GEMINI_API_KEYS in .env.
```

## Data

Primary corpus:

- https://huggingface.co/datasets/th1nhng0/vietnamese-legal-documents

Regenerate the filtered labor-law corpus:

```bash
python -m src.data.build_corpus --max-docs 500 --scan-limit 30000
```

## Experiments

Run these notebooks in order:

```text
notebooks/01_data_exploration.ipynb
notebooks/02_retrieval_experiments.ipynb
notebooks/03_ragas_evaluation.ipynb
```

`02_retrieval_experiments.ipynb` contains BM25, vector, weighted hybrid, Hybrid RRF, MMR, and retrieval metrics directly in the notebook. `03_ragas_evaluation.ipynb` contains answer generation and RAGAS-style AI Studio evaluation directly in the notebook.

The deploy-facing Python code does not contain experimental methods.

Notebook 02 defaults to a fast TF-IDF smoke-test fallback. Set `USE_TFIDF_FALLBACK=false` before running it to generate the production comparison report with multilingual sentence embeddings and FAISS.

## CLI

Inspect retrieved sources without an LLM:

```bash
python app.py "Người lao động đơn phương chấm dứt hợp đồng cần báo trước bao lâu?" --no-llm --tfidf-fallback
```

Generate an answer with Gemini:

```bash
python app.py "Người lao động đơn phương chấm dứt hợp đồng cần báo trước bao lâu?"
```

`--tfidf-fallback` is intended for fast offline smoke tests. Without it, the pipeline uses multilingual sentence embeddings with FAISS.

## Streamlit UI

Run the local demo:

```bash
streamlit run app/streamlit_app.py
```

The UI defaults to retrieval-only mode with TF-IDF fallback so it starts quickly. Enable Gemini answer generation after setting `GEMINI_API_KEY` or `GEMINI_API_KEYS`.

## Selected Deploy Configuration

- Retrieval: Hybrid RRF
- Sparse retrieval: BM25
- Vector backend: FAISS
- Embedding model: `intfloat/multilingual-e5-small`
- LLM: `gemini-3.1-flash-lite`
- API key strategy: round-robin over `GEMINI_API_KEYS`
- UI: Streamlit

## Legal Disclaimer

This demo is for legal information retrieval only. It is not a substitute for professional legal advice.
