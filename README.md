# Vietnamese Legal RAG

Portfolio project for Vietnamese labor-law Retrieval-Augmented Generation.

## Workflow

The project follows a notebook-first workflow:

1. Explore and filter the public Vietnamese legal corpus in `notebooks/01_data_exploration.ipynb`.
2. Compare retrieval methods in `notebooks/02_retrieval_experiments.ipynb`.
3. Select the balanced retrieval method from notebook metrics.
4. Run the manual golden-set answer smoke evaluation in `notebooks/03_ragas_evaluation.ipynb`.
5. Keep only the selected pipeline in `src/` for the CLI and Streamlit demo.

The notebook-selected retrieval method is Hybrid RRF: BM25 + FAISS vector retrieval with Reciprocal Rank Fusion. The CLI and Streamlit demo use a memory-aware local fallback on the full generated corpus so the project runs reliably without prebuilt vector artifacts.

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
python -m src.data.build_corpus
```

The generated `data/processed/labor_corpus.jsonl` file is ignored by Git because the full filtered corpus is large.

## Experiments

Run these notebooks in order:

```text
notebooks/01_data_exploration.ipynb
notebooks/02_retrieval_experiments.ipynb
notebooks/03_ragas_evaluation.ipynb
```

`02_retrieval_experiments.ipynb` contains BM25, vector, weighted hybrid, Hybrid RRF, MMR, and retrieval metrics directly in the notebook. `03_ragas_evaluation.ipynb` contains answer generation and RAGAS-style AI Studio evaluation for an 8-question manually curated golden set.

The deploy-facing Python code does not contain experimental methods.

Notebook 02 defaults to a fast TF-IDF smoke-test fallback. Set `USE_TFIDF_FALLBACK=false` before running it to generate the production comparison report with multilingual sentence embeddings and FAISS.

The notebook 03 scores are smoke/regression signals, not a full benchmark. For a stronger benchmark, generate 30-50 synthetic questions with `ragas.testset.TestsetGenerator`, manually review the generated questions, freeze the accepted set, and then run the same answer-quality metrics.

## CLI

Run a question:

```bash
python app.py "Người lao động đơn phương chấm dứt hợp đồng cần báo trước bao lâu?"
```

If `GEMINI_API_KEY` or `GEMINI_API_KEYS` is configured, the CLI generates an answer with `gemini-3.1-flash-lite`. Without a key, it prints the retrieved legal sources only.

## Streamlit UI

Run the local demo:

```bash
streamlit run app/streamlit_app.py
```

The UI starts in retrieval-only mode so it works without a Gemini key. Enable Gemini answer generation after setting `GEMINI_API_KEY` or `GEMINI_API_KEYS`.

## Selected Deploy Configuration

- Retrieval: Hybrid RRF experiments in notebooks; memory-aware local retrieval in CLI/UI
- Sparse retrieval: BM25 / memory-aware keyword scan for large local corpora
- Vector backend: FAISS for embedding experiments
- Embedding model: `intfloat/multilingual-e5-small`
- LLM: `gemini-3.1-flash-lite`
- API key strategy: round-robin over `GEMINI_API_KEYS`
- UI: Streamlit

## Legal Disclaimer

This demo is for legal information retrieval only. It is not a substitute for professional legal advice.
