# Vietnamese Legal RAG

Portfolio project for Vietnamese labor-law Retrieval-Augmented Generation.

## Workflow

The project follows a notebook-first workflow:

1. Filter the public Vietnamese legal corpus with Gemini in `notebooks/01_data_exploration.ipynb`.
2. Compare retrieval methods in `notebooks/02_retrieval_experiments.ipynb`.
3. Select the balanced retrieval method from notebook metrics.
4. Generate answers and run RAGAS answer evaluation in `notebooks/03_ragas_evaluation.ipynb`.
5. Keep only the selected pipeline in `src/` for the CLI and Streamlit demo.

The current notebook-selected retrieval method is weighted `hybrid`: BM25 + TF-IDF/FAISS vector retrieval selected by a fixed score over lexical retrieval metrics and RAGAS context metrics. The CLI and Streamlit demo use a memory-aware local fallback on the generated corpus so the project runs reliably without prebuilt vector artifacts.

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

Regenerate the filtered labor-law corpus by running `notebooks/01_data_exploration.ipynb`.
The notebook uses `gemini-3.1-flash-lite` as a cached classifier with a conservative `12 RPM/key` limit.

The generated `data/processed/labor_corpus.jsonl` file is ignored by Git because the full filtered corpus is large.

## Experiments

Run these notebooks in order:

```text
notebooks/01_data_exploration.ipynb
notebooks/02_retrieval_experiments.ipynb
notebooks/03_ragas_evaluation.ipynb
```

`02_retrieval_experiments.ipynb` compares `bm25`, `vector`, `hybrid`, `hybrid_rrf`, and `hybrid_rrf_mmr` on the frozen RAGAS-generated benchmark questions. It writes tables plus bar chart and heatmap artifacts under `reports/`.

`03_ragas_evaluation.ipynb` generates answers with the selected retriever and evaluates them with `ragas.evaluate()` using context precision, context recall, faithfulness, answer relevancy, and answer correctness.

The deploy-facing Python code does not contain experimental branches.

The benchmark questions are frozen in `reports/ragas_questions.json`. When Gemini quota or keys are unavailable, the notebooks load existing cached artifacts and mark cache usage in the report instead of pretending to run a fresh evaluation.

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

- Retrieval: notebook-selected hybrid retrieval; memory-aware local retrieval in CLI/UI
- Sparse retrieval: BM25 / memory-aware keyword scan for large local corpora
- Vector backend: FAISS for embedding experiments
- Embedding model: `intfloat/multilingual-e5-small`
- LLM: `gemini-3.1-flash-lite`
- API key strategy: round-robin over `GEMINI_API_KEYS` with per-key rate limiting
- UI: Streamlit

## Legal Disclaimer

This demo is for legal information retrieval only. It is not a substitute for professional legal advice.
