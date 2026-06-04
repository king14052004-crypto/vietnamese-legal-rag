# Processed Data

Run `notebooks/01_data_exploration.ipynb` to generate `data/processed/labor_corpus.jsonl`.

The notebook filters the public corpus with Gemini and caches classification decisions under `reports/`.
The generated JSONL corpus is ignored by Git because the full filtered dataset is large.
