# Dataset Card

## Primary source

- Dataset: `th1nhng0/vietnamese-legal-documents`
- URL: https://huggingface.co/datasets/th1nhng0/vietnamese-legal-documents
- Upstream source: public Vietnamese legal documents from VBPL.

## Project subset

This project generates the filtered labor-law corpus at:

```text
data/processed/labor_corpus.jsonl
```

The corpus is filtered in `notebooks/01_data_exploration.ipynb` with `gemini-3.1-flash-lite`.
The notebook caches every classification decision in `reports/labor_filter_gemini.jsonl`, which is ignored by Git because it can become large.

Raw full datasets are intentionally not committed.
The generated processed JSONL corpus is also ignored by Git because the full filtered subset is large.
