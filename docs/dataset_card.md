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

The corpus is filtered toward labor-law topics such as:

- Bộ luật Lao động
- hợp đồng lao động
- người lao động / người sử dụng lao động
- tiền lương / lương tối thiểu
- bảo hiểm thất nghiệp
- trợ cấp thôi việc
- an toàn vệ sinh lao động
- tranh chấp lao động

Raw full datasets are intentionally not committed.
The generated processed JSONL corpus is also ignored by Git because the full filtered subset is large.
