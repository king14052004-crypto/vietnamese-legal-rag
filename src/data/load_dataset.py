from src.data.schema import LegalDocument

DATASET_NAME = "th1nhng0/vietnamese-legal-documents"


def load_documents_from_jsonl(path: str) -> list[LegalDocument]:
    import json

    documents = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                documents.append(LegalDocument(**row))
    return documents
