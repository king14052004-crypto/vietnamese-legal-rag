from schema import LegalDocument
from datasets import load_dataset

import pandas as pd

def load_vietnamese_legal_metadata(limit: int | None = None) -> list[dict]:
  meta = load_dataset("th1nhng0/vietnamese-legal-documents", "metadata", split="data")
  if limit:
    meta=meta.select(min(range(limit,len(meta))))
  return [dict(meta) for row in meta] 

def load_vietnamese_legal_content(limit: int | None = None) -> list[dict]:
  content = load_dataset("th1nhng0/vietnamese-legal-documents", "content", split="data")
  if limit:
    content=content.select(min(range(limit,len(content))))
  return [dict(content) for row in content] 

def merge_metadata_content(metadata_rows, content_rows) -> list[LegalDocument]:
  content_map={str(row['id']): row.get('content_html'," ") for row in content_rows}
  legal_documents=[]
  for meta in metadata_rows:
    m_id=str(meta['id'])
    if m_id in content_map:
      doc_data={**meta,'content_html':content_map[m_id]}
      doc_data['id']=m_id
      legal_documents.append(LegalDocument(**doc_data))  
  return  legal_documents
  