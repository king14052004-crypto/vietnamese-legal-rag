import html
import re
from typing import List

from schema import LegalDocument

def _normalize_text(text: str) -> str:
  if not text:
    return ""
  # Strip HTML tags and unescape entities to reduce false positives.
  text = html.unescape(text)
  text = re.sub(r"<[^>]+>", " ", text)
  return re.sub(r"\s+", " ", text).lower()


def is_labor_related(document: LegalDocument) -> bool:
  strong_keywords = [
    "hợp đồng lao động",
    "bộ luật lao động",
    "người lao động",
    "người sử dụng lao động",
    "tiền lương",
    "bảo hiểm thất nghiệp",
    "trợ cấp thôi việc",
    "trợ cấp mất việc",
    "kỷ luật lao động",
    "sa thải",
    "an toàn vệ sinh lao động",
  ]

  weak_keywords = [
    "lao động",
    "quan hệ lao động",
    "việc làm",
    "học nghề",
    "đào tạo nghề",
    "thử việc",
    "tiền công",
    "lương tối thiểu",
    "thưởng",
    "thời giờ làm việc",
    "thời giờ nghỉ ngơi",
    "làm thêm giờ",
    "nghỉ hằng năm",
    "nghỉ lễ",
    "bảo hiểm xã hội",
    "bảo hiểm y tế",
    "nghỉ việc",
    "tai nạn lao động",
    "bệnh nghề nghiệp",
    "công đoàn",
    "thỏa ước lao động tập thể",
    "đình công",
    "tranh chấp lao động",
    "lao động nữ",
    "thai sản",
    "người chưa thành niên",
    "lao động nước ngoài",
    "giấy phép lao động",
  ]

  text_to_search = _normalize_text(f"{document.title or ''} {document.content_html or ''}")

  if any(keyword in text_to_search for keyword in strong_keywords):
    return True

  weak_hits = sum(1 for keyword in weak_keywords if keyword in text_to_search)
  return weak_hits >= 2

def filter_labor_documents(documents: List[LegalDocument]) -> List[LegalDocument]:
    return [doc for doc in documents if is_labor_related(doc)]