from schema import LegalDocument
from typing import List, Optional, Dict

def is_labor_related(document: LegalDocument) -> bool:
  keywords = [
    # lao động chung
    "lao động",
    "bộ luật lao động",
    "quan hệ lao động",
    "việc làm",
    "học nghề",
    "đào tạo nghề",

    # hợp đồng
    "hợp đồng lao động",
    "giao kết hợp đồng",
    "chấm dứt hợp đồng",
    "đơn phương chấm dứt",
    "thử việc",

    # chủ thể
    "người lao động",
    "người sử dụng lao động",

    # lương và thu nhập
    "tiền lương",
    "tiền công",
    "lương tối thiểu",
    "thưởng",

    # thời gian làm việc
    "thời giờ làm việc",
    "thời giờ nghỉ ngơi",
    "làm thêm giờ",
    "nghỉ hằng năm",
    "nghỉ lễ",

    # bảo hiểm
    "bảo hiểm xã hội",
    "bảo hiểm thất nghiệp",
    "bảo hiểm y tế",
    "trợ cấp thôi việc",
    "trợ cấp mất việc",

    # nghỉ việc
    "nghỉ việc",
    "sa thải",
    "kỷ luật lao động",

    # an toàn lao động
    "an toàn vệ sinh lao động",
    "tai nạn lao động",
    "bệnh nghề nghiệp",

    # tổ chức lao động
    "công đoàn",
    "thỏa ước lao động tập thể",
    "đình công",
    "tranh chấp lao động",

    # đối tượng đặc biệt
    "lao động nữ",
    "thai sản",
    "người chưa thành niên",
    "lao động nước ngoài",
    "giấy phép lao động"
    ' '
]
  text_to_search=f"{document.title or ' '} {document.content_html or ' '}".lower()
  return any(keyword in text_to_search for keyword in keywords)

def filter_labor_documents(documents: List[LegalDocument]) -> List[LegalDocument]:
    return [doc for doc in documents if is_labor_related(doc)]