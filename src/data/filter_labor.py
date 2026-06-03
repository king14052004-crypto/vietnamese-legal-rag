from src.data.clean_text import normalize_for_match
from src.data.schema import LegalDocument


def is_labor_related(document: LegalDocument) -> bool:
    title_fields = " ".join(
        value or ""
        for value in [
            document.title,
            document.nganh,
            document.linh_vuc,
            document.thong_tin_ap_dung,
        ]
    )
    title_text = normalize_for_match(title_fields)
    full_text = normalize_for_match(
        f"{title_fields} {document.content_html or document.content_text or ''}"
    )

    title_keywords = [
        "lao động",
        "việc làm",
        "bảo hiểm xã hội",
        "bảo hiểm thất nghiệp",
        "tiền lương",
        "lương tối thiểu",
        "an toàn vệ sinh lao động",
        "dạy nghề",
        "đào tạo nghề",
        "công đoàn",
    ]
    very_strong_phrases = [
        "hợp đồng lao động",
        "bộ luật lao động",
        "người lao động",
        "người sử dụng lao động",
        "bảo hiểm thất nghiệp",
        "trợ cấp thôi việc",
        "trợ cấp mất việc",
        "kỷ luật lao động",
        "tranh chấp lao động",
        "thỏa ước lao động tập thể",
        "an toàn vệ sinh lao động",
        "giấy phép lao động",
        "làm thêm giờ",
    ]
    supporting_terms = [
        "lương tối thiểu",
        "tiền lương",
        "học nghề",
        "đào tạo nghề",
        "bảo hiểm xã hội",
        "tai nạn lao động",
        "bệnh nghề nghiệp",
        "công đoàn",
        "đình công",
        "lao động nữ",
        "thai sản",
        "lao động nước ngoài",
    ]

    if any(keyword in title_text for keyword in title_keywords):
        return True

    if any(phrase in full_text for phrase in very_strong_phrases):
        return True

    supporting_hits = sum(1 for keyword in supporting_terms if keyword in full_text)
    return supporting_hits >= 3 and "lao động" in full_text


def filter_labor_documents(documents: list[LegalDocument]) -> list[LegalDocument]:
    return [doc for doc in documents if is_labor_related(doc)]
