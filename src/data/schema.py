from dataclasses import asdict, dataclass, field
from typing import Any

@dataclass
class LegalDocument:
    id: str
    title: str | None = None
    so_ky_hieu: str | None = None
    ngay_ban_hanh: str | None = None
    loai_van_ban: str | None = None
    ngay_co_hieu_luc: str | None = None
    ngay_het_hieu_luc: str | None = None
    nguon_thu_thap: str | None = None
    ngay_dang_cong_bao: str | None = None
    nganh: str | None = None
    linh_vuc: str | None = None
    co_quan_ban_hanh: str | None = None
    chuc_danh: str | None = None
    nguoi_ky: str | None = None
    pham_vi: str | None = None
    thong_tin_ap_dung: str | None = None
    tinh_trang_hieu_luc: str | None = None
    content_html: str | None = None
    content_text: str | None = None
    source_url: str | None = None
    raw_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LegalChunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    article: str | None = None
    source_url: str | None = None
    doc_type: str | None = None
    effective_date: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SearchResult:
    chunk: LegalChunk
    score: float
    method: str
    rank: int | None = None


@dataclass
class RagAnswer:
    question: str
    answer: str
    citations: list[dict[str, Any]]
    retrieval_method: str
    context_chunks: list[LegalChunk]