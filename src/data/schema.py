from dataclasses import dataclass
from typing import List, Optional, Dict

@dataclass
class LegalDocument:
    id: str
    title: Optional[str] = None
    so_ky_hieu: Optional[str] = None
    ngay_ban_hanh: Optional[str] = None
    loai_van_ban: Optional[str] = None
    ngay_co_hieu_luc: Optional[str] = None
    ngay_het_hieu_luc: Optional[str] = None
    nguon_thu_thap: Optional[str] = None
    ngay_dang_cong_bao: Optional[str] = None
    nganh: Optional[str] = None
    linh_vuc: Optional[str] = None
    co_quan_ban_hanh: Optional[str] = None
    chuc_danh: Optional[str] = None
    nguoi_ky: Optional[str] = None
    pham_vi: Optional[str] = None
    thong_tin_ap_dung: Optional[str] = None
    tinh_trang_hieu_luc: Optional[str] = None
    content_html: Optional[str] = None