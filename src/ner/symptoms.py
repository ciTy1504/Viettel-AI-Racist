"""
Baseline NER cho TRIEU_CHUNG: KHONG phai model, chi la stub tu dien +
mot vai pattern rieng cho mau cau "thuoc ... dieu tri <trieu_chung>".

Day la phan "chua du" duoc ghi trong ke hoach: recall thap voi cac cach dien
dat tu do, can nang cap len encoder BIO (XLM-R/mDeBERTa) o Tuan 2 khi EDA xong
tap tu vung trieu chung thuc te trong data.
"""

import re
from dataclasses import dataclass

# Tu dien nho, dat cho phep segment cac cum trieu chung dinh lien nhau
# (vd "lo âu mất ngủ" -> "lo âu" + "mất ngủ"). Sap xep dai truoc de match tham lam.
SYMPTOM_VOCAB = sorted([
    "đánh trống ngực", "khó thở", "đau nhức", "đau bụng", "buồn nôn",
    "chóng mặt", "mệt mỏi", "táo bón", "tiêu chảy", "sốt đau", "lo âu",
    "mất ngủ", "ho", "sốt", "nôn", "ngất xỉu", "phù", "đau ngực",
], key=len, reverse=True)

_TREATS_RE = re.compile(r"điều trị\s+([^\d]+?)(?=\s+\d+\.\s|\s*$)", re.IGNORECASE)


@dataclass
class SymptomMention:
    text: str
    start: int
    end: int


def _segment_vocab(blob: str, base_offset: int) -> list[SymptomMention]:
    """Cat mot doan van thanh cac cum trieu chung biet truoc trong SYMPTOM_VOCAB
    (longest-match, khong chong lan). Neu khong khop tu nao, tra ve ca doan
    nhu mot entity duy nhat (fallback, uu tien recall hon la bo sot)."""
    mentions: list[SymptomMention] = []
    i = 0
    lower = blob.lower()
    n = len(blob)
    while i < n:
        if blob[i].isspace():
            i += 1
            continue
        matched = False
        for term in SYMPTOM_VOCAB:
            if lower.startswith(term, i):
                mentions.append(SymptomMention(
                    text=blob[i:i + len(term)],
                    start=base_offset + i,
                    end=base_offset + i + len(term),
                ))
                i += len(term)
                matched = True
                break
        if not matched:
            # gom ky tu khong khop vao token ke tiep (khoang trang phan cach)
            j = i
            while j < n and not blob[j].isspace():
                j += 1
            i = j

    if not mentions and blob.strip():
        stripped = blob.strip()
        start = base_offset + blob.index(stripped)
        mentions.append(SymptomMention(text=stripped, start=start, end=start + len(stripped)))

    return mentions


def find_symptom_mentions_after_drug(text: str) -> list[SymptomMention]:
    """Mau 'thuốc ... điều trị <triệu chứng>' - dac trung danh sach thuoc PMH."""
    results: list[SymptomMention] = []
    for m in _TREATS_RE.finditer(text):
        results.extend(_segment_vocab(m.group(1), m.start(1)))
    return results


def find_symptom_mentions_in_section(text: str, section_start: int, section_end: int) -> list[SymptomMention]:
    """Tim cac cum trong SYMPTOM_VOCAB xuat hien trong mot section (vd cac dong
    bullet duoi header 'Triệu chứng hiện tại'). Recall gioi han boi tu dien."""
    section_text = text[section_start:section_end]
    return _segment_vocab(section_text, section_start)
