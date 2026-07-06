"""
Entity linking cho THUOC: ten thuoc (+ lieu trich xuat tu NER) -> RXCUI.

Chien luoc (theo bay diem "SCD" va "EMPTY" trong ke hoach):
  1. Tim tat ca ung vien theo ten hoat chat/san pham (substring search co san
     trong KnowledgeBase.search_drug).
  2. Uu tien ung vien co TTY o muc "co the ke don" (SCD/SBD/GPCK/BPCK) hon la
     muc hoat chat don thuan (IN/PIN), vi RxCUI de bai can la muc thuoc-du-lieu-dang.
  3. Neu trich duoc lieu luong tu NER, uu tien ung vien co lieu luong khop.
  4. Neu khong co ung vien nao du tin cay (khong khop ten hoac khong ro muc do),
     tra ve rong thay vi doan bua -> dung theo nguyen tac "tha de trong" (J=1
     khi ground truth cung rong).

Day la baseline substring-based; Tuan 2 se nang cap them fuzzy (RapidFuzz) va
parser lieu/dang/duong dung day du hon.
"""

import re
from dataclasses import dataclass

from src.knowledge_base.lookup import DrugMatch, KnowledgeBase

_PRESCRIBABLE_TTY_RANK = {
    "SCD": 3, "SBD": 3, "GPCK": 2, "BPCK": 2,  # co lieu/dang -> uu tien cao nhat
    "IN": 1, "PIN": 1, "MIN": 1,               # chi la hoat chat -> du phong
}

_DOSE_NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")


def _dose_numbers(dose_text: str | None) -> set[str]:
    if not dose_text:
        return set()
    return {n.replace(",", ".") for n in _DOSE_NUM_RE.findall(dose_text)}


def _rank_key(match: DrugMatch, dose_numbers: set[str]) -> tuple[int, int, int]:
    tty_rank = _PRESCRIBABLE_TTY_RANK.get(match.tty, 0)
    dose_hit = 1 if dose_numbers and any(n in match.name for n in dose_numbers) else 0
    return (-dose_hit, -tty_rank, len(match.name))


def link_drug(
    kb: KnowledgeBase,
    name: str,
    dose: str | None = None,
    limit: int = 3,
    require_dose_match_if_available: bool = True,
) -> list[str]:
    """Tra ve danh sach RXCUI ung vien cho mot ten thuoc trich xuat tu NER.

    Tra ve [] neu khong tim thay ung vien nao dang tin cay, thay vi doan bua.
    """
    candidates = kb.search_drug(name, limit=50)
    if not candidates:
        return []

    dose_numbers = _dose_numbers(dose)
    candidates.sort(key=lambda c: _rank_key(c, dose_numbers))

    if dose_numbers and require_dose_match_if_available:
        with_dose = [c for c in candidates if any(n in c.name for n in dose_numbers)]
        if with_dose:
            candidates = with_dose
        else:
            # co lieu luong nhung khong ung vien nao khop -> khong du tin cay
            return []

    seen: set[str] = set()
    result: list[str] = []
    for c in candidates:
        if c.rxcui in seen:
            continue
        seen.add(c.rxcui)
        result.append(c.rxcui)
        if len(result) >= limit:
            break
    return result
