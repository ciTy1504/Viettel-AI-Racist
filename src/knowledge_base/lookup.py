"""
Unified in-memory lookup over the processed RxNorm and ICD-10-CM tables.

>>> from src.knowledge_base.lookup import KnowledgeBase
>>> kb = KnowledgeBase.load()
>>> kb.rxnorm.search("Chlorpheniramine 0.4 MG/ML")
>>> kb.icd10.search("trào ngược dạ dày")
"""

import csv
import json
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass

from .paths import ICD10_TT06_JSONL, ICD10CM_PROCESSED_DIR, RXNORM_PROCESSED_DIR

_NON_ALNUM = re.compile(r"[^0-9a-z]+")


def _normalize(text: str) -> str:
    """Lowercase + strip accents + gop dau cau/khoang trang, cho match long (ho tro
    tieng Viet). VD 'da day - thuc quan' va 'da day thuc quan' se khop nhau."""
    text = text.strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = _NON_ALNUM.sub(" ", text)
    return text.strip()


@dataclass(frozen=True)
class DrugMatch:
    rxcui: str
    name: str
    tty: str


@dataclass(frozen=True)
class DiagnosisMatch:
    code: str
    description: str


class RxNormLookup:
    """Name <-> RXCUI lookup, built from RXNCONSO.csv (SAB=RXNORM only)."""

    def __init__(self):
        self.rxcui_to_names: dict[str, list[tuple[str, str]]] = defaultdict(list)
        self._normalized_index: list[tuple[str, str, str, str]] = []  # (normalized, rxcui, str, tty)

    @classmethod
    def load(cls) -> "RxNormLookup":
        self = cls()
        path = RXNORM_PROCESSED_DIR / "RXNCONSO.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} khong ton tai. Chay 'python -m src.knowledge_base.convert_rxnorm' truoc."
            )

        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                if row["SAB"] != "RXNORM":
                    continue
                rxcui, name, tty = row["RXCUI"], row["STR"], row["TTY"]
                self.rxcui_to_names[rxcui].append((name, tty))
                self._normalized_index.append((_normalize(name), rxcui, name, tty))

        return self

    def names_for(self, rxcui: str) -> list[tuple[str, str]]:
        return self.rxcui_to_names.get(rxcui, [])

    def exact(self, name: str) -> list[DrugMatch]:
        target = _normalize(name)
        return [
            DrugMatch(rxcui, s, tty)
            for normalized, rxcui, s, tty in self._normalized_index
            if normalized == target
        ]

    def search(self, query: str, limit: int = 10) -> list[DrugMatch]:
        """Substring search over drug names; shortest/most-specific match first."""
        target = _normalize(query)
        if not target:
            return []

        hits = [
            (len(s), rxcui, s, tty)
            for normalized, rxcui, s, tty in self._normalized_index
            if target in normalized
        ]
        hits.sort(key=lambda h: h[0])
        seen_rxcui = set()
        results = []
        for _, rxcui, s, tty in hits:
            if rxcui in seen_rxcui:
                continue
            seen_rxcui.add(rxcui)
            results.append(DrugMatch(rxcui, s, tty))
            if len(results) >= limit:
                break
        return results


class ICD10Lookup:
    """Code <-> description lookup, built from icd10cm-codes-2026.csv (billable codes)."""

    def __init__(self):
        self.code_to_description: dict[str, str] = {}
        self._normalized_index: list[tuple[str, str, str]] = []  # (normalized, code, description)

    @classmethod
    def load(cls) -> "ICD10Lookup":
        self = cls()
        path = ICD10CM_PROCESSED_DIR / "icd10cm-codes-2026.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} khong ton tai. Chay 'python -m src.knowledge_base.convert_icd10' truoc."
            )

        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                code, description = row["code"], row["description"]
                self.code_to_description[code] = description
                self._normalized_index.append((_normalize(description), code, description))

        return self

    def describe(self, code: str) -> str | None:
        return self.code_to_description.get(code)

    def search(self, query: str, limit: int = 10) -> list[DiagnosisMatch]:
        """Substring search over diagnosis descriptions; shortest match first."""
        target = _normalize(query)
        if not target:
            return []

        hits = [
            (len(desc), code, desc)
            for normalized, code, desc in self._normalized_index
            if target in normalized
        ]
        hits.sort(key=lambda h: h[0])
        return [DiagnosisMatch(code, desc) for _, code, desc in hits[:limit]]


class ICD10TT06Lookup:
    """Code <-> ten tieng Viet, doc tu data/raw/icd10_tt06/icd10_tt06.jsonl.

    Day la ban ICD-10 TT06 cua Bo Y te (tieng Viet) - phu hop de khop chan doan
    tieng Viet trong ca benh hon ban ICD-10-CM tieng Anh."""

    def __init__(self):
        self.code_to_name: dict[str, str] = {}       # code -> ten tieng Viet
        self.code_to_name_en: dict[str, str] = {}     # code -> ten tieng Anh
        # (normalized_name, code, name_vi, is_leaf); index ca ten VI va EN
        self._normalized_index: list[tuple[str, str, str, bool]] = []

    @classmethod
    def load(cls) -> "ICD10TT06Lookup":
        self = cls()
        if not ICD10_TT06_JSONL.exists():
            raise FileNotFoundError(
                f"{ICD10_TT06_JSONL} khong ton tai. "
                "Chay 'python scripts/crawl_icd10_tt06.py' truoc."
            )

        with open(ICD10_TT06_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                code = rec.get("code", "")
                name_vi = rec.get("name_vi", "")
                name_en = rec.get("name_en", "")
                if not code or not name_vi:
                    continue
                is_leaf = bool(rec.get("is_leaf"))
                self.code_to_name.setdefault(code, name_vi)
                if name_en:
                    self.code_to_name_en.setdefault(code, name_en)
                # index ten tieng Viet, va tieng Anh (neu co) -> match duoc ca hai
                self._normalized_index.append((_normalize(name_vi), code, name_vi, is_leaf))
                if name_en:
                    self._normalized_index.append((_normalize(name_en), code, name_vi, is_leaf))

        return self

    def describe(self, code: str) -> str | None:
        return self.code_to_name.get(code)

    def describe_en(self, code: str) -> str | None:
        return self.code_to_name_en.get(code)

    def search(self, query: str, limit: int = 10, leaf_only: bool = False) -> list[DiagnosisMatch]:
        """Substring search tren ten (ca VI lan EN); uu tien khop ngan nhat (cu the nhat).
        leaf_only=True chi tra ve ma la (billable) thay vi ma nhom."""
        target = _normalize(query)
        if not target:
            return []

        hits = [
            (len(normalized), code, name_vi)
            for normalized, code, name_vi, is_leaf in self._normalized_index
            if target in normalized and (not leaf_only or is_leaf)
        ]
        hits.sort(key=lambda h: h[0])
        seen: set[str] = set()
        results: list[DiagnosisMatch] = []
        for _, code, name_vi in hits:
            if code in seen:
                continue
            seen.add(code)
            results.append(DiagnosisMatch(code, name_vi))
            if len(results) >= limit:
                break
        return results


@dataclass
class KnowledgeBase:
    """Bundles the terminology sources for the entity-linking step of the pipeline."""

    rxnorm: RxNormLookup | None
    icd10: ICD10Lookup | None
    icd10_tt06: ICD10TT06Lookup | None

    @classmethod
    def load(cls, use_icd10cm: bool = False) -> "KnowledgeBase":
        """Nap KnowledgeBase voi cac nguon co san (nguon thieu file se bi bo qua,
        khong lam fail toan bo). Mac dinh dung ICD-10 TT06 (tieng Viet) cho chan doan;
        bat use_icd10cm=True neu muon nap them ban ICD-10-CM tieng Anh."""
        try:
            rxnorm = RxNormLookup.load()
        except FileNotFoundError:
            rxnorm = None
        try:
            icd10 = ICD10Lookup.load() if use_icd10cm else None
        except FileNotFoundError:
            icd10 = None
        try:
            icd10_tt06 = ICD10TT06Lookup.load()
        except FileNotFoundError:
            icd10_tt06 = None
        return cls(rxnorm=rxnorm, icd10=icd10, icd10_tt06=icd10_tt06)

    def search_drug(self, text: str, limit: int = 10) -> list[DrugMatch]:
        if self.rxnorm is None:
            return []
        return self.rxnorm.search(text, limit=limit)

    def search_diagnosis(self, text: str, limit: int = 10) -> list[DiagnosisMatch]:
        """Uu tien ICD-10 TT06 tieng Viet; roi ve ICD-10-CM neu khong co."""
        if self.icd10_tt06 is not None:
            return self.icd10_tt06.search(text, limit=limit)
        if self.icd10 is not None:
            return self.icd10.search(text, limit=limit)
        return []
