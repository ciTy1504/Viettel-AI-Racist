"""
Unified in-memory lookup over the processed RxNorm and ICD-10-CM tables.

>>> from src.knowledge_base.lookup import KnowledgeBase
>>> kb = KnowledgeBase.load()
>>> kb.rxnorm.search("Chlorpheniramine 0.4 MG/ML")
>>> kb.icd10.search("trào ngược dạ dày")
"""

import csv
import unicodedata
from collections import defaultdict
from dataclasses import dataclass

from .paths import ICD10CM_PROCESSED_DIR, RXNORM_PROCESSED_DIR


def _normalize(text: str) -> str:
    """Lowercase + strip accents, for loose matching (handles Vietnamese text too)."""
    text = text.strip().lower()
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))


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


@dataclass
class KnowledgeBase:
    """Bundles both terminology sources for the entity-linking step of the pipeline."""

    rxnorm: RxNormLookup
    icd10: ICD10Lookup

    @classmethod
    def load(cls) -> "KnowledgeBase":
        return cls(rxnorm=RxNormLookup.load(), icd10=ICD10Lookup.load())

    def search_drug(self, text: str, limit: int = 10) -> list[DrugMatch]:
        return self.rxnorm.search(text, limit=limit)

    def search_diagnosis(self, text: str, limit: int = 10) -> list[DiagnosisMatch]:
        return self.icd10.search(text, limit=limit)
