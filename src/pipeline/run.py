"""
Pipeline baseline: van ban -> danh sach entity theo dung schema de bai
(xem data/Ví dụ.txt): {text, type, candidates, assertions, position}.

Ghep 3 module da co:
    src.ner.sections    - chia section, xac dinh ngu canh isHistorical
    src.ner.drugs       - regex bat cum THUOC
    src.ner.symptoms    - tu dien nho bat TRIEU_CHUNG (stub, recall thap)
    src.assertions.rules - gan assertion theo section + phu dinh cuc bo
    src.linking.drug_linker - THUOC -> RXCUI qua KnowledgeBase

>>> from src.pipeline.run import extract_entities
>>> extract_entities(open("data/test/input/1.txt", encoding="utf-8").read())
"""

import re

from src.assertions.rules import assign_assertions
from src.knowledge_base.lookup import KnowledgeBase
from src.linking.drug_linker import link_drug
from src.ner.drugs import find_drug_mentions
from src.ner.sections import (
    DIAGNOSIS_HEADER_KEYWORDS,
    DRUG_HEADER_KEYWORDS,
    SYMPTOM_HEADER_KEYWORDS,
    Section,
    split_sections,
)
from src.ner.symptoms import find_symptom_mentions_after_drug, find_symptom_mentions_in_section

_INTRO_HISTORICAL_RE = re.compile(r"trước (khi )?nhập viện|tiền sử", re.IGNORECASE)


def _whole_doc_section(text: str) -> Section:
    return Section(header="", header_stack=[], start=0, end=len(text), text=text)


def _dedupe_by_span(entities: list[dict]) -> list[dict]:
    """Neu hai entity trung/chong lan vi tri, giu entity co span dai hon
    (uu tien mau bat duoc lieu/duong dung thay vi mau tu dien ngan hon)."""
    entities = sorted(entities, key=lambda e: (e["position"][0], -(e["position"][1] - e["position"][0])))
    kept: list[dict] = []
    for e in entities:
        overlap = False
        for k in kept:
            if e["position"][0] < k["position"][1] and k["position"][0] < e["position"][1]:
                overlap = True
                break
        if not overlap:
            kept.append(e)
    return sorted(kept, key=lambda e: e["position"][0])


def extract_entities(text: str, kb: KnowledgeBase | None = None) -> list[dict]:
    sections = split_sections(text)
    has_headers = any(s.header for s in sections)
    doc_intro_historical = bool(_INTRO_HISTORICAL_RE.search(text[:150]))

    entities: list[dict] = []

    # --- THUOC ---
    if has_headers:
        drug_sections = [s for s in sections if s.header_matches(DRUG_HEADER_KEYWORDS)]
        if not drug_sections:
            drug_sections = sections
    else:
        drug_sections = [_whole_doc_section(text)]

    for sec in drug_sections:
        for dm in find_drug_mentions(sec.text):
            abs_start = sec.start + dm.start
            abs_end = sec.start + dm.end
            is_hist = sec.is_historical_context or doc_intro_historical
            assertions = assign_assertions(text, abs_start, abs_end, is_hist)

            candidates = link_drug(kb, dm.name, dm.dose) if kb is not None else []

            entities.append({
                "text": dm.text,
                "type": "THUỐC",
                "candidates": candidates,
                "assertions": assertions,
                "position": [abs_start, abs_end],
            })

    # --- TRIEU_CHUNG: mau "... dieu tri <trieu chung>" (danh sach thuoc PMH) ---
    for sm in find_symptom_mentions_after_drug(text):
        assertions = assign_assertions(text, sm.start, sm.end, is_historical_section=False)
        entities.append({
            "text": sm.text,
            "type": "TRIỆU_CHỨNG",
            "assertions": assertions,
            "position": [sm.start, sm.end],
        })

    # --- TRIEU_CHUNG: tu dien trong section co header lien quan trieu chung ---
    for sec in sections:
        if sec.header_matches(SYMPTOM_HEADER_KEYWORDS) and not sec.header_matches(DIAGNOSIS_HEADER_KEYWORDS):
            for sm in find_symptom_mentions_in_section(text, sec.start, sec.end):
                assertions = assign_assertions(
                    text, sm.start, sm.end, is_historical_section=sec.is_historical_context
                )
                entities.append({
                    "text": sm.text,
                    "type": "TRIỆU_CHỨNG",
                    "assertions": assertions,
                    "position": [sm.start, sm.end],
                })

    return _dedupe_by_span(entities)
