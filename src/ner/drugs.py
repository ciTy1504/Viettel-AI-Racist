"""
Baseline NER cho THUOC: bat cum "ten thuoc [+ lieu] [+ duong dung] [+ tan suat]"
bang regex, giu nguyen position tuyet doi trong van ban goc.

Day la baseline theo dung tinh than ke hoach Tuan 1 ("regex bat dong thuoc"):
khong doi hoi hoan hao, muc tieu la co diem candidates/assertions that de
so sanh tren harness local, roi nang cap dan (fuzzy, model BIO) o Tuan 2.
"""

import re
from dataclasses import dataclass

_UNIT = r"(?:mg|mcg|g|ml|meq|iu|%|mmol)"
_DOSE = rf"\d+(?:[.,]\d+)?(?:\s*-\s*\d+(?:[.,]\d+)?)?\s*{_UNIT}(?:/{_UNIT.strip('()?:')}|/ml)?"
_ROUTE = r"(?:po|iv|im|sc|sq|pr|sl|top|ophth|inh)"
_FREQ = r"(?:q\d+h|qid|tid|bid|qam|qhs|qod|qd|daily|prn)(?::prn)?"

# Tu khoa duong dung/tan suat khong duoc de bi "nuot" vao ten thuoc (regex ten
# thuoc von tham lam, se an het cac tu phia sau neu khong chan lai o day).
_STOP_TOKEN = r"(?:po|iv|im|sc|sq|pr|sl|top|ophth|inh|daily|bid|tid|qid|qam|qhs|qod|qd|prn|q\d+h)"

# Ten thuoc: 1-4 tu, chu cai (co dau tieng Viet) + so/chu thuong dinh kem (vd "xl", "50")
_NAME_TOKEN = rf"(?!{_STOP_TOKEN}\b)[A-Za-zÀ-ỹ][A-Za-zÀ-ỹ0-9\-]*"
_NAME = rf"{_NAME_TOKEN}(?:\s+{_NAME_TOKEN}){{0,3}}"

MED_PATTERN = re.compile(
    rf"(?P<name>{_NAME})"
    rf"(?P<dose>\s+{_DOSE})?"
    rf"(?P<route>\s+{_ROUTE})?"
    rf"(?P<freq>\s+{_FREQ})?",
    re.IGNORECASE,
)

# Tu khoa khong duoc de la ten thuoc du dung dau cum regex (header, tu noi...)
_STOPWORDS = {
    "thuốc", "trước", "khi", "nhập", "viện", "điều", "trị", "cho", "và", "các",
    "hôm", "nay", "uống", "tiêm", "reduced", "from", "to",
}


@dataclass
class DrugMention:
    text: str
    start: int
    end: int
    name: str
    dose: str | None
    route: str | None
    freq: str | None


def _clean_span(s: str) -> str:
    return s.strip().rstrip(",;.")


def find_drug_mentions(text: str) -> list[DrugMention]:
    """Tim cac cum tu co dang 'ten_thuoc [lieu] [duong dung] [tan suat]'.

    Chi tra ve match co it nhat mot trong (dose, route, freq) di kem ten thuoc,
    de tranh nhan nham tu thuong thanh ten thuoc.
    """
    mentions: list[DrugMention] = []
    for m in MED_PATTERN.finditer(text):
        dose, route, freq = m.group("dose"), m.group("route"), m.group("freq")
        if not (dose or route or freq):
            continue

        name = m.group("name").strip()
        first_word = name.split()[0].lower() if name else ""
        if first_word in _STOPWORDS:
            continue

        end = m.end()
        # cat bo khoang trang/dau cau du o cuoi match
        raw = text[m.start():end]
        trimmed = raw.rstrip()
        end = m.start() + len(trimmed)
        span_text = _clean_span(text[m.start():end])
        if not span_text:
            continue

        mentions.append(DrugMention(
            text=span_text,
            start=m.start(),
            end=m.start() + len(span_text),
            name=name,
            dose=dose.strip() if dose else None,
            route=route.strip() if route else None,
            freq=freq.strip() if freq else None,
        ))

    return _drop_overlaps(mentions)


def _drop_overlaps(mentions: list[DrugMention]) -> list[DrugMention]:
    """Regex co the sinh match long nhau (do {0,3} tu tuy chon) -> giu match dai nhat."""
    mentions = sorted(mentions, key=lambda m: (m.start, -(m.end - m.start)))
    kept: list[DrugMention] = []
    last_end = -1
    for m in mentions:
        if m.start >= last_end:
            kept.append(m)
            last_end = m.end
    return kept
