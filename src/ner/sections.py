"""
Tach van ban ca benh thanh cac section theo header, dung lam ngu canh cho
NER (giơi han vung tim thuoc/trieu chung) va assertion (isHistorical theo section).

Cac file vi du (data/test/input/*.txt) co cau truc lap lai:
    1.  Tieu de section lon
        Tieu de section con:
        - dong noi dung
        - dong noi dung

Section header duoc nhan dien bang heuristic: dong khong bat dau bang "-",
ket thuc bang ":" hoac la dong ngan (< 60 ky tu) khong co dau cham cuoi cau.
"""

import re
from dataclasses import dataclass

_NUMBERED_HEADER_RE = re.compile(r"^\s*\d+\.\s*(.+)$")

HISTORICAL_HEADER_KEYWORDS = (
    "tiền sử",
    "trước khi nhập viện",
    "trước đây",
    "tiền căn",
)

DRUG_HEADER_KEYWORDS = ("thuốc",)
SYMPTOM_HEADER_KEYWORDS = ("triệu chứng", "lý do nhập viện")
DIAGNOSIS_HEADER_KEYWORDS = ("chẩn đoán", "bệnh lý", "kết luận")
LAB_HEADER_KEYWORDS = ("xét nghiệm", "kết quả")


@dataclass
class Section:
    header: str
    header_stack: list[str]  # header cua section nay va cac section cha (tu ngoai vao trong)
    start: int
    end: int
    text: str

    def header_matches(self, keywords: tuple[str, ...]) -> bool:
        joined = " ".join(self.header_stack).lower()
        return any(kw in joined for kw in keywords)

    @property
    def is_historical_context(self) -> bool:
        return self.header_matches(HISTORICAL_HEADER_KEYWORDS)


def _is_header_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped or stripped.startswith("-"):
        return False
    if _NUMBERED_HEADER_RE.match(stripped):
        return True
    if stripped.endswith(":"):
        return True
    # dong ngan, khong ket thuc bang dau cau cua mot cau van thuong -> coi la header
    if len(stripped) < 60 and not stripped.endswith((".", ",", ")")):
        return True
    return False


def split_sections(text: str) -> list[Section]:
    """Chia van ban thanh cac section lien tiep (khong long nhau) kem header_stack."""
    lines = text.splitlines(keepends=True)
    offsets = []
    pos = 0
    for line in lines:
        offsets.append(pos)
        pos += len(line)

    sections: list[Section] = []
    stack: list[tuple[int, str]] = []  # (indent_level, header)
    current_header_stack: list[str] = []
    section_start = 0
    section_header = ""

    def flush(end: int):
        if end > section_start:
            sections.append(Section(
                header=section_header,
                header_stack=list(current_header_stack),
                start=section_start,
                end=end,
                text=text[section_start:end],
            ))

    for i, line in enumerate(lines):
        stripped = line.strip()
        if _is_header_line(line):
            header_text = _NUMBERED_HEADER_RE.match(stripped)
            header_text = header_text.group(1) if header_text else stripped.rstrip(":")

            indent = len(line) - len(line.lstrip())
            flush(offsets[i])

            # cap nhat stack theo indent: header thut le nhieu hon la con cua header truoc
            while stack and stack[-1][0] >= indent:
                stack.pop()
            stack.append((indent, header_text))
            current_header_stack = [h for _, h in stack]
            section_start = offsets[i]
            section_header = header_text

    flush(len(text))
    return sections


def section_at(sections: list[Section], pos: int) -> Section | None:
    for s in sections:
        if s.start <= pos < s.end:
            return s
    return None
