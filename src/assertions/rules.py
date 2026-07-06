"""
Rule-based assertion assignment: isHistorical / isNegated / isHypothetical /
isFamily, dua theo section ngu canh (paragraph header) + tu khoa phu dinh
trong mot cua so quanh vi tri entity.

Day la baseline theo dung tinh than ke hoach ("Rule-based + classifier nhe" -
phan classifier chua lam, se can khi rule khong du - Tuan 3).
"""

import re

NEGATION_CUES = (
    "không", "chưa", "phủ nhận", "không có", "không thấy", "loại trừ",
)
HYPOTHETICAL_CUES = (
    "nếu", "có thể", "nghi ngờ", "khi cần", "trong trường hợp",
)
FAMILY_CUES = (
    "gia đình", "mẹ", "bố", "cha", "anh", "chị", "em", "con", "người thân",
)

_WINDOW_CHARS = 40


def _window_before(text: str, start: int, size: int = _WINDOW_CHARS) -> str:
    return text[max(0, start - size):start].lower()


def assign_assertions(
    text: str,
    entity_start: int,
    entity_end: int,
    is_historical_section: bool,
) -> list[str]:
    assertions: list[str] = []

    before = _window_before(text, entity_start)
    line_start = text.rfind("\n", 0, entity_start) + 1
    line_context = text[line_start:entity_end].lower()

    if any(cue in before or cue in line_context for cue in NEGATION_CUES):
        assertions.append("isNegated")

    if any(cue in before for cue in HYPOTHETICAL_CUES):
        assertions.append("isHypothetical")

    if any(cue in before for cue in FAMILY_CUES):
        assertions.append("isFamily")

    if is_historical_section and "isNegated" not in assertions:
        assertions.append("isHistorical")

    return assertions
