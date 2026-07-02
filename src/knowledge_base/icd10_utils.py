"""Formatting helpers for ICD-10-CM codes."""


def format_icd10_code(raw_code: str) -> str:
    """Insert the decimal point NCHS strips from its distribution files.

    CDC/NCHS ships codes without the dot (e.g. "K210"); the standard
    ICD-10-CM notation used everywhere else is category (first 3 chars)
    + "." + the remaining subdivision (e.g. "K21.0").
    """
    raw_code = raw_code.strip().upper()
    if len(raw_code) <= 3:
        return raw_code
    return f"{raw_code[:3]}.{raw_code[3:]}"
