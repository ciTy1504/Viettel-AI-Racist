"""Central place for the project's data directory layout."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RXNORM_RAW_DIR = RAW_DIR / "rxnorm" / "rrf"
RXNORM_PROCESSED_DIR = PROCESSED_DIR / "rxnorm"

ICD10CM_RAW_DIR = RAW_DIR / "icd10cm"
ICD10CM_PROCESSED_DIR = PROCESSED_DIR / "icd10cm"
