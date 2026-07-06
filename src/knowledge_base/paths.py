"""Central place for the project's data directory layout."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RXNORM_RAW_DIR = RAW_DIR / "RxNorm_full_prescribe_06012026" / "rrf"
RXNORM_PROCESSED_DIR = PROCESSED_DIR / "rxnorm"

ICD10CM_RAW_DIR = RAW_DIR / "icd10cm"
ICD10CM_PROCESSED_DIR = PROCESSED_DIR / "icd10cm"

# ICD-10 TT06 (ban tieng Viet cua Bo Y te, crawl tu icd.kcb.vn)
ICD10_TT06_RAW_DIR = RAW_DIR / "icd10_tt06"
ICD10_TT06_JSONL = ICD10_TT06_RAW_DIR / "icd10_tt06.jsonl"
