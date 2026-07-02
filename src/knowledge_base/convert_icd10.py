"""
Chuyen cac file .txt cua ICD-10-CM (CDC/NCHS, FY2026) sang .csv co header,
voi ma benh da duoc dinh dang lai theo chuan (co dau cham, vd "K21.0").

Doc tu:  data/raw/icd10cm/*.txt
Ghi ra:  data/processed/icd10cm/*.csv

Cach dung:
    python -m src.knowledge_base.convert_icd10
"""

import csv

from .icd10_utils import format_icd10_code
from .paths import ICD10CM_PROCESSED_DIR, ICD10CM_RAW_DIR


def convert_codes_file(input_path, output_path) -> int:
    """icd10cm-codes-2026.txt: 'CODE<spaces>Description' (billable codes only)."""
    print(f"Dang chuyen: {input_path.name} -> {output_path.name}")

    with open(input_path, "r", encoding="utf-8", errors="replace") as fin, \
         open(output_path, "w", encoding="utf-8-sig", newline="") as fout:

        writer = csv.writer(fout)
        writer.writerow(["code", "raw_code", "description"])

        row_count = 0
        for line in fin:
            line = line.rstrip("\n").rstrip("\r")
            if not line.strip():
                continue

            raw_code, _, description = line.partition(" ")
            raw_code = raw_code.strip()
            writer.writerow([format_icd10_code(raw_code), raw_code, description.strip()])
            row_count += 1

    print(f"  Hoan tat: {row_count:,} dong ghi vao {output_path}")
    return row_count


def convert_order_file(input_path, output_path) -> int:
    """
    icd10cm-order-2026.txt: dinh dang fixed-width theo tai lieu NCHS
    (icd10OrderFiles.pdf):
        cot 1-5   (index 0:5)   order_number
        cot 7-14  (index 6:14)  code (khong co dau cham)
        cot 15    (index 14:15) billable_flag (0/1)
        cot 17-76 (index 16:76) short_description (60 ky tu, cat ngan)
        cot 77+   (index 76:)   long_description (mo ta day du)
    """
    print(f"Dang chuyen: {input_path.name} -> {output_path.name}")

    with open(input_path, "r", encoding="utf-8", errors="replace") as fin, \
         open(output_path, "w", encoding="utf-8-sig", newline="") as fout:

        writer = csv.writer(fout)
        writer.writerow([
            "order_number", "code", "raw_code", "billable",
            "short_description", "long_description",
        ])

        row_count = 0
        for line in fin:
            line = line.rstrip("\n").rstrip("\r")
            if not line.strip():
                continue

            order_number = line[0:5].strip()
            raw_code = line[6:14].strip()
            billable = line[14:15].strip()
            short_description = line[16:76].strip()
            long_description = line[76:].strip()

            writer.writerow([
                order_number, format_icd10_code(raw_code), raw_code,
                billable, short_description, long_description,
            ])
            row_count += 1

    print(f"  Hoan tat: {row_count:,} dong ghi vao {output_path}")
    return row_count


def convert_all(force: bool = False) -> None:
    ICD10CM_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    codes_in = ICD10CM_RAW_DIR / "icd10cm-codes-2026.txt"
    codes_out = ICD10CM_PROCESSED_DIR / "icd10cm-codes-2026.csv"
    if codes_in.exists() and (force or not codes_out.exists()):
        convert_codes_file(codes_in, codes_out)
    elif not codes_in.exists():
        print(f"Khong tim thay: {codes_in}")

    order_in = ICD10CM_RAW_DIR / "icd10cm-order-2026.txt"
    order_out = ICD10CM_PROCESSED_DIR / "icd10cm-order-2026.csv"
    if order_in.exists() and (force or not order_out.exists()):
        convert_order_file(order_in, order_out)
    elif not order_in.exists():
        print(f"Khong tim thay: {order_in}")


if __name__ == "__main__":
    convert_all(force=True)
