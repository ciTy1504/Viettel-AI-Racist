"""
Chuyen cac file .RRF cua RxNorm (pipe-delimited, khong header) sang .csv co header.

Doc tu:  data/raw/rxnorm/rrf/*.RRF
Ghi ra:  data/processed/rxnorm/*.csv

Cach dung:
    python -m src.knowledge_base.convert_rxnorm
"""

import csv
import os

from .paths import RXNORM_PROCESSED_DIR, RXNORM_RAW_DIR

# Ten cot chuan cua tung file RRF, theo tai lieu RxNorm Rich Release Format (NLM).
# Moi dong trong file .RRF ket thuc bang mot dau "|" thua, nen sau khi tach theo
# dau "|" se co du phan tu rong o cuoi -> phan tu do bi bo di khi ghi CSV.
RRF_COLUMNS = {
    "RXNCONSO.RRF": [
        "RXCUI", "LAT", "TS", "LUI", "STT", "SUI", "ISPREF", "RXAUI",
        "SAUI", "SCUI", "SDUI", "SAB", "TTY", "CODE", "STR", "SRL",
        "SUPPRESS", "CVF",
    ],
    "RXNREL.RRF": [
        "RXCUI1", "RXAUI1", "STYPE1", "REL", "RXCUI2", "RXAUI2", "STYPE2",
        "RELA", "RUI", "SRUI", "SAB", "SL", "RG", "DIR", "SUPPRESS", "CVF",
    ],
    "RXNSAT.RRF": [
        "RXCUI", "LUI", "SUI", "RXAUI", "STYPE", "CODE", "ATUI", "SATUI",
        "ATN", "SAB", "ATV", "SUPPRESS", "CVF",
    ],
}


def convert_file(input_path: str, output_path: str, columns: list[str]) -> None:
    print(f"Dang chuyen: {os.path.basename(input_path)} -> {os.path.basename(output_path)}")

    with open(input_path, "r", encoding="utf-8", errors="replace") as fin, \
         open(output_path, "w", encoding="utf-8-sig", newline="") as fout:

        writer = csv.writer(fout)
        writer.writerow(columns)

        n_cols = len(columns)
        row_count = 0

        for line in fin:
            line = line.rstrip("\n").rstrip("\r")
            if not line:
                continue

            fields = line.split("|")
            if fields and fields[-1] == "":
                fields = fields[:-1]

            if len(fields) < n_cols:
                fields += [""] * (n_cols - len(fields))
            elif len(fields) > n_cols:
                fields = fields[:n_cols]

            writer.writerow(fields)
            row_count += 1

            if row_count % 500_000 == 0:
                print(f"  ... da xu ly {row_count:,} dong")

    print(f"  Hoan tat: {row_count:,} dong ghi vao {output_path}")


def convert_all(force: bool = False) -> None:
    RXNORM_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for filename, columns in RRF_COLUMNS.items():
        input_path = RXNORM_RAW_DIR / filename
        output_path = RXNORM_PROCESSED_DIR / (os.path.splitext(filename)[0] + ".csv")

        if not input_path.exists():
            print(f"Bo qua {filename}: khong tim thay tai {input_path}")
            continue

        if output_path.exists() and not force:
            print(f"Da co san: {output_path} (bo qua, dung force=True de ghi de)")
            continue

        convert_file(str(input_path), str(output_path), columns)


if __name__ == "__main__":
    convert_all()
