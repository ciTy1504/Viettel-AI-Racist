"""
Build toan bo knowledge base (RxNorm + ICD-10-CM) tu du lieu raw sang CSV
da xu ly, roi chay vai truy van demo de kiem tra nhanh.

Cach dung (chay tu thu muc goc cua project):
    python scripts/build_knowledge_base.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.knowledge_base.convert_icd10 import convert_all as convert_icd10
from src.knowledge_base.convert_rxnorm import convert_all as convert_rxnorm
from src.knowledge_base.lookup import KnowledgeBase


def main():
    print("=== Buoc 1: Convert RxNorm (.RRF -> .csv) ===")
    convert_rxnorm()

    print("\n=== Buoc 2: Convert ICD-10-CM (.txt -> .csv) ===")
    convert_icd10()

    print("\n=== Buoc 3: Nap knowledge base vao bo nho ===")
    kb = KnowledgeBase.load()
    print(f"  RxNorm: {len(kb.rxnorm.rxcui_to_names):,} RXCUI")
    print(f"  ICD-10-CM: {len(kb.icd10.code_to_description):,} ma benh")

    print("\n=== Buoc 4: Demo truy van (theo vi du de bai) ===")

    print('\nsearch_drug("Chlorpheniramine 0.4 MG/ML"):')
    for m in kb.search_drug("Chlorpheniramine 0.4 MG/ML", limit=5):
        print(f"  RXCUI={m.rxcui}  TTY={m.tty}  {m.name}")

    print('\nsearch_drug("Capsaicin 0.38 MG/ML"):')
    for m in kb.search_drug("Capsaicin 0.38 MG/ML", limit=5):
        print(f"  RXCUI={m.rxcui}  TTY={m.tty}  {m.name}")

    print('\nsearch_diagnosis("gastro-esophageal reflux"):')
    for m in kb.search_diagnosis("gastro-esophageal reflux", limit=5):
        print(f"  {m.code}  {m.description}")


if __name__ == "__main__":
    main()
