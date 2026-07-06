"""
Chay pipeline baseline tren toan bo data/test/input/*.txt, ghi ket qua JSON
vao data/test/output/*.json (dung schema de bai: text/type/candidates/
assertions/position).

Cach dung:
    python scripts\\run_pipeline.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.knowledge_base.lookup import KnowledgeBase
from src.knowledge_base.paths import DATA_DIR
from src.pipeline.run import extract_entities

INPUT_DIR = DATA_DIR / "test" / "input"
OUTPUT_DIR = DATA_DIR / "test" / "output"


def main() -> None:
    try:
        kb = KnowledgeBase.load()
        print("Da nap KnowledgeBase (RxNorm + ICD-10-CM).")
    except FileNotFoundError as e:
        kb = None
        print(f"CANH BAO: khong nap duoc KnowledgeBase ({e}). Candidates se de rong.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(INPUT_DIR.glob("*.txt"), key=lambda p: int(p.stem))
    for path in files:
        text = path.read_text(encoding="utf-8")
        entities = extract_entities(text, kb)
        out_path = OUTPUT_DIR / f"{path.stem}.json"
        out_path.write_text(
            json.dumps(entities, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print(f"Da xu ly {len(files)} file -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
