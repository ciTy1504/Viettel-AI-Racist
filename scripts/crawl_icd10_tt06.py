"""
Crawl ICD-10 TT06 (ban tieng Viet cua Bo Y te) tu API cong khai cua icd.kcb.vn.

API backend (khong can dang nhap):
    https://ccs.whiteneuron.com/api/ICD10_TT06/

Cay 4 cap:  chapter -> section -> type -> disease (la)
    GET root?lang=vi                      -> danh sach chapter
    GET childs/{model}?id={id}&lang=vi    -> con cua mot node
    GET data/disease?id={id}&lang=vi      -> chi tiet (include/exclude/note) - tuy chon

Moi node tra ve dang:
    {"model": "...", "id": "...", "is_leaf": bool,
     "data": {"code": "A00.0", "id": "A000", "name": "...", "html": ...}}

Ghi ra JSONL (moi dong 1 ma):
    data/raw/icd10_tt06/icd10_tt06.jsonl

Cach dung:
    python scripts\\crawl_icd10_tt06.py
    python scripts\\crawl_icd10_tt06.py --details   # lay them include/exclude (cham hon)
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.knowledge_base.paths import RAW_DIR

BASE = "https://ccs.whiteneuron.com/api/ICD10_TT06/"
OUT_DIR = RAW_DIR / "icd10_tt06"
OUT_PATH = OUT_DIR / "icd10_tt06.jsonl"

_HEADERS = {
    "Origin": "https://icd.kcb.vn",
    "Referer": "https://icd.kcb.vn/",
    "User-Agent": "Mozilla/5.0 (icd-tt06-crawler)",
    "Accept": "application/json",
}


def _get(url: str, retries: int = 4) -> dict:
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=_HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            if payload.get("status") != "success":
                raise ValueError(f"status != success: {payload.get('status')}")
            return payload
        except (urllib.error.URLError, ValueError, json.JSONDecodeError) as e:
            last_err = e
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"GET that bai sau {retries} lan: {url} ({last_err})")


def _childs(model: str, node_id: str, lang: str) -> list[dict]:
    url = f"{BASE}childs/{model}?id={urllib.parse.quote(node_id)}&lang={lang}"
    return _get(url).get("data", [])


def _root(lang: str) -> list[dict]:
    return _get(f"{BASE}root?lang={lang}").get("data", [])


def _flatten(node: dict, parent_id: str | None, parent_code: str | None,
             level: int, chapter: str | None) -> dict:
    d = node.get("data", {})
    return {
        "code": d.get("code", "").strip(),
        "name": (d.get("name") or "").strip(),
        "id": node.get("id", d.get("id", "")),
        "model": node.get("model", ""),
        "level": level,
        "is_leaf": bool(node.get("is_leaf", False)),
        "parent_id": parent_id,
        "parent_code": parent_code,
        "chapter": chapter,
    }


def crawl(lang: str, max_workers: int = 12) -> list[dict]:
    records: list[dict] = []
    seen: set[tuple[str, str]] = set()  # (model, id) de tranh trung

    def emit(node, parent_id, parent_code, level, chapter):
        key = (node.get("model", ""), node.get("id", ""))
        if key in seen:
            return None
        seen.add(key)
        rec = _flatten(node, parent_id, parent_code, level, chapter)
        records.append(rec)
        return rec

    print(f"[{lang}] Lay danh sach chuong (root)...")
    chapters = _root(lang)
    for ch in chapters:
        emit(ch, None, None, 1, ch.get("id"))

    # Duyet theo tung cap, moi cap chay song song cac request childs.
    def expand_level(nodes: list[dict], level: int) -> list[dict]:
        next_nodes: list[dict] = []

        def work(node):
            if node.get("is_leaf"):
                return []
            model = node.get("model", "")
            node_id = node.get("id", "")
            try:
                return _childs(model, node_id, lang)
            except RuntimeError as e:
                print(f"  [bo qua] {model}:{node_id} - {e}")
                return []

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            results = list(pool.map(work, nodes))

        chapter_by_key = {(r["model"], r["id"]): r["chapter"] for r in records}
        for parent, children in zip(nodes, results):
            p_id = parent.get("id")
            p_code = parent.get("data", {}).get("code")
            p_chapter = chapter_by_key.get((parent.get("model"), p_id)) or p_id
            for child in children:
                rec = emit(child, p_id, p_code, level + 1, p_chapter)
                if rec is not None and not child.get("is_leaf"):
                    next_nodes.append(child)
        return next_nodes

    level = 1
    frontier = chapters
    while frontier:
        # loc ra node chua phai la de mo rong tiep
        expandable = [n for n in frontier if not n.get("is_leaf")]
        if not expandable:
            break
        print(f"[{lang}] Mo rong cap {level} ({len(expandable)} node)...")
        frontier = expand_level(expandable, level)
        level += 1

    return records


def enrich_details(records: list[dict], lang: str = "vi", max_workers: int = 12) -> None:
    """(Tuy chon) Lay them include/exclude/note cho cac ma disease qua data/disease."""
    diseases = [r for r in records if r["model"] == "disease"]
    print(f"Lay chi tiet cho {len(diseases)} ma disease...")

    def work(rec):
        url = f"{BASE}data/disease?id={urllib.parse.quote(rec['id'])}&lang={lang}"
        try:
            d = _get(url).get("data", {}).get("data", {})
            rec["include"] = (d.get("include") or "").strip()
            rec["exclude"] = (d.get("exclude") or "").strip()
            rec["note"] = (d.get("note") or "").strip()
        except RuntimeError:
            pass

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        list(pool.map(work, diseases))


def crawl_bilingual(max_workers: int = 12) -> list[dict]:
    """Crawl ca hai ban vi + en (cung bo ma), gop thanh record song ngu:
    {code, name_vi, name_en, id, model, level, is_leaf, parent_id, parent_code, chapter}."""
    vi_records = crawl("vi", max_workers=max_workers)
    en_records = crawl("en", max_workers=max_workers)

    # gop theo (model, id) - khoa on dinh giua hai lan crawl
    en_name_by_key = {(r["model"], r["id"]): r["name"] for r in en_records}

    merged: list[dict] = []
    for r in vi_records:
        key = (r["model"], r["id"])
        merged.append({
            "code": r["code"],
            "name_vi": r["name"],
            "name_en": en_name_by_key.get(key, ""),
            "id": r["id"],
            "model": r["model"],
            "level": r["level"],
            "is_leaf": r["is_leaf"],
            "parent_id": r["parent_id"],
            "parent_code": r["parent_code"],
            "chapter": r["chapter"],
        })
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", choices=["vi", "en", "both"], default="both",
                        help="Ngon ngu can crawl (mac dinh 'both' -> file song ngu)")
    parser.add_argument("--details", action="store_true",
                        help="Lay them include/exclude/note (cham hon, ~14k request)")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    t0 = time.time()
    if args.lang == "both":
        records = crawl_bilingual(max_workers=args.workers)
    else:
        raw = crawl(args.lang, max_workers=args.workers)
        key = f"name_{args.lang}"
        records = [{**r, key: r.pop("name")} for r in raw]
    print(f"Da crawl {len(records)} node trong {time.time()-t0:.1f}s")

    if args.details:
        enrich_details(records, lang="vi", max_workers=args.workers)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_model: dict[str, int] = {}
    for r in records:
        by_model[r["model"]] = by_model.get(r["model"], 0) + 1
    missing_en = sum(1 for r in records if "name_en" in r and not r["name_en"])
    print(f"Ghi ra: {OUT_PATH}")
    print(f"Thong ke theo cap: {by_model}")
    if args.lang == "both":
        print(f"So ma thieu ten tieng Anh: {missing_en}")


if __name__ == "__main__":
    main()
