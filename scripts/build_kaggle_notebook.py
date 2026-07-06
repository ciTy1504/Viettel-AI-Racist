"""
Sinh notebook Kaggle tu chua: chay Qwen2.5-7B-Vietnamese-Medical-NER de trich
entity, roi ep position + map type + gan assertion + tra candidate RxNorm/ICD,
xuat ra submission dung schema de bai.

Chay:  python scripts/build_kaggle_notebook.py
Ket qua: notebooks/ner_qwen_pipeline.ipynb  (upload / tao Kaggle notebook tu file nay)
"""

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "ner_qwen_pipeline.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}


CELLS = []

# ---------------------------------------------------------------- intro
CELLS.append(md(r"""# AI Race 2026 — NER y tế bằng Qwen2.5-7B (Kaggle)

Pipeline 1-file: **model NER → ép position → map type → assertion → candidate RxNorm/ICD → submission**.

## Cách setup trên Kaggle (làm 1 lần)
1. **Settings → Accelerator → GPU T4 x2** (bắt buộc, model 7B).
2. **Settings → Internet → On** (để tải model từ HuggingFace).
3. **Add Input**: upload **toàn bộ folder repo** làm 1 Kaggle Dataset (giữ nguyên cấu trúc). Cell config
   mặc định trỏ tới `/kaggle/input/datasets/thiinhcng/viettel/Viettel-AI-Racist-`; nếu Kaggle mount ở
   path khác thì notebook **tự dò lại** (`RXNCONSO.csv`, `icd10_tt06.jsonl`, thư mục `data/test/input`).
4. Chạy toàn bộ (Run All). Kết quả: `submission.zip` + `submission_combined.json` trong `/kaggle/working`.

> Đầu vào lấy hết từ repo đã upload: `data/test/input/*.txt`, `data/processed/rxnorm/RXNCONSO.csv`,
> `data/raw/icd10_tt06/icd10_tt06.jsonl`. Nếu thiếu file nào, cell config sẽ báo lỗi rõ (không chạy tiếp).
> Model chỉ trích `{entity, type}` (**không có position**) — notebook tự dò offset trong văn bản gốc.
"""))

# ---------------------------------------------------------------- install
CELLS.append(code(r"""
# Kaggle da co torch + transformers; chi can them bitsandbytes cho 4-bit.
!pip -q install -U bitsandbytes accelerate
"""))

# ---------------------------------------------------------------- config
CELLS.append(code(r'''
import os, glob, json, re

MODEL_ID = "PeterPaker123/Qwen2.5-7B-Vietnamese-Medical-NER"

# Repo duoc upload nguyen ven lam Kaggle dataset. Uu tien duong dan tuong minh,
# roi fallback tu dong do neu Kaggle mount o cho khac.
REPO_BASE_CANDIDATES = [
    "/kaggle/input/datasets/thiinhcng/viettel/Viettel-AI-Racist-",
    "/kaggle/input/viettel/Viettel-AI-Racist-",
    "/kaggle/input/viettel",
]

def _first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None

def _glob_one(pattern):
    hits = sorted(glob.glob(pattern, recursive=True))
    return hits[0] if hits else None

REPO_BASE = _first_existing(REPO_BASE_CANDIDATES)

# --- INPUT: thu muc 100 file .txt ---
INPUT_DIR = _first_existing([os.path.join(b, "data/test/input") for b in REPO_BASE_CANDIDATES if b])
if not INPUT_DIR:
    # fallback: thu muc co nhieu .txt nhat trong /kaggle/input
    cnt = {}
    for p in glob.glob("/kaggle/input/**/*.txt", recursive=True):
        d = os.path.dirname(p); cnt[d] = cnt.get(d, 0) + 1
    INPUT_DIR = max(cnt, key=cnt.get) if cnt else None

# --- KB: RxNorm + ICD-10 TT06 (luon co vi upload ca repo) ---
RXNCONSO_CSV = _first_existing(
    [os.path.join(b, "data/processed/rxnorm/RXNCONSO.csv") for b in REPO_BASE_CANDIDATES if b]
) or _glob_one("/kaggle/input/**/RXNCONSO.csv")
ICD_TT06_JSONL = _first_existing(
    [os.path.join(b, "data/raw/icd10_tt06/icd10_tt06.jsonl") for b in REPO_BASE_CANDIDATES if b]
) or _glob_one("/kaggle/input/**/icd10_tt06.jsonl")

OUTPUT_DIR = "/kaggle/working/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("REPO_BASE   :", REPO_BASE)
print("INPUT_DIR   :", INPUT_DIR)
print("RXNCONSO.csv:", RXNCONSO_CSV)
print("ICD tt06    :", ICD_TT06_JSONL)
assert INPUT_DIR, "Khong tim thay data/test/input — kiem tra lai duong dan dataset."
assert RXNCONSO_CSV, "Khong thay RXNCONSO.csv — chay convert_rxnorm truoc khi upload, hoac kiem tra path."
assert ICD_TT06_JSONL, "Khong thay icd10_tt06.jsonl — chay crawl_icd10_tt06 truoc khi upload."
files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.txt")), key=lambda p: (len(os.path.basename(p)), p))
print("So file test:", len(files))
'''))

# ---------------------------------------------------------------- load model
CELLS.append(code(r'''
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", quantization_config=bnb, torch_dtype=torch.bfloat16,
)
model.eval()
print("Da nap model 4-bit.")
'''))

# ---------------------------------------------------------------- NER
CELLS.append(code(r'''
# He thong yeu cau trich dung cac loai cua de. Model duoc SFT theo mo ta trong
# system prompt (dynamic extraction) nen ta mo ta 5 loai bang tieng Viet + Anh.
SYSTEM_PROMPT = """You are a medical expert extracting entities from Vietnamese clinical notes.
Extract EXACT substrings (verbatim, keep original casing/spacing) for these types:
- DRUG: ten thuoc/vitamin/thuc pham chuc nang, LAY CA lieu-duong dung-tan suat neu di lien (vd "amlodipine 10 mg po daily").
- SYMPTOM: trieu chung, dau hieu benh (vd "dau bung", "kho tho").
- DIAGNOSIS: benh ly / chan doan (vd "tang huyet ap", "xo gan do ruou").
- LAB_TEST: ten xet nghiem / thu thuat chan doan (vd "cong thuc mau", "sieu am o bung").
- LAB_RESULT: ket qua / gia tri xet nghiem (vd "bach cau tang", "cea 4.9").
Return ONLY a JSON list of {"entity": "...", "type": "..."}. If none, return []."""

def _chunks(text, max_chars=3000):
    """Chia van ban dai theo dong de khong vuot ngu canh; position se duoc do lai
    tren van ban goc nen offset cua chunk khong quan trong."""
    if len(text) <= max_chars:
        return [text]
    parts, cur = [], ""
    for line in text.splitlines(keepends=True):
        if len(cur) + len(line) > max_chars and cur:
            parts.append(cur); cur = ""
        cur += line
    if cur:
        parts.append(cur)
    return parts

def _parse_json_list(s):
    m = re.search(r"\[.*\]", s, re.DOTALL)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except Exception:
        return []
    out = []
    for d in data if isinstance(data, list) else []:
        if isinstance(d, dict) and d.get("entity") and d.get("type"):
            out.append({"entity": str(d["entity"]).strip(), "type": str(d["type"]).strip()})
    return out

@torch.inference_mode()
def run_ner(text):
    ents = []
    for chunk in _chunks(text):
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Text: " + chunk}]
        enc = tokenizer.apply_chat_template(
            msgs, return_tensors="pt", add_generation_prompt=True, return_dict=True
        ).to(model.device)
        out = model.generate(**enc, max_new_tokens=1024, do_sample=False,
                             pad_token_id=(tokenizer.pad_token_id or tokenizer.eos_token_id))
        resp = tokenizer.decode(out[0][enc["input_ids"].shape[-1]:], skip_special_tokens=True)
        ents.extend(_parse_json_list(resp))
    return ents

# thu 1 file de kiem tra
_demo = open(files[0], encoding="utf-8").read()
print(run_ner(_demo)[:8])
'''))

# ---------------------------------------------------------------- run all
CELLS.append(code(r'''
from time import time
raw_by_file = {}
t0 = time()
for i, fp in enumerate(files, 1):
    text = open(fp, encoding="utf-8").read()
    raw_by_file[os.path.basename(fp)] = {"text": text, "entities": run_ner(text)}
    if i % 10 == 0 or i == len(files):
        print(f"  {i}/{len(files)}  ({time()-t0:.0f}s)")
print("Xong NER.")
'''))

# ---------------------------------------------------------------- align + type map
CELLS.append(code(r'''
# Map type cua model -> type cua de. SYMPTOM_AND_DISEASE (neu model tra) tam coi la TRIEU_CHUNG.
TYPE_MAP = {
    "DRUG": "THUỐC",
    "SYMPTOM": "TRIỆU_CHỨNG",
    "SYMPTOM_AND_DISEASE": "TRIỆU_CHỨNG",
    "DISEASE": "CHẨN_ĐOÁN",
    "DIAGNOSIS": "CHẨN_ĐOÁN",
    "MEDICAL_PROCEDURE": "TÊN_XÉT_NGHIỆM",
    "LAB_TEST": "TÊN_XÉT_NGHIỆM",
    "LAB_RESULT": "KẾT_QUẢ_XÉT_NGHIỆM",
}

def align_spans(text, entities):
    """Ep position: tim moi entity trong van ban goc, chon lan xuat hien chua bi chiem.
    Uu tien khop chinh xac; roi ve khong phan biet hoa/thuong."""
    low = text.lower()
    claimed = []  # danh sach (start, end) da dung
    result = []

    def overlaps(a, b):
        return any(a < e and s < b for s, e in claimed)

    for e in entities:
        s = e["entity"].strip()
        if not s:
            continue
        typ = TYPE_MAP.get(e["type"].upper(), None)
        if typ is None:
            continue
        # tim vi tri dau tien khong dam nhau
        found = None
        for hay, needle in ((text, s), (low, s.lower())):
            start = 0
            while True:
                idx = hay.find(needle, start)
                if idx < 0:
                    break
                if not overlaps(idx, idx + len(s)):
                    found = (idx, idx + len(s))
                    break
                start = idx + 1
            if found:
                break
        if not found:
            continue
        claimed.append(found)
        result.append({"text": text[found[0]:found[1]], "type": typ,
                       "position": [found[0], found[1]]})
    result.sort(key=lambda x: x["position"][0])
    return result
'''))

# ---------------------------------------------------------------- assertions
CELLS.append(code(r'''
HIST_KW = ["tiền sử", "trước khi nhập viện", "tiền căn", "trước đây", "bệnh sử"]
HIST_RESET_KW = ["hiện tại", "lý do nhập viện", "khi nhập viện", "tại khoa", "khám"]
NEG_KW = ["không", "chưa", "phủ nhận", "loại trừ", "không có", "không thấy"]
HYP_KW = ["nếu", "nghi ngờ", "có thể", "khi cần"]
FAM_KW = ["gia đình", "mẹ", "bố", "cha", "anh", "chị", "người thân"]

def _is_header(line):
    s = line.strip()
    if not s or s.startswith("-"):
        return False
    return bool(re.match(r"^\s*\d+\.", s)) or s.endswith(":") or (len(s) < 60 and not s.endswith((".", ",")))

def historical_mask(text):
    """Tra ve list bool theo tung ky tu: True neu nam trong section 'tien su'."""
    mask = [False] * (len(text) + 1)
    active = False
    pos = 0
    for line in text.splitlines(keepends=True):
        low = line.lower()
        # bat ky dong nao nhac 'tien su / truoc nhap vien' -> bat ngu canh historical
        if any(k in low for k in HIST_KW):
            active = True
        elif _is_header(line) and (re.match(r"^\s*\d+\.", line) or any(k in low for k in HIST_RESET_KW)):
            active = False
        for i in range(pos, pos + len(line)):
            mask[i] = active
        pos += len(line)
    return mask

def assign_assertions(text, start, end, hist_mask):
    a = []
    line_start = text.rfind("\n", 0, start) + 1
    before = text[max(0, start - 40):start].lower()
    line_ctx = text[line_start:end].lower()
    if any(k in before or k in line_ctx for k in NEG_KW):
        a.append("isNegated")
    if any(k in before for k in HYP_KW):
        a.append("isHypothetical")
    if any(k in before for k in FAM_KW):
        a.append("isFamily")
    if start < len(hist_mask) and hist_mask[start] and "isNegated" not in a:
        a.append("isHistorical")
    return a
'''))

# ---------------------------------------------------------------- candidates (optional)
CELLS.append(code(r'''
import csv, unicodedata
_NON_ALNUM = re.compile(r"[^0-9a-z]+")

def norm(t):
    t = unicodedata.normalize("NFKD", t.strip().lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return _NON_ALNUM.sub(" ", t).strip()

# ---- RxNorm (thuoc) ----
rx_index = []   # (normalized_name, rxcui, tty)
if RXNCONSO_CSV:
    csv.field_size_limit(10**7)
    with open(RXNCONSO_CSV, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("SAB") == "RXNORM":
                rx_index.append((norm(row["STR"]), row["RXCUI"], row["TTY"]))
    print("RxNorm names:", len(rx_index))

_TTY_RANK = {"SCD": 3, "SBD": 3, "GPCK": 2, "BPCK": 2, "IN": 1, "PIN": 1, "MIN": 1}
_NUM = re.compile(r"\d+(?:[.,]\d+)?")
_ROUTE_FORM = re.compile(r"\b(po|iv|im|sc|sq|sl|pr|xl|er|sr|oral|suspension|tablet|cap|ml|mg|mcg|g|daily|bid|tid|qid|prn|q\d+h|qhs|qam|qod)\b")

def link_drug(name, limit=3):
    if not rx_index:
        return []
    q = norm(_ROUTE_FORM.sub(" ", name))
    q = re.sub(r"\s+", " ", q).strip()
    if not q:
        return []
    doses = {n.replace(",", ".") for n in _NUM.findall(name)}
    hits = [(nm, cui, tty) for nm, cui, tty in rx_index if q in nm or nm in q]
    if not hits:
        # thu tung tu chinh (ten hoat chat dau tien)
        first = q.split()[0] if q.split() else ""
        if len(first) >= 4:
            hits = [(nm, cui, tty) for nm, cui, tty in rx_index if first in nm]
    if doses:
        with_dose = [h for h in hits if any(d in h[0] for d in doses)]
        hits = with_dose or []   # co lieu ma khong khop -> thao rong
    def key(h):
        nm, cui, tty = h
        dose_hit = 1 if doses and any(d in nm for d in doses) else 0
        return (-dose_hit, -_TTY_RANK.get(tty, 0), len(nm))
    hits.sort(key=key)
    out, seen = [], set()
    for nm, cui, tty in hits:
        if cui in seen:
            continue
        seen.add(cui); out.append(cui)
        if len(out) >= limit:
            break
    return out

# ---- ICD-10 TT06 (chan doan) ----
icd_index = []  # (normalized_name, code, is_leaf)
if ICD_TT06_JSONL:
    with open(ICD_TT06_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("code") and r.get("name_vi"):
                icd_index.append((norm(r["name_vi"]), r["code"], bool(r.get("is_leaf"))))
                if r.get("name_en"):
                    icd_index.append((norm(r["name_en"]), r["code"], bool(r.get("is_leaf"))))
    print("ICD tt06 names:", len(icd_index))

def link_diagnosis(name, limit=1):
    if not icd_index:
        return []
    q = norm(name)
    if not q:
        return []
    hits = [(len(nm), code) for nm, code, leaf in icd_index if q in nm and leaf]
    hits.sort()
    out, seen = [], set()
    for _, code in hits:
        if code in seen:
            continue
        seen.add(code); out.append(code)
        if len(out) >= limit:
            break
    return out
'''))

# ---------------------------------------------------------------- assemble
CELLS.append(code(r'''
final_by_file = {}
for fname, obj in raw_by_file.items():
    text = obj["text"]
    ents = align_spans(text, obj["entities"])
    hmask = historical_mask(text)
    for e in ents:
        s, en = e["position"]
        e["assertions"] = assign_assertions(text, s, en, hmask)
        if e["type"] == "THUỐC":
            c = link_drug(e["text"])
            if c:
                e["candidates"] = c
        elif e["type"] == "CHẨN_ĐOÁN":
            c = link_diagnosis(e["text"])
            if c:
                e["candidates"] = c
    # sap xep field cho giong vi du
    final_by_file[fname] = [
        {k: e[k] for k in ["text", "type", "candidates", "assertions", "position"] if k in e}
        for e in ents
    ]

# thong ke nhanh
import collections
tt = collections.Counter(); nc = 0; ne = 0
for ents in final_by_file.values():
    for e in ents:
        tt[e["type"]] += 1; ne += 1
        if e.get("candidates"):
            nc += 1
print("Tong entity:", ne, "| theo type:", dict(tt), "| co candidate:", nc)
'''))

# ---------------------------------------------------------------- write submission
CELLS.append(code(r'''
import zipfile
# 1) moi file 1 json (ten trung ten input, doi .txt -> .json)
per_dir = "/kaggle/working/output"
os.makedirs(per_dir, exist_ok=True)
for fname, ents in final_by_file.items():
    stem = os.path.splitext(fname)[0]
    with open(os.path.join(per_dir, stem + ".json"), "w", encoding="utf-8") as f:
        json.dump(ents, f, ensure_ascii=False, indent=2)

# 2) zip
with zipfile.ZipFile("/kaggle/working/submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for fname in final_by_file:
        stem = os.path.splitext(fname)[0]
        z.write(os.path.join(per_dir, stem + ".json"), stem + ".json")

# 3) ban gop 1 file
combined = {os.path.splitext(k)[0]: v for k, v in final_by_file.items()}
with open("/kaggle/working/submission_combined.json", "w", encoding="utf-8") as f:
    json.dump(combined, f, ensure_ascii=False, indent=2)

print("Da ghi: /kaggle/working/submission.zip va submission_combined.json")
'''))


def main():
    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"Da tao {OUT} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
