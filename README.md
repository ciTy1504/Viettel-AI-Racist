# Ontological Reasoning in Medical Knowledge Retrieval

Hệ thống trích xuất và chuẩn hóa khái niệm y khoa (triệu chứng, xét nghiệm, chẩn đoán, thuốc) từ văn bản tự do, ánh xạ chẩn đoán sang **ICD-10-CM** và thuốc sang **RxNorm**, phục vụ cuộc thi *Vòng 1 — Sơ loại* (hạn nộp 30/07/2026).

## 1. Cấu trúc thư mục

```
.
├── README.md
├── .gitignore
├── data/
│   ├── raw/                       # Du lieu goc, tai truc tiep tu nguon chinh thuc (KHONG sua tay)
│   │   ├── rxnorm/
│   │   │   ├── rrf/                       # RXNCONSO.RRF, RXNREL.RRF, RXNSAT.RRF (ban goc NLM)
│   │   │   ├── scripts/                   # Script SQL chinh thuc cua NLM (mysql/, oracle/) — khong dung trong pipeline nay
│   │   │   └── Readme_Full_Prescribe_06012026.txt
│   │   └── icd10cm/
│   │       ├── icd10cm-codes-2026.txt     # Danh sach ma benh (chi ma billable) + mo ta
│   │       ├── icd10cm-order-2026.txt     # Danh sach day du (ca ma cha), dinh dang fixed-width
│   │       ├── *-addenda-2026.txt         # Phu luc thay doi trong nam
│   │       └── *.pdf                      # Tai lieu mo ta dinh dang cua CDC/NCHS
│   ├── processed/                  # Du lieu da convert sang CSV (sinh tu data/raw/, KHONG commit)
│   │   ├── rxnorm/
│   │   │   ├── RXNCONSO.csv        # RXCUI <-> ten thuoc (18 cot chuan RxNorm)
│   │   │   ├── RXNREL.csv          # Quan he giua cac RXCUI (VD: ingredient_of, tradename_of)
│   │   │   └── RXNSAT.csv          # Thuoc tinh bo sung cho RXCUI (NDC, SPL_SET_ID, ...)
│   │   └── icd10cm/
│   │       ├── icd10cm-codes-2026.csv   # code (co dau cham, VD "K21.0"), raw_code, description
│   │       └── icd10cm-order-2026.csv   # + order_number, billable, short/long description
│   └── test/
│       └── input/                  # Tap test vong 1 (100 file .txt, do BTC cung cap)
├── src/
│   └── knowledge_base/
│       ├── paths.py                # Duong dan chuan cho data/raw va data/processed
│       ├── icd10_utils.py          # format_icd10_code(): them dau cham vao ma ICD-10-CM
│       ├── convert_rxnorm.py       # Convert RRF -> CSV
│       ├── convert_icd10.py        # Convert TXT (fixed-width) -> CSV
│       └── lookup.py               # RxNormLookup, ICD10Lookup, KnowledgeBase (tra cuu candidate)
└── scripts/
    └── build_knowledge_base.py     # Chay toan bo pipeline convert + demo truy van
```

## 2. Nguồn dữ liệu

| Nguồn | Phiên bản | Link tải chính thức |
|---|---|---|
| RxNorm (Prescribable Content) | 06/01/2026 | https://www.nlm.nih.gov/research/umls/rxnorm/docs/rxnormfiles.html |
| ICD-10-CM | FY2026 (hiệu lực 01/10/2025 – 30/09/2026) | https://ftp.cdc.gov/pub/health_statistics/nchs/publications/ICD10CM/2026/ |

> Đã chọn bản **FY2026** thay vì FY2027 vì FY2027 chỉ có hiệu lực từ 01/10/2026, sau hạn nộp bài của vòng 1.

## 3. Cài đặt

Chỉ cần Python 3.10+ (dùng type hint `list[str]`, `X | None`), không có dependency ngoài standard library.

```powershell
py --version   # kiem tra da co Python
```

## 4. Sử dụng

### 4.1. Build toàn bộ knowledge base (convert raw -> CSV + demo)

```powershell
python scripts\build_knowledge_base.py
```

Script sẽ:
1. Convert `data/raw/rxnorm/rrf/*.RRF` -> `data/processed/rxnorm/*.csv` (bỏ qua nếu đã tồn tại).
2. Convert `data/raw/icd10cm/*.txt` -> `data/processed/icd10cm/*.csv`, tự động format lại mã ICD-10-CM có dấu chấm (`K210` -> `K21.0`).
3. Nạp cả hai nguồn vào bộ nhớ và chạy vài truy vấn demo.

Muốn convert lại từ đầu (ghi đè CSV cũ):

```python
from src.knowledge_base.convert_rxnorm import convert_all as convert_rxnorm
from src.knowledge_base.convert_icd10 import convert_all as convert_icd10
convert_rxnorm(force=True)
convert_icd10(force=True)
```

### 4.2. Tra cứu candidate (dùng trong bước entity linking)

```python
from src.knowledge_base.lookup import KnowledgeBase

kb = KnowledgeBase.load()

# Tim RXCUI cho ten thuoc trich xuat duoc tu van ban
for m in kb.search_drug("Chlorpheniramine maleate 2 MG/ML"):
    print(m.rxcui, m.tty, m.name)

# Tim ma ICD-10-CM cho chan doan
for m in kb.search_diagnosis("gastro-esophageal reflux"):
    print(m.code, m.description)
```

`search_drug` / `search_diagnosis` dùng khớp chuỗi con (substring, không phân biệt hoa/thường và dấu tiếng Việt), trả về kết quả ngắn nhất/khớp gần nhất trước — phù hợp làm baseline candidate generation, nhưng **chưa xử lý đồng nghĩa/viết tắt phức tạp** (ví dụ "po", "prn", tên biệt dược không khớp hoạt chất). Bước tiếp theo nên bổ sung fuzzy matching hoặc embedding search cho các trường hợp này.

## 5. Lưu ý quan trọng khi dùng dữ liệu

- **RXNCONSO.csv chỉ giữ các cột chuẩn RxNorm** (RXCUI, LAT, TS, LUI, STT, SUI, ISPREF, RXAUI, SAUI, SCUI, SDUI, SAB, TTY, CODE, STR, SRL, SUPPRESS, CVF) — nhiều cột luôn rỗng vì đó là hành vi đúng của định dạng UMLS dùng chung, RxNorm chỉ dùng một tập con.
- **Bộ RxNorm đang dùng là bản "full_prescribe"**, chỉ chứa các khái niệm *có thể kê đơn* (SCD/SBD/GPCK/BPCK...). Một số RXCUI hoạt chất đơn lẻ (SCDC) không nằm trong sản phẩm phối hợp nào có thể **không tồn tại** trong bộ này — đã kiểm chứng thực tế: RXCUI `360047` ("Chlorpheniramine 0.4 MG/ML") trong ví dụ đề bài **không có** trong dữ liệu 06/01/2026 đã tải. Nếu cần độ phủ đầy đủ hơn, cân nhắc tải bản RxNorm Full Release thay vì Prescribable Content.
- **Mã ICD-10-CM đã được format lại có dấu chấm** trong `data/processed/icd10cm/*.csv` (cột `code`), cột `raw_code` giữ nguyên bản gốc không dấu chấm từ CDC để đối chiếu khi cần.
- FY2026 có độ chi tiết cao hơn một số ví dụ cũ trong đề bài (ví dụ `K21.9` khớp đúng, nhưng `K21.0` trong đề đã được chia nhỏ thành `K21.00`/`K21.01` theo bleeding status ở phiên bản hiện tại) — cần lưu ý khi so khớp với ground truth nếu BTC dùng phiên bản ICD-10-CM cũ hơn.

## 6. Việc tiếp theo

- [ ] Xây module NER/LLM trích xuất khái niệm (`TRIỆU_CHỨNG`, `TÊN_XÉT_NGHIỆM`, `KẾT_QUẢ_XÉT_NGHIỆM`, `CHẨN_ĐOÁN`, `THUỐC`) + vị trí ký tự trong văn bản.
- [ ] Module suy luận assertion (`isNegated`, `isFamily`, `isHistorical`) theo ngữ cảnh câu.
- [ ] Nâng cấp `search_drug`/`search_diagnosis` từ substring-match sang fuzzy/embedding search để tăng recall candidate.
