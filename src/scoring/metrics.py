"""
Bo cham diem local, tai hien theo mo ta trong ke hoach (chua phai cong thuc chinh
thuc cua BTC - can hieu chinh lai khi co phan hoi diem that tu he thong nop bai):

    final = 0.3 * text_score + 0.3 * assertion_score + 0.4 * candidate_score

- text_score: 1 - WER giua chuoi cac entity text (theo thu tu vi tri) cua prediction
  va ground truth, cong voi rang buoc entity phai dung `type` va overlap `position`
  moi duoc tinh la khop (theo bay diem "TYPE" trong ke hoach: sai type = tinh 0
  o ca 3 metric).
- assertion_score / candidate_score: trung binh Jaccard tren tung cap entity da
  khop nhau (J=1 neu ca hai tap rong, dung theo bay diem "EMPTY").

Dung de so sanh tuong doi giua cac phien ban pipeline, KHONG phai diem thi that.

>>> from src.scoring.metrics import score_document
>>> score_document(pred_entities, gold_entities)
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Entity:
    text: str
    type: str
    position: tuple[int, int]
    assertions: list[str] = field(default_factory=list)
    candidates: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "Entity":
        return cls(
            text=d["text"],
            type=d["type"],
            position=tuple(d["position"]),
            assertions=list(d.get("assertions", [])),
            candidates=list(d.get("candidates", [])),
        )


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0  # bay diem "EMPTY": ca hai rong -> J = 1
    union = sa | sb
    if not union:
        return 1.0
    return len(sa & sb) / len(union)


def word_error_rate(ref: list[str], hyp: list[str]) -> float:
    """WER chuan (Levenshtein tren chuoi token) giua hai danh sach token."""
    n, m = len(ref), len(hyp)
    if n == 0:
        return 0.0 if m == 0 else 1.0

    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev_diag = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            tmp = dp[j]
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[j] = min(
                dp[j] + 1,      # deletion
                dp[j - 1] + 1,  # insertion
                prev_diag + cost,  # substitution
            )
            prev_diag = tmp
    return dp[m] / n


@dataclass
class MatchedPair:
    gold: Entity | None
    pred: Entity | None
    type_ok: bool


def _match_entities(pred: list[Entity], gold: list[Entity]) -> list[MatchedPair]:
    """Ghep moi gold entity voi pred entity co overlap position + cung type,
    uu tien overlap dai nhat (tham lam, du cho muc dich so sanh local)."""
    used_pred: set[int] = set()
    pairs: list[MatchedPair] = []

    for g in gold:
        best_idx, best_overlap = None, 0
        for i, p in enumerate(pred):
            if i in used_pred or p.type != g.type:
                continue
            if not _overlaps(p.position, g.position):
                continue
            ov = min(p.position[1], g.position[1]) - max(p.position[0], g.position[0])
            if ov > best_overlap:
                best_idx, best_overlap = i, ov
        if best_idx is not None:
            used_pred.add(best_idx)
            pairs.append(MatchedPair(gold=g, pred=pred[best_idx], type_ok=True))
        else:
            # co the co pred overlap nhung sai type -> tinh la khong khop (0 diem, bay TYPE)
            pairs.append(MatchedPair(gold=g, pred=None, type_ok=False))

    for i, p in enumerate(pred):
        if i not in used_pred:
            pairs.append(MatchedPair(gold=None, pred=p, type_ok=False))

    return pairs


@dataclass
class DocumentScore:
    text_score: float
    assertion_score: float
    candidate_score: float
    final_score: float
    n_gold: int
    n_pred: int
    n_matched: int


def score_document(pred: list[dict] | list[Entity], gold: list[dict] | list[Entity]) -> DocumentScore:
    pred_e = [e if isinstance(e, Entity) else Entity.from_dict(e) for e in pred]
    gold_e = [e if isinstance(e, Entity) else Entity.from_dict(e) for e in gold]

    pairs = _match_entities(pred_e, gold_e)
    matched = [p for p in pairs if p.gold is not None and p.pred is not None]

    ref_tokens = [g.text for g in gold_e]
    hyp_tokens = [
        (p.pred.text if (p.pred is not None and p.type_ok) else "")
        for p in pairs if p.gold is not None
    ]
    text_score = 1.0 - word_error_rate(ref_tokens, hyp_tokens)
    text_score = max(0.0, text_score)

    if gold_e:
        assertion_score = sum(
            jaccard(p.pred.assertions, p.gold.assertions) if p.pred and p.type_ok else 0.0
            for p in pairs if p.gold is not None
        ) / len(gold_e)
        candidate_score = sum(
            jaccard(p.pred.candidates, p.gold.candidates) if p.pred and p.type_ok else 0.0
            for p in pairs if p.gold is not None
        ) / len(gold_e)
    else:
        assertion_score = candidate_score = 1.0 if not pred_e else 0.0

    final = 0.3 * text_score + 0.3 * assertion_score + 0.4 * candidate_score

    return DocumentScore(
        text_score=text_score,
        assertion_score=assertion_score,
        candidate_score=candidate_score,
        final_score=final,
        n_gold=len(gold_e),
        n_pred=len(pred_e),
        n_matched=len(matched),
    )


def score_corpus(preds: dict[str, list[dict]], golds: dict[str, list[dict]]) -> DocumentScore:
    """Trung binh cong don gian tren nhieu document (theo id chung, VD ten file)."""
    keys = sorted(golds.keys())
    scores = [score_document(preds.get(k, []), golds[k]) for k in keys]
    if not scores:
        raise ValueError("Khong co document nao de cham diem")

    n = len(scores)
    return DocumentScore(
        text_score=sum(s.text_score for s in scores) / n,
        assertion_score=sum(s.assertion_score for s in scores) / n,
        candidate_score=sum(s.candidate_score for s in scores) / n,
        final_score=sum(s.final_score for s in scores) / n,
        n_gold=sum(s.n_gold for s in scores),
        n_pred=sum(s.n_pred for s in scores),
        n_matched=sum(s.n_matched for s in scores),
    )
