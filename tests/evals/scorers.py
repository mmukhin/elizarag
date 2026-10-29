"""Deterministic checks on a `QAResult`. Each returns a `Check`; `score` runs the ones a case asks for."""

import re
from dataclasses import dataclass

from app.edgar.qa import CITATION, GROUNDED, QAResult

NUMBER = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")
SCALES = (1, 1e-3, 1e3, 1e-6, 1e6)  # "416,161" (millions) may be written "416.2 billion"
TOLERANCE = 0.005  # relative, for rounding


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


def _text(result: QAResult) -> str:
    return " ".join([result.answer, *(c.text for c in result.claims)])


def numbers(text: str, skip_trivial: bool = True) -> list[float]:
    """Figures in `text`, without [n] citations. With `skip_trivial`, drop bare years and single digits."""
    found = []
    for raw in NUMBER.findall(CITATION.sub(" ", text)):
        plain = raw.rstrip(",").replace(",", "")
        value = float(plain)
        if skip_trivial and "." not in plain and "," not in raw and (value < 10 or 1900 <= value <= 2100):
            continue
        found.append(value)
    return found


def check_status(result: QAResult, allowed: list[str]) -> Check:
    return Check("status", result.status in allowed, f"got {result.status}, want one of {allowed}")


def check_citations_valid(result: QAResult) -> Check:
    bad = [w for w in result.warnings if "invalid passage ids" in w or "unknown passages" in w]
    return Check("citations_valid", not bad, "; ".join(bad))


def check_claims_cited(result: QAResult) -> Check:
    uncited = [c.text[:60] for c in result.claims if not c.chunk_ids]
    if result.status not in GROUNDED:
        return Check("claims_cited", True, "not required")
    cited = bool(CITATION.search(result.answer)) or any(c.chunk_ids for c in result.claims)
    return Check("claims_cited", cited and not uncited, f"uncited: {uncited}" if uncited else "" if cited else "no citations")


def check_tickers_covered(result: QAResult, tickers: list[str]) -> Check:
    covered = {t for c in result.claims for t in c.tickers}
    missing = [t for t in tickers if t not in covered]
    return Check("tickers_covered", not missing, f"no claim for {missing}" if missing else "")


def check_mentions(result: QAResult, terms: list[str]) -> Check:
    """Every term must appear (case-insensitive); `a|b` accepts either."""
    text = _text(result).lower()
    missing = [t for t in terms if not any(alt.lower() in text for alt in t.split("|"))]
    return Check("must_mention", not missing, f"missing {missing}" if missing else "")


def check_not_mentions(result: QAResult, terms: list[str]) -> Check:
    text = _text(result).lower()
    found = [t for t in terms if t.lower() in text]
    return Check("must_not_mention", not found, f"found {found}" if found else "")


def check_cites_gold(result: QAResult, gold: list[str]) -> Check:
    cited = {c for claim in result.claims for c in claim.chunk_ids}
    return Check("cites_gold", bool(cited & set(gold)), f"cited {sorted(cited)}, gold {gold}")


def check_no_claims(result: QAResult) -> Check:
    cites = CITATION.findall(result.answer)
    return Check("no_claims", not result.claims and not cites, f"{len(result.claims)} claims, citations {cites}")


def check_caveat(result: QAResult) -> Check:
    return Check("caveat_present", bool(result.caveats), "")


def check_numbers_grounded(result: QAResult, passages: list[str]) -> Check:
    """Every figure in the answer and claims appears in some passage (allowing unit rescaling and rounding)."""
    known = {n for text in passages for n in numbers(text, skip_trivial=False)}
    invented = [
        n
        for n in numbers(_text(result))
        if not any(abs(n - k * s) <= TOLERANCE * max(abs(n), 1e-9) for k in known for s in SCALES)
    ]
    return Check("numbers_grounded", not invented, f"not in passages: {invented}" if invented else "")


def score(result: QAResult, expect: dict, passages: list[str]) -> list[Check]:
    """Always checks citation validity and figures; the rest only when `expect` names them."""
    checks = [check_status(result, expect["status"]), check_citations_valid(result), check_numbers_grounded(result, passages)]
    if set(expect["status"]) <= set(GROUNDED):
        checks.append(check_claims_cited(result))
    if expect.get("tickers"):
        checks.append(check_tickers_covered(result, expect["tickers"]))
    if expect.get("must_mention"):
        checks.append(check_mentions(result, expect["must_mention"]))
    if expect.get("must_not_mention"):
        checks.append(check_not_mentions(result, expect["must_not_mention"]))
    if expect.get("gold_chunk_ids"):
        checks.append(check_cites_gold(result, expect["gold_chunk_ids"]))
    if expect.get("no_claims"):
        checks.append(check_no_claims(result))
    if expect.get("caveat"):
        checks.append(check_caveat(result))
    return checks
