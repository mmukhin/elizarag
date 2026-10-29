"""Turn a question into metadata filters: companies (names, aliases, tickers), years, quarters, filing types.

Deterministic, no model. Company names and aliases come from data/companies.json (scripts/build_companies.py).
Years are matched against `fiscal_year` (year of the period end); quarters are calendar quarters of the period end.
"""

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from app.edgar.vectorstore import Filters

COMPANIES = Path(__file__).resolve().parents[2] / "data" / "companies.json"
COUNTS = {"two": 2, "three": 3, "four": 4, "five": 5, "2": 2, "3": 3, "4": 4, "5": 5}
ORDINALS = {"first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3, "fourth": 4, "4th": 4}

YEAR = re.compile(r"(?<![\w$,.])(?:fy\s?|fiscal\s+(?:year\s+)?)?(20\d{2})(?!\d)", re.IGNORECASE)
SHORT_FY = re.compile(r"\bfy\s?'?(\d{2})\b", re.IGNORECASE)
YEAR_RANGE = re.compile(
    r"(?:\bbetween\s+(?:fy\s?)?(20\d{2})\s+and\s+(?:fy\s?)?(20\d{2}))"
    r"|(?:(?<!\d)(20\d{2})\s*(?:-|–|to|through|until)\s*(?:fy\s?)?(20\d{2})(?!\d))",
    re.IGNORECASE,
)
LATEST = re.compile(r"\b(?:latest|most\s+recent|newest|current|last\s+year|past\s+year)\b", re.IGNORECASE)
LAST_N_YEARS = re.compile(r"\b(?:last|past|previous|prior)\s+(two|three|four|five|[2-5])\s+(?:fiscal\s+)?years\b", re.IGNORECASE)
QUARTER = re.compile(r"(?<![a-z])q([1-4])(?!\d)", re.IGNORECASE)
ORDINAL_QUARTER = re.compile(r"\b(first|second|third|fourth|1st|2nd|3rd|4th)[\s-]+(?:fiscal\s+)?quarter\b", re.IGNORECASE)
TEN_K = re.compile(r"\b10-?ks?\b|\bannual\s+(?:report|filing)s?\b", re.IGNORECASE)
TEN_Q = re.compile(r"\b10-?qs?\b|\bquarterly\b", re.IGNORECASE)


@dataclass(frozen=True)
class Company:
    ticker: str
    name: str = ""
    aliases: tuple[str, ...] = ()  # case-insensitive
    ambiguous_aliases: tuple[str, ...] = ()  # also ordinary words: match only with this capitalisation


def load_companies(tickers: Iterable[str] | None = None, path: Path = COMPANIES) -> list[Company]:
    """Companies from companies.json, limited to `tickers` if given (tickers missing from the file get no aliases)."""
    data = json.loads(path.read_text()) if path.exists() else {}
    wanted = sorted(set(tickers)) if tickers is not None else sorted(data)
    return [
        Company(
            ticker,
            data.get(ticker, {}).get("company", ""),
            tuple(data.get(ticker, {}).get("aliases", [])),
            tuple(data.get(ticker, {}).get("ambiguous_aliases", [])),
        )
        for ticker in wanted
    ]


def _bounded(text: str) -> str:
    return rf"(?<![\w&]){re.escape(text)}(?![\w&])"


def _first_mention(question: str, company: Company) -> int | None:
    """Offset of the first mention of `company` in `question`, or None."""
    ticker = re.escape(company.ticker)
    # bare tickers must be uppercase; one-letter tickers (T, V) only as a cashtag like $T
    ticker_pattern = rf"(?<![\w$&.-])\$?{ticker}(?![\w&-])" if len(company.ticker) > 1 else rf"\${ticker}(?![\w&-])"
    patterns = [(_bounded(n), re.IGNORECASE) for n in (company.name, *company.aliases) if n]
    patterns += [(_bounded(n), 0) for n in company.ambiguous_aliases] + [(ticker_pattern, 0)]
    starts = [m.start() for pattern, flags in patterns if (m := re.search(pattern, question, flags))]
    return min(starts, default=None)


def _explicit_years(question: str, max_year: int | None) -> set[int]:
    years = {int(y) for y in YEAR.findall(question)} | {2000 + int(y) for y in SHORT_FY.findall(question)}
    for match in YEAR_RANGE.finditer(question):
        start, end = (int(y) for y in match.groups() if y)
        if start <= end <= start + 10:
            years |= set(range(start, end + 1))
    if max_year is not None:  # "net zero by 2050" is a target, not a filing year
        years = {y for y in years if y <= max_year + 1}
    return years


def _relative_years(question: str, tickers: list[str], years_by_ticker: Mapping[str, Iterable[int]]) -> set[int]:
    if match := LAST_N_YEARS.search(question):
        n = COUNTS[match.group(1).lower()]
    elif LATEST.search(question):
        n = 1
    else:
        return set()
    pools = [sorted(years_by_ticker.get(t, ())) for t in tickers] or [sorted({y for ys in years_by_ticker.values() for y in ys})]
    return {y for pool in pools for y in pool[-n:]}


def parse_filters(
    question: str,
    companies: Iterable[Company],
    years_by_ticker: Mapping[str, Iterable[int]] | None = None,
) -> Filters:
    """Filters implied by `question`. `years_by_ticker` (years in the index) resolves "latest" / "last two years"."""
    years_by_ticker = years_by_ticker or {}
    found = {c.ticker: at for c in companies if (at := _first_mention(question, c)) is not None}
    tickers = sorted(found, key=found.get)  # in the order the question names them
    all_years = [y for ys in years_by_ticker.values() for y in ys]
    years = _explicit_years(question, max(all_years) if all_years else None)
    if not years:
        years = _relative_years(question, tickers, years_by_ticker)
    quarters = {int(q) for q in QUARTER.findall(question)} | {ORDINALS[o.lower()] for o in ORDINAL_QUARTER.findall(question)}
    filing_types = {"10-K"} if TEN_K.search(question) else set()
    if TEN_Q.search(question):
        filing_types.add("10-Q")
    return Filters(tickers, years, filing_types, quarters)
