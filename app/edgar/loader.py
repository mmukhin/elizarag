"""Load an EDGAR corpus text file: parse the metadata header and clean the body."""

import re
from dataclasses import dataclass, field
from pathlib import Path

HEADER_RULE = re.compile(r"^={10,}\s*$", re.MULTILINE)
URL_PERIOD = re.compile(r"[-_](\d{4})(\d{2})(\d{2})(?:x10k)?\.htm", re.IGNORECASE)
TOC_LINE = re.compile(r"^\s*(?:item\s+\d{1,2}[a-c]?\b.*\|.*|part\s+[ivx]+\b.{0,5})\s*$", re.IGNORECASE)
MIN_TOC_RUN = 10
XBRL_MARKERS = ("fasb.org/us-gaap", "xbrli:", "xbrl.org")
XBRL_MIN_LINE = 500


@dataclass(frozen=True)
class FilingMeta:
    company: str
    ticker: str
    cik: str
    filing_type: str
    filing_date: str
    fiscal_year: int
    source_file: str
    url: str = ""
    period_end: str = ""
    quarter: int = 0  # calendar quarter (1-4) of period_end, 0 if unknown


@dataclass(frozen=True)
class Filing:
    meta: FilingMeta
    body: str = field(repr=False)


def parse_header(text: str) -> tuple[dict[str, str], str]:
    """Split `Key: value` header lines from the body at the first `====` rule."""
    rule = HEADER_RULE.search(text)
    if rule is None:
        raise ValueError("no '====' header rule found")
    fields = {}
    for line in text[: rule.start()].splitlines():
        key, sep, value = line.partition(":")
        if sep:
            fields[key.strip().lower()] = value.strip()
    return fields, text[rule.end() :].lstrip("\n")


def _period_end(fields: dict[str, str]) -> str:
    if fields.get("report period"):
        return fields["report period"]
    match = URL_PERIOD.search(fields.get("url", ""))
    return "-".join(match.groups()) if match else ""


def _quarter(fields: dict[str, str], period_end: str) -> int:
    """From the `Quarter: 2025Q1` header, else from the period-end month (the header uses the same rule)."""
    header = re.fullmatch(r"\d{4}Q([1-4])", fields.get("quarter", ""))
    if header:
        return int(header.group(1))
    return (int(period_end[5:7]) - 1) // 3 + 1 if period_end else 0


def clean_body(body: str) -> str:
    """Drop XBRL tag soup and the pipe-table TOC; NBSP -> space (1:1, offsets stable)."""
    lines = [
        "" if len(line) > XBRL_MIN_LINE and any(m in line for m in XBRL_MARKERS) else line
        for line in body.replace("\xa0", " ").split("\n")
    ]
    toc = [bool(TOC_LINE.match(line)) for line in lines]
    i = 0
    while i < len(lines):
        if not toc[i]:
            i += 1
            continue
        j = i
        while j < len(lines) and toc[j]:
            j += 1
        if j - i >= MIN_TOC_RUN:
            lines[i:j] = [""] * (j - i)
        i = j
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def load_filing(path: str | Path) -> Filing:
    path = Path(path)
    fields, body = parse_header(path.read_text(encoding="utf-8"))
    period_end = _period_end(fields)
    filing_date = fields.get("filing date", "")
    year = period_end[:4] or filing_date[:4]
    meta = FilingMeta(
        company=fields.get("company", ""),
        ticker=fields.get("ticker", ""),
        cik=fields.get("cik", ""),
        filing_type=fields.get("filing type", "").split(" ")[0],
        filing_date=filing_date,
        fiscal_year=int(year) if year else 0,
        source_file=path.name,
        url=fields.get("url", ""),
        period_end=period_end,
        quarter=_quarter(fields, period_end),
    )
    return Filing(meta=meta, body=clean_body(body))
