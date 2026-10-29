import json
import re
from dataclasses import asdict

import pytest

from app.edgar import chunk_filing, clean_body, load_filing, parse_header
from tests.helpers import CORPUS, FIXTURES, needs_corpus

pytestmark = needs_corpus

EXPECTED = json.loads((FIXTURES / "expected.json").read_text())
CHUNK_TOKENS = 800
TOC_ROW = re.compile(r"Item\s+1A\.?\s*\|\s*Risk Factors\s*\|", re.IGNORECASE)


@pytest.fixture(scope="module", params=sorted(EXPECTED))
def case(request):
    name = request.param
    filing = load_filing(CORPUS / name)
    return name, EXPECTED[name], filing, chunk_filing(filing, chunk_tokens=CHUNK_TOKENS)


def test_header_metadata(case):
    _, expected, filing, _ = case
    actual = asdict(filing.meta)
    assert {k: actual[k] for k in expected["meta"]} == expected["meta"]


def test_metadata_on_every_chunk(case):
    _, expected, _, docs = case
    for doc in docs:
        assert {k: doc.metadata[k] for k in expected["meta"]} == expected["meta"]
    assert [d.metadata["chunk_index"] for d in docs] == list(range(len(docs)))
    assert {d.metadata["n_chunks"] for d in docs} == {len(docs)}


def test_noise_removed(case):
    _, _, filing, docs = case
    assert "fasb.org" not in filing.body and "xbrli:" not in filing.body
    assert not any(TOC_ROW.search(d.page_content) for d in docs)
    assert "\xa0" not in filing.body


def test_chunk_sizes(case):
    _, expected, _, docs = case
    assert all(0 < d.metadata["token_count"] <= CHUNK_TOKENS for d in docs)
    assert all(d.page_content.strip() for d in docs)
    low, high = expected["chunks"]
    assert low <= len(docs) <= high


def test_offsets_and_coverage(case):
    _, _, filing, docs = case
    for doc in docs:
        start = doc.metadata["start_index"]
        assert filing.body[start : start + len(doc.page_content)] == doc.page_content
    # consecutive chunks overlap, or are separated only by whitespace the splitter stripped
    for prev, nxt in zip(docs, docs[1:]):
        prev_end = prev.metadata["start_index"] + len(prev.page_content)
        assert not filing.body[prev_end : nxt.metadata["start_index"]].strip()
    assert docs[0].metadata["start_index"] == 0
    last = docs[-1]
    assert last.metadata["start_index"] + len(last.page_content) == len(filing.body)


def test_known_content_is_chunked(case):
    _, expected, _, docs = case
    for snippet in expected["snippets"]:
        assert any(snippet in d.page_content for d in docs), snippet


def test_deterministic(case):
    _, _, filing, docs = case
    again = chunk_filing(filing, chunk_tokens=CHUNK_TOKENS)
    assert [(d.page_content, d.metadata) for d in again] == [(d.page_content, d.metadata) for d in docs]


def test_parse_header():
    text = "Company: Acme\nTicker: ACM\nURL: https://x/y.htm\n==========\nbody here\n"
    fields, body = parse_header(text)
    assert fields == {"company": "Acme", "ticker": "ACM", "url": "https://x/y.htm"}
    assert body == "body here\n"


def test_parse_header_missing_rule():
    with pytest.raises(ValueError, match="header rule"):
        parse_header("Company: Acme\nno rule here")


def test_clean_body_drops_xbrl_line():
    soup = "x" * 600 + " http://fasb.org/us-gaap/2025#Foo"
    assert clean_body(f"before\n{soup}\nafter") == "before\n\nafter"


def test_clean_body_keeps_short_xbrl_mention():
    assert "fasb.org/us-gaap" in clean_body("See http://fasb.org/us-gaap/2025 for details")


def test_clean_body_drops_toc_run_only_when_long():
    toc = "\n".join(f"Item {n}. | Heading | {n}" for n in range(1, 13))
    assert clean_body(f"intro\nPart I\n{toc}\nPart II\nbody text") == "intro\n\nbody text"
    short = "Item 1. | Business | 1\nItem 2. | Properties | 2"
    assert clean_body(short) == short


def test_clean_body_normalises_nbsp():
    assert clean_body("Item\xa01A.\xa0\xa0Risk") == "Item 1A.  Risk"


@pytest.mark.parametrize(
    "name, fiscal_year, quarter",
    [
        ("AAPL_10Q_2025Q1_2025-05-02_full.txt", 2025, 1),  # Quarter: header (Apple's fiscal Q2; calendar Q1)
        ("AAPL_10K_2022Q3_2022-10-28_full.txt", 2022, 3),  # 10-K with a Quarter: header
        ("MS_10K_2026-02-19_full.txt", 2025, 4),  # no header: from the period-end month
        ("GE_10K_2015-02-27_full.txt", 2015, 0),  # no period end at all
    ],
)
def test_quarter_is_calendar_quarter_of_period_end(name, fiscal_year, quarter):
    meta = load_filing(CORPUS / name).meta
    assert (meta.fiscal_year, meta.quarter) == (fiscal_year, quarter)
