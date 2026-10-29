import json

import pytest

from app.edgar.query_filters import COMPANIES, Company, load_companies, parse_filters, resolve_filters
from app.edgar.vectorstore import Filters

YEARS = {"AAPL": [2022, 2023, 2024, 2025], "NVDA": [2022, 2023, 2024, 2025], "GE": [2015], "KO": [2022, 2023, 2024]}


@pytest.fixture(scope="module")
def companies():
    return load_companies()


@pytest.mark.parametrize(
    "question, expected",
    [
        # companies: names, aliases, tickers, ambiguous words
        ("What risks does Apple face?", {"tickers": ["AAPL"]}),
        ("How do I make an apple pie?", {}),
        ("Compare Coke and Pepsi", {"tickers": ["KO", "PEP"]}),
        ("Compare Tesla and Apple", {"tickers": ["TSLA", "AAPL"]}),  # question order, for balanced retrieval
        ("Coca-Cola's bottling partners", {"tickers": ["KO"]}),
        ("What is the target market for TSLA?", {"tickers": ["TSLA"]}),
        ("Target Corp store growth", {"tickers": ["TGT"]}),
        ("Facebook and Google ad revenue", {"tickers": ["META", "GOOG"]}),
        ("J&J talc litigation", {"tickers": ["JNJ"]}),
        ("AT&T and Verizon debt", {"tickers": ["T", "VZ"]}),
        ("T-Mobile competition", {}),  # T only as a cashtag or AT&T
        ("$V cross-border volume", {"tickers": ["V"]}),
        ("visa requirements for travel", {}),
        ("NVDA and AMD data center", {"tickers": ["NVDA", "AMD"]}),
        ("JPMorgan Chase and Bank of America deposits", {"tickers": ["JPM", "BAC"]}),
        ("McDonald's franchise fees", {"tickers": ["MCD"]}),
        ("artificial intelligence spending", {}),  # "Intel" inside a word
        # years
        ("Apple revenue in 2024", {"tickers": ["AAPL"], "fiscal_years": [2024]}),
        ("Apple FY2023 and FY24 margins", {"tickers": ["AAPL"], "fiscal_years": [2023, 2024]}),
        ("Apple results from 2022 to 2024", {"tickers": ["AAPL"], "fiscal_years": [2022, 2023, 2024]}),
        ("Apple results between 2023 and 2025", {"tickers": ["AAPL"], "fiscal_years": [2023, 2024, 2025]}),
        ("Apple results in 2022 and 2025", {"tickers": ["AAPL"], "fiscal_years": [2022, 2025]}),  # two years, not a range
        ("NVIDIA growth over the last two years", {"tickers": ["NVDA"], "fiscal_years": [2024, 2025]}),
        ("Apple's latest 10-K", {"tickers": ["AAPL"], "fiscal_years": [2025], "filing_types": ["10-K"]}),
        ("GE's most recent annual report", {"tickers": ["GE"], "fiscal_years": [2015], "filing_types": ["10-K"]}),
        ("latest results for Apple and Coca-Cola", {"tickers": ["AAPL", "KO"], "fiscal_years": [2024, 2025]}),
        ("Exxon's plan for net zero by 2050", {"tickers": ["XOM"]}),  # a target, not a filing year
        ("revenue of $2,025 million", {}),
        # quarters and filing types
        ("Apple's Q2 2024 10-Q", {"tickers": ["AAPL"], "fiscal_years": [2024], "filing_types": ["10-Q"], "quarters": [2]}),
        ("Microsoft 2024Q3 results", {"tickers": ["MSFT"], "fiscal_years": [2024], "quarters": [3]}),
        ("third-quarter margins at Nike", {"tickers": ["NKE"], "quarters": [3]}),
        ("Tesla quarterly deliveries", {"tickers": ["TSLA"], "filing_types": ["10-Q"]}),
        ("risk factors in 10-Ks and 10-Qs", {"filing_types": ["10-K", "10-Q"]}),
        # nothing to filter on
        ("What are the main risks for banks?", {}),
    ],
)
def test_parse_filters(companies, question, expected):
    assert parse_filters(question, companies, YEARS).as_dict() == expected


def test_relative_years_need_index_years(companies):
    assert parse_filters("Apple's latest results", companies).as_dict() == {"tickers": ["AAPL"]}


def test_load_companies_limits_to_indexed_tickers_and_tolerates_unknown():
    companies = load_companies(["AAPL", "ZZZZ"])
    assert [c.ticker for c in companies] == ["AAPL", "ZZZZ"]
    assert "Apple" in companies[0].ambiguous_aliases and companies[1] == Company("ZZZZ")
    assert parse_filters("ZZZZ outlook", companies).tickers == ("ZZZZ",)  # ticker still matches


def test_companies_file_covers_every_corpus_ticker():
    data = json.loads(COMPANIES.read_text())
    assert len(data) == 54 and all(entry["company"] for entry in data.values())
    assert all(entry["aliases"] or entry["ambiguous_aliases"] for entry in data.values())


# --- every company in the corpus ---------------------------------------------------

ALL_COMPANIES = load_companies()


def company_questions(company: Company) -> list[str]:
    names = [company.name, *company.aliases, *company.ambiguous_aliases]
    questions = [f"How has {n}'s revenue and growth outlook changed over the last two years?" for n in names]
    questions += [f"What risks does {n} describe in its latest 10-K?" for n in names]
    questions.append(f"{company.ticker} revenue trend" if len(company.ticker) > 1 else f"${company.ticker} revenue trend")
    return questions


@pytest.mark.parametrize("company", ALL_COMPANIES, ids=lambda c: c.ticker)
def test_every_company_name_alias_and_ticker_selects_only_that_company(company):
    wrong = {q: parse_filters(q, ALL_COMPANIES).tickers for q in company_questions(company)}
    wrong = {q: got for q, got in wrong.items() if got != (company.ticker,)}
    assert not wrong, wrong


def test_every_pair_of_companies_is_detected_in_question_order():
    name = {c.ticker: (c.aliases or c.ambiguous_aliases or (c.name,))[0] for c in ALL_COMPANIES}
    wrong = []
    for a in ALL_COMPANIES:
        for b in ALL_COMPANIES:
            if a.ticker != b.ticker:
                got = parse_filters(f"Compare {name[a.ticker]} and {name[b.ticker]} revenue", ALL_COMPANIES).tickers
                if got != (a.ticker, b.ticker):
                    wrong.append((name[a.ticker], name[b.ticker], got))
    assert not wrong, wrong[:10]


# --- explicit choices vs the question -------------------------------------------------


def test_resolve_filters_explicit_wins_per_field_and_reports_conflicts(companies):
    question = "How has NVIDIA's revenue changed over the last two years?"
    resolved = resolve_filters(question, companies, YEARS, Filters(tickers=["BA", "BAC"]))
    assert resolved.filters.as_dict() == {"tickers": ["BA", "BAC"], "fiscal_years": [2024, 2025]}  # years still from the question
    assert resolved.detected.tickers == ("NVDA",)
    assert resolved.overridden == {"tickers": (["NVDA"], ["BA", "BAC"])}


def test_resolve_filters_no_conflict_when_explicit_agrees_or_detection_is_off(companies):
    question = "Apple revenue in 2024"
    assert resolve_filters(question, companies, YEARS, Filters(tickers=["AAPL"])).overridden == {}
    off = resolve_filters(question, companies, YEARS, detect=False)
    assert not off.filters and not off.detected
