import pytest

from app.edgar.hybrid import HybridRetriever, tokenize
from app.edgar.vectorstore import Filters, build_store
from tests.fakes import FixedEmbeddings, chunk, fake_retriever


@pytest.fixture(scope="module")
def retriever():
    return fake_retriever()


def controlled_retriever():
    """Vector order for query 'zeta' is A > C > B > filler; keyword order is C > B (A has no 'zeta')."""
    docs = [
        chunk("alpha beta gamma", index=0),  # A: closest vector, no keyword match
        chunk("zeta delta", index=1),  # B: weaker keyword match, farther vector
        chunk("zeta zeta eta", index=2),  # C: vector #2 and best keyword match
        chunk("theta iota kappa", index=3),
        chunk("lambda mu nu", index=4),
        chunk("xi omicron pi", index=5),
    ]
    table = {
        "zeta": [1.0, 0.0],
        docs[0].page_content: [1.0, 0.0],
        docs[2].page_content: [0.9, 0.436],
        docs[1].page_content: [0.5, 0.866],
        docs[3].page_content: [0.0, 1.0],
        docs[4].page_content: [-0.5, 0.866],
        docs[5].page_content: [-1.0, 0.0],
    }
    return HybridRetriever(build_store(docs, FixedEmbeddings(table)))


def texts(hits):
    return [h.doc.page_content for h in hits]


def test_tokenize_keeps_numbers_and_decimals():
    assert tokenize("Revenue was $96.9 billion, up 1,234 units; Q3") == [
        "revenue", "was", "96.9", "billion", "up", "1,234", "units", "q3"
    ]


def test_metadata_listings(retriever):
    assert retriever.tickers == ["AAPL", "JPM", "NVDA", "TSLA"]
    assert retriever.fiscal_years == [2024, 2025]
    assert retriever.filing_types == ["10-K", "10-Q"]
    assert len(retriever.docs) == 12


def test_vector_mode_follows_vector_order():
    hits = controlled_retriever().search("zeta", k=3, mode="vector")
    assert texts(hits) == ["alpha beta gamma", "zeta zeta eta", "zeta delta"]
    assert [h.vector_rank for h in hits] == [1, 2, 3]
    assert all(h.keyword_rank is None for h in hits)


def test_keyword_mode_returns_only_term_matches_best_first():
    hits = controlled_retriever().search("zeta", k=5, mode="keyword")
    assert texts(hits) == ["zeta zeta eta", "zeta delta"]
    assert [h.keyword_rank for h in hits] == [1, 2]


def test_hybrid_ranks_doc_found_by_both_rankers_first():
    hits = controlled_retriever().search("zeta", k=3, mode="hybrid")
    assert hits[0].doc.page_content == "zeta zeta eta"  # vector #2 + keyword #1 beats vector-only #1
    assert (hits[0].vector_rank, hits[0].keyword_rank) == (2, 1)
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_hybrid_surfaces_keyword_only_hit_outside_vector_pool():
    # pool=1: vector search only returns A, keyword search only returns C
    hits = controlled_retriever().search("zeta", k=2, mode="hybrid", pool=1)
    assert set(texts(hits)) == {"alpha beta gamma", "zeta zeta eta"}
    by_text = {h.doc.page_content: h for h in hits}
    assert by_text["alpha beta gamma"].keyword_rank is None
    assert by_text["zeta zeta eta"].vector_rank is None


def test_keyword_finds_exact_figure(retriever):
    hits = retriever.search("23.5 billion", k=1, mode="keyword")
    assert hits[0].doc.metadata["ticker"] == "JPM" and "23.5 billion" in hits[0].doc.page_content


@pytest.mark.parametrize("mode", ["hybrid", "vector", "keyword"])
def test_filters_apply_in_every_mode(retriever, mode):
    hits = retriever.search("tariffs revenue", k=10, mode=mode, filters=Filters(["tsla", "aapl"], [2025]))
    assert hits
    assert all(h.doc.metadata["ticker"] in {"TSLA", "AAPL"} and h.doc.metadata["fiscal_year"] == 2025 for h in hits)


def test_quarter_filter_and_listings(retriever):
    hits = retriever.search("net interest income", k=10, filters=Filters(quarters=[2]))
    assert [(h.doc.metadata["filing_type"], h.doc.metadata["quarter"]) for h in hits] == [("10-Q", 2)]
    assert retriever.quarters == [2, 4]
    assert retriever.years_by_ticker["NVDA"] == [2024, 2025] and set(retriever.years_by_ticker) == {"AAPL", "JPM", "NVDA", "TSLA"}


def test_filing_type_filter(retriever):
    hits = retriever.search("net interest income", k=10, filters=Filters(filing_types=["10-Q"]))
    assert [h.doc.metadata["filing_type"] for h in hits] == ["10-Q"]


def test_per_ticker_returns_k_for_each_company(retriever):
    hits = retriever.search("risk", k=2, mode="vector", filters=Filters(["AAPL", "TSLA", "JPM"]), per_ticker=True)
    tickers = [h.doc.metadata["ticker"] for h in hits]
    assert tickers == ["AAPL", "AAPL", "TSLA", "TSLA", "JPM", "JPM"]


def test_without_per_ticker_one_company_can_dominate(retriever):
    hits = retriever.search(
        "competition smartphones tablets wearables market share tariffs imports China component costs gross margin",
        k=2,
        mode="vector",
        filters=Filters(["AAPL", "TSLA"]),
    )
    assert {h.doc.metadata["ticker"] for h in hits} == {"AAPL"}


def test_no_keyword_overlap_gives_empty_keyword_and_still_hybrid_via_vector(retriever):
    assert retriever.search("qqqq", mode="keyword") == []
    assert retriever.search("qqqq zzzz", mode="hybrid", k=3)  # vector side still answers


@pytest.mark.parametrize("query", ["", "   "])
def test_blank_query_returns_nothing(retriever, query):
    assert retriever.search(query) == []


def test_unknown_mode_raises(retriever):
    with pytest.raises(ValueError, match="unknown mode"):
        retriever.search("x", mode="fuzzy")


def test_k_limits_results(retriever):
    assert len(retriever.search("revenue", k=3)) == 3


def test_empty_store():
    empty = HybridRetriever(build_store([], FixedEmbeddings({})))
    assert empty.docs == [] and empty.search("anything", mode="keyword") == []


def test_trace_records_requests_and_all_candidates():
    trace = []
    hits = controlled_retriever().search("zeta", k=2, mode="hybrid", filters=Filters(fiscal_years=[2025]), trace=trace)
    (t,) = trace
    assert t.hits == hits and len(t.candidates) == 6 and len(t.unused) == 4  # 6 vector candidates, 2 also keyword
    assert t.chroma_request["n_results"] == 50 and t.chroma_request["where"] == {"fiscal_year": {"$in": [2025]}}
    assert t.bm25_request["tokens"] == ["zeta"] and t.bm25_request["filter_after_scoring"] == {"fiscal_years": [2025]}
    top = hits[0]
    assert (top.vector_rank, top.keyword_rank) == (2, 1) and top.vector_score > 0.8 and top.keyword_score > 0


def test_trace_per_ticker_and_single_ranker_modes(retriever):
    trace = []
    retriever.search("risk", k=1, mode="keyword", filters=Filters(["AAPL", "TSLA"]), per_ticker=True, trace=trace)
    assert len(trace) == 2 and all(t.chroma_request is None and t.bm25_request for t in trace)
    trace = []
    retriever.search("risk", k=1, mode="vector", trace=trace)
    assert trace[0].bm25_request is None and trace[0].chroma_request["where"] is None
