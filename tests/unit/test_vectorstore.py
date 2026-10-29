import math

import chromadb
import httpx
import pytest
from langchain_core.documents import Document

from app.edgar.vectorstore import (
    DOCUMENT_PREFIX,
    QUERY_PREFIX,
    Filters,
    GemmaEmbeddings,
    build_store,
    get_embeddings,
    load_store,
    make_filter,
    make_where,
    search,
)
from tests.fakes import BackendDownEmbeddings, BagOfWordsEmbeddings

def make_docs():
    rows = [
        ("AAPL", 2025, "10-K", "tariffs on imports from China may raise component costs and hurt gross margin"),
        ("AAPL", 2024, "10-K", "the company faces intense competition in smartphones and wearables"),
        ("TSLA", 2025, "10-K", "battery cell supply shortages and lithium prices could delay vehicle production"),
        ("TSLA", 2025, "10-Q", "vehicle deliveries and battery production increased this quarter"),
        ("JPM", 2025, "10-K", "credit losses and interest rate changes affect net interest income"),
    ]
    return [
        Document(page_content=text, metadata={"ticker": t, "fiscal_year": y, "filing_type": ft})
        for t, y, ft, text in rows
    ]


@pytest.fixture(scope="module")
def store():
    return build_store(make_docs(), BagOfWordsEmbeddings())


def test_search_ranks_relevant_chunk_first(store):
    doc, _ = search(store, "battery supply and lithium", k=1)[0]
    assert doc.metadata["ticker"] == "TSLA" and "lithium" in doc.page_content


def test_search_filters_by_ticker(store):
    results = search(store, "risk", 10, Filters(tickers=["aapl", "jpm"]))
    assert {d.metadata["ticker"] for d, _ in results} == {"AAPL", "JPM"}


def test_search_filters_by_year_and_filing_type(store):
    results = search(store, "battery", 10, Filters(fiscal_years=[2025], filing_types=["10-K"]))
    assert all(d.metadata["fiscal_year"] == 2025 and d.metadata["filing_type"] == "10-K" for d, _ in results)
    assert len(results) == 3


def test_search_no_match_returns_empty(store):
    assert search(store, "anything", filters=Filters(tickers=["NVDA"])) == []


def test_scores_are_cosine_similarity_best_first(store):
    results = search(store, "battery cell supply shortages and lithium prices could delay vehicle production", k=5)
    scores = [s for _, s in results]
    assert math.isclose(scores[0], 1.0, abs_tol=1e-5) and scores == sorted(scores, reverse=True)


def test_build_then_load_collection_by_name(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    query = "interest rate credit losses"
    before = [(d.page_content, round(s, 6)) for d, s in search(build_store(make_docs(), BagOfWordsEmbeddings(), "filings", client), query, k=3)]
    rebuilt = build_store(make_docs(), BagOfWordsEmbeddings(), "filings", client)  # replaces, does not append
    loaded = load_store("filings", BagOfWordsEmbeddings(), client)
    assert loaded.collection.count() == rebuilt.collection.count() == 5
    after = [(d.page_content, round(s, 6)) for d, s in search(loaded, query, k=3)]
    assert before[0] == after[0] and [s for _, s in before] == [s for _, s in after]  # tie order is unspecified
    assert [d.page_content for d in loaded.documents(page_size=2)] == [d.page_content for d in make_docs()]


def test_add_documents_with_precomputed_vectors_skips_embedding():
    store = build_store([], BackendDownEmbeddings())
    store.add_documents(make_docs()[:2], vectors=[[1.0, 0.0], [0.0, 1.0]])
    assert store.collection.count() == 2


def test_make_where():
    assert make_where() is None and make_where(Filters()) is None
    assert make_where(Filters(tickers=["aapl"])) == {"ticker": {"$in": ["AAPL"]}}
    assert make_where(Filters(["tsla", "aapl"], [2025, 2024], ["10-K"], [4])) == {
        "$and": [
            {"ticker": {"$in": ["AAPL", "TSLA"]}},
            {"fiscal_year": {"$in": [2024, 2025]}},
            {"filing_type": {"$in": ["10-K"]}},
            {"quarter": {"$in": [4]}},
        ]
    }


def test_filters_normalise_and_report_only_restricting_fields():
    filters = Filters(["tsla", "aapl", "TSLA"], [2025, 2024, 2025], None, [2])
    assert filters.tickers == ("TSLA", "AAPL")  # order kept, for balanced search
    assert filters.as_dict() == {"tickers": ["TSLA", "AAPL"], "fiscal_years": [2024, 2025], "quarters": [2]}
    assert filters.only("AAPL").tickers == ("AAPL",) and filters.only("AAPL").quarters == (2,)
    assert not Filters() and Filters(quarters=[1])


def test_make_filter_matches_make_where():
    docs = make_docs()
    for filters in [Filters(), Filters(tickers=["aapl"]), Filters(fiscal_years=[2025], filing_types=["10-Q"])]:
        keep = make_filter(filters)
        kept = [d for d in docs if keep(d)]
        where = make_where(filters)
        assert kept == [d for d in docs if where is None or _matches(where, d.metadata)]


def _matches(where: dict, meta: dict) -> bool:
    if "$and" in where:
        return all(_matches(clause, meta) for clause in where["$and"])
    ((key, cond),) = where.items()
    return meta.get(key) in cond["$in"]


def test_get_embeddings_provider_selection(monkeypatch):
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    assert isinstance(get_embeddings(), GemmaEmbeddings)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    assert type(get_embeddings()).__name__ == "OpenAIEmbeddings"
    with pytest.raises(ValueError, match="unknown embedding provider"):
        get_embeddings("nope")


def test_gemma_embeddings_adds_task_prefixes_and_batches():
    sent = []

    def handler(request):
        import json

        inputs = json.loads(request.content)["input"]
        sent.append(inputs)
        return httpx.Response(200, json={"embeddings": [[1.0, 0.0]] * len(inputs)})

    emb = GemmaEmbeddings(batch_size=2)
    emb.client = httpx.Client(base_url="http://x", transport=httpx.MockTransport(handler))
    assert len(emb.embed_documents(["a", "b", "c"])) == 3
    emb.embed_query("q")
    assert sent == [[DOCUMENT_PREFIX + "a", DOCUMENT_PREFIX + "b"], [DOCUMENT_PREFIX + "c"], [QUERY_PREFIX + "q"]]

