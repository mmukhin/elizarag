import math

import pytest

from app.edgar import chunk_filing, load_filing
from app.edgar.vectorstore import Filters, GemmaEmbeddings, build_store, search
from tests.helpers import CORPUS, ollama_has, needs_corpus


@needs_corpus
@pytest.mark.slow
@pytest.mark.skipif(not ollama_has("embeddinggemma"), reason="Ollama with embeddinggemma not available")
def test_real_embeddinggemma_retrieval_on_filings():
    docs = []
    for name in ["AAPL_10K_2025-10-31_full.txt", "TSLA_10K_2026-01-29_full.txt"]:
        risky = [d for d in chunk_filing(load_filing(CORPUS / name)) if "risk" in d.page_content.lower()]
        docs += risky[:25]
    embeddings = GemmaEmbeddings()
    store = build_store(docs, embeddings)

    vector = embeddings.embed_query("what are the risks")
    assert len(vector) == 768 and math.isclose(sum(x * x for x in vector), 1.0, rel_tol=1e-3)

    results = search(store, "risks from tariffs and international trade", 3, Filters(tickers=["AAPL"]))
    assert len(results) == 3
    assert {d.metadata["ticker"] for d, _ in results} == {"AAPL"}
    assert results[0][1] >= results[-1][1]
