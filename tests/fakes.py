"""Offline test doubles: deterministic embeddings and a small multi-company index."""

import hashlib
import json
import math
import re

import httpx
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.edgar.hybrid import Hit, HybridRetriever
from app.edgar.llm import ChatModel
from app.edgar.vectorstore import build_store


class BagOfWordsEmbeddings(Embeddings):
    """Hashed word counts, L2-normalised."""

    DIM = 256

    def _vec(self, text: str) -> list[float]:
        vec = [0.0] * self.DIM
        for word in re.findall(r"[a-z]+", text.lower()):
            vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.DIM] += 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class FixedEmbeddings(Embeddings):
    """Explicit text -> vector table, so a test controls the vector ranking exactly."""

    def __init__(self, table: dict[str, list[float]]):
        self.table = table

    def embed_documents(self, texts):
        return [self.table[t] for t in texts]

    def embed_query(self, text):
        return self.table[text]


class BackendDownEmbeddings(BagOfWordsEmbeddings):
    """Indexes fine, but queries fail like a stopped Ollama."""

    def embed_query(self, text):
        raise httpx.ConnectError("connection refused")


def chunk(text: str, ticker: str = "AAPL", year: int = 2025, filing_type: str = "10-K", index: int = 0) -> Document:
    return Document(
        page_content=text,
        metadata={
            "ticker": ticker,
            "company": f"{ticker} Corp",
            "fiscal_year": year,
            "filing_type": filing_type,
            "quarter": 2 if filing_type == "10-Q" else 4,
            "filing_date": f"{year}-12-01",
            "source_file": f"{ticker}_{filing_type}_{year}.txt",
            "chunk_index": index,
            "n_chunks": 10,
        },
    )


def sample_chunks() -> list[Document]:
    rows = [
        ("AAPL", 2025, "10-K", "tariffs on imports from China may raise component costs and hurt gross margin"),
        ("AAPL", 2025, "10-K", "intense competition in smartphones tablets and wearables could reduce market share"),
        ("AAPL", 2024, "10-K", "cybersecurity attacks on our systems could disrupt operations and harm reputation"),
        ("TSLA", 2025, "10-K", "battery cell supply shortages and lithium prices could delay vehicle production"),
        ("TSLA", 2025, "10-K", "regulatory credits revenue may decline as emissions rules change"),
        ("TSLA", 2024, "10-K", "tariffs and trade restrictions affect our global supply chain costs"),
        ("JPM", 2025, "10-K", "credit losses and interest rate changes affect net interest income"),
        ("JPM", 2025, "10-K", "capital requirements and stress testing regulation limit dividends and buybacks"),
        ("JPM", 2025, "10-Q", "net interest income was 23.5 billion for the quarter"),
        ("NVDA", 2025, "10-K", "data center revenue grew strongly driven by demand for accelerated computing"),
        ("NVDA", 2024, "10-K", "export controls on advanced chips to China may reduce our revenue"),
        ("NVDA", 2024, "10-K", "gaming revenue was flat while data center compute demand rose"),
    ]
    counters: dict[str, int] = {}
    docs = []
    for ticker, year, filing_type, text in rows:
        key = f"{ticker}{year}{filing_type}"
        docs.append(chunk(text, ticker, year, filing_type, counters.get(key, 0)))
        counters[key] = counters.get(key, 0) + 1
    return docs


def fake_retriever(embeddings: Embeddings | None = None) -> HybridRetriever:
    return HybridRetriever(build_store(sample_chunks(), embeddings or BagOfWordsEmbeddings()))


class ScriptedChat(ChatModel):
    """Returns queued responses (JSON text or schema instances) and records every call."""

    model = "scripted"

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def _complete(self, system, user, schema):
        self.calls.append({"system": system, "user": user, "schema": schema})
        self._count(1200, 80)
        return self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]


def answer_json(**overrides) -> str:
    body = {
        "status": "answered",
        "answer": "Apple cites tariffs on imports from China [1].",
        "claims": [{"text": "Tariffs raise component costs.", "passages": [1]}],
        "caveats": [],
    } | overrides
    return json.dumps(body)


def hits_for(*chunks: Document) -> list[Hit]:
    return [Hit(c, 1.0 / (i + 1)) for i, c in enumerate(chunks)]
