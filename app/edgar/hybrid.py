"""Hybrid retrieval: BM25 keyword ranking + vector ranking, fused with reciprocal rank fusion."""

import re
from dataclasses import dataclass, field

import numpy as np
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from app.edgar.vectorstore import QUERY_PREFIX, ChromaStore, Filters, GemmaEmbeddings, make_filter, make_where, search

WORD = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")
RRF_K = 60  # standard RRF damping constant
MODES = ("hybrid", "vector", "keyword")


def tokenize(text: str) -> list[str]:
    return WORD.findall(text.lower())


@dataclass(frozen=True)
class Hit:
    doc: Document
    score: float  # fused RRF score (hybrid) or the raw score of the single ranker used
    vector_rank: int | None = None  # 1-based, None if the vector search did not return it
    keyword_rank: int | None = None
    vector_score: float | None = None  # cosine similarity from Chroma
    keyword_score: float | None = None  # raw BM25 score


@dataclass
class SearchTrace:
    """One ranked search: the requests sent to Chroma and BM25 and every candidate, best first."""

    query: str
    mode: str
    k: int
    pool: int
    chroma_request: dict | None  # None when the vector side did not run (keyword mode)
    bm25_request: dict | None  # None when BM25 did not run (vector mode)
    candidates: list[Hit] = field(default_factory=list)  # the first k were returned, the rest were not used

    @property
    def hits(self) -> list[Hit]:
        return self.candidates[: self.k]

    @property
    def unused(self) -> list[Hit]:
        return self.candidates[self.k :]


def _key(doc: Document) -> tuple[str, int]:
    return doc.metadata["source_file"], doc.metadata["chunk_index"]


class HybridRetriever:
    def __init__(self, store: ChromaStore):
        self.store = store
        self.docs = store.documents()  # BM25 needs the whole corpus in memory
        self.bm25 = BM25Okapi([tokenize(d.page_content) for d in self.docs]) if self.docs else None

    @property
    def tickers(self) -> list[str]:
        return sorted({d.metadata["ticker"] for d in self.docs})

    @property
    def fiscal_years(self) -> list[int]:
        return sorted({d.metadata["fiscal_year"] for d in self.docs})

    @property
    def years_by_ticker(self) -> dict[str, list[int]]:
        years: dict[str, set[int]] = {}
        for d in self.docs:
            years.setdefault(d.metadata["ticker"], set()).add(d.metadata["fiscal_year"])
        return {ticker: sorted(ys) for ticker, ys in sorted(years.items())}

    @property
    def filing_types(self) -> list[str]:
        return sorted({d.metadata["filing_type"] for d in self.docs})

    @property
    def quarters(self) -> list[int]:
        """Calendar quarters present (chunks indexed before quarters were stored have none)."""
        return sorted({d.metadata["quarter"] for d in self.docs if d.metadata.get("quarter")})

    def _keyword(self, query: str, pool: int, keep) -> list[tuple[Document, float]]:
        tokens = tokenize(query)
        if self.bm25 is None or not tokens:
            return []
        scores = self.bm25.get_scores(tokens)  # idf over the whole corpus, then filter
        order = np.argsort(-scores, kind="stable")
        hits = []
        for i in order:
            if scores[i] <= 0:
                break
            if keep(self.docs[i]):
                hits.append((self.docs[i], float(scores[i])))
                if len(hits) == pool:
                    break
        return hits

    def search(
        self,
        query: str,
        k: int = 5,
        mode: str = "hybrid",
        filters: Filters | None = None,
        per_ticker: bool = False,
        pool: int = 50,
        trace: list[SearchTrace] | None = None,
    ) -> list[Hit]:
        """Top-k hits matching `filters`. With `per_ticker`, retrieve k for each filtered ticker (balanced comparison).

        Pass a list as `trace` to have one `SearchTrace` appended per ranked search that runs.
        """
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r} (expected one of {MODES})")
        if not query.strip():
            return []
        filters = filters or Filters()
        if per_ticker and filters.tickers:
            hits = []
            for ticker in filters.tickers:
                hits += self.search(query, k, mode, filters.only(ticker), pool=pool, trace=trace)
            return hits

        use_vector, use_keyword = mode in ("hybrid", "vector"), mode in ("hybrid", "keyword")
        vector = search(self.store, query, pool, filters) if use_vector else []
        keyword = self._keyword(query, pool, make_filter(filters)) if use_keyword else []

        if mode == "vector":
            candidates = [Hit(d, s, vector_rank=i, vector_score=s) for i, (d, s) in enumerate(vector, 1)]
        elif mode == "keyword":
            candidates = [Hit(d, 0.0, keyword_rank=i, keyword_score=s) for i, (d, s) in enumerate(keyword, 1)]
        else:
            vector_rank = {_key(d): (i, s) for i, (d, s) in enumerate(vector, 1)}
            keyword_rank = {_key(d): (i, s) for i, (d, s) in enumerate(keyword, 1)}
            by_key = {_key(d): d for d, _ in vector} | {_key(d): d for d, _ in keyword}
            fused = {
                key: sum(1 / (RRF_K + ranks[key][0]) for ranks in (vector_rank, keyword_rank) if key in ranks)
                for key in by_key
            }
            candidates = [
                Hit(by_key[key], fused[key], *_rank_score(vector_rank, keyword_rank, key))
                for key in sorted(fused, key=lambda key: (-fused[key], key))
            ]

        if trace is not None:
            trace.append(
                SearchTrace(
                    query,
                    mode,
                    k,
                    pool,
                    self._chroma_request(query, pool, filters) if use_vector else None,
                    self._bm25_request(query, pool, filters) if use_keyword else None,
                    candidates,
                )
            )
        return candidates[:k]

    def _chroma_request(self, query: str, pool: int, filters: Filters) -> dict:
        embeddings = self.store.embeddings
        prefix = QUERY_PREFIX if isinstance(embeddings, GemmaEmbeddings) else ""
        return {
            "collection": self.store.collection.name,
            "query_embeddings": f"embed({prefix + query!r})",
            "embedding_model": getattr(embeddings, "model", type(embeddings).__name__),
            "n_results": pool,
            "where": make_where(filters),
            "include": ["documents", "metadatas", "distances"],
        }

    def _bm25_request(self, query: str, pool: int, filters: Filters) -> dict:
        return {
            "tokens": tokenize(query),
            "corpus_chunks": len(self.docs),
            "filter_after_scoring": filters.as_dict() or "any",
            "max_results": pool,
        }


def _rank_score(vector_rank: dict, keyword_rank: dict, key) -> tuple:
    """(vector_rank, keyword_rank, vector_score, keyword_score) for one fused candidate."""
    v, kw = vector_rank.get(key, (None, None)), keyword_rank.get(key, (None, None))
    return v[0], kw[0], v[1], kw[1]
