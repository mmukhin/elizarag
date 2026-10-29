"""Embeddings (local EmbeddingGemma on Ollama by default) + a ChromaDB vector store."""

import os
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from urllib.parse import urlparse

import chromadb
import httpx
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

OLLAMA_URL = "http://localhost:11434"
CHROMA_URL = "http://localhost:8000"
# EmbeddingGemma expects task prefixes; Ollama does not add them for us.
QUERY_PREFIX = "task: search result | query: "
DOCUMENT_PREFIX = "title: none | text: "


class GemmaEmbeddings(Embeddings):
    """EmbeddingGemma served by a local Ollama (`ollama pull embeddinggemma`)."""

    def __init__(
        self,
        model: str = "embeddinggemma",
        base_url: str = OLLAMA_URL,
        batch_size: int = 32,
        timeout: float = 300.0,
    ):
        self.model = model
        self.batch_size = batch_size
        self.client = httpx.Client(base_url=base_url, timeout=timeout)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        response = self.client.post("/api/embed", json={"model": self.model, "input": texts})
        response.raise_for_status()
        return response.json()["embeddings"]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            vectors.extend(self._embed([DOCUMENT_PREFIX + t for t in batch]))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed([QUERY_PREFIX + text])[0]


def get_embeddings(provider: str | None = None) -> Embeddings:
    """Pick the embedding backend: `ollama` (default) or `openai`.

    Set via argument or EMBEDDING_PROVIDER; EMBEDDING_MODEL overrides the model name.
    Vectors from different providers are not comparable, so rebuild the index when switching.
    """
    provider = (provider or os.environ.get("EMBEDDING_PROVIDER", "ollama")).lower()
    model = os.environ.get("EMBEDDING_MODEL")
    if provider == "ollama":
        return GemmaEmbeddings(model=model or "embeddinggemma", base_url=os.environ.get("OLLAMA_URL", OLLAMA_URL))
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings  # needs OPENAI_API_KEY

        return OpenAIEmbeddings(model=model or "text-embedding-3-small")
    raise ValueError(f"unknown embedding provider {provider!r} (expected 'ollama' or 'openai')")


def get_client(url: str | None = None) -> ClientAPI:
    """HTTP client for the local Chroma server (`chroma run --path data/chroma`); URL from CHROMA_URL."""
    parsed = urlparse(url or os.environ.get("CHROMA_URL", CHROMA_URL))
    return chromadb.HttpClient(host=parsed.hostname, port=parsed.port or 8000, ssl=parsed.scheme == "https")


def chunk_id(meta: dict) -> str:
    """Stable chunk id, `source_file#chunk_index`; also the Chroma record id."""
    return f"{meta['source_file']}#{meta['chunk_index']}"


def _chunk_id(doc: Document) -> str:
    meta = doc.metadata
    if "source_file" in meta and "chunk_index" in meta:
        return chunk_id(meta)
    return doc.id or uuid.uuid4().hex


class ChromaStore:
    """A Chroma collection plus the embedder that produced its vectors (Chroma's own embedder is unused)."""

    def __init__(self, collection: Collection, embeddings: Embeddings):
        self.collection = collection
        self.embeddings = embeddings

    def add_documents(self, docs: list[Document], vectors: list[list[float]] | None = None) -> None:
        if not docs:
            return
        self.collection.add(
            ids=[_chunk_id(d) for d in docs],
            embeddings=vectors or self.embeddings.embed_documents([d.page_content for d in docs]),
            documents=[d.page_content for d in docs],
            metadatas=[d.metadata for d in docs],
        )

    def documents(self, page_size: int = 1000) -> list[Document]:
        """Every stored chunk, in insertion order."""
        docs = []
        while True:
            page = self.collection.get(include=["documents", "metadatas"], limit=page_size, offset=len(docs))
            docs += [Document(page_content=text, metadata=meta) for text, meta in zip(page["documents"], page["metadatas"])]
            if len(page["ids"]) < page_size:
                return docs

    def similarity_search_with_score(self, query: str, k: int, where: dict | None = None) -> list[tuple[Document, float]]:
        """Top-k by cosine similarity (1 - Chroma's cosine distance), higher is better."""
        result = self.collection.query(
            query_embeddings=[self.embeddings.embed_query(query)],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        return [
            (Document(page_content=text, metadata=meta), 1.0 - dist)
            for text, meta, dist in zip(result["documents"][0], result["metadatas"][0], result["distances"][0])
        ]


def build_store(
    docs: list[Document],
    embeddings: Embeddings,
    collection: str | None = None,
    client: ClientAPI | None = None,
    batch_size: int = 128,
    on_batch: Callable[[int, int], None] | None = None,
) -> ChromaStore:
    """(Re)create `collection` and embed `docs` into it in batches; `on_batch(done, total)` is called after each one.

    Without a client, uses an in-process ephemeral Chroma and a throwaway collection name (tests).
    """
    client = client or chromadb.EphemeralClient()
    name = collection or f"tmp-{uuid.uuid4().hex}"
    if any(c.name == name for c in client.list_collections()):
        client.delete_collection(name)
    store = ChromaStore(
        client.create_collection(name, configuration={"hnsw": {"space": "cosine"}}, embedding_function=None),
        embeddings,
    )
    for i in range(0, len(docs), batch_size):
        store.add_documents(docs[i : i + batch_size])
        if on_batch:
            on_batch(min(i + batch_size, len(docs)), len(docs))
    return store


def load_store(collection: str, embeddings: Embeddings, client: ClientAPI | None = None) -> ChromaStore:
    client = client or get_client()
    return ChromaStore(client.get_collection(collection, embedding_function=None), embeddings)


@dataclass(frozen=True)
class Filters:
    """Metadata restriction; an empty field means no restriction on it.

    `quarters` are calendar quarters (1-4) of the filing's period end, as in the corpus `Quarter:` header.
    """

    tickers: tuple[str, ...] = ()
    fiscal_years: tuple[int, ...] = ()
    filing_types: tuple[str, ...] = ()
    quarters: tuple[int, ...] = ()

    def __post_init__(self):  # accept any iterable (or None); tickers keep their order (balanced search follows it)
        object.__setattr__(self, "tickers", tuple(dict.fromkeys(t.upper() for t in self.tickers or ())))
        for name in ("fiscal_years", "filing_types", "quarters"):
            object.__setattr__(self, name, tuple(sorted(set(getattr(self, name) or ()))))

    def __bool__(self) -> bool:
        return any(self.as_dict().values())

    def as_dict(self) -> dict[str, list]:
        """Only the fields that restrict, as lists (for JSON output)."""
        return {name: list(value) for name, value in asdict(self).items() if value}

    def only(self, ticker: str) -> "Filters":
        return replace(self, tickers=(ticker,))


FILTER_FIELDS = {"tickers": "ticker", "fiscal_years": "fiscal_year", "filing_types": "filing_type", "quarters": "quarter"}


def make_where(filters: Filters | None = None) -> dict | None:
    """The same restriction as `make_filter`, as a Chroma `where` clause (None = no restriction)."""
    values = asdict(filters or Filters())
    clauses = [{key: {"$in": sorted(values[name])}} for name, key in FILTER_FIELDS.items() if values[name]]
    if len(clauses) > 1:
        return {"$and": clauses}
    return clauses[0] if clauses else None


def make_filter(filters: Filters | None = None) -> Callable[[Document], bool]:
    """Metadata predicate for `filters` (BM25 side); matches `make_where`."""
    values = asdict(filters or Filters())
    wanted = {key: set(values[name]) for name, key in FILTER_FIELDS.items() if values[name]}

    def keep(doc: Document) -> bool:
        return all(doc.metadata.get(key) in allowed for key, allowed in wanted.items())

    return keep


def search(store: ChromaStore, query: str, k: int = 5, filters: Filters | None = None) -> list[tuple[Document, float]]:
    """Vector similarity search restricted by metadata."""
    return store.similarity_search_with_score(query, k=k, where=make_where(filters))
