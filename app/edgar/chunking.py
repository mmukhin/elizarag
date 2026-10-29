"""Token-bounded chunking of a cleaned filing into LangChain Documents."""

from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

import tiktoken
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.edgar.loader import Filing, load_filing

ENCODING = "cl100k_base"
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
CORPUS = Path(__file__).resolve().parents[2] / "data" / "edgar_corpus"


def chunk_filing(filing: Filing, chunk_tokens: int = 800, overlap: int = 100) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=ENCODING,
        chunk_size=chunk_tokens,
        chunk_overlap=overlap,
        separators=SEPARATORS,
    )
    encoder = tiktoken.get_encoding(ENCODING)
    texts = splitter.split_text(filing.body)
    docs = []
    start = 0
    for i, text in enumerate(texts):
        # LangChain's add_start_index mixes token overlap with char offsets and returns -1,
        # so locate each chunk ourselves; chunks are ordered and each starts after the previous.
        start = filing.body.find(text, start + 1 if i else 0)
        if start < 0:
            raise ValueError(f"chunk {i} of {filing.meta.source_file} not found in body")
        docs.append(
            Document(
                page_content=text,
                metadata={
                    **asdict(filing.meta),
                    "chunk_index": i,
                    "n_chunks": len(texts),
                    "start_index": start,
                    "token_count": len(encoder.encode(text, disallowed_special=())),
                },
            )
        )
    return docs


@lru_cache(maxsize=32)
def _file_chunks(path: Path) -> tuple[Document, ...]:
    return tuple(chunk_filing(load_filing(path)))


def get_chunk(chunk_id: str, corpus: Path = CORPUS) -> Document:
    """Rebuild one chunk from the corpus by its id (`source_file#chunk_index`); chunking is deterministic."""
    source_file, _, index = chunk_id.rpartition("#")
    chunks = _file_chunks(corpus / source_file)
    if not index.isdigit() or int(index) >= len(chunks):
        raise KeyError(f"no chunk {chunk_id!r} ({source_file} has {len(chunks)} chunks)")
    return chunks[int(index)]
