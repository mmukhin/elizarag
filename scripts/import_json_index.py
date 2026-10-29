"""Copy an old InMemoryVectorStore JSON dump into a Chroma collection, reusing its vectors (no re-embedding).

Start the server first: chroma run --path data/chroma
Usage: python3 scripts/import_json_index.py data/index/10k.json [--collection 10k]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from anywhere, no PYTHONPATH

from langchain_core.documents import Document

from app.edgar.env import load_env
from app.edgar.vectorstore import build_store, get_client, get_embeddings

load_env()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--collection", help="default: the file name without .json")
    parser.add_argument("--chroma-url", help="default: $CHROMA_URL or http://localhost:8000")
    parser.add_argument("--batch-size", type=int, default=1000)
    args = parser.parse_args()

    rows = list(json.loads(args.path.read_text()).values())
    name = args.collection or args.path.stem
    store = build_store([], get_embeddings(), name, get_client(args.chroma_url))
    for i in range(0, len(rows), args.batch_size):
        batch = rows[i : i + args.batch_size]
        docs = [Document(page_content=r["text"], metadata=r["metadata"]) for r in batch]
        store.add_documents(docs, vectors=[r["vector"] for r in batch])
    print(f"imported {store.collection.count()} chunks from {args.path} into collection {name!r}")


if __name__ == "__main__":
    main()
