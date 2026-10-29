"""Chunk every 10-K (or the files given) and print chunk/token stats.

Usage: python3 scripts/chunk_report.py [--samples] [FILE ...]
"""

import argparse
import sys
import statistics
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from anywhere, no PYTHONPATH

from app.edgar import chunk_filing, load_filing

CORPUS = Path(__file__).resolve().parent.parent / "data" / "edgar_corpus"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*", type=Path)
    parser.add_argument("--samples", action="store_true", help="print the first and middle chunk per file")
    args = parser.parse_args()

    files = args.files or sorted(CORPUS.glob("*_10K_*_full.txt"))
    total_chunks = total_tokens = 0
    print(f"{'file':42} {'chunks':>6} {'tokens':>9} {'median':>6} {'max':>5}")
    for path in files:
        docs = chunk_filing(load_filing(path))
        counts = [d.metadata["token_count"] for d in docs]
        total_chunks += len(docs)
        total_tokens += sum(counts)
        print(f"{path.name:42} {len(docs):>6} {sum(counts):>9} {statistics.median(counts):>6.0f} {max(counts):>5}")
        if args.samples:
            for doc in (docs[0], docs[len(docs) // 2]):
                print(f"--- chunk {doc.metadata['chunk_index']} ({doc.metadata['token_count']} tok)")
                print(doc.page_content[:600])
    print(f"\n{len(files)} files, {total_chunks} chunks, {total_tokens} tokens")


if __name__ == "__main__":
    main()
