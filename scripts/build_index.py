"""Chunk + embed 10-Ks (and optionally 10-Qs) (EmbeddingGemma on Ollama by default) into a collection on the local Chroma server.

Start the server first: chroma run --path data/chroma
Usage: python3 scripts/build_index.py [--tickers AAPL TSLA JPM] [--collection 10k] [-v]
       python3 scripts/build_index.py --filing-types 10-K 10-Q --collection filings
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from anywhere, no PYTHONPATH

from app.edgar import chunk_filing, load_filing
from app.edgar.env import load_env
from app.edgar.vectorstore import build_store, get_client, get_embeddings

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "edgar_corpus"
log = logging.getLogger("build_index")

load_env()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="*", help="only these tickers (default: all)")
    parser.add_argument(
        "--filing-types", nargs="+", choices=["10-K", "10-Q"], default=["10-K"], help="default: 10-K only"
    )
    parser.add_argument("--provider", choices=["ollama", "openai"], help="default: $EMBEDDING_PROVIDER or ollama")
    parser.add_argument("--collection", default="10k", help="Chroma collection to (re)create")
    parser.add_argument("--chroma-url", help="default: $CHROMA_URL or http://localhost:8000")
    parser.add_argument("--batch-size", type=int, default=128, help="chunks embedded between progress lines")
    parser.add_argument("-v", "--verbose", action="store_true", help="per-filing details and HTTP request logs")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    if not args.verbose:
        logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per Ollama request only with -v

    wanted = {t.upper() for t in args.tickers or []}
    patterns = [f"*_{t.replace('-', '')}_*_full.txt" for t in args.filing_types]  # 10-K -> *_10K_*
    files = sorted(f for pattern in patterns for f in CORPUS.glob(pattern) if not wanted or f.name.split("_")[0] in wanted)
    if not files:
        sys.exit(f"no {'/'.join(args.filing_types)} files matched tickers {sorted(wanted)}")
    log.info("found %d filings in %s%s", len(files), CORPUS, f" (tickers: {', '.join(sorted(wanted))})" if wanted else "")

    docs = []
    chunk_start = time.time()
    for n, path in enumerate(files, 1):
        filing = load_filing(path)
        chunks = chunk_filing(filing)
        docs.extend(chunks)
        log.debug(
            "[%d/%d] %s: %d chars cleaned, FY%s, %d chunks, %d tokens",
            n, len(files), path.name, len(filing.body), filing.meta.fiscal_year, len(chunks),
            sum(c.metadata["token_count"] for c in chunks),
        )
    total_tokens = sum(d.metadata["token_count"] for d in docs)
    log.info("chunked %d filings -> %d chunks, %d tokens in %.0fs", len(files), len(docs), total_tokens, time.time() - chunk_start)

    client = get_client(args.chroma_url)
    client.heartbeat()  # fail before chunking-and-embedding if the server is not running
    embeddings = get_embeddings(args.provider)
    log.info("embedding with %s (%s)", type(embeddings).__name__, args.provider or "default provider")

    embed_start = time.time()

    def progress(done: int, total: int) -> None:
        elapsed = time.time() - embed_start
        rate = done / elapsed
        log.info(
            "embedded %d/%d chunks (%.0f%%) | %.1f chunks/s | elapsed %.0fs | eta %.0fs",
            done, total, 100 * done / total, rate, elapsed, (total - done) / rate,
        )

    store = build_store(docs, embeddings, args.collection, client, batch_size=args.batch_size, on_batch=progress)
    log.info("done: collection %r has %d chunks, total %.0fs", args.collection, store.collection.count(), time.time() - chunk_start)


if __name__ == "__main__":
    main()
