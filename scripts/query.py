"""Query a collection on the local Chroma server.

Usage: python3 scripts/query.py "supply chain risks" --tickers AAPL TSLA --years 2025 -k 3
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from anywhere, no PYTHONPATH

from app.edgar.env import load_env
from app.edgar.vectorstore import Filters, get_client, get_embeddings, load_store, search

load_env()



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--collection", default="10k")
    parser.add_argument("--chroma-url", help="default: $CHROMA_URL or http://localhost:8000")
    parser.add_argument("--tickers", nargs="*")
    parser.add_argument("--years", nargs="*", type=int)
    parser.add_argument("--filing-types", nargs="*", choices=["10-K", "10-Q"])
    parser.add_argument("--quarters", nargs="*", type=int, choices=[1, 2, 3, 4], help="calendar quarter of the period end")
    parser.add_argument("--provider", choices=["ollama", "openai"], help="must match the provider used to build the index")
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()

    store = load_store(args.collection, get_embeddings(args.provider), get_client(args.chroma_url))
    for doc, score in search(store, args.query, args.k, Filters(args.tickers, args.years, args.filing_types, args.quarters)):
        m = doc.metadata
        print(f"\n[{score:.3f}] {m['ticker']} {m['filing_type']} FY{m['fiscal_year']} chunk {m['chunk_index']}/{m['n_chunks']}")
        print(doc.page_content[:400].replace("\n", " "))


if __name__ == "__main__":
    main()
