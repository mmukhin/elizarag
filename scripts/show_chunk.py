"""Print chunks by id (`source_file#chunk_index`), rebuilt from data/edgar_corpus/ (no Chroma needed).

Usage: python3 scripts/show_chunk.py AAPL_10K_2025-10-31_full.txt#8 [more ids...]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from anywhere, no PYTHONPATH

from app.edgar import get_chunk


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("chunk_ids", nargs="+")
    args = parser.parse_args()
    for chunk_id in args.chunk_ids:
        doc = get_chunk(chunk_id)
        m = doc.metadata
        print(f"=== {chunk_id} · {m['ticker']} {m['filing_type']} FY{m['fiscal_year']} · {m['token_count']} tok")
        print(doc.page_content, end="\n\n")


if __name__ == "__main__":
    main()
