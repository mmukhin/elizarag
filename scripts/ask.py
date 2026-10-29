"""Answer a question from the indexed filings and print the structured result as JSON.

Usage: python3 scripts/ask.py "Compare the main risk factors of Apple and Tesla" --collection 10k -k 4
       (companies, years, quarters and filing types named in the question become filters; flags override them)
       python3 scripts/ask.py "..." --llm-provider openai --llm-model gpt-6-luna
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from anywhere, no PYTHONPATH

from app.edgar.env import load_env
from app.edgar.hybrid import MODES, HybridRetriever
from app.edgar.llm import PROVIDERS, context_profile, get_chat_model
from app.edgar.qa import answer_question, select_hits
from app.edgar.query_filters import load_companies, resolve_filters
from app.edgar.vectorstore import Filters, chunk_id, get_client, get_embeddings, load_store

log = logging.getLogger("ask")

load_env()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--collection", default="10k", help="Chroma collection to search")
    parser.add_argument("--chroma-url", help="default: $CHROMA_URL or http://localhost:8000")
    parser.add_argument("--tickers", nargs="*")
    parser.add_argument("--years", nargs="*", type=int)
    parser.add_argument("--filing-types", nargs="*", choices=["10-K", "10-Q"])
    parser.add_argument("--quarters", nargs="*", type=int, choices=[1, 2, 3, 4], help="calendar quarter of the period end")
    parser.add_argument("--no-auto-filters", action="store_true", help="do not derive filters from the question")
    parser.add_argument("--mode", choices=MODES, default="hybrid")
    parser.add_argument("-k", type=int, help="passages (per company when 2+ tickers); default: the model's profile")
    parser.add_argument("--max-context-tokens", type=int, help="passage token budget; default: the model's profile")
    parser.add_argument("--provider", choices=["ollama", "openai"], help="embedding provider (must match the collection)")
    parser.add_argument("--llm-provider", choices=PROVIDERS, help="default: $LLM_PROVIDER or ollama")
    parser.add_argument("--llm-model", help="default: $LLM_MODEL or the provider's default (ollama: gemma4:e2b)")
    parser.add_argument("--with-text", action="store_true", help="add each source passage's text to the JSON (debugging)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,  # keep stdout clean so the JSON can be piped
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    retriever = HybridRetriever(load_store(args.collection, get_embeddings(args.provider), get_client(args.chroma_url)))
    flags = Filters(args.tickers, args.years, args.filing_types, args.quarters)
    resolved = resolve_filters(
        args.question, load_companies(), retriever.years_by_ticker, flags, not args.no_auto_filters
    )
    filters = resolved.filters
    if missing := [t for t in filters.tickers if t not in retriever.tickers]:
        log.warning("%s not in collection %r; nothing will be retrieved for it", missing, args.collection)
    log.info("filters from the question: %s", resolved.detected.as_dict() or "none")
    for field, (asked, chosen) in resolved.overridden.items():
        flag = {"tickers": "--tickers", "fiscal_years": "--years", "filing_types": "--filing-types", "quarters": "--quarters"}[field]
        log.warning("the question implies %s %s, but %s %s is used", field, asked, flag, chosen)
    llm = get_chat_model(args.llm_provider, args.llm_model)
    profile = context_profile(llm.model)
    k, budget = args.k or profile.k, args.max_context_tokens or profile.budget
    hits = retriever.search(args.question, k=k, mode=args.mode, filters=filters, per_ticker=len(filters.tickers) > 1)
    kept = select_hits(hits, budget)
    log.info("retrieved %d passages (k=%d), sending %d (budget %d tokens) to %s", len(hits), k, len(kept), budget, llm.model)

    result = answer_question(args.question, kept, llm, budget, filters.as_dict())
    if not args.with_text:
        print(result.model_dump_json(indent=2))
        return
    texts = {chunk_id(h.doc.metadata): h.doc.page_content for h in kept}
    output = result.model_dump()
    for source in output["sources"]:
        source["text"] = texts[source["chunk_id"]]
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
