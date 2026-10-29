"""Streamlit page: ask a question, see the answer, the retrieval queries, and which passages the LLM got."""

import httpx
import pandas as pd
import streamlit as st

from app.edgar.hybrid import MODES, Hit, HybridRetriever, SearchTrace
from app.edgar.llm import DEFAULT_MODELS, PROVIDERS, ChatModel, Usage, get_chat_model
from app.edgar.qa import QAResult, answer_question, passage_tokens, select_hits
from app.edgar.query_filters import load_companies, parse_filters
from app.edgar.vectorstore import Filters

MODE_LABELS = {"hybrid": "Hybrid (keyword + vector)", "vector": "Vector only", "keyword": "Keyword (BM25) only"}
EXAMPLE_QUESTIONS = [
    "What are the primary risk factors facing Apple, Tesla, and JPMorgan, and how do they compare?",
    "How has NVIDIA's revenue and growth outlook changed over the last two years?",
    "What regulatory risks do the major pharmaceutical companies face, and how are they addressing them?",
]


def hit_title(hit) -> str:
    m = hit.doc.metadata
    return f"{m['ticker']} · {m['filing_type']} FY{m['fiscal_year']} · chunk {m['chunk_index'] + 1}/{m['n_chunks']}"


def hit_caption(hit, mode: str) -> str:
    m = hit.doc.metadata
    ranks = []
    if hit.vector_rank is not None:
        ranks.append(f"vector #{hit.vector_rank}")
    if hit.keyword_rank is not None:
        ranks.append(f"keyword #{hit.keyword_rank}")
    score = f"score {hit.score:.4f} · " if mode != "keyword" else ""
    return f"{score}{' · '.join(ranks)} · {m['company']} · filed {m['filing_date']} · {m['source_file']}"


CONTEXT_BUDGET = 8000


def hit_rows(hits: list[Hit], mode: str, **extra) -> pd.DataFrame:
    """One row per hit: where it ranked in each search, its scores, and its token cost."""
    rows = []
    for hit in hits:
        m = hit.doc.metadata
        rows.append(
            extra
            | {
                "passage": f"{m['ticker']} · {m['filing_type']} FY{m['fiscal_year']} · chunk {m['chunk_index'] + 1}/{m['n_chunks']}",
                "vector #": hit.vector_rank,
                "cosine": hit.vector_score,
                "bm25 #": hit.keyword_rank,
                "bm25 score": hit.keyword_score,
                "rrf": hit.score if mode == "hybrid" else None,
                "tokens": passage_tokens(hit),
                "text": hit.doc.page_content[:300],
            }
        )
    return pd.DataFrame(rows).astype({"vector #": "Int64", "bm25 #": "Int64"})  # blank, not NaN-float


def render_retrieval(traces: list[SearchTrace]) -> None:
    """The exact requests sent to Chroma (embedding search) and BM25, and the top-k used, per search."""
    st.subheader("Retrieval")
    for n, t in enumerate(traces, 1):
        st.markdown(
            f"**Search {n}/{len(traces)}** · {MODE_LABELS[t.mode]} · **top-k {t.k}** · "
            f"{len(t.candidates)} candidates (up to {t.pool} per ranker) · {len(t.unused)} not used"
        )
        chroma, bm25 = st.columns(2)
        with chroma:
            st.caption("Embedding search → Chroma `collection.query`")
            if t.chroma_request:
                st.json(t.chroma_request)
            else:
                st.caption("not run in keyword mode")
        with bm25:
            st.caption("Keyword search → BM25 over all chunks loaded from Chroma")
            if t.bm25_request:
                st.json(t.bm25_request)
            else:
                st.caption("not run in vector mode")


def render_usage(
    traces: list[SearchTrace], retrieved: list[Hit], kept: list[Hit], mode: str, generate: bool, budget: int
) -> None:
    """What the LLM got, what top-k results were cut by the token budget, and candidates outside top-k."""
    kept_ids = {id(h) for h in kept}
    dropped = [h for h in retrieved if id(h) not in kept_ids]
    used = sum(passage_tokens(h) for h in kept)
    if generate:
        st.subheader(f"{len(kept)} passages sent to the LLM")
        st.caption(f"{used:,} of {budget:,} context tokens · each company's best passages are kept first")
    else:
        st.subheader(f"{len(kept)} passages")
    for i, hit in enumerate(kept, 1):
        with st.expander(f"[{i}] {hit_title(hit)}", expanded=not generate and i == 1):
            st.caption(hit_caption(hit, mode))
            st.text(hit.doc.page_content)

    if generate:
        st.subheader(f"{len(dropped)} top-k passages not sent (over the {budget:,}-token budget)")
        if dropped:
            st.dataframe(hit_rows(dropped, mode), hide_index=True, width="stretch")

    unused = [(n, h) for n, t in enumerate(traces, 1) for h in t.unused]
    st.subheader(f"{len(unused)} candidates not used (ranked below top-k)")
    if unused:
        frames = [hit_rows([h], mode, search=n) for n, h in unused]
        st.dataframe(pd.concat(frames, ignore_index=True), hide_index=True, width="stretch")



def format_cost(usage: Usage, local: bool) -> str:
    if usage.cost_usd is None:
        return "unknown"
    if local:
        return "$0 (local)"
    return f"${usage.cost_usd:.5f}" if usage.cost_usd < 0.01 else f"${usage.cost_usd:.4f}"


def render_llm_usage(usage: Usage, local: bool) -> None:
    """Tokens sent and received, cost and latency of the LLM call (summed over retries)."""
    sent, received, cost, latency = st.columns(4)
    sent.metric("Tokens sent", f"{usage.input_tokens:,}")
    received.metric("Tokens received", f"{usage.output_tokens:,}")
    cost.metric("Cost", format_cost(usage, local))
    latency.metric("LLM latency", f"{usage.seconds:.1f} s")
    retries = f" · {usage.calls} calls (retried after invalid output)" if usage.calls > 1 else ""
    hint = " · set LLM_PRICE_INPUT / LLM_PRICE_OUTPUT ($ per 1M tokens) to price this model" if usage.cost_usd is None else ""
    st.caption(f"{usage.model}{retries}{hint}")


def render_answer(result: QAResult, local: bool = False) -> None:
    st.subheader("Answer")
    if result.usage:
        render_llm_usage(result.usage, local)
    if result.status == "off_topic":
        st.info("This question is not about the indexed SEC filings.")
    elif result.status != "answered":
        st.info("The retrieved passages do not fully answer this question.")
    st.markdown(result.answer)
    if result.claims:
        refs = {s.chunk_id: s.ref for s in result.sources}
        st.markdown("**Key points**")
        for claim in result.claims:
            cites = "".join(f"[{refs[c]}]" for c in claim.chunk_ids)
            st.markdown(f"- **{', '.join(claim.tickers) or 'General'}**: {claim.text} {cites}")
    for caveat in result.caveats:
        st.caption(f"Caveat: {caveat}")
    for warning in result.warnings:
        st.warning(warning)
    with st.expander("Structured output (JSON)"):
        st.json(result.model_dump())


def render(retriever: HybridRetriever, chat_model: ChatModel | None = None, context_budget: int = CONTEXT_BUDGET) -> None:
    st.title("SEC filings Q&A")
    st.caption(f"{len(retriever.docs):,} chunks · {len(retriever.tickers)} companies")

    with st.sidebar:
        st.header("Search")
        mode = st.radio("Mode", MODES, format_func=MODE_LABELS.get)
        k = st.slider("Results (per company if balanced)", 1, 20, 5)
        tickers = st.multiselect("Companies", retriever.tickers)
        balanced = st.checkbox("Balance across selected companies", value=True, disabled=len(tickers) < 2)
        years = st.multiselect("Fiscal years", retriever.fiscal_years)
        filing_type = st.selectbox("Filing type", ["Any", *retriever.filing_types])
        quarters = st.multiselect("Quarters (calendar, of period end)", retriever.quarters) if retriever.quarters else []
        detect = st.checkbox("Detect filters from the question", value=True, help="Companies, years, quarters, 10-K/10-Q")

        st.header("Answer")
        generate = st.checkbox("Generate answer with LLM", value=True)
        if generate and chat_model is None:
            llm_provider = st.selectbox("LLM provider", PROVIDERS)
            llm_model = st.text_input("LLM model", value=DEFAULT_MODELS[llm_provider], key=f"model-{llm_provider}")

    with st.expander("Example questions"):
        for i, example in enumerate(EXAMPLE_QUESTIONS):
            st.button(example, key=f"example-{i}", on_click=st.session_state.__setitem__, args=("query", example))
    query = st.text_input("Ask a question", placeholder="What are the primary risk factors facing Apple and Tesla?", key="query")
    if not query.strip():
        return

    detected = parse_filters(query, load_companies(retriever.tickers), retriever.years_by_ticker) if detect else Filters()
    filters = Filters(  # a sidebar choice beats what was detected in the question, per field
        tickers or detected.tickers,
        years or detected.fiscal_years,
        [filing_type] if filing_type != "Any" else detected.filing_types,
        quarters or detected.quarters,
    )
    if detected:
        st.caption("Filters from the question: " + " · ".join(f"{k} {', '.join(map(str, v))}" for k, v in detected.as_dict().items()))

    answer_box = st.container()  # filled last, shown first
    traces: list[SearchTrace] = []
    try:
        with st.spinner("Searching..."):
            hits = retriever.search(
                query,
                k=k,
                mode=mode,
                filters=filters,
                per_ticker=balanced,
                trace=traces,
            )
    except httpx.HTTPError as exc:
        st.error(f"Embedding backend unreachable ({exc}). Start Ollama, or switch Mode to 'Keyword (BM25) only'.")
        return

    if not hits:
        st.warning("No matching passages. Try removing filters or rephrasing.")
        render_retrieval(traces)
        return

    kept = select_hits(hits, context_budget) if generate else hits  # exactly the passages the model will see
    if st.toggle("Show retrieval details", value=True, key="show-retrieval"):
        render_retrieval(traces)
    if st.toggle("Show passages", value=True, key="show-passages"):
        render_usage(traces, hits, kept, mode, generate, context_budget)

    if generate:
        with answer_box:
            try:
                llm = chat_model or get_chat_model(llm_provider, llm_model)
                with st.spinner(f"Generating answer with {getattr(llm, 'model', 'the model')}..."):
                    render_answer(answer_question(query, kept, llm, context_budget, filters.as_dict()), llm.local)
            except Exception as exc:  # backend down, missing API key, invalid structured output...
                st.error(f"Answer generation failed ({type(exc).__name__}: {exc}). The retrieved passages are shown below.")
