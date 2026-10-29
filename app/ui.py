"""Streamlit page: ask a question, see the answer, the retrieval queries, and which passages the LLM got."""

import html
import statistics

import httpx
import pandas as pd
import streamlit as st

from app.edgar.hybrid import MODES, Hit, HybridRetriever, SearchTrace
from app.edgar.llm import ChatModel, Usage, context_profile, default_choice, get_chat_model, model_choices
from app.edgar.qa import CITATION, QAResult, answer_question, passage_tokens, select_hits
from app.edgar.query_filters import load_companies, resolve_filters
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


# [n] citations link to #passage-n; the targeted passage is highlighted
STYLE = """<style>
.passage { white-space: pre-wrap; font-size: 0.9rem; line-height: 1.45; padding: 0.5rem 0.75rem;
           border-radius: 0.4rem; scroll-margin-top: 5rem; transition: background 0.3s; }
.passage:target { background: rgba(255, 210, 0, 0.25); box-shadow: 0 0 0 2px rgba(255, 190, 0, 0.9); }
a.cite { text-decoration: none; font-weight: 600; }
</style>"""


def linkify(text: str, refs: set[int]) -> str:
    """Markdown with each known [n] linked to its passage; `$` escaped so figures don't render as LaTeX."""
    text = text.replace("$", "\\$")
    return CITATION.sub(lambda m: f'<a class="cite" href="#passage-{m[1]}">[{m[1]}]</a>' if int(m[1]) in refs else m[0], text)


def passage_html(ref: int, text: str) -> str:
    return f'<div class="passage" id="passage-{ref}">{html.escape(text)}</div>'


def typical_passage_tokens(retriever: HybridRetriever) -> int:
    counts = [d.metadata["token_count"] for d in retriever.docs if d.metadata.get("token_count")]
    return int(statistics.median(counts)) + 30 if counts else 800  # + passage header


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
    with st.expander(f"Retrieval · {len(traces)} search{'es' if len(traces) != 1 else ''}"):
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


def render_passages(
    traces: list[SearchTrace],
    retrieved: list[Hit],
    kept: list[Hit],
    mode: str,
    generate: bool,
    budget: int,
    cited: set[int] = frozenset(),
) -> None:
    """What the LLM got (cited passages open), top-k results cut by the token budget, and candidates outside top-k."""
    kept_ids = {id(h) for h in kept}
    dropped = [h for h in retrieved if id(h) not in kept_ids]
    used = sum(passage_tokens(h) for h in kept)
    label = f"{len(kept)} passages sent to the LLM · {used:,} of {budget:,} tokens" if generate else f"{len(kept)} passages"
    with st.expander(label, expanded=True):
        if generate:
            st.caption("Each company's best passages are kept first. Cited passages are open; click a [n] in the answer to jump to it.")
        for i, hit in enumerate(kept, 1):
            with st.expander(f"[{i}] {hit_title(hit)}", expanded=i in cited or (not generate and i == 1)):
                st.caption(hit_caption(hit, mode))
                st.markdown(passage_html(i, hit.doc.page_content), unsafe_allow_html=True)

    if generate:
        with st.expander(f"{len(dropped)} top-k passages not sent (over the {budget:,}-token budget)"):
            if dropped:
                st.dataframe(hit_rows(dropped, mode), hide_index=True, width="stretch")
            else:
                st.caption("Every retrieved passage fit the budget.")

    unused = [(n, h) for n, t in enumerate(traces, 1) for h in t.unused]
    with st.expander(f"{len(unused)} candidates not used (ranked below top-k)"):
        if unused:
            frames = [hit_rows([h], mode, search=n) for n, h in unused]
            st.dataframe(pd.concat(frames, ignore_index=True), hide_index=True, width="stretch")
        else:
            st.caption("No other candidates.")


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
    refs = {s.ref for s in result.sources}
    st.markdown(linkify(result.answer, refs), unsafe_allow_html=True)
    if result.claims:
        ref_of = {s.chunk_id: s.ref for s in result.sources}
        st.markdown("**Key points**")
        for claim in result.claims:
            cites = "".join(f"[{ref_of[c]}]" for c in claim.chunk_ids)
            tickers = ", ".join(claim.tickers) or "General"
            st.markdown(f"- **{tickers}**: {linkify(f'{claim.text} {cites}', refs)}", unsafe_allow_html=True)
    for caveat in result.caveats:
        st.caption(f"Caveat: {caveat}")
    for warning in result.warnings:
        st.warning(warning)
    with st.expander("Structured output (JSON)"):
        st.json(result.model_dump())


def cited_refs(result: QAResult) -> set[int]:
    ref_of = {s.chunk_id: s.ref for s in result.sources}
    in_text = {int(n) for n in CITATION.findall(result.answer)}
    return in_text | {ref_of[c] for claim in result.claims for c in claim.chunk_ids}


@st.fragment
def llm_settings(retriever: HybridRetriever, fixed_model: str | None, fixed_budget: int | None):
    """Model and context settings. A fragment: picking a model re-runs only this block, so the window info and
    the k / budget defaults follow the model at once, without re-running the query."""
    st.header("Answer")
    generate = st.checkbox("Generate answer with LLM", value=True, key="generate")
    if fixed_model is None:
        choices = model_choices()
        provider, model = st.selectbox(
            "Model", choices, index=default_choice(choices), format_func=lambda c: f"{c[0]}: {c[1]}", key="llm"
        )
    else:
        provider, model = None, fixed_model

    st.header("Context")
    profile = context_profile(model)
    k = st.slider("Passages per company (total if not balanced)", 1, 30, profile.k, key=f"k-{model}")
    if fixed_budget is None:
        budget = st.slider(
            "Context budget (tokens of passages)", 1_000, profile.max_budget, profile.budget, step=500, key=f"budget-{model}"
        )
    else:
        budget = fixed_budget
    typical = typical_passage_tokens(retriever)
    st.caption(
        f"**{model}**: {profile.context_window:,}-token window, {profile.output_reserve:,} reserved for the answer. "
        f"The budget fits about {budget // typical} passages of ~{typical} tokens; each company's best passages are kept first. "
        f"Defaults for this model: {profile.k} passages per company, {profile.budget:,} tokens."
    )
    return generate, provider, model, k, budget


def render(retriever: HybridRetriever, chat_model: ChatModel | None = None, context_budget: int | None = None) -> None:
    """The Q&A page. `chat_model` / `context_budget` fix the model and budget (tests); otherwise the sidebar picks them."""
    st.title("SCiFi Assistant")
    st.caption(f"{len(retriever.docs):,} chunks · {len(retriever.tickers)} companies")

    with st.sidebar:
        fixed_model = getattr(chat_model, "model", type(chat_model).__name__) if chat_model else None
        generate, llm_provider, model, k, budget = llm_settings(retriever, fixed_model, context_budget)

    # search settings apply only when the form is submitted, so changing them does not re-run (and re-pay for) the query
    with st.sidebar.form("settings"):
        st.header("Search")
        mode = st.radio("Mode", MODES, format_func=MODE_LABELS.get)
        tickers = st.multiselect("Companies", retriever.tickers)
        balanced = st.checkbox("Balance across companies", value=True, help="k passages per company when 2+ are selected or detected")
        years = st.multiselect("Fiscal years", retriever.fiscal_years)
        filing_type = st.selectbox("Filing type", ["Any", *retriever.filing_types])
        quarters = st.multiselect("Quarters (calendar, of period end)", retriever.quarters) if retriever.quarters else []
        detect = st.checkbox("Detect filters from the question", value=True, help="Companies, years, quarters, 10-K/10-Q")
        st.form_submit_button("Apply and re-run", type="primary", key="apply", help="Re-runs the question with all settings")

    with st.expander("Example questions"):
        for i, example in enumerate(EXAMPLE_QUESTIONS):
            st.button(example, key=f"example-{i}", on_click=st.session_state.__setitem__, args=("query", example))
    query = st.text_input("Ask a question", placeholder="What are the primary risk factors facing Apple and Tesla?", key="query")
    if not query.strip():
        return

    sidebar = Filters(tickers, years, [filing_type] if filing_type != "Any" else [], quarters)
    # detect against every known company, not only the indexed ones, so a missing company is reported, not dropped
    resolved = resolve_filters(query, load_companies(), retriever.years_by_ticker, sidebar, detect)
    filters, detected = resolved.filters, resolved.detected
    if detected:
        st.caption("Filters from the question: " + " · ".join(f"{k} {', '.join(map(str, v))}" for k, v in detected.as_dict().items()))
    if missing := [t for t in filters.tickers if t not in retriever.tickers]:
        st.warning(f"{', '.join(missing)} {'is' if len(missing) == 1 else 'are'} not in this collection. Index it with "
                   f"`python3 scripts/build_index.py`, or pick another collection.")
    for field, (asked, chosen) in resolved.overridden.items():
        st.warning(
            f"The question implies {field.replace('_', ' ')} {', '.join(map(str, asked))}, but the sidebar filter "
            f"({', '.join(map(str, chosen))}) is used. Clear it in the sidebar and press Apply to follow the question."
        )

    answer_box = st.container()  # shown first; filled after retrieval
    traces: list[SearchTrace] = []
    try:
        with st.spinner("Searching..."):
            hits = retriever.search(query, k=k, mode=mode, filters=filters, per_ticker=balanced, trace=traces)
    except httpx.HTTPError as exc:
        st.error(f"Embedding backend unreachable ({exc}). Start Ollama, or switch Mode to 'Keyword (BM25) only'.")
        return

    if not hits:
        st.warning("No matching passages. Try removing filters or rephrasing.")
        render_retrieval(traces)
        return

    kept = select_hits(hits, budget) if generate else hits  # exactly the passages the model will see
    render_retrieval(traces)
    cited: set[int] = set()
    if generate:
        with answer_box:
            try:
                llm = chat_model or get_chat_model(llm_provider, model)
                with st.spinner(f"Generating answer with {model}..."):
                    result = answer_question(query, kept, llm, budget, filters.as_dict())
                render_answer(result, llm.local)
                cited = cited_refs(result)
            except Exception as exc:  # backend down, missing API key, invalid structured output...
                st.error(f"Answer generation failed ({type(exc).__name__}: {exc}). The retrieved passages are shown below.")
    render_passages(traces, hits, kept, mode, generate, budget, cited)
    st.markdown(STYLE, unsafe_allow_html=True)
