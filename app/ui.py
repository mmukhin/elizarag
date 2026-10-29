"""Streamlit chat page: ask about the filings, get a grounded answer with linked sources, inspect the retrieval."""

import html
import statistics
import time
from dataclasses import dataclass, field

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


STATUS_BADGES = {
    "answered": ":green-badge[:material/check_circle: Answered]",
    "partial": ":orange-badge[:material/rule: Partially answered]",
    "not_in_passages": ":gray-badge[:material/search_off: Not in the filings]",
    "off_topic": ":gray-badge[:material/block: Off topic]",
}

# chat look: a prominent pinned input, readable answer text, and [n] citations that jump to a highlighted passage
STYLE = """<style>
.block-container { max-width: 60rem; padding-bottom: 7rem; }
/* one border, on the box Streamlit already draws; focus = stronger border + soft ring, same width (no jump) */
[data-testid="stChatInput"] { border: none !important; border-radius: 1rem !important; background: transparent !important; }
[data-testid="stChatInput"] > div { border: 1.5px solid rgba(99, 102, 241, 0.5) !important; border-radius: 1rem !important;
    box-shadow: 0 6px 24px rgba(99, 102, 241, 0.14); transition: border-color 0.15s, box-shadow 0.15s; }
[data-testid="stChatInput"] > div:focus-within { border-color: rgb(99, 102, 241) !important;
    box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.16), 0 6px 24px rgba(99, 102, 241, 0.2); }
[data-testid="stChatInput"] textarea { font-size: 1.05rem !important; }
.hero { text-align: center; margin: 3rem 0 1.25rem; }
.hero h2 { font-size: 1.9rem; font-weight: 650; margin-bottom: 0.35rem; }
.hero p { opacity: 0.7; margin: 0; }
[class*="st-key-example-"] button { justify-content: flex-start; text-align: left; padding: 0.6rem 0.9rem; }
[class*="st-key-example-"] button > div { justify-content: flex-start; width: 100%; }
[class*="st-key-example-"] button p { white-space: normal; text-overflow: clip; overflow: visible; text-align: left; }
.answer { font-size: 1.1rem; line-height: 1.65; margin: 0.35rem 0 0.6rem; }
/* the question's first line is as tall as the avatar (2rem) and pinned to the top, so it sits level with the icon
   whether or not detected filters follow (Streamlit otherwise re-centres short content, off by its -1rem margin) */
[data-testid="stChatMessage"]:has(.question) [data-testid="stChatMessageContent"] { align-self: flex-start !important; margin-top: 0 !important; }
.question { font-size: 1.05rem; font-weight: 550; line-height: 2rem; min-height: 2rem; }
.passage { white-space: pre-wrap; font-size: 0.9rem; line-height: 1.45; padding: 0.5rem 0.75rem;
           border-radius: 0.4rem; scroll-margin-top: 5rem; transition: background 0.3s; }
.passage:target { background: rgba(255, 210, 0, 0.25); box-shadow: 0 0 0 2px rgba(255, 190, 0, 0.9); }
a.cite { text-decoration: none; font-weight: 650; padding: 0 0.1rem; border-radius: 0.25rem; }
a.cite:hover { background: rgba(99, 102, 241, 0.15); }
</style>"""


# Sources start collapsed, and browsers don't open a closed expander for a #link inside it. This fixed script (no
# user or model text in it) opens the expanders around the cited passage, through their headers so Streamlit keeps
# its own open/closed state, then jumps to the passage, which the .passage:target style highlights.
CITE_SCRIPT = """<script>
(() => {
  if (window.__citeLinks) return;
  window.__citeLinks = true;
  document.addEventListener("click", (event) => {
    const link = event.target.closest("a.cite");
    if (!link) return;
    const id = link.getAttribute("href").slice(1);
    const target = document.getElementById(id);
    if (!target) return;
    event.preventDefault();
    const closed = [];
    for (let el = target.parentElement; el; el = el.parentElement) {
      if (el.tagName === "DETAILS" && !el.open) closed.unshift(el);
    }
    closed.forEach((details) => details.querySelector(":scope > summary")?.click());
    setTimeout(() => {
      if (location.hash === "#" + id) history.replaceState(null, "", " ");
      location.hash = id;
      target.scrollIntoView({ behavior: "smooth", block: "center" });
    }, closed.length ? 250 : 0);
  });
})();
</script>"""


def linkify(text: str, refs: set[int], anchor: str = "passage", markdown: bool = True) -> str:
    """Each known [n] linked to `#{anchor}-n`. For markdown, `$` is escaped so figures don't render as LaTeX;
    otherwise the text is HTML-escaped for use inside an HTML block."""
    text = text.replace("$", "\\$") if markdown else html.escape(text).replace("\n", "<br>")
    return CITATION.sub(lambda m: f'<a class="cite" href="#{anchor}-{m[1]}">[{m[1]}]</a>' if int(m[1]) in refs else m[0], text)


def passage_html(ref: int, text: str, anchor: str = "passage") -> str:
    return f'<div class="passage" id="{anchor}-{ref}">{html.escape(text)}</div>'


def filter_badges(filters: Filters) -> str:
    """Markdown badges for the filters a search used, e.g. NVDA · FY2024 · FY2025."""
    badges = [f":blue-badge[{t}]" for t in filters.tickers]
    badges += [f":violet-badge[FY{y}]" for y in filters.fiscal_years]
    badges += [f":gray-badge[{f}]" for f in filters.filing_types] + [f":gray-badge[Q{q}]" for q in filters.quarters]
    return " ".join(badges)


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
    with st.expander(f"Retrieval · {len(traces)} search{'es' if len(traces) != 1 else ''}", icon=":material/manage_search:"):
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
    anchor: str = "passage",
    expanded: bool = True,
) -> None:
    """In pipeline order: candidates ranked below top-k, top-k results cut by the token budget, what the LLM got."""
    unused = [(n, h) for n, t in enumerate(traces, 1) for h in t.unused]
    with st.expander(f"{len(unused)} candidates not used (ranked below top-k)", icon=":material/low_priority:"):
        if unused:
            frames = [hit_rows([h], mode, search=n) for n, h in unused]
            st.dataframe(pd.concat(frames, ignore_index=True), hide_index=True, width="stretch")
        else:
            st.caption("No other candidates.")

    kept_ids = {id(h) for h in kept}
    dropped = [h for h in retrieved if id(h) not in kept_ids]
    if generate:
        with st.expander(f"{len(dropped)} top-k passages not sent (over the {budget:,}-token budget)", icon=":material/content_cut:"):
            if dropped:
                st.dataframe(hit_rows(dropped, mode), hide_index=True, width="stretch")
            else:
                st.caption("Every retrieved passage fit the budget.")

    used = sum(passage_tokens(h) for h in kept)
    label = f"{len(kept)} passages sent to the LLM · {used:,} of {budget:,} tokens" if generate else f"{len(kept)} passages"
    with st.expander(label, expanded=expanded, icon=":material/description:"):
        if generate:
            st.caption("Each company's best passages are kept first. Cited passages are open; click a [n] in the answer to jump to it.")
        for i, hit in enumerate(kept, 1):
            with st.expander(f"[{i}] {hit_title(hit)}", expanded=i in cited or (not generate and i == 1)):
                st.caption(hit_caption(hit, mode))
                st.markdown(passage_html(i, hit.doc.page_content, anchor), unsafe_allow_html=True)


def format_cost(usage: Usage, local: bool) -> str:
    if usage.cost_usd is None:
        return "cost unknown"
    if local:
        return "$0 (local)"
    return f"${usage.cost_usd:.5f}" if usage.cost_usd < 0.01 else f"${usage.cost_usd:.4f}"


def usage_line(usage: Usage, local: bool, total_seconds: float) -> str:
    """One quiet line under the answer: model, tokens sent / received, cost, latency."""
    retries = f" · {usage.calls} calls (retried after invalid output)" if usage.calls > 1 else ""
    hint = " · set LLM_PRICE_INPUT / LLM_PRICE_OUTPUT ($ per 1M tokens) to price this model" if usage.cost_usd is None else ""
    return (
        f":material/memory: {usage.model} · {usage.input_tokens:,} tokens sent · {usage.output_tokens:,} received · "
        f"{format_cost(usage, local)} · LLM {usage.seconds:.1f} s · total {total_seconds:.1f} s{retries}{hint}"
    ).replace("$", "\\$")


# what associates do after an answer; shown as actions under each answer, not built yet
COMING_SOON = {
    "copy": (
        ":material/content_copy:",
        "Copy with citations",
        "Copy the answer with its [n] citations kept and a numbered source list (company, filing, fiscal year, "
        "SEC link), ready to paste into a memo.",
    ),
    "export": (
        ":material/table_view:",
        "Export tables to Excel or CSV",
        "Export the figures and tables in the cited passages to Excel or CSV, ready for the comps model.",
    ),
    "share": (
        ":material/share:",
        "Share with the deal team",
        "Share a link to this answer and its evidence (passages, filters, model) so the deal team sees exactly what you saw.",
    ),
    "flag": (
        ":material/flag:",
        "Flag a wrong answer",
        "Flag a wrong or incomplete answer with a note. The question, answer and passages go into the eval set, "
        "so the problem becomes a test the assistant must pass.",
    ),
}


@st.dialog("Coming soon", icon=":material/rocket_launch:")
def coming_soon(name: str) -> None:
    icon, title, description = COMING_SOON[name]
    st.markdown(f"#### {icon} {title}")
    st.write(description)
    st.caption("Not available in this prototype yet.")


def cited_refs(result: QAResult) -> set[int]:
    ref_of = {s.chunk_id: s.ref for s in result.sources}
    in_text = {int(n) for n in CITATION.findall(result.answer)}
    return in_text | {ref_of[c] for claim in result.claims for c in claim.chunk_ids}


@dataclass
class Settings:
    mode: str
    k: int
    budget: int
    balanced: bool
    generate: bool
    provider: str | None
    model: str
    sidebar: Filters
    detect: bool


@dataclass
class Turn:
    """One question and everything needed to redraw its answer without re-running search or the LLM."""

    question: str
    settings: Settings
    filters: Filters = field(default_factory=Filters)
    detected: Filters = field(default_factory=Filters)
    notices: list[str] = field(default_factory=list)  # sidebar overrides, companies not indexed
    steps: list[str] = field(default_factory=list)  # what the assistant did, shown in its status box
    traces: list[SearchTrace] = field(default_factory=list)
    hits: list[Hit] = field(default_factory=list)
    kept: list[Hit] = field(default_factory=list)
    result: QAResult | None = None
    local: bool = False
    error: str | None = None
    seconds: float = 0.0

    @property
    def summary(self) -> str:
        if self.error and not self.hits:
            return "Search failed"
        if not self.hits:
            return "No matching passages"
        companies = sorted({h.doc.metadata["ticker"] for h in self.kept})
        who = ", ".join(companies) if len(companies) <= 4 else f"{len(companies)} companies"
        return f"Searched filings · {len(self.kept)} passages from {who} · {self.seconds:.1f} s"


def run_turn(question: str, s: Settings, retriever: HybridRetriever, chat_model: ChatModel | None, status) -> Turn:
    """Answer `question`, writing each step to the live `status` box as it happens."""
    started = time.monotonic()
    turn = Turn(question, s)

    def step(text: str) -> None:
        turn.steps.append(text)
        status.write(text)

    # detect against every known company, not only the indexed ones, so a missing company is reported, not dropped
    resolved = resolve_filters(question, load_companies(), retriever.years_by_ticker, s.sidebar, s.detect)
    turn.filters, turn.detected = resolved.filters, resolved.detected
    step(f":material/filter_alt: Filters: {filter_badges(turn.filters)}" if turn.filters else ":material/filter_alt: No filters: searching every company and year")
    if missing := [t for t in turn.filters.tickers if t not in retriever.tickers]:
        turn.notices.append(
            f"{', '.join(missing)} {'is' if len(missing) == 1 else 'are'} not in this collection. "
            f"Index it with `python3 scripts/build_index.py`, or pick another collection."
        )
    for name, (asked, chosen) in resolved.overridden.items():
        turn.notices.append(
            f"The question implies {name.replace('_', ' ')} {', '.join(map(str, asked))}, but the sidebar filter "
            f"({', '.join(map(str, chosen))}) is used. Clear it in the sidebar and press Apply to follow the question."
        )

    try:
        turn.hits = retriever.search(question, k=s.k, mode=s.mode, filters=turn.filters, per_ticker=s.balanced, trace=turn.traces)
    except httpx.HTTPError as exc:
        turn.error = f"Embedding backend unreachable ({exc}). Start Ollama, or switch Mode to 'Keyword (BM25) only'."
        turn.seconds = time.monotonic() - started
        return turn
    step(f":material/manage_search: {MODE_LABELS[s.mode]} search ({len(turn.traces)} quer{'ies' if len(turn.traces) != 1 else 'y'}): {len(turn.hits)} passages")
    if not turn.hits:
        turn.seconds = time.monotonic() - started
        return turn

    turn.kept = select_hits(turn.hits, s.budget) if s.generate else turn.hits  # exactly the passages the model will see
    if s.generate:
        used = sum(passage_tokens(h) for h in turn.kept)
        step(f":material/send: Sending {len(turn.kept)} passages ({used:,} of {s.budget:,} tokens) to {s.model}")
        try:
            llm = chat_model or get_chat_model(s.provider, s.model)
            turn.local = llm.local
            turn.result = answer_question(question, turn.kept, llm, s.budget, turn.filters.as_dict())
            step(f":material/task_alt: Answer checked: {len(turn.result.claims)} cited claims, {len(turn.result.warnings)} warnings")
        except Exception as exc:  # backend down, missing API key, invalid structured output...
            turn.error = f"Answer generation failed ({type(exc).__name__}: {exc}). The retrieved passages are shown below."
    turn.seconds = time.monotonic() - started
    return turn


def render_result(i: int, turn: Turn, latest: bool) -> set[int]:
    """The answer: status badge, answer text with linked citations, key points, caveats, usage, feedback."""
    result, anchor = turn.result, f"passage-{i}"
    refs = {s.ref for s in result.sources}
    st.markdown(STATUS_BADGES.get(result.status, result.status))
    st.markdown(f'<div class="answer">{linkify(result.answer, refs, anchor, markdown=False)}</div>', unsafe_allow_html=True)
    if result.claims:
        ref_of = {s.chunk_id: s.ref for s in result.sources}
        st.markdown("**Key points**")
        for claim in result.claims:
            cites = "".join(f"[{ref_of[c]}]" for c in claim.chunk_ids)
            tickers = ", ".join(claim.tickers) or "General"
            st.markdown(f"- **{tickers}**: {linkify(f'{claim.text} {cites}', refs, anchor)}", unsafe_allow_html=True)
    for caveat in result.caveats:
        st.caption(f":material/info: {caveat}")
    for warning in result.warnings:
        st.warning(warning, icon=":material/warning:")
    if result.usage:
        st.caption(usage_line(result.usage, turn.local, turn.seconds))
    actions, again = st.columns([4, 1], vertical_alignment="center")
    with actions, st.container(horizontal=True, vertical_alignment="center", gap="small"):
        st.feedback("thumbs", key=f"feedback-{i}")
        for name, (icon, title, _) in COMING_SOON.items():
            if st.button("", key=f"{name}-{i}", icon=icon, type="tertiary", help=title):
                coming_soon(name)
    if latest:
        again.button(
            "Regenerate", key="regenerate", icon=":material/refresh:", type="tertiary", help="Run this question again",
            on_click=st.session_state.__setitem__, args=("rerun-last", True),
        )
    return cited_refs(result)


def render_turn(i: int, turn: Turn, latest: bool) -> None:
    """Draw one stored exchange: the question, then the assistant's steps (collapsed), notices, answer and sources."""
    with st.chat_message("user"):
        st.markdown(f'<div class="question">{html.escape(turn.question)}</div>', unsafe_allow_html=True)
        if turn.detected:
            st.caption(f"Detected in the question: {filter_badges(turn.detected)}")
    with st.chat_message("assistant"):
        state = "error" if turn.error and not turn.hits else "complete"
        with st.status(turn.summary, state=state, expanded=False):
            for line in turn.steps:
                st.write(line)
        for notice in turn.notices:
            st.warning(notice, icon=":material/filter_alt_off:")
        if turn.error:
            st.error(turn.error)
        if not turn.hits:
            if not turn.error:
                st.warning("No matching passages. Try removing filters or rephrasing.")
            render_retrieval(turn.traces)
            return
        cited = render_result(i, turn, latest) if turn.result else set()
        if not turn.settings.generate:
            st.caption("Answer generation is off; showing the most relevant passages.")
        # details in pipeline order: searches -> ranked out -> cut by budget -> sent to the LLM -> what it returned
        render_retrieval(turn.traces)
        render_passages(
            turn.traces, turn.hits, turn.kept, turn.settings.mode, turn.settings.generate, turn.settings.budget,
            cited, anchor=f"passage-{i}", expanded=False,
        )
        if turn.result:
            with st.expander("Structured output (JSON)", icon=":material/data_object:"):
                st.json(turn.result.model_dump())


def render_welcome(retriever: HybridRetriever) -> None:
    st.markdown(
        '<div class="hero"><h2>What do you want to know about these filings?</h2>'
        f"<p>{len(retriever.tickers)} companies · 10-K and 10-Q filings · every answer cites its sources</p></div>",
        unsafe_allow_html=True,
    )


def render_examples() -> None:
    """Example questions: under the welcome text, and after the conversation so they stay one click away."""
    _, middle, _ = st.columns([1, 6, 1])
    with middle:
        st.caption("Try one of these")
        for i, example in enumerate(EXAMPLE_QUESTIONS):
            st.button(
                example, key=f"example-{i}", width="stretch", icon=":material/chat_bubble:",
                on_click=st.session_state.__setitem__, args=("pending", example),
            )


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
    """The chat page. `chat_model` / `context_budget` fix the model and budget (tests); otherwise the sidebar picks them."""
    turns: list[Turn] = st.session_state.setdefault("turns", [])
    st.title("SCiFi Assistant")
    st.caption(f"{len(retriever.docs):,} chunks · {len(retriever.tickers)} companies · each question is answered on its own, from the filings only")

    with st.sidebar:
        if st.button("New chat", key="new-chat", icon=":material/add_comment:", width="stretch"):
            turns.clear()
            for key in [k for k in st.session_state if str(k).startswith("feedback-")]:
                del st.session_state[key]
        fixed_model = getattr(chat_model, "model", type(chat_model).__name__) if chat_model else None
        generate, provider, model, k, budget = llm_settings(retriever, fixed_model, context_budget)

    # search settings apply only when the form is submitted, so changing them does not re-run (and re-pay for) a question
    with st.sidebar.form("settings"):
        st.header("Search")
        mode = st.radio("Mode", MODES, format_func=MODE_LABELS.get)
        tickers = st.multiselect("Companies", retriever.tickers)
        balanced = st.checkbox("Balance across companies", value=True, help="k passages per company when 2+ are selected or detected")
        years = st.multiselect("Fiscal years", retriever.fiscal_years)
        filing_type = st.selectbox("Filing type", ["Any", *retriever.filing_types])
        quarters = st.multiselect("Quarters (calendar, of period end)", retriever.quarters) if retriever.quarters else []
        detect = st.checkbox("Detect filters from the question", value=True, help="Companies, years, quarters, 10-K/10-Q")
        applied = st.form_submit_button("Apply and re-run", type="primary", key="apply", help="Re-runs the last question with all settings")

    settings = Settings(
        mode, k, budget, balanced, generate, provider, model,
        Filters(tickers, years, [filing_type] if filing_type != "Any" else [], quarters), detect,
    )
    question = st.chat_input("Ask about the filings, e.g. How has NVIDIA's revenue changed over the last two years?")
    question = question or st.session_state.pop("pending", None)
    rerun = bool(not question and turns and (st.session_state.pop("rerun-last", False) or applied))
    if rerun:
        question = turns.pop().question  # replace the last answer with a fresh one

    if not turns and not question:
        render_welcome(retriever)
    for i, turn in enumerate(turns):
        render_turn(i, turn, latest=not question and i == len(turns) - 1)
    if not question:
        render_examples()
    if question:
        with st.chat_message("user"):
            st.markdown(f'<div class="question">{html.escape(question)}</div>', unsafe_allow_html=True)
        with st.chat_message("assistant"):
            status = st.status("Working on it…", expanded=True)
            with status:
                turn = run_turn(question, settings, retriever, chat_model, status)
        turns.append(turn)
        st.rerun()  # redraw from the stored turn: one code path for new and earlier answers
    st.markdown(STYLE, unsafe_allow_html=True)
    st.html(CITE_SCRIPT, unsafe_allow_javascript=True)
