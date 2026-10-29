# Eliza — SEC EDGAR RAG prototype

A chatbot that answers cross-company questions about SEC 10-K (and 10-Q) filings, grounded in and citing the filings themselves:

- *What are the primary risk factors facing Apple, Tesla and JPMorgan, and how do they compare?*
- *How has NVIDIA's revenue and growth outlook changed over the last two years?*
- *What regulatory risks do the major pharmaceutical companies face, and how are they addressing them?*

Pipeline: **load + clean → chunk (800 tokens) → embed (EmbeddingGemma) → ChromaDB (local server) → filters parsed from the question → hybrid (BM25 + vector) search → grounded LLM answer (structured JSON, cited) → Streamlit UI**.

Everything runs locally by default (Ollama for embeddings and the LLM); OpenAI is optional.

## Install

1. **Python 3.12** (pinned in `.tool-versions`; with asdf, prefix commands with `ASDF_PYTHON_VERSION=3.12.10` if `python3` errors out).
2. **Python packages:**
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. **[Ollama](https://ollama.com)**, running, with the two default models:
   ```bash
   ollama pull embeddinggemma   # embeddings (768-dim)
   ollama pull gemma4:e2b       # answer generation
   ```
4. **Corpus (not in this repo):** put the SEC filings (`<TICKER>_<10K|10Q>_..._full.txt`, public-domain text from [EDGAR](https://www.sec.gov/edgar/)) in `data/edgar_corpus/`. Without it, indexing, the chunking tests and the generation evals are skipped.
5. **Optional, OpenAI:** put `OPENAI_API_KEY=sk-...` in `.env` in the project root (git-ignored). Read by the UI, scripts and evals.

## Run it

```bash
chroma run --path data/chroma                              # Chroma server on :8000; keep it running
python3 scripts/build_index.py --tickers AAPL TSLA JPM     # once (~5 min); all 89 10-Ks: drop --tickers (~50 min)
streamlit run streamlit_app.py                              # http://localhost:8501
```

In the UI, tick **Generate answer with LLM** in the sidebar. From the command line:

```bash
python3 scripts/ask.py "Compare the main risk factors of Apple and Tesla" -k 3   # structured JSON on stdout
```

## Tests and evals

| What | Command | Needs |
|---|---|---|
| Code tests (offline, deterministic) | `pytest` | nothing |
| Live integration tests | `pytest -m slow` | Ollama + `embeddinggemma` + `gemma4:e2b` |
| All evals | `pytest -m eval` | Ollama + `gemma4:e2b`; retrieval suite also needs the `10k` collection |
| One eval suite | `pytest -m eval tests/evals/test_generation.py` | as above |
| Evals against another model | `LLM_MODEL=gemma4:e4b pytest -m eval` or `LLM_PROVIDER=openai pytest -m eval` | that model / `OPENAI_API_KEY` |
| Evals against a prompt version | `PROMPT_VERSION=v2 pytest -m eval tests/evals/test_generation.py` | as above |
| Compare prompt versions | `python3 scripts/compare_prompts.py --cases` | eval result files |

Each eval run prints pass rates per suite and per check, and writes `tests/evals/results/<timestamp>-<prompt version>.json`. Evals measure quality, not correctness, so some failures are expected and tracked as pass rates.

Baseline, `gemma4:e2b` on 2026-10-28: answerable 13/16, off_topic 6/10, not_in_passages 9/10, retrieval 8/9, filters 17/19 (2 xfail). Typical failures: answers without citations, a cited passage number that doesn't exist, and off-topic questions refused correctly but labelled `not_in_passages`.

### Iterating on prompts

The live prompts are `prompts/qa_system.md` and `prompts/qa_user.md`. Each iteration is a snapshot in `prompts/versions/<vN>/`, with notes at the top of the file in an HTML comment block (what changed, why, eval result). `load_prompt` strips `<!-- ... -->` blocks, so the model never sees the notes.

1. Copy the latest version to `prompts/versions/vN+1/`, edit it, and write the notes.
2. `PROMPT_VERSION=vN+1 pytest -m eval tests/evals/test_generation.py`
3. `python3 scripts/compare_prompts.py --cases` for a Markdown table of every version side by side, plus pass/fail per case.
4. Copy the winner over `prompts/qa_system.md` to make it live.

## Commands

| Task | Command |
|---|---|
| Build an index for a few companies | `python3 scripts/build_index.py --tickers AAPL TSLA JPM --collection demo` |
| Build an index for all 89 10-Ks (collection `10k`) | `python3 scripts/build_index.py` |
| Build an index with 10-Ks and 10-Qs | `python3 scripts/build_index.py --filing-types 10-K 10-Q --collection filings` |
| Build verbosely (per-filing details + one line per Ollama request) | `python3 scripts/build_index.py -v` |
| Change how often progress is logged (chunks per batch, default 128) | `python3 scripts/build_index.py --batch-size 50` |
| Import an old `InMemoryVectorStore` JSON dump (reuses its vectors) | `python3 scripts/import_json_index.py data/index/10k.json` |
| Query a collection | `python3 scripts/query.py "supply chain risks" --collection demo --tickers AAPL --years 2025 --quarters 3 -k 3` |
| Ask a question, get structured JSON (filters come from the question) | `python3 scripts/ask.py "Compare the main risk factors of Apple and Tesla" --collection demo -k 3` |
| Same, with the passage text in the JSON | `python3 scripts/ask.py "..." --with-text` |
| Same, with OpenAI | `LLM_PROVIDER=openai python3 scripts/ask.py "..." --collection demo` |
| Print chunks by id (rebuilt from the corpus, no Chroma) | `python3 scripts/show_chunk.py AAPL_10K_2025-10-31_full.txt#8` |
| Regenerate `data/companies.json` (names, aliases) | `python3 scripts/build_companies.py` |
| Chunk stats for all 10-Ks | `python3 scripts/chunk_report.py` |
| Chunk stats + sample chunks for chosen files | `python3 scripts/chunk_report.py --samples data/edgar_corpus/AAPL_10K_2025-10-31_full.txt` |
| UI tests only | `pytest tests/unit/test_ui.py` |
| Hybrid search tests only | `pytest tests/unit/test_hybrid.py` |

Measured on a local Ollama: 1,515 chunks (AAPL + TSLA + JPM, 9 filings) embed in about 4.5 minutes (~5.4 chunks/s), so the full corpus (16,005 chunks) takes roughly 50 minutes. The default collection is `10k`; rebuilding a collection replaces it.

| View the data | URL |
|---|---|
| Browse records (id, metadata, text) in the app | http://localhost:8501/chroma_data (or whatever port `streamlit run` prints) |
| Chroma REST API (Swagger) | http://localhost:8000/docs |
| List collections (JSON) | http://localhost:8000/api/v2/tenants/default_tenant/databases/default_database/collections |

The scripts and the UI talk to the Chroma server at `CHROMA_URL` (default `http://localhost:8000`). Data persists in `data/chroma/`.

---

# Reference

## Q&A UI and hybrid search

The page is a chat. Type in the input pinned at the bottom (or click one of the example questions on the welcome screen) and press Enter. Each answer appears as an assistant message:

- a **status box** listing what it did (filters detected, searches run, passages sent, answer checked), which collapses to a one-line summary when done;
- a **status badge** (Answered / Partially answered / Not in the filings / Off topic), the **answer** in larger type with `[n]` citations that jump to and highlight passage n, and **key points** per company;
- one quiet line with the model, tokens sent and received, cost and latency, then an action row: 👍/👎 **feedback**, **Regenerate**, and four planned actions that open a "Coming soon" dialog: copy with citations (for a memo), export tables to Excel/CSV (for the comps model), share a link to the answer and its evidence with the deal team, and flag a wrong answer (to feed the eval set). They are defined in `app/ui.py::COMING_SOON`;
- collapsible details in pipeline order: the **retrieval** trace, **candidates not used**, **top-k passages not sent**, **passages sent to the LLM**, and the **structured JSON** the model returned.

The example questions stay available under the conversation. Earlier questions and answers stay on the page, and redrawing them never re-runs search or the LLM. **New chat** clears the conversation. Each question is answered on its own: the assistant does not use earlier turns as context.

The sidebar has two parts:
- **Answer / Context** (top): one **Model** picker (`ollama: gemma4:e2b`, `openai: gpt-6-luna`, `openai: gpt-4o-mini`, plus `$LLM_PROVIDER`/`$LLM_MODEL` if set to something else). It defaults to **gpt-6-luna** when `OPENAI_API_KEY` is set (e.g. in `.env`), otherwise to local gemma, or to `$LLM_PROVIDER`/`$LLM_MODEL` if given. Picking a model immediately updates the context-window info and resets *passages per company* and the *token budget* to that model's defaults (see [Context budget per model](#answer-generation)), without re-running the question.
- **Search** (a form): mode, companies, balance, years, filing type, quarters, filter detection. These take effect when you press **Apply and re-run**, which re-runs the current question with all settings.

| Mode | How it ranks |
|---|---|
| **Hybrid** (default) | BM25 keyword ranking and vector ranking, fused with reciprocal rank fusion (`1/(60+rank)` summed over both lists). Catches exact terms and figures (`"23.5 billion"`, `tariffs`) as well as paraphrases. |
| Vector only | EmbeddingGemma cosine similarity. |
| Keyword (BM25) only | Needs no embedding backend, so it still works if Ollama is down. |

- **Detect filters from the question** (on by default) turns companies, years, quarters and 10-K/10-Q named in the question into filters (see [Filters from the question](#filters-from-the-question)). A sidebar choice wins over what was detected, field by field; the detected filters are shown under the question.
- **Balance across selected companies** (on by default when 2+ companies are selected or detected) retrieves *k* passages per company instead of *k* overall, so one company can't crowd out the rest in comparison questions.
- Each result shows its fused score and its rank in each ranker (`vector #2 · keyword #1`).
- **Retrieval** shows, per search (one per company when balanced), the top-k and the exact requests: the Chroma `collection.query` (collection, text that gets embedded, `n_results`, `where` filter) and the BM25 query (tokens, filter, corpus size).
- Below the answer, every section is a collapsible header, starts collapsed, and follows the pipeline: the **Retrieval** trace, **candidates not used** (ranked below top-k), **top-k passages not sent** (cut by the token budget), **passages sent to the LLM** (with tokens used of the budget; each passage is its own expander), each with vector/BM25 rank, cosine and BM25 scores, RRF score and token cost, then the **structured JSON**. Clicking a `[n]` in the answer opens the sections around passage n, scrolls to it and highlights it (a small fixed script in `app/ui.py`, `CITE_SCRIPT`, since browsers don't open a closed expander for a link inside it).
- **Example questions** (collapsed above the question box) fill in the assignment's sample questions. With generation on, the page shows tokens sent/received, cost and LLM latency, the answer, key points with citations, any warnings, the structured JSON, and the numbered passages `[n]` the model saw. If generation fails (Ollama stopped, no API key), the error is shown and the passages are still displayed.
- If the sidebar's Companies (or years…) contradict the question, a warning says which one is used; a company the question names that isn't in the collection is reported instead of silently searching everything.
- The UI loads whichever collection the provider in `EMBEDDING_PROVIDER` can query, so start it with the same provider you built with.

Code: `app/edgar/hybrid.py` (`HybridRetriever`), `app/ui.py` (`render`), `streamlit_app.py` (entry point).

## Answer generation

`scripts/ask.py` and the UI retrieve passages with hybrid search, then ask an LLM to answer **only from those passages**. The model must return JSON matching a schema, and the result is validated before anyone sees it.

```bash
python3 scripts/ask.py "What tariff risks do Apple and Tesla face?" --collection demo \
    --tickers AAPL TSLA --years 2025 -k 3          # stdout is pure JSON, logs go to stderr (-v)
```

**Structured output** (`app/edgar/qa.py`, pydantic):

| Field | Meaning |
|---|---|
| `status` | `answered`, `partial` (passages cover part of the question), `not_in_passages`, or `off_topic` (not about the filings: recipes, weather, coding, prompt injection) |
| `answer` | short answer with `[n]` passage citations; `n` is `sources[].ref` |
| `claims[]` | `{text, chunk_ids[], tickers[]}`, one or more per company so comparisons line up. `tickers` come from the cited chunks, not from the model |
| `caveats[]` | what the passages do not cover |
| `sources[]` | the passages the model saw: `ref` (1..n), `chunk_id`, ticker, filing type, fiscal year, filing date, source file, chunk index. **No text**: look it up by `chunk_id` (the Chroma id, `source_file#chunk_index`), or pass `--with-text` to `ask.py` |
| `filters` | the metadata filters the retrieval used |
| `warnings[]` | problems found by validation (below) |
| `usage` | `{model, calls, input_tokens, output_tokens, seconds, cost_usd}` of the LLM call (`null` if the model wasn't called; `cost_usd` null if the price is unknown) |
| `model`, `question` | provenance |

The model itself returns only `status`, `answer`, `claims[{text, passages[]}]` and `caveats`: passage *numbers* and judgements, never passage content. The prompt labels passages `[1]..[n]` because small models copy short numbers reliably and mangle long ids; code maps the numbers to chunk ids.

**Context budget per model** (`app/edgar/llm.py::PROFILES`). How much retrieved text to send depends on the model:

| Model | Context window | Passage budget (default) | Passages per company (k) | Reserved for the answer |
|---|---|---|---|---|
| `gemma4:e2b` | 16,384 (the `num_ctx` we request) | 8,000 | 5 | 1,024 (also Ollama's `num_predict`) |
| `gpt-6-luna` | ~1.1M | 24,000 | 8 | 4,000 |
| `gpt-4o-mini` | 128k | 16,000 | 6 | 2,000 |
| other models | 16,384 (assumed) | 8,000 | 5 | 1,024 |

A 2B model gets worse as the prompt grows, so gemma gets ~9 passages. gpt-6-luna could take 1.1M tokens, but ~24k (3 companies × 8 passages × ~800 tokens) covers comparison questions for about $0.003; more mostly adds noise, latency and "lost in the middle" risk. The sidebar shows the window and reserve, lets you change k and the budget (up to the window minus the reserve and ~1k of prompt, capped at 64k), and estimates how many passages fit. `ask.py` uses the same defaults unless `-k` / `--max-context-tokens` are given.

**Guard rails** applied after the model answers: citations to passages that don't exist are removed and reported, uncited claims and uncited `answered`/`partial` answers are flagged, and an `off_topic` or `not_in_passages` answer that still makes claims is flagged. Passages are fitted to the model's token budget (gemma 8,000, gpt-6-luna 24,000) taking each company's best passage first, so one company can't push another out; dropped passages are reported. With no retrieved passages the model is not called at all.

**Dates and coverage.** Each passage header carries its period end (`[1] NVDA · 10-K · FY2025 · period ending 2025-01-26 · filed 2025-02-26`), and the user prompt can include today's date (`$today`) and a per-company summary of which filings and periods the passages cover (`$coverage`). From prompt v4 on, the model reads relative time ("last two years") against today's date, names exact fiscal years and periods with every figure, checks the passages cover every company, period and figure asked about, and answers `partial`, naming what's missing, when they don't. For example, asked in October 2026 about NVIDIA's last two years with only FY2024–FY2025 filings, it should say FY2026 isn't in the passages. Generation evals use a fixed date (2026-10-29), so results don't drift with the calendar.

**Prompts** live in [`prompts/`](prompts/) as plain Markdown, loaded by `app/edgar/prompts.py` (`$question`, `$context`, `$today`, `$coverage` placeholders; a version may use any subset): `qa_system.md` (grounding rules and output contract) and `qa_user.md` (passages + question). Edit them without touching code. `PROMPT_VERSION=vN` loads `prompts/versions/vN/` instead (see [Iterating on prompts](#iterating-on-prompts)).

### LLM providers

`app/edgar/llm.py::get_chat_model` selects the backend (separate from the embedding provider, so you can mix them):

| Setting | Meaning | Default |
|---|---|---|
| `LLM_PROVIDER` | `ollama` or `openai` (or `--llm-provider`) | `ollama` |
| `LLM_MODEL` | model name (or `--llm-model`) | `gemma4:e2b` / `gpt-6-luna` |
| `OLLAMA_URL` | Ollama base URL | `http://localhost:11434` |
| `OPENAI_API_KEY` | required for `openai`; read from `.env` in the project root if not set in the shell | — |
| `LLM_PRICE_INPUT` / `LLM_PRICE_OUTPUT` | $ per 1M tokens, for models missing from `llm.PRICES` | — |
| `PROMPT_VERSION` | prompt snapshot in `prompts/versions/` | live `prompts/` |

- **Ollama** is called on `/api/chat` with the JSON schema as `format`, `temperature: 0`, `think: false` (faster) and `num_ctx` / `num_predict` from the model's context profile (16,384 / 1,024 for gemma). Ollama's default 4k window would otherwise silently cut off the passages.
- **OpenAI** uses LangChain `ChatOpenAI.with_structured_output(schema, include_raw=True)` (the raw message carries token usage). No `temperature` is sent unless you pass one, because `gpt-6-luna` accepts only the default.
- Every call records **usage**: tokens sent and received, latency and cost, summed over retries. It's in `QAResult.usage` and shown under the answer in the UI. Prices come from `llm.PRICES` (gpt-6-luna $0.10 / $0.50, gpt-4o-mini $0.15 / $0.60 per 1M input / output tokens, from third-party listings, Oct 2026); local Ollama is $0.
- Output that fails schema validation is retried once, then raises `StructuredOutputError`.
- To add a provider, subclass `ChatModel` and implement `_complete(system, user, schema)`, then add a branch to `get_chat_model`.

## Filters from the question

`app/edgar/query_filters.py::parse_filters(question, companies, years_by_ticker)` turns a question into a `Filters` (`tickers`, `fiscal_years`, `filing_types`, `quarters`). It is deterministic (regexes, no model). One `Filters` object feeds both the Chroma `where` clause and the BM25 filter.

| In the question | Becomes |
|---|---|
| Company names and aliases from `data/companies.json` (Apple, Coke, J&J, Facebook, Chase…) | `tickers`, in the order the question names them |
| Uppercase tickers (`NVDA`), cashtags (`$TSLA`) | `tickers`. One-letter tickers (`T`, `V`) only as cashtags or via an alias (AT&T, Visa) |
| `2024`, `FY2023`, `FY24`, `2022 to 2024`, `between 2023 and 2025` | `fiscal_years` (years far beyond the index, like "net zero by 2050", are ignored) |
| `latest`, `most recent`, `last two years` | the newest year(s) indexed for each named company |
| `Q2`, `2024Q2`, `third quarter` | `quarters` |
| `10-K`, `annual report` / `10-Q`, `quarterly` | `filing_types` |

`resolve_filters` merges the question's filters with explicit ones (sidebar, CLI flags; explicit wins per field) and reports conflicts: the UI warns when, say, the question names NVDA but the sidebar says BA. Companies are detected against **all** of `data/companies.json`, not just the indexed ones, so a company missing from the collection is reported instead of silently falling back to an unfiltered search. The UI caches the loaded collection per collection id and chunk count, so a rebuilt (or still-building) index is reloaded rather than detecting against a stale company list.

`data/companies.json` is generated by `scripts/build_companies.py` from the filing headers plus a hand-curated alias table in that script. **Ambiguous aliases** (Apple, Target, Visa, Oracle, Meta, Chase, Coke…) match only when capitalised, so "apple pie" and "sales target" don't filter.

**Quarters are calendar quarters of the period end**, matching the corpus `Quarter:` header (filings without it get it from `period_end`). Apple's 10-Q for the period ending 2025-03-29 is `2025` / `Q1`, although Apple calls it fiscal Q2. `fiscal_year` is also the year of the period end. Collections built before quarters were stored have no `quarter` metadata, so rebuild before filtering on quarters.

## Embedding providers

`app/edgar/vectorstore.py::get_embeddings` picks the backend. Default is local EmbeddingGemma on Ollama.

| Setting | Meaning | Default |
|---|---|---|
| `EMBEDDING_PROVIDER` | `ollama` or `openai` (or `--provider` on the scripts) | `ollama` |
| `EMBEDDING_MODEL` | model name for the chosen provider | `embeddinggemma` / `text-embedding-3-small` |
| `OLLAMA_URL` | Ollama base URL | `http://localhost:11434` |
| `OPENAI_API_KEY` | required for `openai` | — |

```bash
# external API example
EMBEDDING_PROVIDER=openai python3 scripts/build_index.py --collection openai
```

Vectors from different providers are not comparable. Build the collection and query it with the same provider. To add another provider, add a branch in `get_embeddings` that returns any LangChain `Embeddings`.

EmbeddingGemma needs task prefixes (`task: search result | query: …` for queries, `title: none | text: …` for documents). `GemmaEmbeddings` adds them, because Ollama does not.

## How it works

```
data/edgar_corpus/*_10K_*.txt (+ *_10Q_* with --filing-types)
   │  app/edgar/loader.py     parse header, drop XBRL soup + TOC, NBSP → space
   ▼
Filing(meta, body)
   │  app/edgar/chunking.py   RecursiveCharacterTextSplitter, 800 tokens / 100 overlap
   ▼
list[Document]   metadata: ticker, company, cik, filing_type, filing_date, fiscal_year,
   │                       period_end, quarter, source_file, url, chunk_index, n_chunks,
   │                       start_index, token_count
   │  app/edgar/vectorstore.py   GemmaEmbeddings → ChromaStore (Chroma collection, cosine HNSW,
   │                             id = source_file#chunk_index)
   ▼
search(store, query, k, Filters(tickers=…, fiscal_years=…, filing_types=…, quarters=…))
```

- **Header metadata** comes from the `Key: value` block above the `====` line. `fiscal_year` is the year of `Report Period`, else of the date in the filing URL, else the filing year.
- **Cleaning** is deterministic and conservative. It removes the one-line XBRL tag soup (~7k tokens per filing) and the pipe-table table of contents, which would otherwise match "risk factors" queries.
- **Chunking** is plain size-based chunking, not Item/TOC-aware. The source text is flattened (headings are glued to the previous sentence), so the splitter falls back to sentence boundaries. `start_index` is an offset into the cleaned body (LangChain's own `add_start_index` returns `-1` with token-based overlap, so offsets are computed in `chunk_filing`).

## Corpus size

| | Value |
|---|---|
| 10-K filings | 89 (54 companies) |
| Chunks | 16,005 |
| Tokens | ~10.1M (cl100k) |
| Median chunk | ~750 tokens |
| Examples (chunks) | AAPL ~78, TSLA ~145, NVDA 111–125, PFE ~300, JPM ~593 |

## Test and eval details

```
tests/
  unit/          code tests: offline, deterministic, the default `pytest` run
  integration/   live Ollama plumbing (`-m slow`)
  evals/         answer and retrieval quality on a live model (`-m eval`), scored, with a JSON report
  fakes.py       offline test doubles (embedder, scripted chat model, sample chunks)
  helpers.py     paths, `ollama_has(model)`
  fixtures/expected.json
```

| Code tests (`tests/unit/`) | Covers |
|---|---|
| `test_chunking.py` | AAPL, TSLA, JPM: header metadata, noise removal, chunk size limit, offsets and coverage, known content present, determinism; header parsing, cleaning, calendar quarter |
| `test_hybrid.py` | Tokenizer, BM25/vector/hybrid ranking with hand-set vectors (fusion, keyword-only and vector-only hits), filters in every mode incl. quarter, per-company balancing, blank query, empty store |
| `test_ui.py` | Streamlit `AppTest`: welcome screen and example prompts, chat history (earlier answers kept, unique citation anchors), new chat, regenerate, status badge, assistant steps, linked citations and opened cited passages, feedback, collapsible sections, tokens/cost/latency line, settings applied only on submit, model picker and per-model context defaults, sidebar and detected filters (conflicts, unindexed companies), balanced comparison, keyword mode, no-results and backend-down handling |
| `test_qa.py` | Prompt loading (notes stripped for every version), context numbering and chunk ids, token-budget selection, validation per status (incl. off-topic), claim → chunk id/ticker resolution, `answer_question` with a scripted model |
| `test_llm.py` | Structured-output retry/raise, Ollama request payload via mocked HTTP, OpenAI path with a fake client, provider selection |
| `test_vectorstore.py` | Search ranking, `Filters` → Chroma `where` and the BM25 predicate (kept in sync), build/reload a collection, precomputed vectors, EmbeddingGemma prefixes and batching |
| `test_query_filters.py` | `parse_filters` table: aliases, ambiguous words, ticker vs word, question order, years, ranges, relative years, quarters, filing types; every company's name, aliases and ticker select only that company; every pair of companies; `resolve_filters` overrides and conflicts |
| `test_scorers.py` | The eval scorers on hand-made results, and that every eval case file is well formed (chunk ids exist) |

Generation evals **pin their passages by chunk id** and rebuild them from `data/edgar_corpus/` (chunking is deterministic), so they need only the LLM, not a built index, and don't shift when retrieval changes.

| Suite (`cases/*.json`) | What | Checks |
|---|---|---|
| `answerable` (16) | single- and multi-company questions with a distractor passage | status, every claim cited, citations valid, each named company has a claim, `must_mention` facts, cites the gold chunk, **every figure appears in a passage** |
| `off_topic` (10) | recipes ("apple pie" with Apple passages), weather, code, trivia, prompt injection | `status=off_topic`, no claims or citations, doesn't do the task |
| `not_in_passages` (10) | company not in corpus, year/figure missing, one of two companies, false premise, future prediction | status, caveat, no invented figures |
| `retrieval` (9) | questions against the `10k` collection (cases for tickers not indexed are skipped) | expected chunk in top-k; per-company recall when balanced |
| `company_purity` (54) | for every company in `data/companies.json`: "How has {name}'s revenue … last two years?" and "What are the main risk factors {name} describes?", run question → `resolve_filters` → hybrid search exactly like the UI | every passage belongs to that company (records unfiltered purity too: 0.92 on average, IBM 0.3, so the filter matters) |
| `filters` (19) | realistic phrasings for `parse_filters` | exact match per field |

Result files hold the model, prompt version, every case, the answer and check details. Cases the current approach is known to fail carry `"xfail": "reason"`. To add a case, find chunks with `scripts/query.py`, read them with `scripts/show_chunk.py`, and add the ids to a case file. `test_scorers.py` checks that the ids exist.

Expected chunking values live in `tests/fixtures/expected.json`. Code tests use a deterministic offline embedder and a scripted chat model, so they never need a model.

## Known limitations

- No Item-level labels (Item 1A, 7, …). A "risk factors" query relies on similarity, and only company, year, quarter and filing type can be filtered.
- `parse_filters` has no sector mapping ("pharmaceutical companies" doesn't select PFE, MRK…), and an ambiguous company name capitalised at the start of a sentence ("Apple pie recipe") still matches. Chunk offsets are stored, so Item tagging can be added later without re-chunking.
- A multi-company question retrieved with a single `k` can be dominated by one company. Use "Balance across selected companies" in the UI, or `HybridRetriever.search(..., per_ticker=True)`.
- Answer quality depends on the model and on retrieval. `gemma4:e2b` (2B) gives schema-valid, cited answers but can be shallow or over-generalise across companies; broad questions retrieve generic chunks. Narrow with company, fiscal-year and filing-type filters, or try a larger model (`--llm-model`).
- Retrieval can mix fiscal years for one company; filter by year for "latest" questions. Local generation takes roughly 20-40 s for ~6 passages.
- Some filings are thin: IBM's 10-K incorporates its annual report by reference (~25k tokens).
- 10-Qs are not in the default `10k` collection; build with `--filing-types 10-K 10-Q` to include them (157 more filings).
- `HybridRetriever` loads every chunk from Chroma into memory at startup for BM25 (Chroma only does the vector side).
- `data/index/*.json` are the old `InMemoryVectorStore` dumps (about 18 KB per chunk), kept only for `import_json_index.py`.
