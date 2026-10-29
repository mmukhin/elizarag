"""Entry point: `chroma run --path data/chroma`, then `streamlit run streamlit_app.py`."""

import streamlit as st

from app.edgar.env import load_env
from app.edgar.hybrid import HybridRetriever
from app.edgar.vectorstore import get_client, get_embeddings, load_store
from app.ui import render

load_env()


@st.cache_resource(show_spinner="Loading collection from Chroma...", max_entries=2)
def load_retriever(collection: str, version: tuple[str, int]) -> HybridRetriever:
    """Cached per collection *version* (id, chunk count): a rebuild or an index still being written gets reloaded,
    so BM25 and company detection never run on a stale copy while vector search sees the live collection."""
    return HybridRetriever(load_store(collection, get_embeddings(), get_client()))


st.set_page_config(page_title="SEC filings Q&A", layout="wide")
try:
    collections = sorted(c.name for c in get_client().list_collections())
except Exception as exc:  # server not running / wrong CHROMA_URL
    st.error(f"Chroma server unreachable ({exc}). Start it first: `chroma run --path data/chroma`")
    st.stop()
if not collections:
    st.error("No Chroma collection found. Build one first: `python3 scripts/build_index.py --tickers AAPL TSLA JPM`")
    st.stop()

default = collections.index("10k") if "10k" in collections else 0
collection = st.sidebar.selectbox("Collection", collections, index=default)
live = get_client().get_collection(collection)
render(load_retriever(collection, (str(live.id), live.count())))
