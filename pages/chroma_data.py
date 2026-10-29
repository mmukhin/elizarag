"""Browse the raw records (id, metadata, text) stored in the local Chroma server."""

import pandas as pd
import streamlit as st

from app.edgar.vectorstore import get_client

PAGE_SIZE = 200

st.set_page_config(page_title="Chroma data", layout="wide")
st.title("Chroma data")
try:
    client = get_client()
    collections = sorted(c.name for c in client.list_collections())
except Exception as exc:
    st.error(f"Chroma server unreachable ({exc}). Start it first: `chroma run --path data/chroma`")
    st.stop()
if not collections:
    st.info("No collections yet.")
    st.stop()

name = st.sidebar.selectbox("Collection", collections)
collection = client.get_collection(name, embedding_function=None)
ticker = st.sidebar.text_input("Ticker filter (optional)").strip().upper()
where = {"ticker": ticker} if ticker else None
total = len(collection.get(where=where, include=[])["ids"]) if where else collection.count()
st.caption(f"collection `{name}` · id `{collection.id}` · {total:,} records · {collection.configuration_json['hnsw']['space']} distance")
if not total:
    st.stop()

page = st.sidebar.number_input("Page", 1, (total - 1) // PAGE_SIZE + 1, 1)
rows = collection.get(where=where, limit=PAGE_SIZE, offset=(page - 1) * PAGE_SIZE, include=["documents", "metadatas"])
st.dataframe(
    pd.DataFrame([{"id": i, **m, "text": d} for i, m, d in zip(rows["ids"], rows["metadatas"], rows["documents"])]),
    hide_index=True,
    width="stretch",
)
