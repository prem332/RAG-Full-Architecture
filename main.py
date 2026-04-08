import os
import time
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

os.environ["ANONYMIZED_TELEMETRY"] = "False"

from rag_pipeline import (
    build_rag_app,
    rag_query,
    load_vectorstore,
    load_pdf,
    chunk_text,
    build_vectorstore,
    PDF_PATH,
    TOP_K_RETRIEVE,
    TOP_K_RERANK,
)

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="RAG Chatbot", page_icon="📄", layout="wide")
st.title("📄 RAG Chatbot")
st.caption("Bi-encoder retrieval → Cross-encoder reranking → Groq LLM")

# ── Cache Resources ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Building RAG pipeline...")
def get_rag_app():
    return build_rag_app()

@st.cache_resource(show_spinner="Loading vectorstore...")
def get_collection():
    return load_vectorstore()

rag_app    = get_rag_app()
collection = get_collection()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    st.info(f"""
    **Model**: `llama-3.1-8b-instant`
    **Embeddings**: `all-MiniLM-L6-v2`
    **Reranker**: `ms-marco-MiniLM-L-6-v2`
    **Retrieve**: top-{TOP_K_RETRIEVE} → rerank → top-{TOP_K_RERANK}
    """)
    show_context = st.toggle("Show retrieved context", value=False)
    show_scores  = st.toggle("Show reranking scores",  value=False)
    st.divider()
    if st.button("🗑️ Clear chat history"):
        st.session_state.messages = []
        st.rerun()

# ── Chat UI ───────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and show_context and "context" in msg:
            with st.expander("📎 Retrieved context (after reranking)"):
                for i, (chunk, score) in enumerate(zip(msg["context"], msg["ce_scores"])):
                    st.markdown(f"**Chunk {i+1}** — CE Score: `{score:.4f}`")
                    st.text(chunk[:300] + "..." if len(chunk) > 300 else chunk)
                    st.divider()

if query := st.chat_input("Ask something about the document..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving → Reranking → Generating..."):
            t0          = time.time()
            final_state = rag_query(query, rag_app)
            elapsed     = time.time() - t0

        st.markdown(final_state["answer"])
        st.caption(f"⏱ {elapsed:.1f}s")

        if show_context:
            with st.expander("📎 Retrieved context (after reranking)"):
                for i, (chunk, score) in enumerate(zip(final_state["reranked_docs"], final_state["ce_scores"])):
                    st.markdown(f"**Chunk {i+1}** — CE Score: `{score:.4f}`")
                    st.text(chunk[:300] + "..." if len(chunk) > 300 else chunk)
                    st.divider()

        if show_scores:
            import pandas as pd
            df = pd.DataFrame({
                "Rank"    : range(1, len(final_state["ce_scores"]) + 1),
                "CE Score": [f"{s:.4f}" for s in final_state["ce_scores"]],
            })
            st.dataframe(df, hide_index=True)

    st.session_state.messages.append({
        "role"     : "assistant",
        "content"  : final_state["answer"],
        "context"  : final_state["reranked_docs"],
        "ce_scores": final_state["ce_scores"],
    })