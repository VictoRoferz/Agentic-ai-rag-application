#!/usr/bin/env python3
"""
Syte Knowledge Assistant — Streamlit Chat Interface
=====================================================
Start with:  streamlit run scripts/app_streamlit.py --server.port 8080

Alternative to the Chainlit app — simpler setup, same functionality.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from loguru import logger

from src.pipeline import RAGPipeline

# ── Page Config ───────────────────────────────────────────────

st.set_page_config(
    page_title="Syte Knowledge Assistant",
    page_icon="🔍",
    layout="wide",
)

# ── Initialize Pipeline ──────────────────────────────────────

@st.cache_resource
def get_pipeline():
    """Initialize pipeline once (cached across reruns)."""
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    p = RAGPipeline()
    p.initialize()
    return p


pipeline = get_pipeline()

# ── Sidebar ──────────────────────────────────────────────────

with st.sidebar:
    st.title("🔍 Syte Knowledge Assistant")
    st.markdown("---")

    # Status
    status = pipeline.get_status()
    st.metric("Chunks in Wissensbasis", status["collection"]["points_count"])
    st.caption(f"Embedding: {status['embedding_model']}")
    st.caption(f"LLM: {status['llm_provider']}")

    st.markdown("---")

    # Output mode
    mode = st.radio(
        "Output-Modus:",
        options=["output_a", "output_b"],
        format_func=lambda x: "📝 Output A — 1-Seiter mit Fußnoten" if x == "output_a" else "📋 Output B — Priorisierte Quellenliste",
        index=0,
    )

    st.markdown("---")

    # File upload
    st.subheader("📄 Dokument hochladen")
    uploaded_file = st.file_uploader(
        "PDF, DOCX oder PPTX",
        type=["pdf", "docx", "pptx"],
        help="Dokument wird automatisch indexiert.",
    )

    is_distractor = st.checkbox("Als Stördokument markieren", value=False)

    if uploaded_file is not None:
        # Save to temp and ingest
        temp_path = PROJECT_ROOT / "data" / "inbox" / uploaded_file.name
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner(f"Indexiere {uploaded_file.name}..."):
            try:
                count = pipeline.ingest_file(temp_path, is_distractor=is_distractor)
                st.success(f"✅ {uploaded_file.name}: {count} Chunks indexiert")
            except Exception as e:
                st.error(f"❌ Fehler: {e}")

    st.markdown("---")

    # Reset
    if st.button("🗑️ Wissensbasis zurücksetzen", type="secondary"):
        pipeline.reset()
        st.warning("Wissensbasis gelöscht und neu erstellt.")
        st.rerun()

# ── Chat Interface ───────────────────────────────────────────

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("📚 Quellen anzeigen"):
                st.markdown(msg["sources"])

# Chat input
if prompt := st.chat_input("Stelle eine Frage an die Syte-Wissensbasis..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Suche und generiere..."):
            try:
                result = pipeline.query(prompt, mode=mode)

                # Display answer
                st.markdown(result["answer"])

                # Display sources
                if result["sources"]:
                    with st.expander("📚 Verwendete Quellen", expanded=False):
                        st.markdown(result["sources_summary"])
                        st.markdown("---")
                        for i, src in enumerate(result["sources"], 1):
                            meta = src["metadata"]
                            score = src.get("rerank_score", src.get("score", 0))
                            dist = " ⚠️ DISTRACTOR" if meta.get("is_distractor") else ""
                            st.markdown(
                                f"**[{i}] {meta['doc_name']}** | S. {meta['page_number']} | "
                                f"Score: {score:.3f}{dist}"
                            )
                            st.caption(src["text"][:300] + "...")

                # Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources_summary"],
                })

            except Exception as e:
                st.error(f"❌ Fehler: {e}")
                logger.error(f"Query failed: {e}")
