"""
Syte Knowledge Assistant — Chainlit Chat Interface
====================================================
Start with:  chainlit run scripts/app.py --port 8080

Features:
- Chat-based interaction
- Toggle Output A (1-Seiter) / Output B (Quellenliste)
- Source panel showing used documents
- Streaming responses
- Document upload for ingestion
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import chainlit as cl
from loguru import logger

from src.pipeline import RAGPipeline

# ── Global Pipeline ───────────────────────────────────────────

pipeline: RAGPipeline = None


@cl.on_chat_start
async def on_chat_start():
    """Initialize the pipeline and welcome the user."""
    global pipeline

    if pipeline is None:
        pipeline = RAGPipeline()
        pipeline.initialize()

    status = pipeline.get_status()
    points = status["collection"]["points_count"]

    # Set default mode
    cl.user_session.set("mode", "output_a")

    # Welcome message
    await cl.Message(
        content=(
            f"## 🔍 Syte Knowledge Assistant\n\n"
            f"**Wissensbasis:** {points} Chunks indexiert\n\n"
            f"Stelle eine Frage und ich suche die passenden Dokumente mit Quellenangaben.\n\n"
            f"**Befehle:**\n"
            f"- `/a` — Output A: Narrativer 1-Seiter mit Fußnoten\n"
            f"- `/b` — Output B: Priorisierte Quellenliste\n"
            f"- `/info` — Status der Wissensbasis\n\n"
            f"_Aktueller Modus: **Output A** (1-Seiter)_"
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """Handle user messages."""
    global pipeline
    text = message.content.strip()

    # ── Commands ──
    if text == "/a":
        cl.user_session.set("mode", "output_a")
        await cl.Message(content="✅ Modus gewechselt: **Output A** (Narrativer 1-Seiter)").send()
        return

    if text == "/b":
        cl.user_session.set("mode", "output_b")
        await cl.Message(content="✅ Modus gewechselt: **Output B** (Priorisierte Quellenliste)").send()
        return

    if text == "/info":
        status = pipeline.get_status()
        await cl.Message(
            content=(
                f"📊 **Status:**\n"
                f"- Chunks: {status['collection']['points_count']}\n"
                f"- Embedding: {status['embedding_model']}\n"
                f"- LLM: {status['llm_provider']}\n"
                f"- Reranker: {'✅' if status['reranker_enabled'] else '❌'}"
            )
        ).send()
        return

    # ── File upload handling ──
    if message.elements:
        for element in message.elements:
            if hasattr(element, "path") and element.path:
                try:
                    count = pipeline.ingest_file(element.path)
                    await cl.Message(
                        content=f"✅ **{element.name}** indexiert: {count} Chunks hinzugefügt."
                    ).send()
                except Exception as e:
                    await cl.Message(
                        content=f"❌ Fehler beim Indexieren von {element.name}: {e}"
                    ).send()
        if not text:
            return

    # ── RAG Query ──
    mode = cl.user_session.get("mode", "output_a")
    mode_label = "📝 Output A (1-Seiter)" if mode == "output_a" else "📋 Output B (Quellenliste)"

    # Start streaming response
    msg = cl.Message(content="")
    await msg.send()

    try:
        sources_shown = False
        full_answer = ""

        for chunk in pipeline.query_stream(text, mode=mode):
            if chunk["type"] == "sources":
                # Show sources as elements
                source_text = chunk["sources_summary"]
                sources = chunk["sources"]

                # Add source elements
                elements = []
                seen_docs = set()
                for src in sources:
                    doc_name = src["metadata"]["doc_name"]
                    if doc_name not in seen_docs:
                        seen_docs.add(doc_name)
                        page = src["metadata"]["page_number"]
                        score = src.get("rerank_score", src.get("score", 0))
                        is_dist = " ⚠️" if src["metadata"].get("is_distractor") else ""
                        elements.append(
                            cl.Text(
                                name=f"{doc_name} (S. {page})",
                                content=f"Score: {score:.3f}{is_dist}\n\n{src['text'][:500]}...",
                                display="side",
                            )
                        )

                msg.elements = elements
                sources_shown = True

            elif chunk["type"] == "token":
                full_answer += chunk["content"]
                await msg.stream_token(chunk["content"])

        # Append source summary at bottom
        if sources_shown:
            footer = f"\n\n---\n_Modus: {mode_label} | Quellen im Seitenpanel →_"
            await msg.stream_token(footer)

        await msg.update()

    except Exception as e:
        logger.error(f"Query failed: {e}")
        await cl.Message(
            content=f"❌ Fehler bei der Verarbeitung: {str(e)}"
        ).send()
