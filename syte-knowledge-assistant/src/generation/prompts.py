"""
Prompt Builder — Formats retrieved chunks into LLM prompts for Output A/B.
Handles context formatting with metadata headers so the LLM can cite sources.
"""

from pathlib import Path
from typing import Any

from loguru import logger


# Default prompt directory
PROMPT_DIR = Path(__file__).parent.parent.parent / "config" / "prompts"


def load_prompt_template(mode: str = "output_a") -> str:
    """Load a prompt template from file."""
    if mode == "output_a":
        path = PROMPT_DIR / "output_a_narrative.txt"
    elif mode == "output_b":
        path = PROMPT_DIR / "output_b_research_list.txt"
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'output_a' or 'output_b'.")

    if not path.exists():
        logger.warning(f"Prompt template not found: {path}. Using built-in default.")
        return _default_prompt(mode)

    return path.read_text(encoding="utf-8")


def format_context(results: list[dict[str, Any]]) -> str:
    """
    Format retrieved chunks into a context block for the LLM.
    Each chunk gets a clear header with source metadata.
    """
    context_parts = []

    for i, result in enumerate(results, 1):
        meta = result.get("metadata", {})
        doc_name = meta.get("doc_name", "Unbekannt")
        page = meta.get("page_number", "?")
        section = meta.get("section_title", "")
        score = result.get("rerank_score", result.get("score", 0))
        text = result.get("text", "")

        header = f"[DOKUMENT {i}: {doc_name} | Seite: {page}"
        if section:
            header += f" | Abschnitt: {section}"
        header += f" | Relevanz: {score:.2f}]"

        context_parts.append(f"{header}\n{text}")

    return "\n\n---\n\n".join(context_parts)


def build_prompt(
    query: str,
    results: list[dict[str, Any]],
    mode: str = "output_a",
) -> tuple[str, str]:
    """
    Build system prompt and user message for the LLM.

    Args:
        query: User question
        results: Retrieved + reranked chunks
        mode: "output_a" (narrative) or "output_b" (research list)

    Returns:
        (system_prompt, user_message)
    """
    template = load_prompt_template(mode)
    context = format_context(results)

    # Split template into system prompt (before {context}) and build user msg
    # The template contains {context} and {query} placeholders
    system_prompt = template.split("## KONTEXT-DOKUMENTE")[0].strip()

    user_message = (
        f"## KONTEXT-DOKUMENTE\n\n"
        f"Die folgenden {len(results)} Textabschnitte stammen aus der "
        f"Syte-Wissensbasis:\n\n{context}\n\n"
        f"## ANFRAGE\n\n{query}"
    )

    logger.info(
        f"Built prompt: mode={mode}, context_chunks={len(results)}, "
        f"context_chars={len(context)}, query='{query[:80]}...'"
    )
    return system_prompt, user_message


def format_sources_summary(results: list[dict[str, Any]]) -> str:
    """
    Format a short source summary for the UI sidebar.
    Shows which documents were used with page numbers.
    """
    seen = {}
    for r in results:
        meta = r.get("metadata", {})
        doc_name = meta.get("doc_name", "")
        page = meta.get("page_number", "?")
        score = r.get("rerank_score", r.get("score", 0))

        if doc_name not in seen:
            seen[doc_name] = {"pages": set(), "score": score, "is_distractor": meta.get("is_distractor", False)}
        seen[doc_name]["pages"].add(str(page))

    lines = []
    for doc_name, info in seen.items():
        pages = ", ".join(sorted(info["pages"], key=lambda x: int(x) if x.isdigit() else 999))
        flag = " ⚠️ DISTRACTOR" if info["is_distractor"] else ""
        lines.append(f"📄 **{doc_name}** (S. {pages}) — Score: {info['score']:.2f}{flag}")

    return "\n".join(lines)


def _default_prompt(mode: str) -> str:
    """Built-in fallback prompts."""
    if mode == "output_a":
        return (
            "Du bist ein Research-Assistant. Erstelle einen 1-Seiter auf Basis der Dokumente.\n"
            "Regeln: Jede Kernaussage braucht eine Fussnote [Dokumentname, S. X].\n"
            "Markiere: [QUELLE] = direkt aus Dokument, [TRANSFER] = Interpretation.\n\n"
            "{context}\n\nAnfrage: {query}"
        )
    else:
        return (
            "Du bist ein Research-Assistant. Erstelle eine priorisierte Quellenliste.\n"
            "Sortiere nach Relevanz. Markiere: ZUERST LESEN / ZUR VERTIEFUNG / HINTERGRUND.\n\n"
            "{context}\n\nAnfrage: {query}"
        )
