#!/usr/bin/env python3
"""
Query the Syte Knowledge Base from the command line.

Usage:
    # Interactive mode
    python scripts/query.py

    # Single query
    python scripts/query.py --question "What are the key AI use cases in pharma?"

    # Output B (research list)
    python scripts/query.py --question "..." --mode output_b

    # Show sources detail
    python scripts/query.py --question "..." --show-sources
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger


def print_result(result: dict, show_sources: bool = False):
    """Pretty-print a query result."""
    mode_label = "📝 Output A (1-Seiter)" if result["mode"] == "output_a" else "📋 Output B (Quellenliste)"

    print(f"\n{'='*70}")
    print(f"  {mode_label}")
    print(f"  Frage: {result['query']}")
    print(f"{'='*70}\n")
    print(result["answer"])
    print(f"\n{'─'*70}")
    print(f"📚 Verwendete Quellen:\n")
    print(result["sources_summary"])

    if show_sources:
        print(f"\n{'─'*70}")
        print(f"🔍 Detail-Quellen ({len(result['sources'])} Chunks):\n")
        for i, src in enumerate(result["sources"], 1):
            meta = src["metadata"]
            score = src.get("rerank_score", src.get("score", 0))
            dist = " ⚠️ DISTRACTOR" if meta.get("is_distractor") else ""
            print(f"  [{i}] {meta['doc_name']} | S. {meta['page_number']} | "
                  f"Score: {score:.3f}{dist}")
            print(f"      {src['text'][:150]}...")
            print()


def interactive_mode(pipeline):
    """Interactive query loop."""
    print("\n🤖 Syte Knowledge Assistant — Interactive Mode")
    print("─" * 50)
    print("Commands:")
    print("  /a  → Switch to Output A (1-Seiter)")
    print("  /b  → Switch to Output B (Quellenliste)")
    print("  /s  → Toggle source details")
    print("  /info → Show collection info")
    print("  /quit → Exit")
    print("─" * 50)

    mode = "output_a"
    show_sources = False

    while True:
        try:
            question = input(f"\n{'[A]' if mode == 'output_a' else '[B]'} Frage: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not question:
            continue

        if question == "/quit":
            break
        elif question == "/a":
            mode = "output_a"
            print("→ Modus: Output A (1-Seiter)")
            continue
        elif question == "/b":
            mode = "output_b"
            print("→ Modus: Output B (Quellenliste)")
            continue
        elif question == "/s":
            show_sources = not show_sources
            print(f"→ Quellen-Detail: {'AN' if show_sources else 'AUS'}")
            continue
        elif question == "/info":
            status = pipeline.get_status()
            print(f"   Points: {status['collection']['points_count']}")
            print(f"   Model:  {status['embedding_model']}")
            continue

        # Execute query
        print("\n⏳ Suche und generiere...")
        result = pipeline.query(question, mode=mode)
        print_result(result, show_sources=show_sources)


def main():
    parser = argparse.ArgumentParser(description="Query the Syte Knowledge Base")
    parser.add_argument("--question", "-q", type=str, help="Question to ask")
    parser.add_argument("--mode", "-m", type=str, default="output_a", choices=["output_a", "output_b"])
    parser.add_argument("--show-sources", "-s", action="store_true", help="Show detailed sources")
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")

    from src.pipeline import RAGPipeline
    pipeline = RAGPipeline()
    pipeline.initialize()

    if args.question:
        result = pipeline.query(args.question, mode=args.mode)
        print_result(result, show_sources=args.show_sources)
    else:
        interactive_mode(pipeline)


if __name__ == "__main__":
    main()
