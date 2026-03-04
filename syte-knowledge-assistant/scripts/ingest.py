#!/usr/bin/env python3
"""
Ingest documents into the Syte Knowledge Base.

Usage:
    # Ingest a single file
    python scripts/ingest.py --file path/to/document.pdf

    # Ingest all files in a directory
    python scripts/ingest.py --dir data/test/relevant/

    # Ingest distractors (marked in metadata)
    python scripts/ingest.py --dir data/test/distractors/ --distractor

    # Ingest both relevant + distractors for testing
    python scripts/ingest.py --test-setup

    # Reset knowledge base and re-ingest
    python scripts/ingest.py --reset --test-setup

    # Show collection info
    python scripts/ingest.py --info
"""

import argparse
import sys
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into Syte Knowledge Base")
    parser.add_argument("--file", type=str, help="Ingest a single file")
    parser.add_argument("--dir", type=str, help="Ingest all files in a directory")
    parser.add_argument("--distractor", action="store_true", help="Mark as distractor documents")
    parser.add_argument("--test-setup", action="store_true", help="Ingest both relevant + distractors from data/test/")
    parser.add_argument("--reset", action="store_true", help="Reset knowledge base before ingesting")
    parser.add_argument("--info", action="store_true", help="Show collection info")
    args = parser.parse_args()

    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")

    from src.pipeline import RAGPipeline
    pipeline = RAGPipeline()
    pipeline.initialize()

    # --- Info ---
    if args.info:
        status = pipeline.get_status()
        print("\n📊 Knowledge Base Status:")
        print(f"   Collection: {status['collection']['name']}")
        print(f"   Points:     {status['collection']['points_count']}")
        print(f"   Status:     {status['collection']['status']}")
        print(f"   Embedding:  {status['embedding_model']}")
        print(f"   LLM:        {status['llm_provider']}")
        print(f"   Reranker:   {'enabled' if status['reranker_enabled'] else 'disabled'}")
        return

    # --- Reset ---
    if args.reset:
        confirm = input("⚠️  Delete ALL vectors and reset? (y/N): ")
        if confirm.lower() == "y":
            pipeline.reset()
            print("✅ Knowledge base reset.")
        else:
            print("Cancelled.")
            return

    # --- Ingest single file ---
    if args.file:
        path = Path(args.file)
        if not path.exists():
            print(f"❌ File not found: {path}")
            sys.exit(1)
        count = pipeline.ingest_file(path, is_distractor=args.distractor)
        label = " (DISTRACTOR)" if args.distractor else ""
        print(f"✅ Ingested '{path.name}'{label}: {count} chunks")

    # --- Ingest directory ---
    elif args.dir:
        path = Path(args.dir)
        if not path.is_dir():
            print(f"❌ Directory not found: {path}")
            sys.exit(1)
        count = pipeline.ingest_directory(path, is_distractor=args.distractor)
        label = " (DISTRACTORS)" if args.distractor else ""
        print(f"✅ Ingested directory '{path}'{label}: {count} chunks total")

    # --- Test setup: ingest both ---
    elif args.test_setup:
        relevant_dir = PROJECT_ROOT / "data" / "test" / "relevant"
        distractor_dir = PROJECT_ROOT / "data" / "test" / "distractors"

        total = 0
        if relevant_dir.is_dir() and any(relevant_dir.iterdir()):
            count = pipeline.ingest_directory(relevant_dir, is_distractor=False)
            print(f"✅ Relevant docs: {count} chunks")
            total += count
        else:
            print(f"⚠️  No files in {relevant_dir}")

        if distractor_dir.is_dir() and any(distractor_dir.iterdir()):
            count = pipeline.ingest_directory(distractor_dir, is_distractor=True)
            print(f"✅ Distractor docs: {count} chunks")
            total += count
        else:
            print(f"⚠️  No files in {distractor_dir}")

        print(f"\n📊 Total ingested: {total} chunks")
        status = pipeline.get_status()
        print(f"   Collection points: {status['collection']['points_count']}")

    else:
        parser.print_help()

    # Show final status
    if args.file or args.dir or args.test_setup:
        print("\n💡 Run the app:  python scripts/app.py")


if __name__ == "__main__":
    main()
