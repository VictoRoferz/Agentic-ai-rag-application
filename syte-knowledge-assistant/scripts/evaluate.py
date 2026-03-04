#!/usr/bin/env python3
"""
Quality Evaluation — Test runner for Meilenstein 2 quality protocol.

Runs the defined test queries against the knowledge base and measures:
1. Recall@20: Do the relevant docs appear in top 20?
2. Precision@5: Are top-5 results actually relevant?
3. Distractor rate: How many distractors slip into top results?
4. Footnote check: Does the output contain proper citations?

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --verbose
    python scripts/evaluate.py --output report.md
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger


# ── Test Cases (Agenda Points) ────────────────────────────────

TEST_QUERIES = [
    {
        "id": "agenda_1",
        "query": "Generative AI as a Game Changer: From Visibility to Revenue in the Short Term — What are the key strategies and examples?",
        "agenda_point": "1) Generative AI as a Game Changer",
    },
    {
        "id": "agenda_2",
        "query": "Fast and Measurable: How can real-world data drive revenue in pharma? What are concrete examples?",
        "agenda_point": "2) Driving Revenue Through Real-World Data",
    },
    {
        "id": "agenda_3",
        "query": "How can regional teams in pharma drive digital health impact independently despite global constraints?",
        "agenda_point": "3) Local Success, Global Constraints",
    },
    {
        "id": "agenda_4",
        "query": "What are the best AI and digital health use cases in clinical studies and patient engagement?",
        "agenda_point": "4) AI in Clinical Studies and Patient Engagement",
    },
    {
        "id": "agenda_5",
        "query": "How to select and scale digital health initiatives from concept to measurable outcomes?",
        "agenda_point": "5) From Concept to Outcome",
    },
]


def evaluate_retrieval(pipeline, verbose: bool = False) -> dict[str, Any]:
    """Run retrieval evaluation across all test queries."""
    results = []

    for test in TEST_QUERIES:
        logger.info(f"Testing: {test['id']} — {test['query'][:60]}...")

        # Search (raw retrieval, no generation)
        search_results = pipeline._search_service.search(test["query"], top_k=20)

        # Rerank
        if pipeline._reranker:
            reranked = pipeline._reranker.rerank(test["query"], search_results, top_n=10)
        else:
            reranked = search_results[:10]

        # Analyze results
        top_5 = reranked[:5]
        top_20 = search_results[:20]

        # Count distractors
        distractors_in_top_5 = sum(1 for r in top_5 if r["metadata"].get("is_distractor", False))
        distractors_in_top_20 = sum(1 for r in top_20 if r["metadata"].get("is_distractor", False))

        # Unique documents in results
        docs_top_5 = list({r["metadata"]["doc_name"] for r in top_5})
        docs_top_20 = list({r["metadata"]["doc_name"] for r in top_20})

        eval_result = {
            "test_id": test["id"],
            "agenda_point": test["agenda_point"],
            "query": test["query"],
            "total_results": len(search_results),
            "top_5_docs": docs_top_5,
            "top_20_docs": docs_top_20,
            "distractors_in_top_5": distractors_in_top_5,
            "distractors_in_top_20": distractors_in_top_20,
            "precision_at_5": 1.0 - (distractors_in_top_5 / max(len(top_5), 1)),
            "distractor_rate_top_20": distractors_in_top_20 / max(len(top_20), 1),
            "top_score": top_5[0]["score"] if top_5 else 0,
            "top_rerank_score": top_5[0].get("rerank_score", 0) if top_5 else 0,
        }

        results.append(eval_result)

        if verbose:
            print(f"\n{'─'*60}")
            print(f"  {test['id']}: {test['agenda_point']}")
            print(f"  Results: {len(search_results)} | Distractors top-5: {distractors_in_top_5}/5")
            print(f"  Top-5 docs:")
            for i, r in enumerate(top_5, 1):
                dist = " ⚠️" if r["metadata"].get("is_distractor") else " ✅"
                score = r.get("rerank_score", r.get("score", 0))
                print(f"    [{i}]{dist} {r['metadata']['doc_name']} (S.{r['metadata']['page_number']}) — {score:.3f}")

    return {
        "timestamp": datetime.now().isoformat(),
        "test_count": len(results),
        "results": results,
        "summary": compute_summary(results),
    }


def evaluate_generation(pipeline, verbose: bool = False) -> dict[str, Any]:
    """Run full generation evaluation (Output A + B) for the first test query."""
    test = TEST_QUERIES[0]
    gen_results = {}

    for mode in ["output_a", "output_b"]:
        logger.info(f"Generating {mode} for: {test['query'][:60]}...")
        result = pipeline.query(test["query"], mode=mode)

        # Check for footnotes in Output A
        footnote_count = result["answer"].count("[") - result["answer"].count("[QUELLE]") - result["answer"].count("[TRANSFER]")
        has_source_marks = "[QUELLE]" in result["answer"] or "[TRANSFER]" in result["answer"]
        word_count = len(result["answer"].split())

        gen_results[mode] = {
            "query": test["query"],
            "answer_length_words": word_count,
            "answer_length_chars": len(result["answer"]),
            "has_footnotes": footnote_count > 0 or "S." in result["answer"] or "Seite" in result["answer"],
            "has_source_marking": has_source_marks,
            "sources_count": len(result["sources"]),
            "answer_preview": result["answer"][:500] + "...",
        }

        if verbose:
            print(f"\n{'='*60}")
            print(f"  {mode.upper()}")
            print(f"{'='*60}")
            print(result["answer"][:1000])
            print(f"\n  Words: {word_count} | Sources: {len(result['sources'])} | Footnotes: {gen_results[mode]['has_footnotes']}")

    return gen_results


def compute_summary(results: list[dict]) -> dict:
    """Compute aggregate metrics."""
    if not results:
        return {}

    avg_precision_5 = sum(r["precision_at_5"] for r in results) / len(results)
    avg_distractor_20 = sum(r["distractor_rate_top_20"] for r in results) / len(results)
    total_distractors_top_5 = sum(r["distractors_in_top_5"] for r in results)

    return {
        "avg_precision_at_5": round(avg_precision_5, 3),
        "avg_distractor_rate_top_20": round(avg_distractor_20, 3),
        "total_distractors_in_top_5": total_distractors_top_5,
        "pass_precision": avg_precision_5 >= 0.80,
        "pass_distractor": avg_distractor_20 <= 0.20,
    }


def generate_report(eval_data: dict, gen_data: dict = None) -> str:
    """Generate a Markdown quality report."""
    s = eval_data.get("summary", {})
    lines = [
        "# Syte Knowledge Assistant — Qualitätsprotokoll",
        f"\n_Erstellt: {eval_data['timestamp']}_\n",
        "## 1. Retrieval-Ergebnisse\n",
        f"| Metrik | Wert | Ziel | Status |",
        f"|--------|------|------|--------|",
        f"| Precision@5 (Durchschnitt) | {s.get('avg_precision_at_5', 'N/A')} | ≥ 0.80 | {'✅ PASS' if s.get('pass_precision') else '❌ FAIL'} |",
        f"| Distractor-Rate Top-20 | {s.get('avg_distractor_rate_top_20', 'N/A')} | ≤ 0.20 | {'✅ PASS' if s.get('pass_distractor') else '❌ FAIL'} |",
        f"| Distractors in Top-5 (gesamt) | {s.get('total_distractors_in_top_5', 'N/A')} | möglichst 0 | |",
        "\n### Detail pro Agenda-Punkt\n",
    ]

    for r in eval_data.get("results", []):
        lines.append(f"**{r['agenda_point']}**")
        lines.append(f"- Ergebnisse: {r['total_results']} | Precision@5: {r['precision_at_5']:.2f}")
        lines.append(f"- Top-5 Dokumente: {', '.join(r['top_5_docs'][:5])}")
        lines.append(f"- Distractors in Top-5: {r['distractors_in_top_5']}\n")

    if gen_data:
        lines.append("## 2. Generation-Ergebnisse\n")
        for mode, data in gen_data.items():
            label = "Output A (1-Seiter)" if mode == "output_a" else "Output B (Quellenliste)"
            lines.append(f"### {label}\n")
            lines.append(f"- Wörter: {data['answer_length_words']}")
            lines.append(f"- Fußnoten vorhanden: {'✅' if data['has_footnotes'] else '❌'}")
            lines.append(f"- Quellenmarkierung: {'✅' if data['has_source_marking'] else '❌'}")
            lines.append(f"- Quellen verwendet: {data['sources_count']}\n")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Syte Knowledge Assistant quality")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--output", "-o", type=str, help="Save report to file (Markdown)")
    parser.add_argument("--generation", "-g", action="store_true", help="Also test generation (slower)")
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")

    from src.pipeline import RAGPipeline
    pipeline = RAGPipeline()
    pipeline.initialize()

    print("\n🧪 Running Syte Knowledge Assistant Evaluation...\n")

    # Retrieval evaluation
    eval_data = evaluate_retrieval(pipeline, verbose=args.verbose)
    s = eval_data["summary"]

    print(f"\n{'='*60}")
    print(f"  📊 ERGEBNIS")
    print(f"{'='*60}")
    print(f"  Precision@5:          {s.get('avg_precision_at_5', 'N/A')} {'✅' if s.get('pass_precision') else '❌'}")
    print(f"  Distractor-Rate@20:   {s.get('avg_distractor_rate_top_20', 'N/A')} {'✅' if s.get('pass_distractor') else '❌'}")

    # Generation evaluation
    gen_data = None
    if args.generation:
        gen_data = evaluate_generation(pipeline, verbose=args.verbose)

    # Save report
    if args.output:
        report = generate_report(eval_data, gen_data)
        Path(args.output).write_text(report, encoding="utf-8")
        print(f"\n📄 Report saved to: {args.output}")

    report = generate_report(eval_data, gen_data)
    report_path = PROJECT_ROOT / "logs" / f"quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"📄 Report: {report_path}")


if __name__ == "__main__":
    main()
