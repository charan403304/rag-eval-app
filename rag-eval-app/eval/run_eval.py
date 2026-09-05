#!/usr/bin/env python3
"""
Run the full evaluation suite against data/eval/eval_set.json and print a
report. This is the file that matters most for the "portfolio" story: it
proves the RAG system's quality is measured, not assumed.

    python eval/run_eval.py                # keyword-based scoring only
    python eval/run_eval.py --llm-judge     # also score with Claude as judge
                                             # (requires ANTHROPIC_API_KEY)
    python eval/run_eval.py --json report.json   # also dump machine-readable report
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.metrics import (  # noqa: E402
    citation_validity,
    keyword_recall,
    llm_judge_score,
    refusal_correct,
    retrieval_precision_recall,
)
from src.config import EVAL_SET_PATH  # noqa: E402
from src.pipeline import RAGPipeline  # noqa: E402


def load_eval_set(path: Path = EVAL_SET_PATH) -> list[dict]:
    return json.loads(path.read_text())


def run(use_llm_judge: bool = False) -> dict:
    eval_set = load_eval_set()
    pipeline = RAGPipeline.from_corpus()

    rows = []
    for item in eval_set:
        response = pipeline.answer(item["query"])

        row = {
            "id": item["id"],
            "query": item["query"],
            "answerable": item["answerable"],
            "refused": response.refused,
            "refusal_correct": refusal_correct(response, item["answerable"]),
            "answer": response.answer,
            "citation_validity": citation_validity(response),
            "cost_usd": response.stats.cost_usd(),
            "total_ms": response.stats.total_ms,
        }

        if item["answerable"]:
            rm = retrieval_precision_recall(response, item["relevant_doc_ids"])
            row["retrieval_precision"] = rm.precision
            row["retrieval_recall"] = rm.recall
            row["retrieval_hit"] = rm.hit_at_k
            row["keyword_recall"] = keyword_recall(response, item["expected_keywords"])
            if use_llm_judge:
                row["llm_judge_score"] = llm_judge_score(
                    item["query"], response.answer, item["expected_keywords"]
                )
        rows.append(row)

    return summarize(rows)


def summarize(rows: list[dict]) -> dict:
    answerable_rows = [r for r in rows if r["answerable"]]
    unanswerable_rows = [r for r in rows if not r["answerable"]]

    summary = {
        "n_total": len(rows),
        "n_answerable": len(answerable_rows),
        "n_unanswerable": len(unanswerable_rows),
        "refusal_accuracy": mean(r["refusal_correct"] for r in rows) if rows else 0.0,
        "citation_validity_avg": mean(r["citation_validity"] for r in rows) if rows else 0.0,
        "avg_cost_usd_per_query": mean(r["cost_usd"] for r in rows) if rows else 0.0,
        "avg_latency_ms": mean(r["total_ms"] for r in rows) if rows else 0.0,
    }
    if answerable_rows:
        summary["retrieval_precision_avg"] = mean(r["retrieval_precision"] for r in answerable_rows)
        summary["retrieval_recall_avg"] = mean(r["retrieval_recall"] for r in answerable_rows)
        summary["retrieval_hit_rate"] = mean(r["retrieval_hit"] for r in answerable_rows)
        summary["keyword_recall_avg"] = mean(r["keyword_recall"] for r in answerable_rows)
        judge_scores = [r["llm_judge_score"] for r in answerable_rows if r.get("llm_judge_score")]
        if judge_scores:
            summary["llm_judge_avg"] = mean(judge_scores)

    summary["rows"] = rows
    return summary


def print_report(summary: dict) -> None:
    print("=" * 72)
    print("RAG EVALUATION REPORT")
    print("=" * 72)
    print(f"Questions evaluated:      {summary['n_total']} "
          f"({summary['n_answerable']} answerable, {summary['n_unanswerable']} unanswerable)")
    print(f"Refusal accuracy:         {summary['refusal_accuracy']:.1%}"
          "   (correctly refused OOD questions / correctly answered in-scope ones)")
    print(f"Citation validity:        {summary['citation_validity_avg']:.1%}"
          "   (citation markers pointing at real retrieved chunks)")
    if "retrieval_precision_avg" in summary:
        print(f"Retrieval precision:     {summary['retrieval_precision_avg']:.1%}")
        print(f"Retrieval recall:        {summary['retrieval_recall_avg']:.1%}")
        print(f"Retrieval hit rate:      {summary['retrieval_hit_rate']:.1%}")
        print(f"Keyword recall (answer): {summary['keyword_recall_avg']:.1%}")
    if "llm_judge_avg" in summary:
        print(f"LLM judge score (1-5):  {summary['llm_judge_avg']:.2f}")
    print(f"Avg cost / query:         ${summary['avg_cost_usd_per_query']:.6f}")
    print(f"Avg latency / query:      {summary['avg_latency_ms']:.1f} ms")
    print("=" * 72)
    print("\nPer-question detail:\n")
    for r in summary["rows"]:
        flag = "OK" if r["refusal_correct"] else "FAIL"
        extra = ""
        if r["answerable"]:
            extra = f" | retr.recall={r['retrieval_recall']:.2f} kw.recall={r['keyword_recall']:.2f}"
        print(f"[{flag}] {r['id']}: {r['query'][:60]}{'...' if len(r['query']) > 60 else ''}"
              f" | refused={r['refused']}{extra}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm-judge", action="store_true",
                         help="Also score answer correctness with Claude as judge (needs ANTHROPIC_API_KEY)")
    parser.add_argument("--json", type=str, default=None,
                         help="Path to also write the full report as JSON")
    args = parser.parse_args()

    summary = run(use_llm_judge=args.llm_judge)
    print_report(summary)

    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=2))
        print(f"\nFull report written to {args.json}")


if __name__ == "__main__":
    main()
