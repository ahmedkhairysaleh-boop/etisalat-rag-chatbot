"""Run the test set and report how the system did.

    python -m scripts.evaluate --retrieval-only    # free, no LLM calls
    python -m scripts.evaluate                     # full run, paced for the free tier

Checks three things per question: did the answer contain the expected facts,
did the agent take the expected route, and how close was the best retrieved
chunk.

The retrieval-only mode skips the language model entirely. It cannot check
answer content, but it measures distances and routing, which is enough to
tune the threshold without consuming quota.

The full run is paced: the Gemini free tier allows 5 requests per minute, so
without a delay the run fails partway through.
"""

import argparse
import logging
import time
from datetime import datetime

from src.config import PROJECT_ROOT
from src.graph.build import ask
from src.graph.nodes import classify_smalltalk, retrieve
from src.language import detect_language
from src.logging_config import setup_logging
from tests.questions import QUESTIONS

log = logging.getLogger(__name__)

REPORT_PATH = PROJECT_ROOT / "logs" / "evaluation.md"

# free tier allows 5 requests per minute; 3 seconds leaves margin
SECONDS_BETWEEN_CALLS = 8


def actual_route(result, question):
    """Work out which branch the agent took, from the final state."""
    if classify_smalltalk(question):
        return "smalltalk"
    return "generate" if result.get("has_context") else "no_context"


def check_answer(answer, must_contain):
    """True if the answer contains any expected keyword.

    'any' rather than 'all' because some entries list alternatives - an Arabic
    answer may write MiFi in Latin or Arabic script and either is correct.
    """
    if not must_contain:
        return True
    return any(word in answer for word in must_contain)


def blank_result(case, route, distance=None):
    """A result row for retrieval-only mode, where no answer text exists."""
    return {
        "label": case["label"],
        "question": case["question"],
        "answer": "",
        "content_ok": True,          # not measurable without the model
        "route_ok": route == case["route"],
        "expected_route": case["route"],
        "actual_route": route,
        "distance": distance,
    }


def run_retrieval_only(case):
    """Measure routing and distance without calling the language model."""
    question = case["question"]

    if classify_smalltalk(question):
        return blank_result(case, "smalltalk")

    state = retrieve({"question": question,
                      "language": detect_language(question)})

    hits = state["hits"]
    route = "generate" if state["has_context"] else "no_context"
    distance = round(hits[0]["distance"], 3) if hits else None

    return blank_result(case, route, distance)


def run_full(case):
    """Run the whole agent, including generation.

    A failed call is recorded rather than raised, so one rate-limit error does
    not discard the results already collected.
    """
    try:
        result = ask(case["question"])
    except RuntimeError as error:
        log.warning("case failed: %s", error)
        return {
            "label": case["label"],
            "question": case["question"],
            "answer": f"[failed: {error}]",
            "content_ok": False,
            "route_ok": False,
            "expected_route": case["route"],
            "actual_route": "error",
            "distance": None,
        }

    answer = result.get("answer", "")
    hits = result.get("hits", [])
    route = actual_route(result, case["question"])

    return {
        "label": case["label"],
        "question": case["question"],
        "answer": answer,
        "content_ok": check_answer(answer, case["must_contain"]),
        "route_ok": route == case["route"],
        "expected_route": case["route"],
        "actual_route": route,
        "distance": round(hits[0]["distance"], 3) if hits else None,
    }


def print_report(results, retrieval_only):
    """Console summary."""
    print()
    print(f"{'content':<9} {'route':<7} {'dist':<7} label")
    print("-" * 70)

    for row in results:
        content = "-" if retrieval_only else (
            "PASS" if row["content_ok"] else "FAIL")
        route = "ok" if row["route_ok"] else "WRONG"
        distance = f"{row['distance']:.3f}" if row["distance"] else "-"
        print(f"{content:<9} {route:<7} {distance:<7} {row['label']}")

    route_passed = sum(row["route_ok"] for row in results)
    total = len(results)

    print("-" * 70)

    if not retrieval_only:
        content_passed = sum(row["content_ok"] for row in results)
        print(f"content: {content_passed}/{total} passed")

    print(f"routing: {route_passed}/{total} correct")

    print_distance_summary(results)


def print_distance_summary(results):
    """The distribution the relevance threshold should be set from."""
    answerable = [row["distance"] for row in results
                  if row["expected_route"] == "generate" and row["distance"]]

    print()
    print("retrieval distances")

    if answerable:
        print(f"  questions reaching retrieval: {min(answerable):.3f} - "
              f"{max(answerable):.3f}  (n={len(answerable)})")


def write_report(results):
    """Save a markdown report for the documentation deliverable."""
    content_passed = sum(row["content_ok"] for row in results)
    route_passed = sum(row["route_ok"] for row in results)
    total = len(results)

    lines = [
        "# Test Report",
        "",
        f"Run: {datetime.now():%Y-%m-%d %H:%M}",
        "",
        f"- Content checks passed: {content_passed}/{total}",
        f"- Routing correct: {route_passed}/{total}",
        "",
        "| Result | Route | Distance | Question | Answer |",
        "|---|---|---|---|---|",
    ]

    for row in results:
        content = "pass" if row["content_ok"] else "FAIL"
        route = row["actual_route"] if row["route_ok"] else (
            f"**{row['actual_route']}** (expected {row['expected_route']})")
        distance = f"{row['distance']:.3f}" if row["distance"] else "-"
        answer = row["answer"].replace("\n", " ").replace("|", "/")[:120]

        lines.append(
            f"| {content} | {route} | {distance} | {row['question']} | {answer} |"
        )

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nreport written to {REPORT_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate the chatbot.")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="measure retrieval and routing without calling "
                             "the language model")
    args = parser.parse_args()

    setup_logging("evaluate.log")

    mode = "retrieval only" if args.retrieval_only else "full"
    print(f"running {len(QUESTIONS)} test questions ({mode})...")

    if not args.retrieval_only:
        estimate = len(QUESTIONS) * SECONDS_BETWEEN_CALLS // 60
        print(f"paced for the free tier, roughly {estimate} minutes\n")

    results = []
    for position, case in enumerate(QUESTIONS, start=1):
        print(f"  {position}/{len(QUESTIONS)}  {case['label']}")
        results.append(
            run_retrieval_only(case) if args.retrieval_only else run_full(case)
        )

        if not args.retrieval_only and position < len(QUESTIONS):
            time.sleep(SECONDS_BETWEEN_CALLS)

    print_report(results, args.retrieval_only)

    if not args.retrieval_only:
        write_report(results)


if __name__ == "__main__":
    main()