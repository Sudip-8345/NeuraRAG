"""
Evaluation pipeline — run test questions and score the RAG system.

Scoring:
  ✅ PASS    — Accurate, grounded, no hallucination
  ⚠️ PARTIAL — Partially correct or missing nuance
  ❌ FAIL    — Wrong, hallucinated, or missed entirely
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from rag.chain import ask
from rag.vectorstore import load_vectorstore, build_vectorstore
from rag.loader import load_documents
from rag.chunker import chunk_documents
from evaluation.questions import EVAL_QUESTIONS
import config


# --- Scoring helpers ---

def check_keywords(answer, keywords):
    """Return (found, missing) keyword lists."""
    answer_lower = answer.lower()
    found = [kw for kw in keywords if kw.lower() in answer_lower]
    missing = [kw for kw in keywords if kw.lower() not in answer_lower]
    return found, missing


def check_hallucination(answer, category):
    """Check for hallucination signals. Returns 'PASS', 'WARN', or 'FAIL'."""
    answer_lower = answer.lower()

    if category == "UNANSWERABLE":
        decline_phrases = [
            "not available", "not found", "no information", "don't have",
            "cannot answer", "not mentioned", "not in the provided", "not covered",
        ]
        return "PASS" if any(p in answer_lower for p in decline_phrases) else "FAIL"

    fabrication_signals = [
        "i think", "i believe", "probably", "it's likely",
        "i assume", "generally speaking", "in most companies",
    ]
    return "WARN" if any(s in answer_lower for s in fabrication_signals) else "PASS"


def score_answer(question_data, result):
    """Score a single answer. Returns a dict with scores and overall verdict."""
    answer = result["answer"]
    category = question_data["category"]
    keywords = question_data["expected_keywords"]

    found, missing = check_keywords(answer, keywords)
    kw_score = len(found) / len(keywords) if keywords else 1.0
    hall_check = check_hallucination(answer, category)

    # Determine overall verdict
    if category == "UNANSWERABLE":
        overall = "✅ PASS" if hall_check == "PASS" else "❌ FAIL"
    elif category == "PARTIALLY_ANSWERABLE":
        if kw_score >= 0.5:
            overall = "✅ PASS"
        elif kw_score >= 0.3 and hall_check != "FAIL":
            overall = "⚠️ PARTIAL"
        else:
            overall = "❌ FAIL"
    else:  # ANSWERABLE
        if kw_score >= 0.5 and hall_check == "PASS":
            overall = "✅ PASS"
        elif kw_score >= 0.25:
            overall = "⚠️ PARTIAL"
        else:
            overall = "❌ FAIL"

    return {
        "question_id": question_data["id"],
        "category": category,
        "keyword_score": round(kw_score, 2),
        "keywords_found": found,
        "keywords_missing": missing,
        "hallucination_check": hall_check,
        "overall": overall,
    }


# --- Main evaluation ---

def get_vectorstore():
    """Load or build vector store."""
    if os.path.exists(config.CHROMA_PERSIST_DIR):
        return load_vectorstore()
    print("Building vector store first...\n")
    docs = load_documents()
    chunks = chunk_documents(docs)
    return build_vectorstore(chunks)


def run_evaluation(prompt_version="v2", use_reranking=True):
    """Run all eval questions and print + save results."""
    print(f"\n{'=' * 60}")
    print(f"  EVALUATION — Prompt: {prompt_version} | Reranking: {use_reranking}")
    print(f"{'=' * 60}\n")

    store = get_vectorstore()
    results = []

    for q in EVAL_QUESTIONS:
        print(f"Q{q['id']}: {q['question']}")
        print(f"  Category: {q['category']}")

        try:
            rag_result = ask(store, q["question"], prompt_version, use_reranking)
            score = score_answer(q, rag_result)
            score["answer_preview"] = rag_result["answer"][:200]
            score["sources_returned"] = rag_result["sources"]
        except Exception as e:
            print(f"  ERROR: {e}")
            score = {
                "question_id": q["id"], "category": q["category"],
                "overall": "❌ FAIL", "error": str(e),
            }

        results.append(score)
        print(f"  Result: {score['overall']}")
        if "keyword_score" in score:
            print(f"  Keywords: {score['keyword_score']} | Hallucination: {score['hallucination_check']}")
        print()

    # Print summary
    pass_n = sum(1 for r in results if "PASS" in r["overall"])
    partial_n = sum(1 for r in results if "PARTIAL" in r["overall"])
    fail_n = sum(1 for r in results if "FAIL" in r["overall"])

    print(f"{'=' * 60}")
    print(f"  SUMMARY: ✅ {pass_n}  ⚠️ {partial_n}  ❌ {fail_n}  (total: {len(results)})")
    print(f"{'=' * 60}\n")

    # Save to file
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "evaluation_results")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"eval_{prompt_version}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Saved to: {out_file}\n")

    return results


def compare_prompts():
    """Run both prompt versions and print side-by-side comparison."""
    r1 = run_evaluation("v1")
    r2 = run_evaluation("v2")

    print(f"\n{'=' * 60}")
    print("  V1 vs V2 COMPARISON")
    print(f"{'=' * 60}\n")
    print(f"{'Question':<48} {'V1':<12} {'V2':<12}")
    print("─" * 72)

    for a, b in zip(r1, r2):
        q = EVAL_QUESTIONS[a["question_id"] - 1]["question"][:45]
        print(f"{q:<48} {a['overall']:<12} {b['overall']:<12}")

    v1_pass = sum(1 for r in r1 if "PASS" in r["overall"])
    v2_pass = sum(1 for r in r2 if "PASS" in r["overall"])
    print(f"\n  V1 Pass: {v1_pass}/{len(r1)}  |  V2 Pass: {v2_pass}/{len(r2)}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="NeuraRAG Evaluation")
    parser.add_argument("--prompt", choices=["v1", "v2"], default="v2")
    parser.add_argument("--compare", action="store_true", help="Compare v1 vs v2")
    parser.add_argument("--no-rerank", action="store_true")
    args = parser.parse_args()

    if args.compare:
        compare_prompts()
    else:
        run_evaluation(args.prompt, not args.no_rerank)
