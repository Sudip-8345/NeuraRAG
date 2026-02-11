"""Evaluation pipeline — score the RAG system using RAGAS."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas.llms import llm_factory
from google import genai
from ragas.embeddings import GoogleEmbeddings
from ragas.embeddings.base import embedding_factory
from agent.workflow import ask
from agent.nodes import init_tool
from rag.vectorstore import load_vectorstore, build_vectorstore
from rag.loader import load_documents
from rag.chunker import chunk_documents
from rag.embeddings import get_embedding_function
from evaluation.questions import EVAL_QUESTIONS
import config

client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

# --- Helpers ---

def get_vectorstore():
    if os.path.exists(config.CHROMA_PERSIST_DIR):
        return load_vectorstore()
    print("Building vector store first...\n")
    docs = load_documents()
    chunks = chunk_documents(docs)
    return build_vectorstore(chunks)


def collect_rag_responses(prompt_version="v2"):
    questions, answers, contexts, ground_truths = [], [], [], []

    for q in EVAL_QUESTIONS:
        print(f"  Q{q['id']}: {q['question'][:60]}...")
        try:
            result = ask(q["question"], prompt_version)
            questions.append(q["question"])
            answers.append(result["answer"])
            contexts.append(result.get("context", [""]))
            ground_truths.append(q["ground_truth"])
        except Exception as e:
            print(f"    ERROR: {e}")
            questions.append(q["question"])
            answers.append(f"Error: {e}")
            contexts.append([""])
            ground_truths.append(q["ground_truth"])

    return {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }


# --- Main evaluation ---

def run_evaluation(prompt_version="v2", use_reranking=True):
    print(f"\n{'=' * 60}")
    print(f"  RAGAS EVALUATION — Prompt: {prompt_version} | Reranking: {use_reranking}")
    print(f"{'=' * 60}\n")

    store = get_vectorstore()
    init_tool(store, use_reranking)

    # Collect RAG responses
    print("Running questions through RAG pipeline...\n")
    data = collect_rag_responses(prompt_version)
    dataset = Dataset.from_dict(data)

    # Configure RAGAS to use Google Gemini
    evaluator_llm = llm_factory(
        "gemini-2.0-flash",
        provider="google",
        client=client
    )
    embeddings = GoogleEmbeddings(client=client, model="gemini-embedding-001")
    metrics = [
        Faithfulness(llm=evaluator_llm),
        AnswerRelevancy(llm=evaluator_llm, embeddings=embeddings),
        ContextPrecision(llm=evaluator_llm),
        ContextRecall(llm=evaluator_llm),
    ]

    # Run RAGAS evaluation
    print("Running RAGAS evaluation ...\n")
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
    )

    # Print results
    print(f"\n{'=' * 60}")
    print(f"  RAGAS SCORES (Prompt: {prompt_version})")
    print(f"{'=' * 60}")
    for metric, score in result.items():
        print(f"  {metric:<25} {score:.4f}")
    print(f"{'=' * 60}\n")

    # Save per-question details
    df = result.to_pandas()
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "evaluation_results")
    os.makedirs(out_dir, exist_ok=True)

    # Save CSV
    csv_file = os.path.join(out_dir, f"ragas_{prompt_version}.csv")
    df.to_csv(csv_file, index=False)
    print(f"  Details saved to: {csv_file}")

    return result


def compare_prompts():
    r1 = run_evaluation("v1")
    r2 = run_evaluation("v2")

    print(f"\n{'=' * 60}")
    print("  V1 vs V2 COMPARISON")
    print(f"{'=' * 60}")
    print(f"  {'Metric':<25} {'V1':>8} {'V2':>8}")
    print(f"  {'─' * 41}")
    for metric in r1:
        print(f"  {metric:<25} {r1[metric]:>8.4f} {r2[metric]:>8.4f}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="NeuraRAG RAGAS Evaluation")
    parser.add_argument("--prompt", choices=["v1", "v2"], default="v2")
    parser.add_argument("--compare", action="store_true", help="Compare v1 vs v2")
    parser.add_argument("--no-rerank", action="store_true")
    args = parser.parse_args()

    if args.compare:
        compare_prompts()
    else:
        run_evaluation(args.prompt, not args.no_rerank)
