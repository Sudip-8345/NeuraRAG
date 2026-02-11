"""
DeepEval evaluation pipeline for NeuraRAG (OpenRouter Free Tier Edition).

Usage:
    python evaluation/deepeval_eval.py --prompt v2
    python evaluation/deepeval_eval.py --compare
    python evaluation/deepeval_eval.py --prompt v1 --no-rerank
"""

import os
import sys
import json
import re
from dotenv import load_dotenv

from openai import AsyncOpenAI, OpenAI
from deepeval.models.base_model import DeepEvalBaseLLM

load_dotenv()

if not os.getenv("OPENROUTER_API_KEY"):
    print("CRITICAL ERROR: OPENROUTER_API_KEY is missing from your .env file.")
    sys.exit(1)

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)

from agent.workflow import ask
from agent.nodes import init_tool
from rag.vectorstore import load_vectorstore, build_vectorstore
from rag.loader import load_documents
from rag.chunker import chunk_documents
from evaluation.questions import EVAL_QUESTIONS
import config


class OpenRouterJudge(DeepEvalBaseLLM):
    """
    A custom wrapper for OpenRouter to act as the evaluator for DeepEval.
    Uses 'google/gemini-2.0-flash-lite-preview-02-05:free' by default.
    """
    def __init__(self, model_name="google/gemini-2.0-flash-001"):
        self.model_name = model_name
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1"
        
        # OpenRouter-specific headers for Free Tier reliability
        self.headers = {
            "HTTP-Referer": "https://localhost:3000",  # Required for rankings
            "X-Title": "NeuraRAG Eval",                # Required for rankings
        }

        # Initialize clients
        self.async_client = AsyncOpenAI(
            base_url=self.base_url, 
            api_key=self.api_key,
            default_headers=self.headers
        )
        self.sync_client = OpenAI(
            base_url=self.base_url, 
            api_key=self.api_key,
            default_headers=self.headers
        )

    def load_model(self):
        return self.async_client

    def _clean_json(self, content: str) -> str:
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(json)?", "", content, flags=re.IGNORECASE)
            content = content.rstrip("`")
        return content.strip()

    def generate(self, prompt: str) -> str:
        chat_completion = self.sync_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model_name,
        )
        return self._clean_json(chat_completion.choices[0].message.content)

    async def a_generate(self, prompt: str) -> str:
        chat_completion = await self.async_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model_name,
        )
        return self._clean_json(chat_completion.choices[0].message.content)

    def get_model_name(self):
        return f"OpenRouter ({self.model_name})"


# --- HELPER FUNCTIONS ---

def _get_vectorstore():
    if os.path.exists(config.CHROMA_PERSIST_DIR):
        return load_vectorstore()
    print("Building vector store first...\n")
    docs = load_documents()
    chunks = chunk_documents(docs)
    return build_vectorstore(chunks)


def _build_test_cases(prompt_version="v2"):
    test_cases = []

    for q in EVAL_QUESTIONS:
        print(f"  Q{q['id']}: {q['question'][:60]}...")
        try:
            result = ask(q["question"], prompt_version)
            tc = LLMTestCase(
                input=q["question"],
                actual_output=result["answer"],
                expected_output=q["ground_truth"],
                retrieval_context=result.get("context", []),
            )
        except Exception as e:
            print(f"    ERROR: {e}")
            tc = LLMTestCase(
                input=q["question"],
                actual_output=f"Error: {e}",
                expected_output=q["ground_truth"],
                retrieval_context=[],
            )
        test_cases.append(tc)

    return test_cases


# --- EVALUATION ---

def run_evaluation(prompt_version="v2", use_reranking=True):
    print(f"\n{'=' * 60}")
    print(f"  DEEPEVAL EVALUATION — Prompt: {prompt_version} | Reranking: {use_reranking}")
    print(f"{'=' * 60}\n")

    store = _get_vectorstore()
    init_tool(store, use_reranking)

    print("Running questions through RAG pipeline...\n")
    test_cases = _build_test_cases(prompt_version)

    # --- INITIALIZE OPENROUTER JUDGE ---
    judge_model = OpenRouterJudge(model_name="google/gemini-2.0-flash-001")

    metrics = [
        AnswerRelevancyMetric(model=judge_model, threshold=0.6),
        FaithfulnessMetric(model=judge_model, threshold=0.6),
        ContextualPrecisionMetric(model=judge_model, threshold=0.6),
        ContextualRecallMetric(model=judge_model, threshold=0.6),
    ]

    # Run evaluation
    print(f"Running DeepEval evaluation using {judge_model.get_model_name()}...\n")
    results = evaluate(
        test_cases=test_cases,
        metrics=metrics,
    )

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"  DEEPEVAL SCORES (Prompt: {prompt_version})")
    print(f"{'=' * 60}")
    
    final_scores = {}
    for metric in metrics:
        print(f"  {metric.__class__.__name__:<30} {metric.score:.4f}  {'PASS' if metric.is_successful() else 'FAIL'}")
        final_scores[metric.__class__.__name__] = metric.score
        
        if metric.reason:
            print(f"    Reason: {metric.reason[:100]}...")

    print(f"{'=' * 60}\n")
    return final_scores


def compare_prompts():
    r1 = run_evaluation("v1")
    r2 = run_evaluation("v2")

    print(f"\n{'=' * 60}")
    print("  V1 vs V2 COMPARISON")
    print(f"{'=' * 60}")
    print(f"  {'Metric':<30} {'V1':>8} {'V2':>8}")
    print(f"  {'─' * 46}")
    for metric in r1:
        print(f"  {metric:<30} {r1[metric]:>8.4f} {r2[metric]:>8.4f}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NeuraRAG DeepEval Evaluation (OpenRouter)")
    parser.add_argument("--prompt", choices=["v1", "v2"], default="v2")
    parser.add_argument("--compare", action="store_true", help="Compare v1 vs v2")
    parser.add_argument("--no-rerank", action="store_true")
    args = parser.parse_args()

    if args.compare:
        compare_prompts()
    else:
        run_evaluation(args.prompt, not args.no_rerank)