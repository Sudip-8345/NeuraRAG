"""NeuraRAG — CLI Entry Point"""
"""
Usage:
    python main.py --build          Build the vector store
    python main.py                  Interactive Q&A (default: prompt v2)
    python main.py --prompt v1      Use prompt v1
    python main.py --no-rerank      Disable reranking
"""

import argparse
import os
import sys

import config
from rag.loader import load_documents
from rag.chunker import chunk_documents
from rag.vectorstore import build_vectorstore, load_vectorstore
from agent.nodes import init_tool
from agent.workflow import ask
from langchain_core.messages import HumanMessage, AIMessage
from collections import deque


def build_index():
    print("\n=== Building Vector Store ===\n")
    docs = load_documents()
    chunks = chunk_documents(docs)
    store = build_vectorstore(chunks)
    print("\nVector store built successfully!\n")
    return store


def get_vectorstore():
    if os.path.exists(config.CHROMA_PERSIST_DIR):
        return load_vectorstore()
    print("No vector store found. Building one first...\n")
    return build_index()


# --- Interactive Question & Answer Loop ---
def interactive_qa(prompt_version, use_reranking):
    store = get_vectorstore()
    init_tool(store, use_reranking)  # Initialize retrieval tool with vectorstore

    # keep last N turns as conversation history
    history = deque(maxlen=config.MEMORY_MAX_TURNS)

    print("\n=== NeuraRAG Policy Assistant ===")
    print(f"Prompt: {prompt_version} | Reranking: {'on' if use_reranking else 'off'}")
    print(f"Memory: last {config.MEMORY_MAX_TURNS} turns")
    print("Type 'quit' to exit, 'clear' to reset memory.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit"):
            print("Bye!")
            break
        if question.lower() == "clear":
            history.clear()
            print("Memory cleared.\n")
            continue

        result = ask(question, prompt_version, chat_history=list(history))

        history.append(HumanMessage(content=question))
        history.append(AIMessage(content=result["answer"]))

        print(f"\n{'─' * 60}")
        print(result["answer"])
        print(f"{'─' * 60}")
        if result["sources"]:
            print(f"Sources: {', '.join(result['sources'])}")
        print(f"Model: {result['model_used']} | Prompt: {result['prompt_version']}\n")


def validate_keys():
    missing = []
    if not config.GROQ_API_KEY or "your_" in (config.GROQ_API_KEY or ""):
        missing.append("GROQ_API_KEY")
    if not config.GOOGLE_API_KEY or "your_" in (config.GOOGLE_API_KEY or ""):
        missing.append("GOOGLE_API_KEY")
    if missing:
        print(f"Error: Set {', '.join(missing)} in your .env file")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="NeuraRAG Policy Q&A")
    parser.add_argument("--build", action="store_true", help="Build the vector store")
    parser.add_argument("--prompt", choices=["v1", "v2"], default="v2", help="Prompt version")
    parser.add_argument("--no-rerank", action="store_true", help="Disable reranking")
    args = parser.parse_args()

    validate_keys()

    if args.build:
        build_index()
    else:
        interactive_qa(args.prompt, not args.no_rerank)


if __name__ == "__main__":
    main()
