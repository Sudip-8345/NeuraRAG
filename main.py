"""
NeuraRAG — CLI Entry Point

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
from rag.chain import ask


def build_index():
    """Load docs → chunk → embed → store in ChromaDB."""
    print("\n=== Building Vector Store ===\n")
    docs = load_documents()
    chunks = chunk_documents(docs)
    store = build_vectorstore(chunks)
    print("\nVector store built successfully!\n")
    return store


def get_vectorstore():
    """Load existing vector store, or build one if it doesn't exist."""
    if os.path.exists(config.CHROMA_PERSIST_DIR):
        return load_vectorstore()
    print("No vector store found. Building one first...\n")
    return build_index()


def interactive_qa(prompt_version, use_reranking):
    """Run the interactive Q&A loop."""
    store = get_vectorstore()

    print("\n=== NeuraRAG Policy Assistant ===")
    print(f"Prompt: {prompt_version} | Reranking: {'on' if use_reranking else 'off'}")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        result = ask(store, question, prompt_version, use_reranking)

        print(f"\n{'─' * 60}")
        print(result["answer"])
        print(f"{'─' * 60}")
        print(f"Sources: {', '.join(result['sources'])}")
        print(f"Model: {result['model_used']} | Prompt: {result['prompt_version']}\n")


def validate_keys():
    """Check that API keys are set in .env."""
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
