#!/usr/bin/env python3
"""
Main CLI script for the RAG system.

This script provides a command-line interface to:
1. Index policy documents
2. Ask questions interactively
3. Run evaluations
"""

import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.rag_system import RAGSystem
from src.evaluation import Evaluator, EvaluationDataset
from src.generation import PromptTemplate


def index_documents(args):
    """Index policy documents."""
    print("\n" + "="*80)
    print("INDEXING DOCUMENTS")
    print("="*80 + "\n")
    
    rag = RAGSystem(
        data_dir=args.data_dir,
        persist_dir=args.persist_dir
    )
    
    rag.index_documents(force_reindex=args.force)
    
    print("✅ Indexing complete!")


def interactive(args):
    """Run in interactive mode."""
    rag = RAGSystem(
        data_dir=args.data_dir,
        persist_dir=args.persist_dir
    )
    
    # Check if documents are indexed
    count = rag.vector_store.get_collection_count()
    if count == 0:
        print("\n⚠️  No documents indexed yet!")
        print("Please run: python main.py index\n")
        return
    
    print(f"\nLoaded {count} document chunks.")
    
    rag.interactive_mode()


def ask(args):
    """Answer a single question."""
    rag = RAGSystem(
        data_dir=args.data_dir,
        persist_dir=args.persist_dir
    )
    
    # Check if documents are indexed
    count = rag.vector_store.get_collection_count()
    if count == 0:
        print("\n⚠️  No documents indexed yet!")
        print("Please run: python main.py index\n")
        return
    
    result = rag.answer_question(args.question, verbose=args.verbose)
    
    if not args.verbose:
        print("\n" + "="*80)
        print("ANSWER:")
        print("="*80)
        print(result['answer'])
        print("\n" + "="*80)
        print(f"Sources: {', '.join(set(result['sources']))}")
        print("="*80 + "\n")


def evaluate(args):
    """Run evaluation."""
    print("\n" + "="*80)
    print("EVALUATION MODE")
    print("="*80 + "\n")
    
    rag = RAGSystem(
        data_dir=args.data_dir,
        persist_dir=args.persist_dir
    )
    
    # Check if documents are indexed
    count = rag.vector_store.get_collection_count()
    if count == 0:
        print("\n⚠️  No documents indexed yet!")
        print("Please run: python main.py index\n")
        return
    
    print(f"Loaded {count} document chunks.\n")
    
    evaluator = Evaluator()
    questions = EvaluationDataset.get_evaluation_questions()
    
    results = evaluator.run_evaluation(rag, questions)
    
    # Save results if requested
    if args.save:
        evaluator.save_results(results, args.save)


def show_prompts(args):
    """Show prompt templates and explanations."""
    print("\n" + "="*80)
    print("PROMPT TEMPLATES")
    print("="*80 + "\n")
    
    print("INITIAL PROMPT (v1):")
    print("-" * 80)
    print(PromptTemplate.INITIAL_PROMPT)
    print()
    
    print("\nIMPROVED PROMPT (v2):")
    print("-" * 80)
    print(PromptTemplate.IMPROVED_PROMPT)
    print()
    
    print(PromptTemplate.get_prompt_explanation())


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="RAG System for Company Policy Q&A",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index documents
  python main.py index
  
  # Ask a question
  python main.py ask "Can I return a product after 30 days?"
  
  # Interactive mode
  python main.py interactive
  
  # Run evaluation
  python main.py evaluate
  
  # Show prompts
  python main.py prompts
        """
    )
    
    parser.add_argument(
        '--data-dir',
        default='./data',
        help='Directory containing policy documents (default: ./data)'
    )
    
    parser.add_argument(
        '--persist-dir',
        default='./chroma_db',
        help='Directory for vector database (default: ./chroma_db)'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Index command
    index_parser = subparsers.add_parser('index', help='Index policy documents')
    index_parser.add_argument(
        '--force',
        action='store_true',
        help='Force reindexing even if documents are already indexed'
    )
    
    # Interactive command
    interactive_parser = subparsers.add_parser('interactive', help='Run in interactive mode')
    
    # Ask command
    ask_parser = subparsers.add_parser('ask', help='Ask a single question')
    ask_parser.add_argument('question', help='Question to ask')
    ask_parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed information'
    )
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Run evaluation')
    eval_parser.add_argument(
        '--save',
        help='Save results to JSON file'
    )
    
    # Prompts command
    prompts_parser = subparsers.add_parser('prompts', help='Show prompt templates')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute command
    commands = {
        'index': index_documents,
        'interactive': interactive,
        'ask': ask,
        'evaluate': evaluate,
        'prompts': show_prompts
    }
    
    commands[args.command](args)


if __name__ == "__main__":
    main()
