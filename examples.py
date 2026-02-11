#!/usr/bin/env python3
"""
Example script demonstrating RAG system usage.

This script shows how to use the RAG system programmatically.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def example_basic_usage():
    """Example 1: Basic usage with a single question."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Usage")
    print("="*80 + "\n")
    
    from src.rag_system import RAGSystem
    
    # Initialize the system
    print("Initializing RAG system...")
    rag = RAGSystem()
    
    # Index documents if needed
    count = rag.vector_store.get_collection_count()
    if count == 0:
        print("Indexing documents for the first time...")
        rag.index_documents()
    else:
        print(f"Using existing index with {count} chunks.")
    
    # Ask a question
    question = "Can I return a product after 30 days?"
    print(f"\nQuestion: {question}")
    print("-" * 80)
    
    result = rag.answer_question(question)
    
    print(f"Answer: {result['answer']}")
    print(f"\nSources: {', '.join(set(result['sources']))}")


def example_multiple_questions():
    """Example 2: Asking multiple questions."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Multiple Questions")
    print("="*80 + "\n")
    
    from src.rag_system import RAGSystem
    
    rag = RAGSystem()
    
    questions = [
        "What is the cost of overnight shipping?",
        "How long does it take to process a refund?",
        "Can I cancel a subscription?",
        "Do you ship internationally?"
    ]
    
    print(f"Asking {len(questions)} questions...\n")
    
    for i, question in enumerate(questions, 1):
        print(f"{i}. Q: {question}")
        result = rag.answer_question(question)
        print(f"   A: {result['answer'][:150]}...")
        print()


def example_with_verbose_output():
    """Example 3: Using verbose mode to see retrieval details."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Verbose Output (See Retrieval Process)")
    print("="*80 + "\n")
    
    from src.rag_system import RAGSystem
    
    rag = RAGSystem()
    
    question = "What happens if I receive a defective product?"
    
    result = rag.answer_question(question, verbose=True)


def example_prompt_comparison():
    """Example 4: Comparing prompt versions."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Prompt Version Comparison")
    print("="*80 + "\n")
    
    from src.rag_system import RAGSystem
    
    rag = RAGSystem()
    
    # Question that might trigger hallucination
    question = "What is your warranty policy for electronics?"
    print(f"Question: {question}")
    print("(This question cannot be answered from the provided policies)\n")
    
    # Test with initial prompt (v1)
    print("Response with INITIAL PROMPT (v1):")
    print("-" * 80)
    result_v1 = rag.answer_question(question, use_improved_prompt=False)
    print(result_v1['answer'])
    
    print("\n" + "="*80 + "\n")
    
    # Test with improved prompt (v2)
    print("Response with IMPROVED PROMPT (v2):")
    print("-" * 80)
    result_v2 = rag.answer_question(question, use_improved_prompt=True)
    print(result_v2['answer'])
    
    print("\n" + "="*80)
    print("OBSERVATION:")
    print("-" * 80)
    print("The improved prompt (v2) should correctly identify that the")
    print("information is not available and provide a fallback response,")
    print("while v1 might attempt to answer without proper grounding.")
    print("="*80)


def example_custom_configuration():
    """Example 5: Custom RAG configuration."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Custom Configuration")
    print("="*80 + "\n")
    
    from src.rag_system import RAGSystem
    
    # Initialize with custom settings
    rag = RAGSystem(
        data_dir="./data",
        chunk_size=256,       # Smaller chunks
        chunk_overlap=25,     # Less overlap
        top_k=5               # Retrieve more chunks
    )
    
    print("Custom configuration:")
    print(f"  Chunk size: 256 characters")
    print(f"  Chunk overlap: 25 characters")
    print(f"  Top-k: 5 chunks")
    
    question = "What are the shipping options?"
    print(f"\nQuestion: {question}")
    print("-" * 80)
    
    result = rag.answer_question(question, verbose=False)
    
    print(f"Answer: {result['answer']}")
    print(f"\nRetrieved {result['context_chunks']} chunks")


def example_document_statistics():
    """Example 6: Analyzing document statistics."""
    print("\n" + "="*80)
    print("EXAMPLE 6: Document Statistics")
    print("="*80 + "\n")
    
    from src.data_preparation import DocumentProcessor
    
    processor = DocumentProcessor()
    
    # Process all documents
    chunks = processor.process_documents("./data")
    
    print(f"Total chunks: {len(chunks)}")
    
    # Statistics by source
    sources = {}
    for chunk in chunks:
        source = chunk['source']
        sources[source] = sources.get(source, 0) + 1
    
    print("\nChunks by document:")
    for source, count in sorted(sources.items()):
        print(f"  - {source}: {count} chunks")
    
    # Chunk size statistics
    sizes = [len(chunk['text']) for chunk in chunks]
    avg_size = sum(sizes) / len(sizes)
    min_size = min(sizes)
    max_size = max(sizes)
    
    print(f"\nChunk size statistics:")
    print(f"  - Average: {avg_size:.0f} characters")
    print(f"  - Min: {min_size} characters")
    print(f"  - Max: {max_size} characters")


def main():
    """Run example demonstrations."""
    print("\n" + "="*80)
    print("NEURARAG USAGE EXAMPLES")
    print("="*80)
    print("\nThese examples demonstrate how to use the RAG system.")
    print("Note: Requires GROQ_API_KEY in .env file for examples 1-5.")
    print("Example 6 can run without API access.\n")
    
    # Check if API key is available
    from dotenv import load_dotenv
    load_dotenv()
    has_api_key = os.getenv("GROQ_API_KEY") is not None
    
    if not has_api_key:
        print("⚠️  GROQ_API_KEY not found in .env file")
        print("Running only examples that don't require API access...\n")
    
    examples = [
        ("Document Statistics (No API required)", example_document_statistics, False),
        ("Basic Usage", example_basic_usage, True),
        ("Multiple Questions", example_multiple_questions, True),
        ("Verbose Output", example_with_verbose_output, True),
        ("Prompt Comparison", example_prompt_comparison, True),
        ("Custom Configuration", example_custom_configuration, True),
    ]
    
    for title, func, requires_api in examples:
        if requires_api and not has_api_key:
            print(f"\n⏭️  Skipping: {title} (requires API key)")
            continue
        
        try:
            func()
            input("\nPress Enter to continue to next example...")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\n❌ Error in {title}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("Examples completed!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
