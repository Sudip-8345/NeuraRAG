"""
Main RAG System

This module integrates all components into a complete RAG system.
"""

from src.data_preparation import DocumentProcessor
from src.retrieval import VectorStore
from src.generation import LLMGenerator
from typing import Dict, List


class RAGSystem:
    """Complete RAG system for policy Q&A."""
    
    def __init__(self, 
                 data_dir: str = "./data",
                 persist_dir: str = "./chroma_db",
                 chunk_size: int = 512,
                 chunk_overlap: int = 50,
                 top_k: int = 3):
        """
        Initialize the RAG system.
        
        Args:
            data_dir: Directory containing policy documents
            persist_dir: Directory for vector database persistence
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            top_k: Number of chunks to retrieve
        """
        self.data_dir = data_dir
        
        # Initialize components
        print("Initializing RAG System...")
        print("-" * 80)
        
        self.document_processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        self.vector_store = VectorStore(
            collection_name="policy_docs",
            persist_directory=persist_dir
        )
        
        self.generator = LLMGenerator()
        
        self.top_k = top_k
        
        print("RAG System initialized successfully!")
        print("-" * 80 + "\n")
    
    def index_documents(self, force_reindex: bool = False) -> None:
        """
        Index all documents in the data directory.
        
        Args:
            force_reindex: If True, clear existing index and reindex
        """
        # Check if already indexed
        current_count = self.vector_store.get_collection_count()
        
        if current_count > 0 and not force_reindex:
            print(f"Vector store already contains {current_count} chunks.")
            print("Use force_reindex=True to rebuild the index.\n")
            return
        
        if force_reindex:
            print("Clearing existing vector store...")
            self.vector_store.clear_collection()
        
        # Process and index documents
        print(f"Processing documents from {self.data_dir}...")
        chunks = self.document_processor.process_documents(self.data_dir)
        
        print(f"Processed {len(chunks)} chunks from documents.")
        
        # Add to vector store
        self.vector_store.add_documents(chunks)
        
        print(f"\nIndexing complete! {len(chunks)} chunks indexed.\n")
    
    def answer_question(self, 
                       question: str, 
                       use_improved_prompt: bool = True,
                       verbose: bool = False) -> Dict:
        """
        Answer a question using the RAG pipeline.
        
        Args:
            question: User's question
            use_improved_prompt: Whether to use improved prompt (default: True)
            verbose: Whether to print detailed information
            
        Returns:
            Dictionary containing answer and metadata
        """
        if verbose:
            print(f"\nQuestion: {question}")
            print("-" * 80)
        
        # Retrieve relevant chunks
        if verbose:
            print(f"Retrieving top {self.top_k} relevant chunks...")
        
        retrieved_chunks = self.vector_store.search(question, top_k=self.top_k)
        
        if verbose:
            print(f"Retrieved {len(retrieved_chunks)} chunks")
            for i, chunk in enumerate(retrieved_chunks, 1):
                print(f"\n  Chunk {i} (from {chunk['source']}, relevance: {chunk['relevance_score']:.3f}):")
                print(f"  {chunk['text'][:150]}...")
            print()
        
        # Generate answer
        if verbose:
            print("Generating answer...")
        
        result = self.generator.generate_answer(
            question=question,
            retrieved_chunks=retrieved_chunks,
            use_improved_prompt=use_improved_prompt
        )
        
        # Add retrieved chunks to result
        result['retrieved_chunks'] = retrieved_chunks
        
        if verbose:
            print(f"\nAnswer: {result['answer']}")
            print("-" * 80 + "\n")
        
        return result
    
    def interactive_mode(self):
        """Run the system in interactive mode."""
        print("\n" + "="*80)
        print("RAG SYSTEM - INTERACTIVE MODE")
        print("="*80)
        print("\nAsk questions about our policies!")
        print("Type 'quit' or 'exit' to stop.\n")
        
        while True:
            try:
                question = input("Your question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("\nThank you for using the RAG system!")
                    break
                
                if not question:
                    print("Please enter a question.\n")
                    continue
                
                # Get answer
                result = self.answer_question(question, verbose=False)
                
                # Display answer
                print("\n" + "-"*80)
                print("ANSWER:")
                print("-"*80)
                print(result['answer'])
                print("\n" + "-"*80)
                print("METADATA:")
                print(f"  • Model: {result['model']}")
                print(f"  • Prompt Version: {result['prompt_version']}")
                print(f"  • Chunks Retrieved: {result['context_chunks']}")
                print(f"  • Sources: {', '.join(set(result['sources']))}")
                print("-"*80 + "\n")
                
            except KeyboardInterrupt:
                print("\n\nThank you for using the RAG system!")
                break
            except Exception as e:
                print(f"\nError: {e}\n")
