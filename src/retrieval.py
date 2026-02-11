"""
Retrieval Module

This module handles embedding generation, vector storage, and semantic retrieval.
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import os


class VectorStore:
    """Manages vector storage and retrieval using ChromaDB."""
    
    def __init__(self, 
                 collection_name: str = "policy_docs",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 persist_directory: str = "./chroma_db"):
        """
        Initialize the vector store.
        
        Args:
            collection_name: Name of the ChromaDB collection
            embedding_model: Sentence transformer model for embeddings
                           all-MiniLM-L6-v2 chosen because:
                           - Fast and efficient (384 dimensions)
                           - Good performance on semantic similarity tasks
                           - Lightweight for quick retrieval
                           - Well-suited for question-answering
            persist_directory: Directory to persist the vector database
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Initialize embedding model
        print(f"Loading embedding model: {embedding_model}...")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()
    
    def add_documents(self, chunks: List[Dict]) -> None:
        """
        Add document chunks to the vector store.
        
        Args:
            chunks: List of document chunks with text and metadata
        """
        if not chunks:
            return
        
        # Extract texts for embedding
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings
        print(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.generate_embeddings(texts)
        
        # Prepare IDs and metadata
        ids = [f"{chunk['source']}_{chunk['chunk_id']}" for chunk in chunks]
        metadatas = [
            {
                'source': chunk['source'],
                'chunk_id': str(chunk['chunk_id']),
                'start': str(chunk['start']),
                'end': str(chunk['end'])
            }
            for chunk in chunks
        ]
        
        # Add to collection
        print(f"Adding {len(chunks)} chunks to vector store...")
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"Successfully added {len(chunks)} chunks to the vector store.")
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Search for relevant document chunks.
        
        Args:
            query: Search query
            top_k: Number of top results to return (default: 3)
                   3 chosen because:
                   - Provides sufficient context without overwhelming the LLM
                   - Balances recall and precision
                   - Fits well within LLM context windows
                   - Allows for diversity in retrieved information
            
        Returns:
            List of relevant chunks with metadata and scores
        """
        # Generate query embedding
        query_embedding = self.generate_embeddings([query])[0]
        
        # Search the collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['documents'][0])):
            formatted_results.append({
                'text': results['documents'][0][i],
                'source': results['metadatas'][0][i]['source'],
                'chunk_id': results['metadatas'][0][i]['chunk_id'],
                'distance': results['distances'][0][i],
                'relevance_score': 1 - results['distances'][0][i]  # Convert distance to similarity
            })
        
        return formatted_results
    
    def clear_collection(self) -> None:
        """Clear all documents from the collection."""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"Cleared collection: {self.collection_name}")
        except Exception as e:
            print(f"Error clearing collection: {e}")
    
    def get_collection_count(self) -> int:
        """Get the number of documents in the collection."""
        return self.collection.count()
