"""
Data Preparation Module

This module handles loading, cleaning, and chunking of policy documents.
"""

import os
from typing import List, Dict
import re


class DocumentProcessor:
    """Handles document loading and text processing."""
    
    # Constants for sentence boundary detection
    SENTENCE_BOUNDARY_LOOKBACK = 100  # chars to look back for sentence end
    SENTENCE_BOUNDARY_LOOKAHEAD = 50  # chars to look ahead for sentence end
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Size of each text chunk in characters.
                        512 chosen because:
                        - Balances context (not too small) and specificity (not too large)
                        - Works well with sentence-transformers embedding models
                        - Allows ~2-3 paragraphs per chunk for semantic coherence
                        - Fits within typical transformer context windows
            chunk_overlap: Number of overlapping characters between chunks.
                          50 chosen to maintain context continuity at chunk boundaries.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def load_document(self, filepath: str) -> str:
        """
        Load a document from a file.
        
        Args:
            filepath: Path to the document file
            
        Returns:
            Document content as string
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    
    def clean_text(self, text: str) -> str:
        """
        Clean text by removing extra whitespace and normalizing.
        
        Args:
            text: Raw text content
            
        Returns:
            Cleaned text
        """
        # Remove multiple consecutive whitespaces
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing whitespace
        text = text.strip()
        return text
    
    def chunk_text(self, text: str) -> List[Dict[str, str]]:
        """
        Split text into overlapping chunks.
        
        Strategy:
        1. Split by paragraphs first to respect document structure
        2. Combine paragraphs until chunk_size is reached
        3. Add overlap to maintain context between chunks
        
        Args:
            text: Text to chunk
            
        Returns:
            List of dictionaries containing chunk text and metadata
        """
        # Clean the text first
        text = self.clean_text(text)
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Determine end position
            end = start + self.chunk_size
            
            # If not at the end of text, try to break at sentence boundary
            if end < len(text):
                # Look for sentence ending punctuation
                substr = text[max(start, end - self.SENTENCE_BOUNDARY_LOOKBACK):end + self.SENTENCE_BOUNDARY_LOOKAHEAD]
                sentence_end = max(
                    substr.rfind('. '),
                    substr.rfind('.\n'),
                    substr.rfind('? '),
                    substr.rfind('! ')
                )
                
                if sentence_end != -1:
                    # Adjust end to sentence boundary
                    end = max(start, end - self.SENTENCE_BOUNDARY_LOOKBACK) + sentence_end + 1
            
            # Extract chunk
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunks.append({
                    'text': chunk_text,
                    'start': start,
                    'end': end
                })
            
            # Move start position with overlap
            start = end - self.chunk_overlap
        
        return chunks
    
    def process_documents(self, data_dir: str) -> List[Dict]:
        """
        Process all documents in a directory.
        
        Args:
            data_dir: Directory containing policy documents
            
        Returns:
            List of processed chunks with metadata
        """
        all_chunks = []
        
        for filename in os.listdir(data_dir):
            if filename.endswith(('.md', '.txt')):
                filepath = os.path.join(data_dir, filename)
                
                # Load and chunk document
                text = self.load_document(filepath)
                chunks = self.chunk_text(text)
                
                # Add metadata to each chunk
                for i, chunk in enumerate(chunks):
                    chunk['source'] = filename
                    chunk['chunk_id'] = i
                    all_chunks.append(chunk)
        
        return all_chunks
