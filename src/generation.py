"""
Generation Module

This module handles LLM interaction and prompt management.
"""

from groq import Groq
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv


class PromptTemplate:
    """Manages prompt templates for the RAG system."""
    
    # Initial Prompt (v1)
    INITIAL_PROMPT = """You are a helpful customer service assistant. Answer the user's question based on the provided context from company policy documents.

Context:
{context}

Question: {question}

Answer:"""
    
    # Improved Prompt (v2)
    IMPROVED_PROMPT = """You are a helpful and accurate customer service assistant for our company. Your role is to answer questions based ONLY on the information provided in the context below.

**Instructions:**
1. Answer the question using ONLY the information from the provided context
2. If the context contains the answer, provide a clear and concise response
3. If the context does NOT contain enough information to answer the question, respond with: "I don't have enough information in our policy documents to answer this question. Please contact customer service at support@company.com for assistance."
4. Do NOT make up information or use external knowledge
5. Structure your answer with clear formatting when appropriate (bullet points, numbered lists)
6. Cite the relevant policy when possible (e.g., "According to our Refund Policy...")

**Context from Policy Documents:**
{context}

**Question:** {question}

**Answer:**"""

    @staticmethod
    def get_prompt_explanation() -> str:
        """
        Explain the changes between prompt versions.
        
        Returns:
            Detailed explanation of prompt iterations
        """
        return """
## Prompt Engineering Iterations

### Initial Prompt (v1)
Simple and straightforward, but had issues:
- No explicit instruction to avoid hallucination
- No guidance on handling missing information
- No structure for formatting answers
- Too generic for customer service context

### Improved Prompt (v2)
Key improvements:
1. **Explicit Grounding:** Added clear instruction to answer ONLY from context
2. **Hallucination Prevention:** Direct statement "Do NOT make up information"
3. **Missing Information Handling:** Specific template response when answer isn't in context
4. **Structured Output:** Instructions for formatting (bullets, numbers)
5. **Citation Guidance:** Encourages referencing source policies
6. **Fallback Action:** Provides contact information when uncertain
7. **Role Clarity:** Better defined the assistant's role and boundaries

### Why These Changes Matter:
- **Accuracy:** Explicit grounding reduces hallucinations by ~60-80%
- **User Trust:** Admitting "I don't know" is better than wrong information
- **Usability:** Structured answers are easier to read and act upon
- **Traceability:** Citations help users verify information in source documents
- **Safety:** Clear boundaries prevent the model from overstepping its knowledge
"""


class LLMGenerator:
    """Handles LLM generation using Groq."""
    
    def __init__(self, 
                 model: str = "llama-3.1-70b-versatile",
                 temperature: float = 0.1,
                 max_tokens: int = 1024):
        """
        Initialize the LLM generator.
        
        Args:
            model: Groq model to use (llama-3.1-70b-versatile for quality)
            temperature: Lower temperature (0.1) for more deterministic, factual responses.
                        Should be between 0.0 and 2.0. Lower values = more deterministic.
            max_tokens: Maximum tokens in response
        """
        load_dotenv()
        
        # Validate temperature
        if not 0.0 <= temperature <= 2.0:
            raise ValueError(f"Temperature must be between 0.0 and 2.0, got {temperature}")
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found. Please set it in .env file.\n"
                "Get your API key from: https://console.groq.com/keys"
            )
        
        self.client = Groq(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def format_context(self, retrieved_chunks: List[Dict]) -> str:
        """
        Format retrieved chunks into context string.
        
        Args:
            retrieved_chunks: List of retrieved document chunks
            
        Returns:
            Formatted context string
        """
        if not retrieved_chunks:
            return "No relevant information found."
        
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            source = chunk['source'].replace('_', ' ').replace('.md', '').title()
            context_parts.append(
                f"[Source {i}: {source}]\n{chunk['text']}\n"
            )
        
        return "\n".join(context_parts)
    
    def generate_answer(self,
                       question: str,
                       retrieved_chunks: List[Dict],
                       use_improved_prompt: bool = True) -> Dict:
        """
        Generate an answer using the LLM.
        
        Args:
            question: User's question
            retrieved_chunks: Retrieved context chunks
            use_improved_prompt: Whether to use improved prompt (default: True)
            
        Returns:
            Dictionary containing answer and metadata
        """
        # Format context
        context = self.format_context(retrieved_chunks)
        
        # Select prompt template
        prompt_template = (PromptTemplate.IMPROVED_PROMPT if use_improved_prompt 
                          else PromptTemplate.INITIAL_PROMPT)
        
        # Fill in the prompt
        prompt = prompt_template.format(context=context, question=question)
        
        # Generate response
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful and accurate customer service assistant."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            answer = response.choices[0].message.content
            
            return {
                'answer': answer,
                'model': self.model,
                'prompt_version': 'v2' if use_improved_prompt else 'v1',
                'context_chunks': len(retrieved_chunks),
                'sources': [chunk['source'] for chunk in retrieved_chunks]
            }
            
        except Exception as e:
            return {
                'answer': f"Error generating response: {str(e)}",
                'model': self.model,
                'prompt_version': 'v2' if use_improved_prompt else 'v1',
                'context_chunks': 0,
                'sources': [],
                'error': str(e)
            }
