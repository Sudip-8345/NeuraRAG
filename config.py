"""
Centralized configuration — all settings in one place.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Paths ---
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
CHROMA_PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")

# --- Chunking ---
# 500 chars ≈ 1 policy section. Short docs benefit from small, precise chunks.
# 50 char overlap avoids cutting mid-sentence at boundaries.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# --- Retrieval ---
TOP_K = 3  # 3 chunks balances relevance vs noise for short policy docs.

# --- LLM ---
LLM_MODEL = "llama-3.1-8b-instant"
LLM_TEMPERATURE = 0.1  # Low temp = factual, deterministic answers

# --- Embedding ---
EMBEDDING_MODEL = "models/embedding-001"
