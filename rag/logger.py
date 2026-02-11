"""Simple logging for RAG pipeline tracing."""

import datetime
import os

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
LOG_FILE = os.path.join(LOG_DIR, "rag_trace.log")


def _write(text):
    """Write text to log file, creating the directory if needed."""
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(text)


def _now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(stage, message):
    """Log a one-line timestamped entry."""
    _write(f"[{_now()}] [{stage}] {message}\n")


def log_query(query, context, answer, prompt_version):
    """Log a full query → answer trace."""
    sep = "=" * 60
    _write(
        f"\n{sep}\n"
        f"[{_now()}] QUERY TRACE (prompt: {prompt_version})\n"
        f"Q: {query}\n"
        f"CONTEXT (500 chars): {context[:500]}...\n"
        f"ANSWER: {answer}\n"
        f"{sep}\n"
    )
