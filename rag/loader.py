"""Load all Markdown policy docs from the data directory."""

from langchain_community.document_loaders import DirectoryLoader, TextLoader
import config


def load_documents():
    """Load all .md files and return as LangChain Documents."""
    loader = DirectoryLoader(
        config.DATA_DIR,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs = loader.load()
    print(f"[Loader] Loaded {len(docs)} document(s)")
    return docs
