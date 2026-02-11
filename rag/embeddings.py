"""Google Generative AI embeddings via LangChain."""

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import config


def get_embedding_function():
    return GoogleGenerativeAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        google_api_key=config.GOOGLE_API_KEY,
    )
