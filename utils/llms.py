from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
import config

def get_groq_llm():
    return ChatGroq(
        model=config.GROQ_MODEL,
        api_key=config.GROQ_API_KEY,
        temperature=config.LLM_TEMPERATURE,
    )

def get_google_llm():
    return ChatGoogleGenerativeAI(
        model=config.GOOGLE_MODEL,
        google_api_key=config.GOOGLE_API_KEY,
        temperature=config.LLM_TEMPERATURE,
    )