from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr

from src.utils import get_env_var

def get_llm(temperature=0.2):
    google_api_key = get_env_var("GOOGLE_API_KEY")
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        api_key=SecretStr(google_api_key),
        temperature=temperature
    )
    
    