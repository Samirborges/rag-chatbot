import logging

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from langchain_google_genai._common import GoogleGenerativeAIError

from src.ingestion.embedder import get_vectorstore


logger = logging.getLogger(__name__)


def get_retriever(k: int = 4):
    vector_store = get_vectorstore()
    return vector_store.as_retriever(search_kwargs={"k": k})


@retry(
    retry=retry_if_exception_type(GoogleGenerativeAIError),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    stop=stop_after_attempt(4),
    reraise=True,
)
def retrieve_relevant_chunks(query: str, k: int = 4):
    retriver = get_retriever(k)
    documents = retriver.invoke(query)
    logger.info("Número de documentos encontrados: %i", len(documents))
    return documents

