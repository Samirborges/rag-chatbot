import logging

from src.ingestion.embedder import get_vectorstore


logger = logging.getLogger(__name__)

def get_retriever(k: int = 4):
    vector_store = get_vectorstore()
    return vector_store.as_retriever(search_kwargs={"k": k})


def retrieve_relevant_chunks(query: str, k: int = 4):
    retriver = get_retriever(k)
    documents = retriver.invoke(query)
    logger.info("Número de documentos encontrados: %i", len(documents))
    return documents

