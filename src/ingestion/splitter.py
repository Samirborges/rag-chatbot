import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


def split_documents_into_chunks(documents: list[Document]) -> list[Document]:
    """
    Divide uma lista de Documents (geralmente páginas de PDF) em chunks
    menores, preservando os metadados originais em cada chunk gerado.
    """
    if not documents:
        logger.warning("Nenhum documento recebido para split.")
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )

    chunks = text_splitter.split_documents(documents)

    logger.info(
        "Split concluído: %d documentos -> %d chunks", len(documents), len(chunks)
    )

    return chunks

