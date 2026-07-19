import logging

from src.ingestion.loader import load_pdfs_from_directory
from src.ingestion.splitter import split_documents_into_chunks

logger = logging.getLogger(__name__)


def data_ingestion_pipeline(verbose: bool = False) -> list:
    """
    Orquestra a fase de ingestão: carrega os PDFs e divide em chunks.
    Retorna a lista final de chunks, pronta para a etapa de embedding.
    """
    documents = load_pdfs_from_directory()

    if not documents:
        logger.error("Nenhum documento carregado. Abortando pipeline de ingestão.")
        return []

    chunks = split_documents_into_chunks(documents)

    if verbose:
        for chunk in chunks[:3]:  # mostra só uma amostra, não tudo
            print("=" * 50)
            print(f"page_content: {chunk.page_content}")
            print(f"metadata: {chunk.metadata}")

    return chunks


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    data_ingestion_pipeline(verbose=True)
    
    
    