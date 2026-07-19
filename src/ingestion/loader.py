import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from src.config import BASE_DIRECTORY_FILES

logger = logging.getLogger(__name__)


def load_pdfs_from_directory(directory: Path = BASE_DIRECTORY_FILES) -> list[Document]:
    """
    Carrega todos os PDFs de um diretório e retorna uma lista única
    de Documents (um Document por página, com metadados de origem).

    PDFs corrompidos ou ilegíveis são pulados e logados, sem
    interromper o processamento dos demais arquivos.
    """
    pdf_files = sorted(directory.glob("*.pdf"))

    if not pdf_files:
        logger.warning("Nenhum PDF encontrado em %s", directory)
        return []

    all_documents: list[Document] = []

    for pdf_path in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()
            all_documents.extend(pages)
            logger.info("Carregado '%s' (%d páginas)", pdf_path.name, len(pages))
        except Exception as e:
            logger.error("Falha ao carregar '%s': %s", pdf_path.name, e)
            continue

    logger.info(
        "Ingestão concluída: %d páginas carregadas de %d arquivos",
        len(all_documents), len(pdf_files),
    )

    return all_documents