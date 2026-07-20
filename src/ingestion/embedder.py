import logging
import os
import time

from dotenv import load_dotenv
from pydantic import SecretStr
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from langchain_google_genai._common import GoogleGenerativeAIError

from src.config import MODEL_EMBEDDINGS, DIMENSION_EMBEDDING, COLLECTION_NAME
from src.utils import get_env_var

load_dotenv()
logger = logging.getLogger(__name__)


def get_qdrant_client() -> QdrantClient:
    """Cria o cliente de conexão com o Qdrant Cloud."""
    url = get_env_var("QDRANT_URL")
    api_key = get_env_var("QDRANT_API_KEY")
    return QdrantClient(url=url, api_key=api_key)


def ensure_collection_exists(
    client: QdrantClient,
    collection_name: str = COLLECTION_NAME,
    force_recreate: bool = False,
) -> None:
    """
    Garante que a collection existe no Qdrant, criando-a se necessário.
    Se force_recreate=True, apaga e recria (útil ao reingerir PDFs atualizados).
    """
    exists = client.collection_exists(collection_name)

    if exists and force_recreate:
        logger.info("Recriando collection '%s'", collection_name)
        client.delete_collection(collection_name)
        exists = False

    if not exists:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=DIMENSION_EMBEDDING, distance=Distance.COSINE),
        )
        logger.info("Collection '%s' criada", collection_name)
    else:
        logger.info("Collection '%s' já existe, reaproveitando", collection_name)


def get_vectorstore(force_recreate: bool = False) -> QdrantVectorStore:
    """
    Retorna uma instância pronta de QdrantVectorStore, conectada
    à collection do projeto. Reaproveitável tanto na ingestão
    quanto na fase de retrieval.
    """
    google_api_key = get_env_var("GOOGLE_API_KEY")

    embeddings = GoogleGenerativeAIEmbeddings(
        model=MODEL_EMBEDDINGS,
        api_key=SecretStr(google_api_key),
    )

    client = get_qdrant_client()
    ensure_collection_exists(client, force_recreate=force_recreate)

    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )


@retry(
    retry=retry_if_exception_type(GoogleGenerativeAIError),
    wait=wait_exponential(multiplier=1, min=15, max=90),
    stop=stop_after_attempt(5),
    reraise=True,
)
def _add_documents_with_retry(vector_store: QdrantVectorStore, batch: list[Document]) -> list[str]:
    """Insere um único lote de documentos, com retry em caso de rate limit (429)."""
    return vector_store.add_documents(batch)


def embed_and_store_documents(
    chunks: list[Document],
    batch_size: int = 50,
    delay_between_batches: float = 2.0,
    force_recreate: bool = False,
) -> list[str]:
    """
    Gera embeddings para os chunks e insere no Qdrant Cloud, em lotes,
    para evitar estourar rate limit da API do Gemini.

    Retorna a lista de IDs dos pontos inseridos.
    """
    if not chunks:
        logger.warning("Nenhum chunk recebido para embedding.")
        return []

    vector_store = get_vectorstore(force_recreate=force_recreate)

    all_ids: list[str] = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        ids = _add_documents_with_retry(vector_store, batch)
        all_ids.extend(ids)
        logger.info("Lote %d-%d inserido (%d chunks)", i, i + len(batch), len(batch))

        if i + batch_size < len(chunks):
            time.sleep(delay_between_batches)

    logger.info("Total de %d chunks inseridos na collection '%s'", len(all_ids), COLLECTION_NAME)
    return all_ids

