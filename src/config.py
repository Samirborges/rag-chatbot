from pathlib import Path
from sentence_transformers import SentenceTransformer


# Raiz do projeto, independente de onde o script for executado
PROJECT_ROOT = Path(__file__).resolve().parent.parent

BASE_DIRECTORY_FILES = PROJECT_ROOT / "data" / "raw"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


MODEL_EMBEDDINGS = "models/gemini-embedding-001"

DIMENSION_EMBEDDING = 3072

COLLECTION_NAME = "curso_ia_collection"