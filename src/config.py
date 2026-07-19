from pathlib import Path

# Raiz do projeto, independente de onde o script for executado
PROJECT_ROOT = Path(__file__).resolve().parent.parent

BASE_DIRECTORY_FILES = PROJECT_ROOT / "data" / "raw"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200