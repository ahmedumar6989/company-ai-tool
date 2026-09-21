from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv


# ---------------------------------------------------------
# PROJECT ROOT
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------
# LOAD .ENV FILE
# ---------------------------------------------------------

load_dotenv(BASE_DIR / ".env")


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

@dataclass(frozen=True)
class Settings:
    APP_NAME: str = "Company AI Assistant"

    DOCUMENTS_DIR: Path = BASE_DIR / "documents"
    CHROMA_DIR: Path = BASE_DIR / "chroma_db"

    COLLECTION_NAME: str = os.getenv(
        "COLLECTION_NAME",
        "company_documents"
    )

    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    OPENAI_MODEL: str = os.getenv(
        "OPENAI_MODEL",
        "gpt-5.6-luna"
    )

    OPENAI_API_KEY: str | None = os.getenv(
        "OPENAI_API_KEY"
    )

    TOP_K: int = int(
        os.getenv("TOP_K", "3")
    )

    CHUNK_SIZE: int = int(
        os.getenv("CHUNK_SIZE", "500")
    )

    CHUNK_OVERLAP: int = int(
        os.getenv("CHUNK_OVERLAP", "100")
    )


settings = Settings()


# ---------------------------------------------------------
# CREATE REQUIRED DIRECTORIES
# ---------------------------------------------------------

def ensure_directories() -> None:
    settings.DOCUMENTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    settings.CHROMA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )