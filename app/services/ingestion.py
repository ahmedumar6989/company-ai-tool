import hashlib
import logging
import re
from typing import Any

from app.config import Settings
from app.services.retrieval import RetrievalService


logger = logging.getLogger(__name__)


SAMPLE_DOCUMENT = """
COMPANY POLICY

Employees receive 20 annual leave days per calendar year.

Employees should submit annual leave requests at least 3 working days
before the requested leave date.

Employees receive up to 10 medical leave days per year.

Medical leave longer than 3 consecutive days may require supporting
documentation.

Employees who leave the company are paid for approved unused annual
leave according to company policy.

The standard working week is Monday through Friday.

Normal working hours are 9:00 AM to 5:00 PM.
""".strip()


class DocumentIngestionService:
    """
    Handles document:

    - loading
    - cleaning
    - hashing
    - chunking
    - vector database synchronization
    """

    def __init__(
        self,
        settings: Settings,
        retrieval_service: RetrievalService,
    ):
        self.settings = settings
        self.retrieval_service = retrieval_service

    # -----------------------------------------------------
    # SAMPLE DOCUMENT
    # -----------------------------------------------------

    def create_sample_document_if_needed(self) -> None:
        """
        Create a sample document if the documents folder
        contains no .txt files.
        """

        existing_files = list(
            self.settings.DOCUMENTS_DIR.glob("*.txt")
        )

        if existing_files:
            return

        sample_path = (
            self.settings.DOCUMENTS_DIR
            / "company_policy.txt"
        )

        sample_path.write_text(
            SAMPLE_DOCUMENT,
            encoding="utf-8",
        )

        logger.info(
            "Created sample document: %s",
            sample_path.name,
        )

    # -----------------------------------------------------
    # LOAD DOCUMENTS
    # -----------------------------------------------------

    def load_documents(self) -> list[dict[str, str]]:
        """
        Read all .txt files from documents/.
        """

        self.create_sample_document_if_needed()

        documents = []

        file_paths = sorted(
            self.settings.DOCUMENTS_DIR.glob("*.txt")
        )

        if not file_paths:

            raise RuntimeError(
                "No .txt documents found in the documents folder."
            )

        for file_path in file_paths:

            text = file_path.read_text(
                encoding="utf-8"
            ).strip()

            if not text:
                logger.warning(
                    "Skipping empty file: %s",
                    file_path.name,
                )
                continue

            documents.append(
                {
                    "filename": file_path.name,
                    "text": text,
                }
            )

        return documents

    # -----------------------------------------------------
    # CLEAN TEXT
    # -----------------------------------------------------

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Remove unnecessary whitespace.
        """

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()

    # -----------------------------------------------------
    # HASH DOCUMENT
    # -----------------------------------------------------

    @staticmethod
    def create_document_hash(text: str) -> str:
        """
        Create SHA-256 hash for document version tracking.
        """

        return hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest()

    # -----------------------------------------------------
    # CHUNK TEXT
    # -----------------------------------------------------

    def chunk_text(
        self,
        text: str,
    ) -> list[str]:
        """
        Break text into overlapping chunks.
        """

        text = self.clean_text(text)

        chunk_size = self.settings.CHUNK_SIZE
        overlap = self.settings.CHUNK_OVERLAP

        if overlap >= chunk_size:

            raise ValueError(
                "CHUNK_OVERLAP must be smaller than CHUNK_SIZE."
            )

        if len(text) <= chunk_size:
            return [text]

        chunks = []

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            if end >= len(text):
                break

            start = end - overlap

        return chunks

    # -----------------------------------------------------
    # PREPARE CHUNKS
    # -----------------------------------------------------

    def prepare_chunks(
        self,
        documents: list[dict[str, str]],
    ) -> list[dict[str, Any]]:

        all_chunks = []

        for document in documents:

            filename = document["filename"]
            cleaned_text = self.clean_text(
                document["text"]
            )

            document_hash = (
                self.create_document_hash(
                    cleaned_text
                )
            )

            chunks = self.chunk_text(
                cleaned_text
            )

            # Hash filename so IDs stay safe and unique.
            filename_hash = hashlib.sha256(
                filename.encode("utf-8")
            ).hexdigest()[:12]

            for index, chunk in enumerate(chunks):

                chunk_id = (
                    f"{filename_hash}-"
                    f"{document_hash[:12]}-"
                    f"{index}"
                )

                all_chunks.append(
                    {
                        "id": chunk_id,
                        "filename": filename,
                        "chunk_index": index,
                        "document_hash": document_hash,
                        "text": chunk,
                    }
                )

        return all_chunks

    # -----------------------------------------------------
    # BUILD INDEX
    # -----------------------------------------------------

    def build_index(self) -> None:
        """
        Complete document ingestion pipeline.
        """

        documents = self.load_documents()

        chunks = self.prepare_chunks(
            documents
        )

        active_filenames = {
            document["filename"]
            for document in documents
        }

        logger.info(
            "Documents found: %s",
            len(documents),
        )

        logger.info(
            "Chunks prepared: %s",
            len(chunks),
        )

        self.retrieval_service.sync_documents(
            chunks=chunks,
            active_filenames=active_filenames,
        )

        logger.info(
            "Document ingestion completed."
        )