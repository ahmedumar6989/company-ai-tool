import logging
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

from app.config import Settings


logger = logging.getLogger(__name__)


class RetrievalService:
    """
    Handles:

    1. Embedding generation
    2. ChromaDB connection
    3. Vector indexing
    4. Semantic retrieval
    """

    def __init__(self, settings: Settings):
        self.settings = settings

        logger.info("Loading embedding model...")

        self.embedding_model = SentenceTransformer(
            self.settings.EMBEDDING_MODEL
        )

        logger.info("Embedding model loaded.")

        # -------------------------------------------------
        # CHROMA DATABASE
        # -------------------------------------------------

        self.chroma_client = chromadb.PersistentClient(
            path=str(self.settings.CHROMA_DIR)
        )

        self.collection = (
            self.chroma_client.get_or_create_collection(
                name=self.settings.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        )

        logger.info(
            "ChromaDB collection ready: %s",
            self.settings.COLLECTION_NAME,
        )

    # -----------------------------------------------------
    # CREATE EMBEDDINGS
    # -----------------------------------------------------

    def create_embeddings(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """
        Convert text into numerical vectors.
        """

        if not texts:
            return []

        embeddings = self.embedding_model.encode(
            texts,
            normalize_embeddings=True,
        )

        return embeddings.tolist()

    # -----------------------------------------------------
    # SYNC DOCUMENTS WITH VECTOR DATABASE
    # -----------------------------------------------------

    def sync_documents(
        self,
        chunks: list[dict[str, Any]],
        active_filenames: set[str],
    ) -> None:
        """
        Synchronize local documents with ChromaDB.

        Handles:
        - new files
        - modified files
        - deleted files
        """

        existing = self.collection.get(
            include=["metadatas"]
        )

        existing_ids = existing.get("ids", [])
        existing_metadatas = existing.get("metadatas", [])

        existing_by_filename: dict[
            str,
            list[tuple[str, dict[str, Any]]]
        ] = {}

        for document_id, metadata in zip(
            existing_ids,
            existing_metadatas,
        ):
            if not metadata:
                continue

            filename = metadata.get("filename")

            if not filename:
                continue

            existing_by_filename.setdefault(
                filename,
                []
            ).append(
                (
                    document_id,
                    metadata,
                )
            )

        # -------------------------------------------------
        # DELETE FILES THAT NO LONGER EXIST
        # -------------------------------------------------

        for filename, records in existing_by_filename.items():

            if filename not in active_filenames:

                ids_to_delete = [
                    record[0]
                    for record in records
                ]

                if ids_to_delete:

                    logger.info(
                        "Removing deleted document: %s",
                        filename,
                    )

                    self.collection.delete(
                        ids=ids_to_delete
                    )

        # -------------------------------------------------
        # PROCESS CURRENT DOCUMENTS
        # -------------------------------------------------

        chunks_by_filename: dict[
            str,
            list[dict[str, Any]]
        ] = {}

        for chunk in chunks:
            chunks_by_filename.setdefault(
                chunk["filename"],
                []
            ).append(chunk)

        for filename, file_chunks in (
            chunks_by_filename.items()
        ):

            if not file_chunks:
                continue

            current_document_hash = (
                file_chunks[0]["document_hash"]
            )

            existing_records = (
                existing_by_filename.get(
                    filename,
                    []
                )
            )

            existing_hashes = {
                metadata.get("document_hash")
                for _, metadata in existing_records
            }

            # -------------------------------------------------
            # SKIP UNCHANGED DOCUMENT
            # -------------------------------------------------

            if (
                current_document_hash in existing_hashes
                and len(existing_records) == len(file_chunks)
            ):
                logger.info(
                    "Document already indexed: %s",
                    filename,
                )
                continue

            # -------------------------------------------------
            # DELETE OLD VERSION
            # -------------------------------------------------

            old_ids = [
                record[0]
                for record in existing_records
            ]

            if old_ids:

                logger.info(
                    "Updating document: %s",
                    filename,
                )

                self.collection.delete(
                    ids=old_ids
                )

            # -------------------------------------------------
            # CREATE EMBEDDINGS
            # -------------------------------------------------

            texts = [
                chunk["text"]
                for chunk in file_chunks
            ]

            logger.info(
                "Creating embeddings for %s (%s chunks)",
                filename,
                len(texts),
            )

            embeddings = self.create_embeddings(
                texts
            )

            # -------------------------------------------------
            # STORE IN CHROMADB
            # -------------------------------------------------

            self.collection.add(
                ids=[
                    chunk["id"]
                    for chunk in file_chunks
                ],
                documents=texts,
                embeddings=embeddings,
                metadatas=[
                    {
                        "filename": chunk["filename"],
                        "chunk_index": chunk["chunk_index"],
                        "document_hash": chunk["document_hash"],
                    }
                    for chunk in file_chunks
                ],
            )

            logger.info(
                "Indexed document: %s",
                filename,
            )

    # -----------------------------------------------------
    # RETRIEVE RELEVANT DOCUMENTS
    # -----------------------------------------------------

    def retrieve(
        self,
        question: str,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Find the most relevant document chunks.
        """

        number_of_results = (
            top_k
            if top_k is not None
            else self.settings.TOP_K
        )

        query_embedding = self.create_embeddings(
            [question]
        )[0]

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=number_of_results,
        )

        documents = results.get(
            "documents",
            [[]],
        )[0]

        metadatas = results.get(
            "metadatas",
            [[]],
        )[0]

        retrieved: list[dict[str, Any]] = []

        for text, metadata in zip(
            documents,
            metadatas,
        ):

            if not metadata:
                metadata = {}

            retrieved.append(
                {
                    "text": text,
                    "metadata": metadata,
                }
            )

        return retrieved