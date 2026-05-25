import os
import shutil
from typing import List, Optional

from chromadb.errors import InvalidArgumentError
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma


class MedicalVectorDB:
    """
    Vector Database manager untuk Medical RAG.

    Features:
    - ChromaDB persistence
    - Save & load vector database
    - Similarity search
    - Metadata storage
    """

    def __init__(
        self,
        embedding_model,
        persist_directory: str = "./chroma_db",
        collection_name: str = "clinical_rag",
    ):
        """
        Parameters
        ----------
        embedding_model:
            HuggingFace embedding model

        persist_directory:
            lokasi penyimpanan ChromaDB

        collection_name:
            nama collection vector database
        """

        self.embedding_model = embedding_model

        self.persist_directory = persist_directory

        self.collection_name = collection_name

        self.vectordb: Optional[Chroma] = None

    # =========================================================
    # BUILD VECTOR DATABASE
    # =========================================================
    def build(
        self,
        documents: List[Document],
    ):
        """
        Build vector database dari chunks.
        """

        print(
            f"[INFO] building ChromaDB "
            f"with {len(documents)} chunks..."
        )

        try:
            self.vectordb = Chroma.from_documents(
                documents=documents,

                embedding=self.embedding_model,

                persist_directory=self.persist_directory,

                collection_name=self.collection_name,
            )
        except InvalidArgumentError as exc:
            message = str(exc).lower()
            if (
                "expecting embedding with dimension" in message
                or ("dimension" in message and "got" in message)
            ):
                print(
                    "[WARN] build failed due to stale Chroma collection",
                    "-> cleaning persist directory and retrying..."
                )
                self._cleanup_persist_directory()
                self.vectordb = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embedding_model,
                    persist_directory=self.persist_directory,
                    collection_name=self.collection_name,
                )
            else:
                raise

        print("[INFO] ChromaDB created successfully.")

        return self.vectordb

    # =========================================================
    # LOAD EXISTING DATABASE
    # =========================================================
    def load(self):
        """
        Load existing ChromaDB.
        """

        print(
            f"[INFO] loading ChromaDB from "
            f"{self.persist_directory}"
        )

        self.vectordb = Chroma(
            persist_directory=self.persist_directory,

            embedding_function=self.embedding_model,

            collection_name=self.collection_name,
        )

        print("[INFO] ChromaDB loaded successfully.")

        return self.vectordb

    # =========================================================
    # PERSISTENCE CHECK
    # =========================================================
    def exists(self) -> bool:
        """Check whether persisted Chroma data exists."""

        if not os.path.isdir(self.persist_directory):
            return False

        try:
            return bool(os.listdir(self.persist_directory))
        except OSError:
            return False

    # =========================================================
    # LOAD OR BUILD DATABASE
    # =========================================================
    def _cleanup_persist_directory(self):
        if os.path.isdir(self.persist_directory):
            shutil.rmtree(self.persist_directory, ignore_errors=True)

    def _check_embedding_dimension(self) -> bool:
        """Return True when the loaded DB does not match current embedding dimension."""
        if self.vectordb is None:
            return False

        try:
            self.vectordb.similarity_search_with_score("test", k=1)
            return False
        except InvalidArgumentError as exc:
            message = str(exc).lower()
            if "expecting embedding with dimension" in message or (
                "dimension" in message and "got" in message
            ):
                print(
                    "[WARN] embedding dimension mismatch detected",
                    "-> existing ChromaDB was built with a different model."
                )
                return True
            raise
        except Exception:
            return False

    def load_or_build(
        self,
        documents: List[Document],
    ):
        """Load an existing ChromaDB or build it when missing."""

        if self.exists():
            try:
                vectordb = self.load()
                if self._check_embedding_dimension():
                    self._cleanup_persist_directory()
                    return self.build(documents)
                return vectordb
            except Exception as exc:
                print(
                    "[WARN] Failed to load existing ChromaDB:",
                    exc,
                )
                print("[INFO] Rebuilding ChromaDB from documents...")
                self._cleanup_persist_directory()

        return self.build(documents)

    # =========================================================
    # SIMILARITY SEARCH
    # =========================================================
    def similarity_search(
        self,
        query: str,
        k: int = 3,
    ):
        """
        Dense retrieval menggunakan vector similarity.
        """

        if self.vectordb is None:
            raise ValueError(
                "Vector database belum di-load/build."
            )

        results = self.vectordb.similarity_search(
            query,
            k=k,
        )

        return results

    # =========================================================
    # SIMILARITY SEARCH WITH SCORE
    # =========================================================
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 3,
    ):
        """
        Retrieval dengan similarity score.
        """

        if self.vectordb is None:
            raise ValueError(
                "Vector database belum di-load/build."
            )

        results = (
            self.vectordb.similarity_search_with_score(
                query,
                k=k,
            )
        )

        return results

    # =========================================================
    # DATABASE INFO
    # =========================================================
    def info(self):
        """
        Menampilkan informasi database.
        """

        print("\n===== VECTOR DB INFO =====")

        print(f"Collection: {self.collection_name}")

        print(f"Persist Dir: {self.persist_directory}")

        if self.vectordb:
            try:
                count = self.vectordb._collection.count()

                print(f"Total vectors: {count}")

            except Exception:
                print("Unable to get vector count.")

        print("=" * 35)