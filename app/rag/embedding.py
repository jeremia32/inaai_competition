import torch

from app.config import EMBEDDING_MODEL
from langchain_community.embeddings import HuggingFaceEmbeddings


class MedicalEmbeddingModel:
    """
    Embedding model untuk Medical RAG.

    Features:
    - GPU auto detection
    - Normalized embeddings
    - Configurable model
    - Compatible with ChromaDB
    """

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
    ):
        """
        Parameters
        ----------
        model_name:
            HuggingFace embedding model
        """

        self.model_name = model_name

        # auto detect GPU
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(f"[INFO] device: {self.device}")
        print(f"[INFO] embedding model: {self.model_name}")

        self.embedding_model = HuggingFaceEmbeddings(
            model_name=self.model_name,

            model_kwargs={
                "device": self.device,
            },

            encode_kwargs={
                "normalize_embeddings": True,
            },
        )

    # =========================================================
    # GET MODEL
    # =========================================================
    def get_model(self):
        """
        Return embedding model.
        """

        return self.embedding_model

    # =========================================================
    # EMBED QUERY
    # =========================================================
    def embed_query(self, query: str):
        """
        Generate embedding untuk query.
        """

        embedding = self.embedding_model.embed_query(
            query
        )

        return embedding

    # =========================================================
    # EMBED DOCUMENTS
    # =========================================================
    def embed_documents(self, texts):
        """
        Generate embedding untuk banyak text.
        """

        embeddings = (
            self.embedding_model.embed_documents(
                texts
            )
        )

        return embeddings

    # =========================================================
    # TEST EMBEDDING
    # =========================================================
    def test_embedding(self):
        """
        Test embedding model.
        """

        sample_text = (
            "Apa gejala diabetes mellitus?"
        )

        vector = self.embed_query(sample_text)

        print("\n===== EMBEDDING TEST =====")

        print(f"Text: {sample_text}")

        print(f"Vector dimension: {len(vector)}")

        print(f"First 10 values:")
        print(vector[:10])

        print("\n[INFO] embedding model working.")