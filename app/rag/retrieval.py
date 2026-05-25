from collections import defaultdict
from typing import List

from rank_bm25 import BM25Okapi
from langchain_core.documents import Document


class MedicalHybridRetriever:
    """
    Hybrid Retrieval System untuk Medical RAG.

    Features:
    - Dense retrieval (vector search)
    - Sparse retrieval (BM25)
    - Hybrid fusion
    - Recency weighting
    - Citation formatting
    """

    def __init__(
        self,
        vectordb,
        documents: List[Document],
        top_k: int = 5,
    ):
        """
        Parameters
        ----------
        vectordb:
            ChromaDB instance

        documents:
            semua chunks untuk BM25

        top_k:
            jumlah hasil retrieval
        """

        self.vectordb = vectordb

        self.documents = documents

        self.top_k = top_k

        # =====================================================
        # BM25 SETUP
        # =====================================================

        self.tokenized_docs = [
            doc.page_content.split()
            for doc in self.documents
        ]

        self.bm25 = BM25Okapi(
            self.tokenized_docs
        )

        print(
            f"[INFO] BM25 initialized "
            f"with {len(self.documents)} chunks"
        )

    # =========================================================
    # DENSE RETRIEVAL
    # =========================================================
    def dense_search(
        self,
        query: str,
        k: int = 5,
    ) -> List[tuple]:
        """
        Dense retrieval menggunakan ChromaDB.
        """

        results = self.vectordb.similarity_search_with_score(
            query,
            k=k,
        )

        return results

    # =========================================================
    # SPARSE RETRIEVAL
    # =========================================================
    def sparse_search(
        self,
        query: str,
        k: int = 5,
    ) -> List[tuple]:
        """
        Sparse retrieval menggunakan BM25.
        """

        tokenized_query = query.split()

        scores = self.bm25.get_scores(
            tokenized_query
        )

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:k]

        return [
            (self.documents[i], float(scores[i]))
            for i in ranked_indices
        ]

    # =========================================================
    # SCORE NORMALIZATION
    # =========================================================
    def normalize_scores(
        self,
        score_list: List[float],
    ) -> List[float]:
        if not score_list:
            return []

        min_score = min(score_list)
        max_score = max(score_list)

        if min_score == max_score:
            return [1.0 for _ in score_list]

        return [
            (score - min_score) / (max_score - min_score)
            for score in score_list
        ]

    # =========================================================
    # RECENCY WEIGHTING
    # =========================================================
    def recency_boost(
        self,
        document: Document,
    ) -> float:
        """
        Tambahan score berdasarkan tahun guideline.
        """

        year = document.metadata.get("year")

        if year and isinstance(year, int):

            # contoh:
            # 2025 -> 0.25
            # 2024 -> 0.24

            return (year - 2000) * 0.01

        return 0.0

    # =========================================================
    # DOCUMENT UNIQUE KEY
    # =========================================================
    def get_doc_key(
        self,
        document: Document,
    ) -> str:
        """
        Unique identifier document.
        """

        metadata = document.metadata

        return (
            f"{metadata.get('file_name')}_"
            f"{metadata.get('page')}_"
            f"{metadata.get('chunk_id')}"
        )

    # =========================================================
    # HYBRID RETRIEVAL
    # =========================================================
    def hybrid_search(
        self,
        query: str,
        k_dense: int = 5,
        k_sparse: int = 5,
        alpha: float = 0.7,
        beta: float = 0.3,
    ) -> List[Document]:
        """
        Hybrid retrieval menggunakan:
        - Dense Retrieval
        - BM25
        - Score fusion dengan normalisasi
        """

        dense_results = self.dense_search(
            query,
            k=k_dense,
        )

        sparse_results = self.sparse_search(
            query,
            k=k_sparse,
        )

        dense_scores = [score for _, score in dense_results]
        sparse_scores = [score for _, score in sparse_results]

        dense_norm = self.normalize_scores(dense_scores)
        sparse_norm = self.normalize_scores(sparse_scores)

        scores = defaultdict(float)
        docs_map = {}

        for (doc, _), score_norm in zip(dense_results, dense_norm):
            doc_key = self.get_doc_key(doc)
            docs_map[doc_key] = doc
            scores[doc_key] += alpha * score_norm

        for (doc, _), score_norm in zip(sparse_results, sparse_norm):
            doc_key = self.get_doc_key(doc)
            docs_map[doc_key] = doc
            scores[doc_key] += beta * score_norm

        for doc_key, doc in docs_map.items():
            scores[doc_key] += self.recency_boost(doc)

        ranked_docs = sorted(
            docs_map.values(),
            key=lambda d: scores[self.get_doc_key(d)],
            reverse=True,
        )

        return ranked_docs[: self.top_k]

    # =========================================================
    # FORMAT CONTEXT
    # =========================================================
    def format_context(
        self,
        documents: List[Document],
    ) -> str:
        """
        Format retrieval context untuk prompt LLM.
        """

        context_blocks = []

        for idx, doc in enumerate(
            documents,
            start=1,
        ):

            metadata = doc.metadata

            citation = (
                f"{metadata.get('file_name')} "
                f"(halaman {metadata.get('page')})"
            )

            block = (
                f"[SUMBER {idx}]\n"
                f"{citation}\n\n"
                f"{doc.page_content}"
            )

            context_blocks.append(block)

        return "\n\n".join(context_blocks)

    # =========================================================
    # DEBUG RETRIEVAL
    # =========================================================
    def debug_results(
        self,
        documents: List[Document],
    ):
        """
        Print hasil retrieval.
        """

        print(
            "\n========== RETRIEVAL RESULTS ==========\n"
        )

        for idx, doc in enumerate(
            documents,
            start=1,
        ):

            print(f"[{idx}]")

            print("Metadata:")
            print(doc.metadata)

            print("\nContent Preview:")
            print(doc.page_content[:500])

            print("\n" + "=" * 50 + "\n")