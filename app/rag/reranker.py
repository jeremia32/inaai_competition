from sentence_transformers import (
    CrossEncoder
)


class MedicalReranker:
    """
    Medical retrieval reranker.
    """

    def __init__(self):

        print(
            "\nLoading reranker model...\n"
        )

        self.model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    # =====================================================
    # RERANK
    # =====================================================

    def rerank(
        self,
        query: str,
        results: list,
        top_k: int = 3,
    ):
        """
        Rerank retrieval results.
        """

        # ================================================
        # CREATE QUERY-DOCUMENT PAIRS
        # ================================================

        pairs = []

        for doc in results:

            pairs.append(
                (
                    query,
                    doc.page_content
                )
            )

        # ================================================
        # PREDICT SCORES
        # ================================================

        scores = self.model.predict(
            pairs
        )

        # ================================================
        # COMBINE RESULTS
        # ================================================

        scored_results = []

        for doc, score in zip(
            results,
            scores
        ):

            scored_results.append({

                "document": doc,

                "rerank_score": float(score)
            })

        # ================================================
        # SORT
        # ================================================

        scored_results = sorted(

            scored_results,

            key=lambda x:
            x["rerank_score"],

            reverse=True
        )

        return scored_results[:top_k]

    # =====================================================
    # DEBUG
    # =====================================================

    def debug_rerank(
        self,
        reranked_results,
    ):

        print(
            "\n========== RERANK RESULTS ==========\n"
        )

        for idx, item in enumerate(
            reranked_results
        ):

            doc = item["document"]

            score = item[
                "rerank_score"
            ]

            print(
                f"\n[{idx+1}] Score: {score:.4f}"
            )

            print(
                f"Source: "
                f"{doc.metadata.get('source')}"
            )

            print(
                doc.page_content[:300]
            )

            print("\n-------------------")