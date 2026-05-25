import re


class MedicalRecencyWeighter:
    """
    Medical recency weighting system.
    """

    def __init__(self):

        # ================================================
        # RECENCY BONUS
        # ================================================

        self.year_bonus = {

            2025: 1.0,
            2024: 0.9,
            2023: 0.8,
            2022: 0.7,
            2021: 0.6,
            2020: 0.5,
        }

    # =====================================================
    # EXTRACT YEAR
    # =====================================================

    def extract_year(
        self,
        text: str,
    ):
        """
        Extract year from source text.
        """

        years = re.findall(
            r"(20\d{2})",
            text
        )

        if years:

            return int(years[0])

        return None

    # =====================================================
    # APPLY RECENCY WEIGHT
    # =====================================================

    def apply_weighting(
        self,
        reranked_results,
    ):
        """
        Apply recency bonus.
        """

        weighted_results = []

        for item in reranked_results:

            doc = item["document"]

            rerank_score = item[
                "rerank_score"
            ]

            source = str(
                doc.metadata.get(
                    "source",
                    ""
                )
            )

            year = self.extract_year(
                source
            )

            bonus = 0.0

            if year in self.year_bonus:

                bonus = self.year_bonus[
                    year
                ]

            final_score = (
                rerank_score + bonus
            )

            weighted_results.append({

                "document": doc,

                "rerank_score":
                rerank_score,

                "year": year,

                "recency_bonus":
                bonus,

                "final_score":
                final_score,
            })

        # ================================================
        # SORT FINAL SCORE
        # ================================================

        weighted_results = sorted(

            weighted_results,

            key=lambda x:
            x["final_score"],

            reverse=True
        )

        return weighted_results

    # =====================================================
    # DEBUG
    # =====================================================

    def debug_results(
        self,
        weighted_results,
    ):

        print(
            "\n========== RECENCY RESULTS ==========\n"
        )

        for idx, item in enumerate(
            weighted_results
        ):

            doc = item["document"]

            print(
                f"\n[{idx+1}]"
            )

            print(
                f"Year: {item['year']}"
            )

            print(
                f"Rerank Score: "
                f"{item['rerank_score']:.4f}"
            )

            print(
                f"Recency Bonus: "
                f"{item['recency_bonus']:.2f}"
            )

            print(
                f"Final Score: "
                f"{item['final_score']:.4f}"
            )

            print(
                f"Source: "
                f"{doc.metadata.get('source')}"
            )

            print(
                doc.page_content[:200]
            )

            print("\n-------------------")