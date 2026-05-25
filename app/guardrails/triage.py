from typing import Dict, List


class MedicalTriageGuardrails:
    """
    Medical safety guardrails.

    Features:
    - Emergency symptom detection
    - High-risk query detection
    - Self-harm detection
    """

    def __init__(self):

        # =====================================================
        # EMERGENCY KEYWORDS
        # =====================================================

        self.emergency_keywords = [

            # cardiac
            "nyeri dada",
            "sesak napas",
            "serangan jantung",

            # neuro
            "stroke",
            "kejang",
            "tidak sadar",

            # respiratory
            "sulit bernapas",

            # emergency bleeding
            "pendarahan hebat",

            # self harm
            "bunuh diri",
            "ingin mati",

            # severe symptoms
            "kelumpuhan",
            "pingsan",
        ]

        # =====================================================
        # HIGH RISK KEYWORDS
        # =====================================================

        self.high_risk_keywords = [

            "dosis obat",

            "hentikan obat",

            "diagnosis sendiri",

            "overdosis",

            "campur obat",
        ]

    # =========================================================
    # DETECT EMERGENCY
    # =========================================================
    def detect_emergency(
        self,
        query: str,
    ) -> bool:
        """
        Detect emergency symptoms.
        """

        query_lower = query.lower()

        for keyword in (
            self.emergency_keywords
        ):

            if keyword in query_lower:
                return True

        return False

    # =========================================================
    # DETECT HIGH RISK
    # =========================================================
    def detect_high_risk(
        self,
        query: str,
    ) -> bool:
        """
        Detect dangerous medical intent.
        """

        query_lower = query.lower()

        for keyword in (
            self.high_risk_keywords
        ):

            if keyword in query_lower:
                return True

        return False

    # =========================================================
    # TRIAGE QUERY
    # =========================================================
    def triage(
        self,
        query: str,
    ) -> Dict:
        """
        Medical query triage.
        """

        emergency = self.detect_emergency(
            query
        )

        high_risk = self.detect_high_risk(
            query
        )

        risk_level = "LOW"

        if emergency:
            risk_level = "EMERGENCY"

        elif high_risk:
            risk_level = "HIGH"

        return {
            "query": query,
            "emergency": emergency,
            "high_risk": high_risk,
            "risk_level": risk_level,
        }

    # =========================================================
    # SAFE RESPONSE
    # =========================================================
    def generate_guardrail_message(
        self,
        triage_result: Dict,
    ) -> str:
        """
        Generate safety response.
        """

        risk_level = (
            triage_result["risk_level"]
        )

        if risk_level == "EMERGENCY":

            return (
                "⚠️ Gejala yang Anda sebutkan "
                "dapat mengindikasikan kondisi "
                "darurat medis. "
                "Segera hubungi dokter "
                "atau kunjungi IGD terdekat."
            )

        elif risk_level == "HIGH":

            return (
                "⚠️ Pertanyaan Anda termasuk "
                "kategori medis berisiko tinggi. "
                "Konsultasikan dengan "
                "tenaga medis profesional."
            )

        return (
            "✅ Query termasuk kategori aman."
        )

    # =========================================================
    # DEBUG
    # =========================================================
    def debug_triage(
        self,
        query: str,
    ):

        result = self.triage(query)

        print(
            "\n========== TRIAGE RESULT ==========\n"
        )

        print(result)

        print("\nSafety Message:\n")

        print(
            self.generate_guardrail_message(
                result
            )
        )