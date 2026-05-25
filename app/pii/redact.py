import re
from typing import Dict


class MedicalPIIRedactor:
    """
    PII Redaction untuk Medical AI.

    Features:
    - NIK masking
    - Phone masking
    - Email masking
    - Simple name masking
    """

    def __init__(self):

        # =====================================================
        # REGEX PATTERNS
        # =====================================================

        self.patterns = {

            # NIK Indonesia (16 digit)
            "NIK": r"\b\d{16}\b",

            # nomor telepon Indonesia
            "PHONE": (
                r"(\+62|62|0)"
                r"8[1-9][0-9]{6,10}"
            ),

            # email
            "EMAIL": (
                r"\b[A-Za-z0-9._%+-]+"
                r"@[A-Za-z0-9.-]+"
                r"\.[A-Z|a-z]{2,}\b"
            ),

            # nama sederhana
            # contoh:
            # Nama: Budi Santoso
            "NAME": (
                r"(Nama\s*:\s*)"
                r"([A-Z][a-z]+"
                r"(?:\s[A-Z][a-z]+)*)"
            ),
        }

    # =========================================================
    # REDACT TEXT
    # =========================================================
    def redact(
        self,
        text: str,
    ) -> str:
        """
        Redact PII dari text.
        """

        redacted_text = text

        # =====================================================
        # REDACT NIK
        # =====================================================

        redacted_text = re.sub(
            self.patterns["NIK"],
            "[REDACTED_NIK]",
            redacted_text,
        )

        # =====================================================
        # REDACT PHONE
        # =====================================================

        redacted_text = re.sub(
            self.patterns["PHONE"],
            "[REDACTED_PHONE]",
            redacted_text,
        )

        # =====================================================
        # REDACT EMAIL
        # =====================================================

        redacted_text = re.sub(
            self.patterns["EMAIL"],
            "[REDACTED_EMAIL]",
            redacted_text,
        )

        # =====================================================
        # REDACT NAME
        # =====================================================

        redacted_text = re.sub(
            self.patterns["NAME"],
            r"\1[REDACTED_NAME]",
            redacted_text,
        )

        return redacted_text

    # =========================================================
    # DETECT PII
    # =========================================================
    def detect_pii(
        self,
        text: str,
    ) -> Dict:
        """
        Detect PII tanpa redact.
        """

        findings = {}

        for pii_type, pattern in (
            self.patterns.items()
        ):

            matches = re.findall(
                pattern,
                text,
            )

            findings[pii_type] = matches

        return findings

    # =========================================================
    # DEBUG FUNCTION
    # =========================================================
    def debug_redaction(
        self,
        text: str,
    ):
        """
        Debug hasil redaction.
        """

        print("\n========== ORIGINAL ==========\n")

        print(text)

        redacted = self.redact(text)

        print("\n========== REDACTED ==========\n")

        print(redacted)