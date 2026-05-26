import json
import re
from difflib import SequenceMatcher
from typing import Any, Dict, Optional

from app.reasoning.gemini_client import GeminiMedicalLLM


class MedicalLLMJudge:
    """Judge medical answers for faithfulness and relevance."""

    def __init__(self, llm: Optional[GeminiMedicalLLM] = None):
        self.llm = llm

    @staticmethod
    def normalize_text(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)
        return text

    @staticmethod
    def _parse_json_output(text: str) -> Optional[Dict[str, Any]]:
        text = text.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            return None

        try:
            payload = text[start : end + 1]
            return json.loads(payload)
        except json.JSONDecodeError:
            return None

    def build_prompt(
        self,
        query: str,
        generated_answer: str,
        reference_answer: str,
        context: Optional[str] = None,
    ) -> str:
        safe_context = context or "Tidak ada context tambahan."
        return f"""
Anda adalah penilai medis objektif.

Berikan dua skor integer antara 0 dan 100:
1. faithfulness: seberapa baik jawaban sesuai dengan referensi medis tanpa berhalusinasi.
2. relevance: seberapa relevan jawaban terhadap pertanyaan pasien.

Context:
{safe_context}

Pertanyaan pasien:
{query}

Jawaban model:
{generated_answer}

Jawaban referensi:
{reference_answer}

Output harus berupa JSON valid dengan kunci: faithfulness, relevance, comment.
Contoh:
{{"faithfulness": 84, "relevance": 92, "comment": "Jawaban mencakup inti pertanyaan tetapi melewatkan beberapa detail."}}
""".strip()

    def evaluate(
        self,
        query: str,
        generated_answer: str,
        reference_answer: str,
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        if self.llm is not None:
            prompt = self.build_prompt(
                query=query,
                generated_answer=generated_answer,
                reference_answer=reference_answer,
                context=context,
            )
            try:
                raw_output = self.llm.generate(prompt)
                parsed = self._parse_json_output(raw_output)
                if parsed is not None:
                    return {
                        "faithfulness": int(parsed.get("faithfulness", 0)),
                        "relevance": int(parsed.get("relevance", 0)),
                        "comment": parsed.get("comment", "") or "",
                        "raw_output": raw_output,
                    }
            except Exception:
                pass

        return self._heuristic_score(
            query=query,
            generated_answer=generated_answer,
            reference_answer=reference_answer,
        )

    def _heuristic_score(
        self,
        query: str,
        generated_answer: str,
        reference_answer: str,
    ) -> Dict[str, Any]:
        reference_norm = self.normalize_text(reference_answer)
        generated_norm = self.normalize_text(generated_answer)
        faithfulness = SequenceMatcher(None, reference_norm, generated_norm).ratio()

        query_terms = set(self.normalize_text(query).split())
        generated_terms = set(generated_norm.split())
        if not query_terms:
            relevance = 0.0
        else:
            overlap = len(query_terms & generated_terms)
            relevance = overlap / len(query_terms)

        return {
            "faithfulness": int(faithfulness * 100),
            "relevance": int(relevance * 100),
            "comment": (
                "Fallback judge used heuristics because a dedicated LLM judge was unavailable."
            ),
            "raw_output": "",
        }
