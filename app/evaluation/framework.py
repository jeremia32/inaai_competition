import re
import statistics
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

from app.guardrails.triage import MedicalTriageGuardrails
from app.pii.redact import MedicalPIIRedactor
from app.rag.retrieval import MedicalHybridRetriever
from app.reasoning.cot import MedicalPromptBuilder
from app.reasoning.gemini_client import GeminiMedicalLLM


@dataclass
class EvaluationExample:
    query: str
    reference_answer: str
    relevant_sources: List[Dict[str, Any]]
    description: Optional[str] = None


@dataclass
class EvaluationResult:
    query: str
    blocked: bool
    safety_message: str
    mrr_at_5: Optional[float]
    factual_score: Optional[float]
    redaction_rate: float
    latency_seconds: float
    generated_answer: Optional[str] = None
    error: Optional[str] = None


class MedicalEvaluator:
    """Evaluation framework for the Clinical RAG agent."""

    def __init__(
        self,
        redactor: MedicalPIIRedactor,
        guardrails: MedicalTriageGuardrails,
        retriever: MedicalHybridRetriever,
        prompt_builder: MedicalPromptBuilder,
        llm: Optional[GeminiMedicalLLM] = None,
        top_k: int = 5,
    ):
        self.redactor = redactor
        self.guardrails = guardrails
        self.retriever = retriever
        self.prompt_builder = prompt_builder
        self.llm = llm
        self.top_k = top_k

    @staticmethod
    def normalize_text(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^a-z0-9\[\]]+", " ", text)
        return text

    def match_relevant_source(
        self,
        document: Any,
        relevant_source: Dict[str, Any],
    ) -> bool:
        for key, value in relevant_source.items():
            if document.metadata.get(key) != value:
                return False
        return True

    def compute_mrr_at_k(
        self,
        query: str,
        relevant_sources: List[Dict[str, Any]],
        k: int = 5,
    ) -> Optional[float]:
        if not relevant_sources:
            return None

        results = self.retriever.hybrid_search(
            query=query,
            k_dense=k,
            k_sparse=k,
        )

        for index, doc in enumerate(results, start=1):
            for relevant in relevant_sources:
                if self.match_relevant_source(doc, relevant):
                    return 1.0 / index

        return 0.0

    def compute_pii_redaction_rate(
        self,
        original_text: str,
    ) -> float:
        original_findings = self.redactor.detect_pii(
            original_text
        )
        original_count = sum(
            len(matches)
            for matches in original_findings.values()
        )

        if original_count == 0:
            return 1.0

        redacted = self.redactor.redact(original_text)
        after_findings = self.redactor.detect_pii(
            redacted
        )
        after_count = sum(
            len(matches)
            for matches in after_findings.values()
        )

        return 1.0 - (after_count / original_count)

    def compute_factual_score(
        self,
        generated_answer: str,
        reference: str,
    ) -> float:
        if not generated_answer or not reference:
            return 0.0

        generated_norm = self.normalize_text(generated_answer)
        reference_norm = self.normalize_text(reference)

        return SequenceMatcher(
            None,
            generated_norm,
            reference_norm,
        ).ratio()

    def evaluate_example(
        self,
        example: EvaluationExample,
    ) -> EvaluationResult:
        start = time.perf_counter()

        safe_query = self.redactor.redact(example.query)
        triage_result = self.guardrails.triage(safe_query)
        safety_message = self.guardrails.generate_guardrail_message(
            triage_result
        )

        if triage_result["risk_level"] != "LOW":
            latency = time.perf_counter() - start
            redaction_rate = self.compute_pii_redaction_rate(
                example.query
            )
            return EvaluationResult(
                query=example.query,
                blocked=True,
                safety_message=safety_message,
                mrr_at_5=None,
                factual_score=None,
                redaction_rate=redaction_rate,
                latency_seconds=latency,
                generated_answer=safety_message,
            )

        try:
            mrr_value = self.compute_mrr_at_k(
                safe_query,
                example.relevant_sources,
                k=self.top_k,
            )

            results = self.retriever.hybrid_search(
                query=safe_query,
                k_dense=self.top_k,
                k_sparse=self.top_k,
            )

            context = self.retriever.format_context(results)
            safe_context = self.redactor.redact(context)
            final_prompt = self.prompt_builder.build_prompt(
                query=safe_query,
                context=safe_context,
            )

            if self.llm is None:
                generated_answer = None
                factual_score = None
            else:
                generated_answer = self.llm.generate(final_prompt)
                factual_score = self.compute_factual_score(
                    generated_answer,
                    example.reference_answer,
                )

            redaction_rate = self.compute_pii_redaction_rate(
                example.query + "\n" + context
            )
            latency = time.perf_counter() - start

            return EvaluationResult(
                query=example.query,
                blocked=False,
                safety_message=safety_message,
                mrr_at_5=mrr_value,
                factual_score=factual_score,
                redaction_rate=redaction_rate,
                latency_seconds=latency,
                generated_answer=generated_answer,
            )

        except Exception as exc:
            latency = time.perf_counter() - start
            return EvaluationResult(
                query=example.query,
                blocked=False,
                safety_message=safety_message,
                mrr_at_5=None,
                factual_score=None,
                redaction_rate=0.0,
                latency_seconds=latency,
                generated_answer=None,
                error=str(exc),
            )

    def summarize_results(
        self,
        results: List[EvaluationResult],
    ) -> Dict[str, Any]:
        mrr_scores = [r.mrr_at_5 for r in results if r.mrr_at_5 is not None]
        factual_scores = [
            r.factual_score for r in results if r.factual_score is not None
        ]
        redaction_rates = [r.redaction_rate for r in results]
        latencies = [r.latency_seconds for r in results]

        return {
            "num_examples": len(results),
            "mrr_at_5": statistics.mean(mrr_scores) if mrr_scores else None,
            "factual_faithfulness": statistics.mean(factual_scores) if factual_scores else None,
            "pii_redaction_rate": statistics.mean(redaction_rates),
            "average_latency_seconds": statistics.mean(latencies),
            "p95_latency_seconds": statistics.quantiles(latencies, n=20)[-1]
            if len(latencies) >= 2
            else None,
        }
