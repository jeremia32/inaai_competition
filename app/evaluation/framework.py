import re
import statistics
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

from app.evaluation.cost import CostTracker
from app.evaluation.judge import MedicalLLMJudge
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
    faithfulness_score: Optional[float]
    relevance_score: Optional[float]
    estimated_cost_usd: Optional[float]
    prompt_tokens: Optional[int]
    response_tokens: Optional[int]
    redaction_rate: float
    latency_seconds: float
    judge_comment: Optional[str] = None
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
        cost_tracker: Optional[CostTracker] = None,
        judge: Optional[MedicalLLMJudge] = None,
        top_k: int = 5,
    ):
        self.redactor = redactor
        self.guardrails = guardrails
        self.retriever = retriever
        self.prompt_builder = prompt_builder
        self.llm = llm
        self.cost_tracker = cost_tracker or CostTracker()
        self.judge = judge or MedicalLLMJudge(llm=llm)
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
                faithfulness_score=None,
                relevance_score=None,
                estimated_cost_usd=None,
                prompt_tokens=None,
                response_tokens=None,
                redaction_rate=redaction_rate,
                latency_seconds=latency,
                judge_comment=None,
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

            generated_answer = None
            factual_score = None
            faithfulness_score = None
            relevance_score = None
            judge_comment = None
            estimated_cost = None
            prompt_token_count = None
            response_token_count = None

            if self.llm is not None:
                generated_answer = self.llm.generate(final_prompt)
                factual_score = self.compute_factual_score(
                    generated_answer,
                    example.reference_answer,
                )

            prompt_token_count = self.cost_tracker.count_tokens(
                final_prompt
            )
            response_token_count = self.cost_tracker.estimate_response_tokens(
                prompt_token_count
            )
            estimated_cost = self.cost_tracker.estimate_query_cost(
                prompt_tokens=prompt_token_count,
                response_tokens=response_token_count,
            )

            judge_result = self.judge.evaluate(
                query=safe_query,
                generated_answer=generated_answer or "",
                reference_answer=example.reference_answer,
                context=safe_context,
            )
            faithfulness_score = judge_result.get("faithfulness")
            relevance_score = judge_result.get("relevance")
            judge_comment = judge_result.get("comment")

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
                faithfulness_score=faithfulness_score,
                relevance_score=relevance_score,
                estimated_cost_usd=estimated_cost,
                prompt_tokens=prompt_token_count,
                response_tokens=response_token_count,
                redaction_rate=redaction_rate,
                latency_seconds=latency,
                judge_comment=judge_comment,
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
                faithfulness_score=None,
                relevance_score=None,
                estimated_cost_usd=None,
                prompt_tokens=None,
                response_tokens=None,
                redaction_rate=0.0,
                latency_seconds=latency,
                judge_comment=None,
                generated_answer=None,
                error=str(exc),
            )

    def simulate_costs(
        self,
        examples: List[EvaluationExample],
        target_queries: int = 1000,
    ) -> Dict[str, Any]:
        samples = examples or []
        if not samples:
            return {
                "num_queries": 0,
                "total_estimated_cost_usd": 0.0,
                "average_cost_usd": 0.0,
                "estimated_cost_1000_usd": 0.0,
                "average_latency_seconds": None,
            }

        total_cost = 0.0
        latencies = []
        for index in range(target_queries):
            example = samples[index % len(samples)]
            safe_query = self.redactor.redact(example.query)
            triage_result = self.guardrails.triage(safe_query)
            if triage_result["risk_level"] != "LOW":
                latencies.append(0.0)
                continue

            start = time.perf_counter()
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
            prompt_tokens = self.cost_tracker.count_tokens(final_prompt)
            response_tokens = self.cost_tracker.estimate_response_tokens(
                prompt_tokens
            )
            estimated_cost = self.cost_tracker.estimate_query_cost(
                prompt_tokens=prompt_tokens,
                response_tokens=response_tokens,
            )
            total_cost += estimated_cost
            latencies.append(time.perf_counter() - start)

        return {
            "num_queries": target_queries,
            "total_estimated_cost_usd": round(total_cost, 6),
            "average_cost_usd": round(total_cost / target_queries, 8),
            "estimated_cost_1000_usd": round(total_cost, 6),
            "average_latency_seconds": (
                statistics.mean(latencies) if latencies else None
            ),
        }

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

        cost_values = [r.estimated_cost_usd for r in results if r.estimated_cost_usd is not None]
        faithfulness_values = [r.faithfulness_score for r in results if r.faithfulness_score is not None]
        relevance_values = [r.relevance_score for r in results if r.relevance_score is not None]

        return {
            "num_examples": len(results),
            "mrr_at_5": statistics.mean(mrr_scores) if mrr_scores else None,
            "factual_faithfulness": statistics.mean(factual_scores) if factual_scores else None,
            "faithfulness_score": statistics.mean(faithfulness_values) if faithfulness_values else None,
            "relevance_score": statistics.mean(relevance_values) if relevance_values else None,
            "average_cost_usd": statistics.mean(cost_values) if cost_values else None,
            "estimated_cost_1000_usd": (
                statistics.mean(cost_values) * 1000.0
                if cost_values
                else None
            ),
            "pii_redaction_rate": statistics.mean(redaction_rates),
            "average_latency_seconds": statistics.mean(latencies),
            "p95_latency_seconds": statistics.quantiles(latencies, n=20)[-1]
            if len(latencies) >= 2
            else None,
        }
