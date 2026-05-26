import re
from typing import Optional

try:
    import tiktoken
except ImportError:
    tiktoken = None


class CostTracker:
    """Estimate token load and query cost for the medical RAG pipeline."""

    DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
    EMBEDDING_COST_PER_1K = 0.0004
    PROMPT_COST_PER_1K = 0.0012
    RESPONSE_COST_PER_1K = 0.0018
    BASE_QUERY_COST = 0.0001

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name

    def count_tokens(self, text: str, model_name: Optional[str] = None) -> int:
        """Estimate token usage for a text string."""
        model_name = model_name or self.model_name

        if not text:
            return 0

        if tiktoken is not None:
            try:
                encoder = tiktoken.encoding_for_model(model_name)
            except Exception:
                try:
                    encoder = tiktoken.get_encoding("cl100k_base")
                except Exception:
                    encoder = None

            if encoder is not None:
                return len(encoder.encode(text))

        return self._fallback_token_count(text)

    @staticmethod
    def _fallback_token_count(text: str) -> int:
        tokens = re.findall(r"\w+|[^\w\s]", text)
        return max(1, len(tokens))

    def estimate_response_tokens(self, prompt_tokens: int) -> int:
        """Estimate response token count from prompt size."""
        return max(32, int(prompt_tokens * 0.25))

    def estimate_query_cost(
        self,
        prompt_tokens: int,
        response_tokens: int,
        embedding_tokens: Optional[int] = None,
    ) -> float:
        """Estimate USD cost for a single query."""
        embedding_tokens = prompt_tokens if embedding_tokens is None else embedding_tokens
        prompt_cost = prompt_tokens / 1000.0 * self.PROMPT_COST_PER_1K
        response_cost = response_tokens / 1000.0 * self.RESPONSE_COST_PER_1K
        embedding_cost = embedding_tokens / 1000.0 * self.EMBEDDING_COST_PER_1K
        return round(self.BASE_QUERY_COST + prompt_cost + response_cost + embedding_cost, 8)

    def summarize_costs(self, total_cost: float, num_queries: int) -> dict:
        average_cost = total_cost / max(1, num_queries)
        return {
            "num_queries": num_queries,
            "total_cost_usd": round(total_cost, 6),
            "average_cost_usd": round(average_cost, 8),
            "estimated_cost_1000_usd": round(average_cost * 1000.0, 6),
        }
