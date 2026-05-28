import logging
import time
from typing import Any, Dict, List, Optional, Sequence
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from core.utils.token_tracker import TokenTracker

logger = logging.getLogger(__name__)

# Feature detection from run metadata or tool names
FEATURE_MAP = {
    "search_book_context": "search",
    "generate_mcq": "generate_mcq",
    "generate_essay_questions": "generate_essay",
    "generate_complete_podcast": "podcast",
    "summarization": "summarization",
    "agent_reasoning": "agent_reasoning",
}


class TokenUsageCallbackHandler(BaseCallbackHandler):
    """LangChain callback handler for tracking token usage."""

    def __init__(
        self,
        session_id: Optional[str] = None,
        course_id: Optional[str] = None,
        feature: Optional[str] = None,
    ):
        super().__init__()
        self.session_id = session_id
        self.course_id = course_id
        self.feature = feature or "unknown"
        self.tracker = TokenTracker()
        self._start_times: Dict[UUID, float] = {}

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> Any:
        """Track LLM start time for latency calculation."""
        self._start_times[run_id] = time.time()

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> Any:
        """Extract token usage from LLM response and record it."""
        # Calculate latency
        start_time = self._start_times.pop(run_id, time.time())
        latency_ms = int((time.time() - start_time) * 1000)

        # Extract usage from response
        for generation_list in response.generations:
            for generation in generation_list:
                # Check for usage_metadata in generation (LangChain standard)
                usage_metadata = getattr(generation, "usage_metadata", None)
                if usage_metadata:
                    self._record_usage(
                        input_tokens=usage_metadata.get("input_tokens", 0),
                        output_tokens=usage_metadata.get("output_tokens", 0),
                        model=usage_metadata.get("model", "unknown"),
                        latency_ms=latency_ms,
                        metadata={"run_id": str(run_id)},
                    )
                    continue

                # Check in response_metadata (provider-specific)
                resp_meta = getattr(generation, "response_metadata", {})
                if "token_usage" in resp_meta:
                    token_usage = resp_meta["token_usage"]
                    self._record_usage(
                        input_tokens=token_usage.get("prompt_tokens", 0),
                        output_tokens=token_usage.get("completion_tokens", 0),
                        model=resp_meta.get("model_name", "unknown"),
                        latency_ms=latency_ms,
                        metadata={"run_id": str(run_id)},
                    )
                    continue

                # Check llm_output for usage
                if response.llm_output and "token_usage" in response.llm_output:
                    token_usage = response.llm_output["token_usage"]
                    self._record_usage(
                        input_tokens=token_usage.get("prompt_tokens", 0),
                        output_tokens=token_usage.get("completion_tokens", 0),
                        model=response.llm_output.get("model_name", "unknown"),
                        latency_ms=latency_ms,
                        metadata={"run_id": str(run_id)},
                    )
                    return  # Already recorded, skip other generations

        # Fallback: check llm_output directly
        if response.llm_output and "token_usage" in response.llm_output:
            token_usage = response.llm_output["token_usage"]
            self._record_usage(
                input_tokens=token_usage.get("prompt_tokens", 0),
                output_tokens=token_usage.get("completion_tokens", 0),
                model=response.llm_output.get("model_name", "unknown"),
                latency_ms=latency_ms,
                metadata={"run_id": str(run_id)},
            )

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> Any:
        """Handle LLM errors."""
        self._start_times.pop(run_id, None)
        logger.warning(f"LLM error (run_id={run_id}): {error}")

    def _record_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str,
        latency_ms: int = 0,
        metadata: Optional[Dict] = None,
    ):
        """Record token usage to the tracker."""
        # Detect provider from model name
        provider = "unknown"
        model_lower = model.lower()
        if "gpt" in model_lower or "o1" in model_lower or "o3" in model_lower:
            provider = "openai"
        elif "claude" in model_lower:
            provider = "anthropic"
        elif "llama" in model_lower or "mistral" in model_lower:
            provider = "openrouter"

        self.tracker.record(
            model=model,
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            feature=self.feature,
            session_id=self.session_id,
            course_id=self.course_id,
            latency_ms=latency_ms,
            metadata=metadata,
        )

    def set_context(
        self,
        session_id: Optional[str] = None,
        course_id: Optional[str] = None,
        feature: Optional[str] = None,
    ):
        """Update context for the callback handler."""
        if session_id is not None:
            self.session_id = session_id
        if course_id is not None:
            self.course_id = course_id
        if feature is not None:
            self.feature = feature


def create_token_callback(
    session_id: Optional[str] = None,
    course_id: Optional[str] = None,
    feature: Optional[str] = None,
) -> TokenUsageCallbackHandler:
    """Factory function to create a configured token callback handler."""
    return TokenUsageCallbackHandler(
        session_id=session_id,
        course_id=course_id,
        feature=feature,
    )
