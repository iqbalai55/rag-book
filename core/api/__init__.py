"""HTTP credit/burn/refund protocol for the KitaPandu rag-book service.

Owns the entire *credit-check → burn → invoke service → refund-on-Exception →
JSON error shape* protocol for credit-backed HTTP endpoints. Service modules
(`core.services.*`) and the streaming agent do not import `CreditManager` —
the credit concern is owned at this seam.

Public interface (the test surface):

- `book_feature_endpoint(feature_name, user_id, payload, credit_manager,
  service_call)` returns a FastAPI `JSONResponse`. The protocol is:
    1. `credit_manager.check_sufficient_credits(user_id, feature_name)` →
       402 if not sufficient (no burn, no service call).
    2. `credit_manager.burn_credits(user_id, feature_name)` → 500 if burn
       fails (no service call).
    3. `service_call(user_id, payload)` → service result, wrapped in
       `JSONResponse`. A `dict` return value is JSON-serialised. An
       already-built `JSONResponse` is passed through.
    4. On `HTTPException(4xx)` re-raised from the service: re-raise without
       refund (a 4xx must not consume a credit).
    5. On any other `Exception`: `credit_manager.refund_credits(...)`, then
       return the existing JSON-500 shape with the exception message.

- `book_feature_endpoint_streaming(feature_name, user_id, payload,
  credit_manager, stream_call)` returns a FastAPI `StreamingResponse`. The
  credit is burned *before* the first byte is yielded. Insufficient credits
  return 402 before the stream starts. A 4xx `HTTPException` raised before
  any yield re-raises without refund. An unexpected exception after burning
  triggers a refund and yields the existing SSE error chunk followed by
  `[DONE]`.

`rate-limit` (`@limiter.limit`) decoration is *not* the concern of this
module; route declarations carry the decorator. This module does not import
`slowapi` or `fastapi.Depends`.

The module is the *only* place (outside `core/utils/credit_manager.py` and
the Ingest routes, which are out of scope for C1) that imports
`CreditManager`. Service modules must not import `CreditManager`.
"""
from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, Awaitable, Callable, Union

from fastapi import HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

logger = logging.getLogger(__name__)


# Service-callable signature for the JSON variant. Returns either a dict
# (serialised into a JSONResponse) or a pre-built JSONResponse.
ServiceCall = Callable[[str, Any], Awaitable[Union[dict, JSONResponse]]]

# Stream-callable signature for the streaming variant. Returns an async
# iterator of bytes/strings (SSE chunks).
StreamCall = Callable[[str, Any], AsyncIterator[str]]


def _error_response(message: str) -> JSONResponse:
    """Match the existing 500 shape used by the pre-refactor handlers:
    `{"status": "error", "message": <str>}`."""
    return JSONResponse({"status": "error", "message": message}, status_code=500)


def _insufficient_credits_response(feature_name: str, cost: float) -> JSONResponse:
    """Match the existing 402 shape used by the pre-refactor handlers:
    `{"detail": "Insufficient credits. Balance too low for feature '<name>'
    (cost: <cost>)"}`."""
    return JSONResponse(
        {
            "detail": (
                f"Insufficient credits. Balance too low for feature "
                f"'{feature_name}' (cost: {cost})"
            )
        },
        status_code=402,
    )


async def book_feature_endpoint(
    *,
    feature_name: str,
    user_id: str,
    credit_manager: Any,
    service_call: ServiceCall,
    payload: Any = None,
) -> JSONResponse:
    """JSON-variant of the credit protocol. See module docstring."""
    sufficient, cost = credit_manager.check_sufficient_credits(user_id, feature_name)
    if not sufficient:
        return _insufficient_credits_response(feature_name, cost)

    if not credit_manager.burn_credits(user_id, feature_name):
        return _error_response("Failed to deduct credits")

    try:
        result = await service_call(user_id, payload)
    except HTTPException:
        # 4xx must not consume a credit.
        raise
    except Exception as exc:
        logger.exception("book_feature_endpoint[%s] service raised", feature_name)
        try:
            credit_manager.refund_credits(user_id, feature_name)
        except Exception:
            logger.exception(
                "book_feature_endpoint[%s] refund failed for user %s", feature_name, user_id
            )
        return _error_response(str(exc))

    if isinstance(result, JSONResponse):
        return result
    return JSONResponse(result)


async def book_feature_endpoint_streaming(
    *,
    feature_name: str,
    user_id: str,
    credit_manager: Any,
    stream_call: StreamCall,
    payload: Any = None,
) -> StreamingResponse:
    """Streaming variant of the credit protocol. The credit is burned *before*
    the first byte is yielded. Insufficient credits return 402 before the
    stream starts. An unexpected exception during streaming triggers
    `refund_credits` and yields the existing SSE error chunk followed by
    `[DONE]`. A 4xx `HTTPException` raised before any yield re-raises
    without refund (no bytes were sent, the response is still
    `HTTPException`)."""
    sufficient, cost = credit_manager.check_sufficient_credits(user_id, feature_name)
    if not sufficient:
        return _insufficient_credits_response(feature_name, cost)

    if not credit_manager.burn_credits(user_id, feature_name):
        return _error_response("Failed to deduct credits")

    async def event_generator():
        try:
            async for chunk in stream_call(user_id, payload):
                yield chunk
        except HTTPException:
            # 4xx before any byte: re-raise to FastAPI (no refund).
            raise
        except Exception as exc:
            logger.exception(
                "book_feature_endpoint_streaming[%s] stream raised", feature_name
            )
            try:
                credit_manager.refund_credits(user_id, feature_name)
            except Exception:
                logger.exception(
                    "book_feature_endpoint_streaming[%s] refund failed for user %s",
                    feature_name,
                    user_id,
                )
            yield f"data: {json.dumps({'id': 'chatcmpl', 'type': 'error', 'content': str(exc), 'metadata': {}})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
