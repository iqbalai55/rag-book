"""Tests for the credit/burn/refund/HTTP-error protocol in
`core.api.book_feature_endpoint`. These tests cross the new module's public
interface with a fake CreditManager and a fake service callable.

Behavioural contract verified here:
- 200 happy path: burn on entry, service called, no refund
- 402 insufficient credits: NO burn, NO service call
- 500 unexpected exception: burn + refund
- HTTPException(4xx) re-raise: NO refund (4xx must not consume a credit)
"""
from __future__ import annotations

import os
import sys
import json
from typing import List

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.api.book_feature_endpoint import (
    book_feature_endpoint,
    book_feature_endpoint_streaming,
)
from core.utils.credit_manager import FEATURE_COST


class FakeCreditManager:
    """Records the sequence of credit calls. No DB."""

    def __init__(self, *, sufficient: bool = True):
        self.calls: List[tuple] = []
        self._sufficient = sufficient

    def check_sufficient_credits(self, user_id: str, feature: str, estimated_cost: float = 0.0):
        self.calls.append(("check", user_id, feature, estimated_cost))
        cost = estimated_cost if estimated_cost > 0 else FEATURE_COST.get(feature, FEATURE_COST["unknown"])
        return self._sufficient, cost

    def burn_credits(self, user_id: str, feature: str, tokens_used: int = 0, estimated_cost: float = 0.0, metadata=None) -> bool:
        self.calls.append(("burn", user_id, feature, estimated_cost))
        return True

    def refund_credits(self, user_id: str, feature: str, tokens_used: int = 0, estimated_cost: float = 0.0, metadata=None) -> bool:
        self.calls.append(("refund", user_id, feature, estimated_cost))
        return True


def _build_app(fake_credit: FakeCreditManager, service_callable):
    """Build a minimal FastAPI app with a single route that uses
    `book_feature_endpoint`. The protocol module is the seam under test."""
    app = FastAPI()

    @app.post("/test/feature")
    async def route(user_id: str, payload: dict):
        return await book_feature_endpoint(
            feature_name="test_feature",
            user_id=user_id,
            credit_manager=fake_credit,
            service_call=service_callable,
        )

    return app


# ----------------------------------------------------------------- 200 path

def test_happy_path_calls_burn_then_service_no_refund():
    fake_credit = FakeCreditManager(sufficient=True)
    seen: List[dict] = []

    async def service(user_id: str, payload: dict):
        seen.append({"user_id": user_id, "payload": payload})
        return {"ok": True, "echo": payload}

    app = _build_app(fake_credit, service)
    client = TestClient(app)
    resp = client.post("/test/feature?user_id=u1", json={"x": 1})

    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "echo": {"x": 1}}
    assert seen == [{"user_id": "u1", "payload": {"x": 1}}]

    op_names = [c[0] for c in fake_credit.calls]
    assert op_names == ["check", "burn"], f"expected check+burn, got {op_names}"
    assert ("refund", *_ ) not in fake_credit.calls


# ----------------------------------------------------------------- 402 path

def test_insufficient_credits_returns_402_no_burn_no_service():
    fake_credit = FakeCreditManager(sufficient=False)
    service_called = []

    async def service(user_id: str, payload: dict):
        service_called.append(True)
        return {"ok": True}

    app = _build_app(fake_credit, service)
    client = TestClient(app)
    resp = client.post("/test/feature?user_id=u1", json={"x": 1})

    assert resp.status_code == 402
    body = resp.json()
    assert "Insufficient credits" in body["detail"]
    assert "test_feature" in body["detail"]
    assert service_called == [], "service must not be called on insufficient credits"

    op_names = [c[0] for c in fake_credit.calls]
    assert op_names == ["check"], f"only check should run, got {op_names}"


# ----------------------------------------------------------------- 500 path

def test_unexpected_exception_triggers_refund_and_500():
    fake_credit = FakeCreditManager(sufficient=True)
    service_called = []

    async def service(user_id: str, payload: dict):
        service_called.append(True)
        raise RuntimeError("kaboom")

    app = _build_app(fake_credit, service)
    client = TestClient(app)
    resp = client.post("/test/feature?user_id=u1", json={"x": 1})

    assert resp.status_code == 500
    body = resp.json()
    assert body.get("status") == "error"
    assert "kaboom" in body.get("message", "")

    op_names = [c[0] for c in fake_credit.calls]
    assert op_names == ["check", "burn", "refund"], f"check+burn+refund expected, got {op_names}"


# ----------------------------------------------- 4xx re-raise, no refund

def test_httpexception_reraise_does_not_refund():
    fake_credit = FakeCreditManager(sufficient=True)
    service_called = []

    async def service(user_id: str, payload: dict):
        service_called.append(True)
        raise HTTPException(status_code=404, detail="not found")

    app = _build_app(fake_credit, service)
    client = TestClient(app)
    resp = client.post("/test/feature?user_id=u1", json={"x": 1})

    assert resp.status_code == 404
    assert service_called == [True]

    op_names = [c[0] for c in fake_credit.calls]
    assert op_names == ["check", "burn"], f"4xx must not trigger refund, got {op_names}"


# --------------------------------------- streaming variant: 200 + chunks

def test_streaming_happy_path_yields_then_dones():
    fake_credit = FakeCreditManager(sufficient=True)

    async def stream_call(user_id: str, payload: dict):
        for chunk in ["hello", "world"]:
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"

    app = FastAPI()

    @app.post("/test/stream")
    async def route(user_id: str, payload: dict):
        return await book_feature_endpoint_streaming(
            feature_name="agent_reasoning",
            user_id=user_id,
            credit_manager=fake_credit,
            stream_call=stream_call,
        )

    client = TestClient(app)
    resp = client.post("/test/stream?user_id=u1", json={"q": "hi"})

    assert resp.status_code == 200
    body = "".join(chunk if isinstance(chunk, str) else chunk.decode() for chunk in resp.iter_lines())
    assert "hello" in body
    assert "world" in body
    assert "[DONE]" in body

    op_names = [c[0] for c in fake_credit.calls]
    assert op_names == ["check", "burn"]


def test_streaming_insufficient_credits_returns_402_before_any_chunk():
    fake_credit = FakeCreditManager(sufficient=False)
    stream_started = []

    async def stream_call(user_id: str, payload: dict):
        stream_started.append(True)
        yield "data: nope\n\n"

    app = FastAPI()

    @app.post("/test/stream")
    async def route(user_id: str, payload: dict):
        return await book_feature_endpoint_streaming(
            feature_name="agent_reasoning",
            user_id=user_id,
            credit_manager=fake_credit,
            stream_call=stream_call,
        )

    client = TestClient(app)
    resp = client.post("/test/stream?user_id=u1", json={"q": "hi"})

    assert resp.status_code == 402
    assert stream_started == [], "stream must not start when credits are insufficient"
    op_names = [c[0] for c in fake_credit.calls]
    assert op_names == ["check"]


def test_streaming_unexpected_exception_triggers_refund():
    fake_credit = FakeCreditManager(sufficient=True)

    async def stream_call(user_id: str, payload: dict):
        yield "data: partial\n\n"
        raise RuntimeError("stream blew up")

    app = FastAPI()

    @app.post("/test/stream")
    async def route(user_id: str, payload: dict):
        return await book_feature_endpoint_streaming(
            feature_name="agent_reasoning",
            user_id=user_id,
            credit_manager=fake_credit,
            stream_call=stream_call,
        )

    client = TestClient(app)
    resp = client.post("/test/stream?user_id=u1", json={"q": "hi"})

    assert resp.status_code == 200  # streaming already started
    body = "".join(chunk if isinstance(chunk, str) else chunk.decode() for chunk in resp.iter_lines())
    assert "error" in body  # error chunk is yielded
    assert "[DONE]" in body

    op_names = [c[0] for c in fake_credit.calls]
    assert op_names == ["check", "burn", "refund"]
