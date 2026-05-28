"""Tests for token tracker."""
import pytest
from unittest.mock import patch, MagicMock

from core.utils.token_tracker import TokenTracker, TokenUsageRecord, MODEL_PRICING


class TestTokenTrackerSingleton:
    def test_singleton_returns_same_instance(self):
        tracker1 = TokenTracker()
        tracker2 = TokenTracker()
        assert tracker1 is tracker2


class TestModelPricing:
    def test_gpt4o_pricing_exists(self):
        assert "gpt-4o" in MODEL_PRICING
        assert MODEL_PRICING["gpt-4o"]["input"] == 2.50

    def test_default_pricing_exists(self):
        assert "default" in MODEL_PRICING
        assert MODEL_PRICING["default"]["input"] == 0.0


class TestTokenTracker:
    def test_calculate_cost_known_model(self):
        tracker = TokenTracker()
        cost = tracker.calculate_cost("gpt-4o", input_tokens=1000, output_tokens=500)
        expected = (1000 * 2.50 + 500 * 10.00) / 1_000_000
        assert abs(cost - expected) < 0.000001

    def test_calculate_cost_unknown_model(self):
        tracker = TokenTracker()
        cost = tracker.calculate_cost("unknown-model", input_tokens=1000, output_tokens=500)
        assert cost == 0.0

    def test_record_updates_totals(self):
        tracker = TokenTracker()
        initial_tokens = tracker._totals["total_tokens"]
        tracker.record(
            model="gpt-4o",
            provider="openai",
            input_tokens=100,
            output_tokens=50,
            feature="test",
        )
        assert tracker._totals["total_tokens"] == initial_tokens + 150

    def test_record_updates_by_feature(self):
        tracker = TokenTracker()
        tracker.record(
            model="gpt-4o",
            provider="openai",
            input_tokens=100,
            output_tokens=50,
            feature="search",
        )
        assert "search" in tracker._by_feature
        assert tracker._by_feature["search"]["calls"] == 1

    def test_record_updates_by_model(self):
        tracker = TokenTracker()
        tracker.record(
            model="gpt-4o",
            provider="openai",
            input_tokens=100,
            output_tokens=50,
            feature="test",
        )
        assert "gpt-4o" in tracker._by_model

    def test_record_updates_by_course(self):
        tracker = TokenTracker()
        tracker.record(
            model="gpt-4o",
            provider="openai",
            input_tokens=100,
            output_tokens=50,
            feature="test",
            course_id="course_1",
        )
        assert "course_1" in tracker._by_course

    def test_record_adds_to_buffer(self):
        tracker = TokenTracker()
        initial_buffer_size = len(tracker._buffer)
        tracker.record(
            model="gpt-4o",
            provider="openai",
            input_tokens=100,
            output_tokens=50,
            feature="test",
        )
        assert len(tracker._buffer) == initial_buffer_size + 1

    def test_get_summary_structure(self):
        tracker = TokenTracker()
        summary = tracker.get_summary()
        assert "totals" in summary
        assert "by_feature" in summary
        assert "by_model" in summary
        assert "by_course" in summary
        assert "buffer_size" in summary


class TestTokenUsageRecord:
    def test_valid_record(self):
        record = TokenUsageRecord(
            session_id="s1",
            course_id="c1",
            feature="test",
            model="gpt-4o",
            provider="openai",
            input_tokens=100,
            output_tokens=50,
        )
        assert record.total_tokens == 0  # Not auto-calculated in dataclass
        assert record.feature == "test"

    def test_defaults(self):
        record = TokenUsageRecord()
        assert record.session_id is None
        assert record.feature == "unknown"
        assert record.input_tokens == 0
