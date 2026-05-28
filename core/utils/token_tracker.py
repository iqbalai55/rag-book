import os
import logging
import asyncio
from typing import Optional, Dict, List
from datetime import datetime
from dataclasses import dataclass, field
from collections import defaultdict

import psycopg

logger = logging.getLogger(__name__)

# Cost per 1M tokens (USD) - as of 2024
MODEL_PRICING = {
    # OpenAI
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    # Anthropic
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.25, "output": 1.25},
    "claude-3-opus": {"input": 15.00, "output": 75.00},
    # Free/unknown models
    "default": {"input": 0.0, "output": 0.0},
}


@dataclass
class TokenUsageRecord:
    session_id: Optional[str] = None
    course_id: Optional[str] = None
    feature: str = "unknown"
    model: str = "unknown"
    provider: str = "unknown"
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    latency_ms: int = 0
    metadata: Dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


class TokenTracker:
    """Singleton token usage tracker with PostgreSQL persistence."""

    _instance: Optional["TokenTracker"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self.db_url = os.getenv("SUPABASE_DB_URL")
        self._buffer: List[TokenUsageRecord] = []
        self._buffer_size = 10
        self._lock = asyncio.Lock()

        # In-memory aggregation
        self._totals = {
            "total_tokens": 0,
            "total_cost_usd": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
        }
        self._by_feature: Dict[str, Dict] = defaultdict(lambda: {
            "tokens": 0, "cost": 0.0, "calls": 0
        })
        self._by_model: Dict[str, Dict] = defaultdict(lambda: {
            "tokens": 0, "cost": 0.0, "calls": 0
        })
        self._by_course: Dict[str, Dict] = defaultdict(lambda: {
            "tokens": 0, "cost": 0.0, "calls": 0
        })

        logger.info("TokenTracker initialized")

    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate estimated cost in USD."""
        model_key = model.lower()
        for key in MODEL_PRICING:
            if key in model_key:
                pricing = MODEL_PRICING[key]
                break
        else:
            pricing = MODEL_PRICING["default"]

        cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000
        return round(cost, 6)

    def record(
        self,
        model: str,
        provider: str,
        input_tokens: int,
        output_tokens: int,
        feature: str = "unknown",
        session_id: Optional[str] = None,
        course_id: Optional[str] = None,
        latency_ms: int = 0,
        metadata: Optional[Dict] = None,
    ):
        """Record token usage (sync, adds to buffer)."""
        total_tokens = input_tokens + output_tokens
        cost = self.calculate_cost(model, input_tokens, output_tokens)

        record = TokenUsageRecord(
            session_id=session_id,
            course_id=course_id,
            feature=feature,
            model=model,
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=cost,
            latency_ms=latency_ms,
            metadata=metadata or {},
        )

        # Update in-memory aggregation
        self._totals["total_tokens"] += total_tokens
        self._totals["total_cost_usd"] += cost
        self._totals["input_tokens"] += input_tokens
        self._totals["output_tokens"] += output_tokens
        self._by_feature[feature]["tokens"] += total_tokens
        self._by_feature[feature]["cost"] += cost
        self._by_feature[feature]["calls"] += 1
        self._by_model[model]["tokens"] += total_tokens
        self._by_model[model]["cost"] += cost
        self._by_model[model]["calls"] += 1
        if course_id:
            self._by_course[course_id]["tokens"] += total_tokens
            self._by_course[course_id]["cost"] += cost
            self._by_course[course_id]["calls"] += 1

        self._buffer.append(record)
        logger.debug(
            f"Token usage: {feature} | {model} | "
            f"in={input_tokens} out={output_tokens} | "
            f"cost=${cost:.6f}"
        )

        # Flush if buffer is full
        if len(self._buffer) >= self._buffer_size:
            asyncio.get_event_loop().create_task(self.flush())

    async def flush(self):
        """Flush buffered records to PostgreSQL."""
        async with self._lock:
            if not self._buffer:
                return

            records_to_flush = self._buffer.copy()
            self._buffer.clear()

        if not self.db_url:
            logger.warning("No DB URL configured, skipping flush")
            return

        try:
            async with await psycopg.AsyncConnection.connect(self.db_url) as conn:
                async with conn.cursor() as cur:
                    for record in records_to_flush:
                        await cur.execute(
                            """
                            INSERT INTO token_usage
                            (session_id, course_id, feature, model, provider,
                             input_tokens, output_tokens, total_tokens,
                             estimated_cost_usd, latency_ms, metadata)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """,
                            (
                                record.session_id,
                                record.course_id,
                                record.feature,
                                record.model,
                                record.provider,
                                record.input_tokens,
                                record.output_tokens,
                                record.total_tokens,
                                record.estimated_cost_usd,
                                record.latency_ms,
                                str(record.metadata) if record.metadata else "{}",
                            ),
                        )
                await conn.commit()
            logger.info(f"Flushed {len(records_to_flush)} token usage records to DB")
        except Exception as e:
            logger.error(f"Failed to flush token usage records: {e}")

    def get_summary(self) -> Dict:
        """Get in-memory aggregated summary."""
        return {
            "totals": self._totals.copy(),
            "by_feature": dict(self._by_feature),
            "by_model": dict(self._by_model),
            "by_course": dict(self._by_course),
            "buffer_size": len(self._buffer),
        }

    async def query_usage(
        self,
        course_id: Optional[str] = None,
        session_id: Optional[str] = None,
        feature: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict]:
        """Query token_usage table with filters."""
        if not self.db_url:
            return []

        conditions = []
        params = []

        if course_id:
            conditions.append("course_id = %s")
            params.append(course_id)
        if session_id:
            conditions.append("session_id = %s")
            params.append(session_id)
        if feature:
            conditions.append("feature = %s")
            params.append(feature)
        if from_date:
            conditions.append("created_at >= %s")
            params.append(from_date)
        if to_date:
            conditions.append("created_at <= %s")
            params.append(to_date)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        query = f"""
            SELECT
                SUM(total_tokens) as total_tokens,
                SUM(estimated_cost_usd) as total_cost,
                SUM(input_tokens) as input_tokens,
                SUM(output_tokens) as output_tokens,
                COUNT(*) as total_calls,
                feature,
                model
            FROM token_usage
            WHERE {where_clause}
            GROUP BY feature, model
            ORDER BY total_tokens DESC
            LIMIT %s
        """
        params.append(limit)

        try:
            async with await psycopg.AsyncConnection.connect(self.db_url) as conn:
                async with conn.cursor() as cur:
                    await cur.execute(query, params)
                    rows = await cur.fetchall()
                    columns = [desc[0] for desc in cur.description]
                    return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            logger.error(f"Failed to query token usage: {e}")
            return []

    async def query_daily_usage(
        self,
        course_id: Optional[str] = None,
        days: int = 7,
    ) -> List[Dict]:
        """Query daily token usage aggregation."""
        if not self.db_url:
            return []

        conditions = ["created_at >= NOW() - INTERVAL %s"]
        params = [f"{days} days"]

        if course_id:
            conditions.append("course_id = %s")
            params.append(course_id)

        where_clause = " AND ".join(conditions)

        query = f"""
            SELECT
                DATE(created_at) as date,
                SUM(total_tokens) as total_tokens,
                SUM(estimated_cost_usd) as total_cost,
                COUNT(*) as total_calls
            FROM token_usage
            WHERE {where_clause}
            GROUP BY DATE(created_at)
            ORDER BY date DESC
        """

        try:
            async with await psycopg.AsyncConnection.connect(self.db_url) as conn:
                async with conn.cursor() as cur:
                    await cur.execute(query, params)
                    rows = await cur.fetchall()
                    columns = [desc[0] for desc in cur.description]
                    return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            logger.error(f"Failed to query daily usage: {e}")
            return []
