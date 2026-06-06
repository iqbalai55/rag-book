import os
import logging
import asyncio
from typing import Optional, Dict, List
from datetime import datetime, date
from dataclasses import dataclass, field
from decimal import Decimal

import psycopg

logger = logging.getLogger(__name__)

FEATURE_COST = {
    "search": 0.001,
    "generate_mcq": 0.01,
    "generate_essay": 0.015,
    "summarize": 0.005,
    "mindmap": 0.02,
    "agent_reasoning": 0.005,
    "dataset": 0.025,
    "unknown": 0.001,
}


@dataclass
class CreditTransaction:
    user_id: str
    amount: float
    transaction_type: str
    feature: str = "unknown"
    tokens_used: int = 0
    estimated_cost_usd: float = 0.0
    metadata: Dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


class CreditManager:
    _instance: Optional["CreditManager"] = None

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
        self._lock = asyncio.Lock()

        logger.info("CreditManager initialized")

    def get_balance(self, user_id: str) -> float:
        """Get user's current credit balance."""
        if not self.db_url:
            logger.warning("No DB URL configured")
            return 0.0

        try:
            import psycopg
            with psycopg.connect(self.db_url) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT credits_balance FROM profiles WHERE id = %s",
                        (user_id,)
                    )
                    row = cur.fetchone()
                    if row:
                        return float(row[0]) if row[0] is not None else 0.0
                    return 0.0
        except Exception as e:
            logger.error(f"Failed to get balance for user {user_id}: {e}")
            return 0.0

    async def _async_get_balance(self, user_id: str) -> float:
        """Async get user's current credit balance."""
        if not self.db_url:
            return 0.0

        try:
            async with await psycopg.AsyncConnection.connect(self.db_url) as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "SELECT credits_balance FROM profiles WHERE id = %s",
                        (user_id,)
                    )
                    row = await cur.fetchone()
                    if row:
                        return float(row[0]) if row[0] is not None else 0.0
                    return 0.0
        except Exception as e:
            logger.error(f"Failed to get balance for user {user_id}: {e}")
            return 0.0

    def burn_credits(
        self,
        user_id: str,
        feature: str,
        tokens_used: int = 0,
        estimated_cost: float = 0.0,
        metadata: Optional[Dict] = None,
    ) -> bool:
        """Synchronous burn credits - use this for pre-check before LLM call."""
        cost = estimated_cost if estimated_cost > 0 else FEATURE_COST.get(feature, FEATURE_COST["unknown"])

        if not self.db_url:
            logger.warning("No DB URL configured, skipping credit burn")
            return True

        try:
            with psycopg.connect(self.db_url) as conn:
                with conn.cursor() as cur:
                    cur.execute("BEGIN")

                    cur.execute(
                        "SELECT credits_balance FROM profiles WHERE id = %s FOR UPDATE",
                        (user_id,)
                    )
                    row = cur.fetchone()
                    current_balance = float(row[0]) if row and row[0] is not None else 0.0

                    if current_balance < cost:
                        logger.warning(f"Insufficient credits for user {user_id}: {current_balance} < {cost}")
                        return False

                    new_balance = current_balance - cost

                    cur.execute(
                        """
                        UPDATE profiles 
                        SET credits_balance = %s 
                        WHERE id = %s
                        """,
                        (new_balance, user_id)
                    )

                    cur.execute(
                        """
                        INSERT INTO credit_transactions 
                        (user_id, amount, transaction_type, feature, tokens_used, estimated_cost_usd, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            user_id,
                            -cost,
                            "burn",
                            feature,
                            tokens_used,
                            cost,
                            str(metadata or {}),
                        )
                    )

                    conn.commit()
                    logger.info(f"Burned {cost} credits for user {user_id}, feature: {feature}")
                    return True

        except Exception as e:
            logger.error(f"Failed to burn credits for user {user_id}: {e}")
            try:
                with psycopg.connect(self.db_url) as conn:
                    conn.rollback()
            except:
                pass
            return False

    async def burn_credits_async(
        self,
        user_id: str,
        feature: str,
        tokens_used: int = 0,
        estimated_cost: float = 0.0,
        metadata: Optional[Dict] = None,
    ) -> bool:
        """Async burn credits - use this within async endpoint."""
        async with self._lock:
            cost = estimated_cost if estimated_cost > 0 else FEATURE_COST.get(feature, FEATURE_COST["unknown"])

            if not self.db_url:
                logger.warning("No DB URL configured, skipping credit burn")
                return True

            try:
                async with await psycopg.AsyncConnection.connect(self.db_url) as conn:
                    async with conn.cursor() as cur:
                        await cur.execute("BEGIN")

                        await cur.execute(
                            "SELECT credits_balance FROM profiles WHERE id = %s FOR UPDATE",
                            (user_id,)
                        )
                        row = await cur.fetchone()
                        current_balance = float(row[0]) if row and row[0] is not None else 0.0

                        if current_balance < cost:
                            logger.warning(f"Insufficient credits for user {user_id}: {current_balance} < {cost}")
                            await cur.execute("ROLLBACK")
                            return False

                        new_balance = current_balance - cost

                        await cur.execute(
                            "UPDATE profiles SET credits_balance = %s WHERE id = %s",
                            (new_balance, user_id)
                        )

                        await cur.execute(
                            """
                            INSERT INTO credit_transactions 
                            (user_id, amount, transaction_type, feature, tokens_used, estimated_cost_usd, metadata)
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                            """,
                            (
                                user_id,
                                -cost,
                                "burn",
                                feature,
                                tokens_used,
                                cost,
                                str(metadata or {}),
                            )
                        )

                        await conn.commit()
                        logger.info(f"Burned {cost} credits for user {user_id}, feature: {feature}")
                        return True

            except Exception as e:
                logger.error(f"Failed to burn credits for user {user_id}: {e}")
                try:
                    async with await psycopg.AsyncConnection.connect(self.db_url) as conn:
                        await conn.rollback()
                except:
                    pass
                return False

    def refund_credits(
        self,
        user_id: str,
        feature: str,
        tokens_used: int = 0,
        estimated_cost: float = 0.0,
        metadata: Optional[Dict] = None,
    ) -> bool:
        """Refund credits to user (e.g., on failure)."""
        cost = estimated_cost if estimated_cost > 0 else FEATURE_COST.get(feature, FEATURE_COST["unknown"])

        if not self.db_url:
            logger.warning("No DB URL configured, skipping credit refund")
            return True

        try:
            with psycopg.connect(self.db_url) as conn:
                with conn.cursor() as cur:
                    cur.execute("BEGIN")

                    cur.execute(
                        "SELECT credits_balance FROM profiles WHERE id = %s FOR UPDATE",
                        (user_id,)
                    )
                    row = cur.fetchone()
                    current_balance = float(row[0]) if row and row[0] is not None else 0.0

                    new_balance = current_balance + cost

                    cur.execute(
                        "UPDATE profiles SET credits_balance = %s WHERE id = %s",
                        (new_balance, user_id)
                    )

                    cur.execute(
                        """
                        INSERT INTO credit_transactions 
                        (user_id, amount, transaction_type, feature, tokens_used, estimated_cost_usd, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            user_id,
                            cost,
                            "refund",
                            feature,
                            tokens_used,
                            cost,
                            str(metadata or {}),
                        )
                    )

                    conn.commit()
                    logger.info(f"Refunded {cost} credits to user {user_id}, feature: {feature}")
                    return True

        except Exception as e:
            logger.error(f"Failed to refund credits for user {user_id}: {e}")
            try:
                with psycopg.connect(self.db_url) as conn:
                    conn.rollback()
            except:
                pass
            return False

    async def refund_credits_async(
        self,
        user_id: str,
        feature: str,
        tokens_used: int = 0,
        estimated_cost: float = 0.0,
        metadata: Optional[Dict] = None,
    ) -> bool:
        """Async refund credits."""
        async with self._lock:
            cost = estimated_cost if estimated_cost > 0 else FEATURE_COST.get(feature, FEATURE_COST["unknown"])

            if not self.db_url:
                return True

            try:
                async with await psycopg.AsyncConnection.connect(self.db_url) as conn:
                    async with conn.cursor() as cur:
                        await cur.execute("BEGIN")

                        await cur.execute(
                            "SELECT credits_balance FROM profiles WHERE id = %s FOR UPDATE",
                            (user_id,)
                        )
                        row = await cur.fetchone()
                        current_balance = float(row[0]) if row and row[0] is not None else 0.0

                        new_balance = current_balance + cost

                        await cur.execute(
                            "UPDATE profiles SET credits_balance = %s WHERE id = %s",
                            (new_balance, user_id)
                        )

                        await cur.execute(
                            """
                            INSERT INTO credit_transactions 
                            (user_id, amount, transaction_type, feature, tokens_used, estimated_cost_usd, metadata)
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                            """,
                            (
                                user_id,
                                cost,
                                "refund",
                                feature,
                                tokens_used,
                                cost,
                                str(metadata or {}),
                            )
                        )

                        await conn.commit()
                        logger.info(f"Refunded {cost} credits to user {user_id}, feature: {feature}")
                        return True

            except Exception as e:
                logger.error(f"Failed to refund credits for user {user_id}: {e}")
                try:
                    async with await psycopg.AsyncConnection.connect(self.db_url) as conn:
                        await conn.rollback()
                except:
                    pass
                return False

    def add_credits(
        self,
        user_id: str,
        amount: float,
        transaction_type: str,
        feature: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> bool:
        """Add credits to user account (purchase, bonus, etc.)."""
        if not self.db_url:
            logger.warning("No DB URL configured, skipping credit add")
            return True

        try:
            with psycopg.connect(self.db_url) as conn:
                with conn.cursor() as cur:
                    cur.execute("BEGIN")

                    cur.execute(
                        "SELECT credits_balance FROM profiles WHERE id = %s FOR UPDATE",
                        (user_id,)
                    )
                    row = cur.fetchone()
                    current_balance = float(row[0]) if row and row[0] is not None else 0.0

                    new_balance = current_balance + amount

                    cur.execute(
                        "UPDATE profiles SET credits_balance = %s WHERE id = %s",
                        (new_balance, user_id)
                    )

                    cur.execute(
                        """
                        INSERT INTO credit_transactions 
                        (user_id, amount, transaction_type, feature, tokens_used, estimated_cost_usd, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            user_id,
                            amount,
                            transaction_type,
                            feature or transaction_type,
                            0,
                            amount,
                            str(metadata or {}),
                        )
                    )

                    conn.commit()
                    logger.info(f"Added {amount} credits to user {user_id}, type: {transaction_type}")
                    return True

        except Exception as e:
            logger.error(f"Failed to add credits for user {user_id}: {e}")
            try:
                with psycopg.connect(self.db_url) as conn:
                    conn.rollback()
            except:
                pass
            return False

    async def get_recent_transactions(
        self,
        user_id: str,
        limit: int = 10,
    ) -> List[Dict]:
        """Get recent credit transactions for a user."""
        if not self.db_url:
            return []

        try:
            async with await psycopg.AsyncConnection.connect(self.db_url) as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        """
                        SELECT id, amount, transaction_type, feature, tokens_used, 
                               estimated_cost_usd, created_at
                        FROM credit_transactions
                        WHERE user_id = %s
                        ORDER BY created_at DESC
                        LIMIT %s
                        """,
                        (user_id, limit)
                    )
                    rows = await cur.fetchall()
                    columns = [desc[0] for desc in cur.description]
                    result = []
                    for row in rows:
                        record = dict(zip(columns, row))
                        if record.get("id"):
                            record["id"] = str(record["id"])
                        if record.get("user_id"):
                            record["user_id"] = str(record["user_id"])
                        # Convert Decimal, datetime, and other non-serializable types
                        for key, value in record.items():
                            if isinstance(value, Decimal):
                                record[key] = float(value)
                            elif isinstance(value, (datetime, date)):
                                record[key] = value.isoformat()
                        result.append(record)
                    return result
        except Exception as e:
            logger.error(f"Failed to get transactions for user {user_id}: {e}")
            return []

    def check_sufficient_credits(self, user_id: str, feature: str, estimated_cost: float = 0.0) -> tuple[bool, float]:
        """Check if user has sufficient credits. Returns (is_sufficient, cost)."""
        cost = estimated_cost if estimated_cost > 0 else FEATURE_COST.get(feature, FEATURE_COST["unknown"])
        balance = self.get_balance(user_id)
        return balance >= cost, cost