"""Tests for CacheManager (lock + agent caching)."""
import asyncio
import pytest
from unittest.mock import AsyncMock, Mock, patch

from core.utils.cache_manager import CacheManager


class TestAcquireBookLock:
    async def test_concurrent_ops_on_same_book_serialise(self):
        cache = CacheManager(Mock(), Mock())
        execution_order = []
        release = asyncio.Event()

        async def slow_op():
            async with cache.acquire_book_lock("b1"):
                execution_order.append("slow:start")
                await release.wait()
                execution_order.append("slow:end")

        async def fast_op():
            # give slow_op time to grab the lock first
            await asyncio.sleep(0.05)
            async with cache.acquire_book_lock("b1"):
                execution_order.append("fast")

        t_slow = asyncio.create_task(slow_op())
        t_fast = asyncio.create_task(fast_op())

        # let fast_op try to acquire (it should block on the lock)
        await asyncio.sleep(0.1)
        release.set()

        await asyncio.wait_for(asyncio.gather(t_slow, t_fast), timeout=2.0)

        # The fast op must NOT enter the lock until slow releases it
        assert execution_order == ["slow:start", "slow:end", "fast"]

    async def test_different_books_run_in_parallel(self):
        cache = CacheManager(Mock(), Mock())
        gates = {"b1": asyncio.Event(), "b2": asyncio.Event()}
        started = {"b1": False, "b2": False}

        async def worker(book_id: str):
            async with cache.acquire_book_lock(book_id):
                started[book_id] = True
                await gates[book_id].wait()

        t1 = asyncio.create_task(worker("b1"))
        t2 = asyncio.create_task(worker("b2"))

        # wait for both to be inside the lock
        for _ in range(200):
            if started["b1"] and started["b2"]:
                break
            await asyncio.sleep(0.01)

        assert started["b1"] and started["b2"], "Both books should hold their locks concurrently"

        gates["b1"].set()
        gates["b2"].set()
        await asyncio.gather(t1, t2)

    async def test_lock_released_on_exception(self):
        cache = CacheManager(Mock(), Mock())
        with pytest.raises(RuntimeError, match="boom"):
            async with cache.acquire_book_lock("b1"):
                raise RuntimeError("boom")

        async def quick():
            async with cache.acquire_book_lock("b1"):
                return "ok"

        result = await asyncio.wait_for(quick(), timeout=1.0)
        assert result == "ok"


class TestClearBook:
    async def test_clear_book_evicts_agent_and_lock(self):
        cache = CacheManager(Mock(), Mock())
        cache._agents["b1:user1"] = Mock()
        cache._ingest_locks["b1"] = asyncio.Lock()

        await cache.clear_book("b1")

        assert "b1:user1" not in cache._agents
        assert "b1" not in cache._ingest_locks

    async def test_clear_book_does_not_touch_other_books(self):
        cache = CacheManager(Mock(), Mock())
        cache._agents["b1:user1"] = Mock()
        cache._agents["b2:user1"] = Mock()
        cache._ingest_locks["b1"] = asyncio.Lock()
        cache._ingest_locks["b2"] = asyncio.Lock()

        await cache.clear_book("b1")

        assert "b1:user1" not in cache._agents
        assert "b2:user1" in cache._agents
        assert "b1" not in cache._ingest_locks
        assert "b2" in cache._ingest_locks
