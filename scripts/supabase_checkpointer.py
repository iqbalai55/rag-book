import asyncio
import json
import logging
from typing import Any, Optional, Sequence
from uuid import UUID

from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    ChannelVersions,
    get_checkpoint_id,
)
from supabase import Client

logger = logging.getLogger(__name__)


class SupabaseCheckpointer(BaseCheckpointSaver):
    """Durable LangGraph checkpointing via Supabase REST API (PostgREST)."""

    client: Client

    def __init__(self, client: Client):
        super().__init__()
        self.client = client

    def _serialize(self, obj: Any) -> str:
        return json.dumps(obj, default=str)

    def _deserialize(self, data: str) -> Any:
        if isinstance(data, str):
            return json.loads(data)
        return data

    def put(
        self,
        config: dict,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> dict:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = checkpoint["id"]
        parent_checkpoint_id = config["configurable"].get("checkpoint_id")

        row = {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint_id,
            "parent_checkpoint_id": parent_checkpoint_id,
            "type": "checkpoint",
            "checkpoint": self._serialize(checkpoint),
            "metadata": self._serialize(metadata),
        }

        try:
            self.client.table("langgraph_checkpoints").upsert(row).execute()
        except Exception as e:
            logger.error(f"put checkpoint error: {e}")

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    def put_writes(
        self,
        config: dict,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
    ) -> None:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]

        rows = []
        for idx, (channel, value) in enumerate(writes):
            rows.append({
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
                "task_id": task_id,
                "idx": idx,
                "channel": channel,
                "type": "write",
                "blob": self._serialize(value),
            })

        if rows:
            try:
                self.client.table("langgraph_writes").upsert(rows).execute()
            except Exception as e:
                logger.error(f"put_writes error: {e}")

    def get_tuple(self, config: dict) -> Optional[CheckpointTuple]:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = get_checkpoint_id(config)

        try:
            query = (
                self.client.table("langgraph_checkpoints")
                .select("*")
                .eq("thread_id", thread_id)
                .eq("checkpoint_ns", checkpoint_ns)
                .order("created_at", desc=True)
                .limit(1)
            )

            if checkpoint_id:
                query = query.eq("checkpoint_id", checkpoint_id)

            result = query.execute()

            if not result.data:
                return None

            row = result.data[0]
            checkpoint = self._deserialize(row["checkpoint"])
            metadata = self._deserialize(row.get("metadata", "{}"))

            # Fetch pending writes
            writes_result = (
                self.client.table("langgraph_writes")
                .select("*")
                .eq("thread_id", thread_id)
                .eq("checkpoint_ns", checkpoint_ns)
                .eq("checkpoint_id", row["checkpoint_id"])
                .order("task_id")
                .order("idx")
                .execute()
            )

            pending_writes = []
            if writes_result.data:
                for w in writes_result.data:
                    pending_writes.append((
                        w["task_id"],
                        w["channel"],
                        self._deserialize(w["blob"]) if w.get("blob") else None,
                    ))

            parent_config = None
            if row.get("parent_checkpoint_id"):
                parent_config = {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": row["parent_checkpoint_id"],
                    }
                }

            config_out = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": row["checkpoint_id"],
                }
            }

            return CheckpointTuple(
                config=config_out,
                checkpoint=checkpoint,
                metadata=metadata,
                parent_config=parent_config,
                pending_writes=pending_writes,
            )

        except Exception as e:
            logger.error(f"get_tuple error: {e}")
            return None

    def list(
        self,
        config: dict,
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[dict] = None,
        limit: Optional[int] = None,
    ):
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        try:
            query = (
                self.client.table("langgraph_checkpoints")
                .select("*")
                .eq("thread_id", thread_id)
                .eq("checkpoint_ns", checkpoint_ns)
                .order("created_at", desc=True)
            )

            if limit:
                query = query.limit(limit)

            result = query.execute()

            if not result.data:
                return

            for row in result.data:
                checkpoint = self._deserialize(row["checkpoint"])
                metadata = self._deserialize(row.get("metadata", "{}"))

                parent_config = None
                if row.get("parent_checkpoint_id"):
                    parent_config = {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": row["parent_checkpoint_id"],
                        }
                    }

                yield CheckpointTuple(
                    config={
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": row["checkpoint_id"],
                        }
                    },
                    checkpoint=checkpoint,
                    metadata=metadata,
                    parent_config=parent_config,
                )

        except Exception as e:
            logger.error(f"list error: {e}")
            return

    async def aget_tuple(self, config: dict) -> Optional[CheckpointTuple]:
        return await asyncio.to_thread(self._aget_tuple_sync, config)

    def _aget_tuple_sync(self, config: dict) -> Optional[CheckpointTuple]:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = get_checkpoint_id(config)

        try:
            query = (
                self.client.table("langgraph_checkpoints")
                .select("*")
                .eq("thread_id", thread_id)
                .eq("checkpoint_ns", checkpoint_ns)
                .order("created_at", desc=True)
                .limit(1)
            )

            if checkpoint_id:
                query = query.eq("checkpoint_id", checkpoint_id)

            result = query.execute()

            if not result.data:
                return None

            row = result.data[0]
            checkpoint = self._deserialize(row["checkpoint"])
            metadata = self._deserialize(row.get("metadata", "{}"))

            writes_result = (
                self.client.table("langgraph_writes")
                .select("*")
                .eq("thread_id", thread_id)
                .eq("checkpoint_ns", checkpoint_ns)
                .eq("checkpoint_id", row["checkpoint_id"])
                .order("task_id")
                .order("idx")
                .execute()
            )

            pending_writes = []
            if writes_result.data:
                for w in writes_result.data:
                    pending_writes.append((
                        w["task_id"],
                        w["channel"],
                        self._deserialize(w["blob"]) if w.get("blob") else None,
                    ))

            parent_config = None
            if row.get("parent_checkpoint_id"):
                parent_config = {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": row["parent_checkpoint_id"],
                    }
                }

            config_out = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": row["checkpoint_id"],
                }
            }

            return CheckpointTuple(
                config=config_out,
                checkpoint=checkpoint,
                metadata=metadata,
                parent_config=parent_config,
                pending_writes=pending_writes,
            )

        except Exception as e:
            logger.error(f"aget_tuple error: {e}")
            return None

    async def aput(
        self,
        config: dict,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> dict:
        return await asyncio.to_thread(self._aput_sync, config, checkpoint, metadata, new_versions)

    def _aput_sync(
        self,
        config: dict,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> dict:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = checkpoint["id"]
        parent_checkpoint_id = config["configurable"].get("checkpoint_id")

        row = {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint_id,
            "parent_checkpoint_id": parent_checkpoint_id,
            "type": "checkpoint",
            "checkpoint": self._serialize(checkpoint),
            "metadata": self._serialize(metadata),
        }

        try:
            self.client.table("langgraph_checkpoints").upsert(row).execute()
        except Exception as e:
            logger.error(f"aput checkpoint error: {e}")

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    async def aput_writes(
        self,
        config: dict,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
    ) -> None:
        await asyncio.to_thread(self._aput_writes_sync, config, writes, task_id)

    def _aput_writes_sync(
        self,
        config: dict,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
    ) -> None:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]

        rows = []
        for idx, (channel, value) in enumerate(writes):
            rows.append({
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
                "task_id": task_id,
                "idx": idx,
                "channel": channel,
                "type": "write",
                "blob": self._serialize(value),
            })

        if rows:
            try:
                self.client.table("langgraph_writes").upsert(rows).execute()
            except Exception as e:
                logger.error(f"aput_writes error: {e}")

    async def alist(
        self,
        config: dict,
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[dict] = None,
        limit: Optional[int] = None,
    ):
        result = await asyncio.to_thread(self._alist_sync, config, filter, before, limit)
        for item in result:
            yield item

    def _alist_sync(
        self,
        config: dict,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[dict] = None,
        limit: Optional[int] = None,
    ):
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        try:
            query = (
                self.client.table("langgraph_checkpoints")
                .select("*")
                .eq("thread_id", thread_id)
                .eq("checkpoint_ns", checkpoint_ns)
                .order("created_at", desc=True)
            )

            if limit:
                query = query.limit(limit)

            result = query.execute()

            if not result.data:
                return

            for row in result.data:
                checkpoint = self._deserialize(row["checkpoint"])
                metadata = self._deserialize(row.get("metadata", "{}"))

                parent_config = None
                if row.get("parent_checkpoint_id"):
                    parent_config = {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": row["parent_checkpoint_id"],
                        }
                    }

                yield CheckpointTuple(
                    config={
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": row["checkpoint_id"],
                        }
                    },
                    checkpoint=checkpoint,
                    metadata=metadata,
                    parent_config=parent_config,
                )

        except Exception as e:
            logger.error(f"alist error: {e}")
            return
