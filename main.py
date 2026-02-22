import asyncio
import os
from dotenv import load_dotenv
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
import selectors

# Load env
load_dotenv()
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")

async def setup_checkpointer():
    async with AsyncPostgresSaver.from_conn_string(SUPABASE_DB_URL) as checkpointer:
        await checkpointer.setup()
        print("✅ Checkpointer setup selesai!")

if __name__ == "__main__":
    asyncio.run(setup_checkpointer(), loop_factory=lambda: asyncio.SelectorEventLoop(selectors.SelectSelector()))