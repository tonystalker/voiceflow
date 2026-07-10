import asyncio
from app.mcp.swiggy_client import get_swiggy_tools

async def run():
    async with get_swiggy_tools() as tools:
        for t in tools:
            print(f"Tool: {t.name}, desc={getattr(t, 'description', '')[:50]}")

asyncio.run(run())
