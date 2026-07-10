import asyncio
from app.mcp.swiggy_client import get_swiggy_tools

async def main():
    print("Setting up Swiggy Food MCP Auth...")
    async with get_swiggy_tools("food") as tools:
        print(f"Successfully loaded {len(tools)} tools!")
        for t in tools:
            print(f" - {t.name}: {t.description}")
    print("Auth complete! Tokens saved.")

if __name__ == "__main__":
    asyncio.run(main())
