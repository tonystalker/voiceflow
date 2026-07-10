import asyncio
import json
import urllib.parse
import webbrowser
from pathlib import Path

from mcp.client.auth import OAuthClientProvider, TokenStorage
from mcp.client.streamable_http import streamablehttp_client
from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthToken
from mcp import ClientSession
from langchain_mcp_adapters.tools import load_mcp_tools


class FileTokenStorage(TokenStorage):
    """Persists tokens + DCR client info to disk so re-auth isn't needed every run."""
    def __init__(self, path: str):
        self._path = Path(path)
        self._tokens = None
        self._client_info = None
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text())
                if data.get("tokens"):
                    self._tokens = OAuthToken(**data["tokens"])
                if data.get("client_info"):
                    self._client_info = OAuthClientInformationFull(**data["client_info"])
            except Exception as e:
                print(f"[FileTokenStorage] Error loading tokens: {e}")

    async def get_tokens(self) -> OAuthToken | None:
        return self._tokens

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        return self._client_info

    async def set_tokens(self, tokens: OAuthToken) -> None:
        self._tokens = tokens
        self._save()

    async def set_client_info(self, info: OAuthClientInformationFull) -> None:
        self._client_info = info
        self._save()

    def _save(self):
        self._path.write_text(json.dumps({
            "tokens": self._tokens.model_dump(mode="json") if self._tokens else None,
            "client_info": self._client_info.model_dump(mode="json") if self._client_info else None,
        }, indent=2))


async def _redirect_handler(url: str) -> None:
    """Open the browser for Swiggy OAuth login."""
    print(f"\n[MCP] Opening browser for Swiggy Authentication...")
    print(f"[MCP] If it doesn't open automatically, click here: {url}\n")
    webbrowser.open(url)


async def _callback_handler() -> tuple[str, str | None]:
    """
    Stand up a tiny local HTTP server on the redirect_uri port to catch
    Swiggy's ?code=...&state=... redirect.
    """
    loop = asyncio.get_running_loop()
    future = loop.create_future()

    async def handle_client(reader, writer):
        try:
            request_line = await reader.readline()
            request_line = request_line.decode('utf-8').strip()
            # Parse GET /callback?code=xxx&state=yyy HTTP/1.1
            if "GET " in request_line and " /" in request_line:
                path = request_line.split(" ")[1]
                if "?" in path:
                    query = path.split("?")[1]
                    params = urllib.parse.parse_qs(query)
                    code = params.get('code', [None])[0]
                    state = params.get('state', [None])[0]
                    if not future.done() and code:
                        future.set_result((code, state))
            
            response = b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n<html><body style='font-family:sans-serif;text-align:center;margin-top:50px;'><h1>Success!</h1><p>Authentication complete. You can close this tab.</p></body></html>"
            writer.write(response)
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    # Start the server on port 8765 as registered in redirect_uris
    server = await asyncio.start_server(handle_client, '127.0.0.1', 8765)
    print("[MCP] Waiting for callback from browser on http://127.0.0.1:8765/callback ...")
    
    # Wait for the future to be resolved
    code, state = await future
    print("[MCP] Callback received! Shutting down local server.")
    
    server.close()
    await server.wait_closed()
    return code, state


from contextlib import asynccontextmanager

@asynccontextmanager
async def get_swiggy_tools(server: str = "food"):
    """
    Open a Swiggy MCP session with framework-managed OAuth (DCR + PKCE + auto-refresh).
    Yields LangChain-compatible tools via load_mcp_tools(), keeping the session alive.
    """
    auth = OAuthClientProvider(
        server_url=f"https://mcp.swiggy.com/{server}",
        client_metadata=OAuthClientMetadata(
            client_name="VoiceFlow Personal Assistant",
            redirect_uris=["http://localhost:8765/callback"],
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
        ),
        storage=FileTokenStorage(f".swiggy_token_{server}.json"),
        redirect_handler=_redirect_handler,
        callback_handler=_callback_handler,
    )
    
    print(f"[MCP] Initializing Swiggy {server} MCP client...")
    
    async with streamablehttp_client(
        f"https://mcp.swiggy.com/{server}",
        auth=auth,
    ) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print(f"[MCP] Swiggy {server} session initialized.")
            tools = await load_mcp_tools(session)
            yield tools
