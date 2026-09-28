from __future__ import annotations

import importlib
import inspect
import logging
from collections.abc import Awaitable, Callable, Iterable
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any, Protocol

from nexus_ai_harness.protocols.context import Context
from nexus_ai_harness.protocols.lifecycle import Lifecycle
from nexus_ai_harness.protocols.tool import Tool
from nexus_ai_harness.protocols.tool_provider import ToolProvider

logger = logging.getLogger(__name__)


@dataclass
class MCPServer:
    """Connection config for a single Model Context Protocol server.

    A server is reached over exactly one transport: a stdio subprocess launched
    from ``command``, or an HTTP/SSE endpoint at ``url``. Both the launch argv
    and the URL come from trusted plugin configuration, never from model input,
    so a tool call can't point the client at an arbitrary process or host.

    Attributes:
        name: Stable label for the server; prefixed onto each discovered tool's
            name (``"<name>__<tool>"``) so tools from different servers can't
            collide. Use only ``[A-Za-z0-9_-]`` characters — provider tool-name
            APIs reject others (e.g. ``.``), and the name is sent to the model
            verbatim.
        command: The stdio launch argv (``argv[0]`` is the program); ``None``
            when the server is reached over ``url`` instead.
        url: The HTTP/SSE endpoint; ``None`` when the server is launched via
            ``command`` instead.
        env: Extra environment for a stdio launch. May hold secrets, so it is
            never logged.
    """

    name: str
    command: list[str] | None = None
    url: str | None = None
    env: dict[str, str] | None = None


class _MCPSession(Protocol):
    """The minimal session surface the client needs from a connector.

    A connector returns any object satisfying this: enough to enumerate a
    server's tools and invoke one. It keeps the client independent of the
    concrete transport (the ``mcp`` SDK, or a test fake).
    """

    async def list_tools(self) -> list[dict[str, Any]]:
        """Return tool specs, each a ``{name, description, input_schema}`` dict."""
        ...

    async def call_tool(self, name: str, arguments: dict) -> str:
        """Invoke tool ``name`` with ``arguments`` and return its text result."""
        ...


class _MCPTool(Tool):
    """A harness :class:`~protocols.tool.Tool` proxying one MCP server tool.

    Identity is per-instance rather than the class-level metadata a static tool
    declares: the name, description, and JSON-schema parameters all come from
    the server's advertised spec. The exposed name is namespaced with the server
    label, but calls route to the underlying MCP tool by its bare name.
    """

    def __init__(
        self, session: _MCPSession, server_name: str, spec: dict[str, Any]
    ) -> None:
        self._session = session
        # The bare name call_tool uses; the harness sees the namespaced name below.
        self._tool_name = str(spec.get("name", ""))
        # Separator is "__" not "." — provider tool-name APIs require [A-Za-z0-9_-].
        self.name = f"{server_name}__{self._tool_name}"  # type: ignore[misc]
        self.description = str(spec.get("description", "") or "")  # type: ignore[misc]
        schema = spec.get("input_schema")
        # Fall back to a permissive object schema when the server omits one, so
        # the model can still pass arbitrary arguments.
        self.parameters = (  # type: ignore[misc]
            schema
            if isinstance(schema, dict)
            else {"type": "object", "properties": {}, "additionalProperties": True}
        )

    async def run(self, arguments: dict, ctx: Context) -> str:
        """Call the underlying MCP tool and return its text.

        A call that raises propagates, so the tool runner turns it into the
        usual ``"error: ..."`` result, logs the traceback, and records the
        exception on :class:`~core.events.ToolCallCompleted`.
        """
        return await self._session.call_tool(self._tool_name, arguments)


class _SDKSession:
    """Adapts an ``mcp`` SDK ``ClientSession`` to the :class:`_MCPSession` shape."""

    def __init__(self, session: Any) -> None:
        self._session = session

    async def list_tools(self) -> list[dict[str, Any]]:
        """Translate the SDK's tool listing into plain spec dicts."""
        result = await self._session.list_tools()
        return [
            {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": tool.inputSchema,
            }
            for tool in result.tools
        ]

    async def call_tool(self, name: str, arguments: dict) -> str:
        """Invoke a tool and join its text content blocks into one string.

        A result flagged ``isError`` is returned as an ``"error: ..."`` string so
        the model sees the failure rather than a silent empty reply.
        """
        result = await self._session.call_tool(name, arguments)
        parts = [
            block.text
            for block in result.content
            if getattr(block, "text", None) is not None
        ]
        text = "\n".join(parts)
        if getattr(result, "isError", False):
            return f"error: {text}" if text else "error: tool call failed"
        return text


class MCPClient(ToolProvider, Lifecycle):
    """Connects to configured MCP servers and exposes their tools to the harness.

    On :meth:`start` the client opens a session per server, discovers each
    server's advertised tools, and wraps every one in an :class:`_MCPTool`. Those
    proxies are handed to the loop via :meth:`provide_tools`, so remote MCP tools
    behave exactly like statically registered ones. :meth:`stop` closes every
    session.

    The transport is reached through an injectable ``connector`` seam: tests pass
    a fake so no subprocess or socket is opened, while the default connector
    lazily adapts the official ``mcp`` SDK. Server launch commands and URLs come
    only from :class:`MCPServer` config, and environment values are never logged.
    """

    def __init__(
        self,
        servers: Iterable[MCPServer],
        *,
        connector: Callable[[MCPServer], Awaitable[_MCPSession]] | None = None,
    ) -> None:
        self._servers = list(servers)
        self._connector = connector or self._default_connect
        self._exit_stack = AsyncExitStack()
        self._sessions: list[_MCPSession] = []
        self._tools: list[Tool] = []

    async def start(self, ctx: Context) -> None:
        """Connect to every server and cache a proxy tool per discovered tool.

        Idempotent: a second call while already connected is a no-op rather than
        duplicating sessions and tools. If any server fails mid-connect, the
        sessions opened so far are closed before the error propagates, so a
        partial start doesn't leak subprocesses.
        """
        if self._sessions or self._tools:  # already started
            return
        try:
            for server in self._servers:
                session = await self._connector(server)
                self._sessions.append(session)
                discovered = 0
                for spec in await session.list_tools():
                    if not str(spec.get("name", "")).strip():
                        logger.debug(
                            "MCP server %r advertised a tool with no name; skipped",
                            server.name,
                        )
                        continue  # skip malformed specs with no usable name
                    self._tools.append(_MCPTool(session, server.name, spec))
                    discovered += 1
                # Only the label is logged; command, url, and env stay out of logs.
                logger.info(
                    "connected to MCP server %r (%d tools)", server.name, discovered
                )
        except BaseException:
            logger.debug("MCP client start failed; closing opened sessions")
            await self.stop()  # close what opened; don't leak on partial start
            raise

    async def stop(self) -> None:
        """Close every session and the transport stack, best-effort.

        Cleanup is best-effort by design: a failing close must not stop the
        harness from tearing down the rest of its plugins. Two closers run,
        covering both connector kinds: sessions that expose ``aclose``/``close``
        (test fakes) are closed directly, while SDK sessions — whose transports
        and :class:`ClientSession` were entered on ``_exit_stack`` during
        :meth:`start` — are released by closing that stack. The stack is then
        replaced with a fresh one so the client is cleanly restartable.

        Known limitation: the ``mcp`` SDK's stdio/SSE transports open anyio
        cancel scopes bound to the task that entered them, so ``_exit_stack``
        must be closed on the same task that ran :meth:`start`. Driving a run on
        a background task and stopping from another can leave a stdio subprocess
        orphaned; keep start/stop on one task.
        """
        for session in reversed(self._sessions):
            closer = getattr(session, "aclose", None) or getattr(session, "close", None)
            if closer is None:
                continue
            try:
                result = closer()
                if inspect.isawaitable(result):
                    await result
            except Exception:  # best-effort; keep closing the rest
                logger.debug("closing an MCP session failed", exc_info=True)
        self._sessions = []
        self._tools = []
        try:
            await self._exit_stack.aclose()
        except Exception:  # best-effort, as above
            logger.debug("closing the MCP transport stack failed", exc_info=True)
        # Fresh stack so a subsequent start() never reuses a torn-down one.
        self._exit_stack = AsyncExitStack()

    def provide_tools(self, ctx: Context) -> list[Tool]:
        """Return the cached proxy tools (empty until :meth:`start` has run)."""
        return list(self._tools)

    async def _default_connect(self, server: MCPServer) -> _MCPSession:
        """Open a session using the official ``mcp`` SDK.

        Imports the SDK lazily so the package is only required when the default
        connector is actually used; a missing ``mcp`` install surfaces here as a
        clear ImportError rather than at module import time.
        """
        try:
            mcp = importlib.import_module("mcp")
            stdio = importlib.import_module("mcp.client.stdio")
            sse = importlib.import_module("mcp.client.sse")
        except ImportError as exc:  # SDK absent — say so plainly
            raise ImportError(
                "MCPClient's default connector requires the 'mcp' package; "
                "install it (pip install mcp) or pass a custom connector"
            ) from exc

        if server.command:
            params = mcp.StdioServerParameters(
                command=server.command[0],
                args=list(server.command[1:]),
                env=server.env,
            )
            transport = await self._exit_stack.enter_async_context(
                stdio.stdio_client(params)
            )
        elif server.url:
            transport = await self._exit_stack.enter_async_context(
                sse.sse_client(server.url)
            )
        else:
            raise ValueError(
                f"MCP server {server.name!r} has neither a command nor a url"
            )

        read, write = transport[0], transport[1]
        session = await self._exit_stack.enter_async_context(
            mcp.ClientSession(read, write)
        )
        await session.initialize()
        return _SDKSession(session)
