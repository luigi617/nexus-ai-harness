from __future__ import annotations

import asyncio

import pytest

from nexus_ai_harness.plugins.mcp import MCPClient, MCPServer
from nexus_ai_harness.protocols.tool import Tool
from tests.conftest import make_ctx

READ_SPEC = {
    "name": "read",
    "description": "Read a file",
    "input_schema": {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    },
}
WRITE_SPEC = {
    "name": "write",
    "description": "Write a file",
    "input_schema": {
        "type": "object",
        "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
    },
}


class FakeSession:
    """A canned MCP session: two tools, echoing call_tool, tracked close."""

    def __init__(self, server_name: str = "fs") -> None:
        self.server_name = server_name
        self.calls: list[tuple[str, dict]] = []
        self.closed = False

    async def list_tools(self) -> list[dict]:
        return [READ_SPEC, WRITE_SPEC]

    async def call_tool(self, name: str, arguments: dict) -> str:
        self.calls.append((name, arguments))
        return f"{name} -> {arguments}"

    async def aclose(self) -> None:
        self.closed = True


def make_connector(session: FakeSession):
    async def connector(server: MCPServer) -> FakeSession:
        return session

    return connector


def test_start_populates_provide_tools():
    session = FakeSession()
    client = MCPClient(
        [MCPServer(name="fs", command=["run-fs"])],
        connector=make_connector(session),
    )
    ctx = make_ctx()

    # Before start, nothing is contributed.
    assert client.provide_tools(ctx) == []

    asyncio.run(client.start(ctx))
    tools = client.provide_tools(ctx)

    assert len(tools) == 2
    assert all(isinstance(t, Tool) for t in tools)
    by_name = {t.name: t for t in tools}
    # Names are namespaced by server.
    assert set(by_name) == {"fs__read", "fs__write"}
    assert by_name["fs__read"].description == "Read a file"
    assert by_name["fs__read"].parameters == READ_SPEC["input_schema"]
    assert by_name["fs__write"].parameters == WRITE_SPEC["input_schema"]


def test_run_proxies_to_call_tool():
    session = FakeSession()
    client = MCPClient(
        [MCPServer(name="fs", command=["run-fs"])],
        connector=make_connector(session),
    )
    ctx = make_ctx()
    asyncio.run(client.start(ctx))
    read = {t.name: t for t in client.provide_tools(ctx)}["fs__read"]

    result = asyncio.run(read.run({"path": "a.txt"}, ctx))

    # The bare tool name (not the namespaced one) reaches the session.
    assert session.calls == [("read", {"path": "a.txt"})]
    assert result == "read -> {'path': 'a.txt'}"


def test_run_returns_error_on_failure():
    class BoomSession(FakeSession):
        async def call_tool(self, name: str, arguments: dict) -> str:
            raise RuntimeError("boom")

    client = MCPClient(
        [MCPServer(name="fs", command=["run-fs"])],
        connector=make_connector(BoomSession()),
    )
    ctx = make_ctx()
    asyncio.run(client.start(ctx))
    read = {t.name: t for t in client.provide_tools(ctx)}["fs__read"]

    result = asyncio.run(read.run({"path": "a.txt"}, ctx))

    assert result.startswith("error:")
    assert "boom" in result


def test_names_namespaced_across_servers():
    sessions = {"alpha": FakeSession("alpha"), "beta": FakeSession("beta")}

    async def connector(server: MCPServer) -> FakeSession:
        return sessions[server.name]

    client = MCPClient(
        [MCPServer(name="alpha", command=["a"]), MCPServer(name="beta", url="x")],
        connector=connector,
    )
    ctx = make_ctx()
    asyncio.run(client.start(ctx))

    names = {t.name for t in client.provide_tools(ctx)}
    assert names == {"alpha__read", "alpha__write", "beta__read", "beta__write"}


def test_stop_is_safe_and_closes_sessions():
    session = FakeSession()
    client = MCPClient(
        [MCPServer(name="fs", command=["run-fs"])],
        connector=make_connector(session),
    )
    ctx = make_ctx()
    asyncio.run(client.start(ctx))

    asyncio.run(client.stop())

    assert session.closed is True
    # After stop, no tools remain and a second stop is still safe.
    assert client.provide_tools(ctx) == []
    asyncio.run(client.stop())


def test_start_is_idempotent():
    # A second start() without an intervening stop() must not duplicate tools.
    session = FakeSession()
    client = MCPClient(
        [MCPServer(name="fs", command=["run-fs"])],
        connector=make_connector(session),
    )
    ctx = make_ctx()
    asyncio.run(client.start(ctx))
    asyncio.run(client.start(ctx))
    assert {t.name for t in client.provide_tools(ctx)} == {"fs__read", "fs__write"}


def test_start_skips_tools_with_no_name():
    class NamelessSession(FakeSession):
        async def list_tools(self) -> list[dict]:
            return [READ_SPEC, {"description": "no name"}, {"name": "  "}]

    client = MCPClient(
        [MCPServer(name="fs", command=["run-fs"])],
        connector=make_connector(NamelessSession()),
    )
    ctx = make_ctx()
    asyncio.run(client.start(ctx))
    assert {t.name for t in client.provide_tools(ctx)} == {"fs__read"}


def test_restartable_after_stop():
    # start -> stop -> start must re-discover tools rather than leave the client
    # dead (a fresh exit stack is installed on stop).
    session = FakeSession()
    client = MCPClient(
        [MCPServer(name="fs", command=["run-fs"])],
        connector=make_connector(session),
    )
    ctx = make_ctx()
    asyncio.run(client.start(ctx))
    asyncio.run(client.stop())
    asyncio.run(client.start(ctx))
    assert {t.name for t in client.provide_tools(ctx)} == {"fs__read", "fs__write"}


def test_stop_clears_tools_even_if_close_raises():
    # A session whose close raises must not stop teardown: tools are still
    # cleared (best-effort), so the client is left in a clean state.
    class BadCloseSession(FakeSession):
        async def aclose(self) -> None:
            raise RuntimeError("close boom")

    client = MCPClient(
        [MCPServer(name="fs", command=["run-fs"])],
        connector=make_connector(BadCloseSession()),
    )
    ctx = make_ctx()
    asyncio.run(client.start(ctx))
    asyncio.run(client.stop())  # must not raise
    assert client.provide_tools(ctx) == []


def test_missing_or_invalid_schema_falls_back_to_permissive():
    # A spec with no input_schema (or a non-dict one) yields a permissive
    # object schema rather than an invalid parameters value.
    class LooseSession(FakeSession):
        async def list_tools(self) -> list[dict]:
            return [
                {"name": "noschema", "description": "no schema"},
                {"name": "badschema", "input_schema": "not-a-dict"},
            ]

    client = MCPClient(
        [MCPServer(name="fs", command=["run-fs"])],
        connector=make_connector(LooseSession()),
    )
    ctx = make_ctx()
    asyncio.run(client.start(ctx))
    tools = {t.name: t for t in client.provide_tools(ctx)}
    permissive = {"type": "object", "properties": {}, "additionalProperties": True}
    assert tools["fs__noschema"].parameters == permissive
    assert tools["fs__badschema"].parameters == permissive


def test_partial_start_failure_closes_opened_sessions():
    # Server "alpha" connects; "beta" fails. The opened session must be closed
    # and no tools left cached, rather than leaking a live session.
    alpha = FakeSession("alpha")

    async def connector(server: MCPServer) -> FakeSession:
        if server.name == "beta":
            raise RuntimeError("connect failed")
        return alpha

    client = MCPClient(
        [MCPServer(name="alpha", command=["a"]), MCPServer(name="beta", command=["b"])],
        connector=connector,
    )
    ctx = make_ctx()
    with pytest.raises(RuntimeError, match="connect failed"):
        asyncio.run(client.start(ctx))
    assert alpha.closed is True
    assert client.provide_tools(ctx) == []
