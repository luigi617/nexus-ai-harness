from __future__ import annotations

import asyncio

from core.response import Response
from plugins.loops import ChatLoop
from tests.conftest import ScriptedProvider, make_ctx


def test_chat_loop_returns_text_and_appends_assistant_message():
    ctx = make_ctx(ScriptedProvider(Response(text="hello")))
    result = asyncio.run(ChatLoop().run(ctx))
    assert result == "hello"
    assert ctx.history[-1].role == "assistant"
    assert ctx.history[-1].content == "hello"


def test_chat_loop_raises_without_provider():
    try:
        asyncio.run(ChatLoop().run(make_ctx()))
        raised = False
    except LookupError:
        raised = True
    assert raised
