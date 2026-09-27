from __future__ import annotations

import asyncio
from typing import ClassVar

import pytest

from core.events import Event, SkillInvoked
from core.response import Response
from harness import NexusAIHarness
from harness.session import Session
from plugins.loops import AgenticLoop
from plugins.permissions import AutoApprove
from plugins.skills import MarkdownSkill, load_skills
from plugins.skills.markdown import _parse_frontmatter
from protocols.interceptor import Interceptor
from protocols.skill import Skill
from protocols.tool import Tool
from services.tool_runner import ToolRunner
from tests.conftest import ScriptedModel, make_ctx

# --- code-authored skills for the in-memory tests --------------------------


class _StubSkill(Skill):
    def __init__(self, name: str, description: str, body: str = "do the thing") -> None:
        self.name = name
        self.description = description
        self._body = body

    def instructions(self) -> str:
        return self._body


class _AsyncSkill(Skill):
    name = "async-skill"
    description = "Instructions come from an async def."

    async def instructions(self) -> str:
        return "Async body loaded."


class _BoomSkill(Skill):
    name = "boom"
    description = "Its body read raises."

    def instructions(self) -> str:
        raise RuntimeError("cannot read body")


# --- a skill is a tool ------------------------------------------------------


def test_skill_is_a_tool_with_no_argument_schema():
    skill = _StubSkill("s", "desc")
    assert isinstance(skill, Tool)  # registered and used like any other tool
    assert skill.parameters == {"type": "object", "properties": {}}


def test_skill_run_returns_instructions_and_emits_event():
    tool = _StubSkill("alpha", "A", body="Follow step one.")
    ctx = make_ctx(tool)
    events: list[Event] = []
    ctx.on(SkillInvoked, lambda e, _c: events.append(e))
    result = asyncio.run(tool.run({}, ctx))
    assert result.startswith("Loaded skill 'alpha'")
    assert "Follow step one." in result
    assert events == [SkillInvoked("alpha")]


def test_skill_awaits_async_instructions():
    tool = _AsyncSkill()
    result = asyncio.run(tool.run({}, make_ctx(tool)))
    assert "Async body loaded." in result


def test_skill_emits_event_before_body_read_even_if_it_raises():
    """A failed lazy load is still observable: the event fires before the read."""
    tool = _BoomSkill()
    ctx = make_ctx(tool)
    events: list[Event] = []
    ctx.on(SkillInvoked, lambda e, _c: events.append(e))
    with pytest.raises(RuntimeError):
        asyncio.run(tool.run({}, ctx))
    assert events == [SkillInvoked("boom")]


def test_skill_call_fires_tool_interceptors_once():
    """instructions() is an internal step, not a second interceptor entrypoint."""
    calls: list[str] = []

    class _Spy(Interceptor):
        target: ClassVar[type] = Tool

        def before(self, ctx) -> None:
            calls.append("before")

        def after(self, ctx) -> None:
            calls.append("after")

    skill = _StubSkill("s", "d", body="BODY")
    ctx = make_ctx(skill, _Spy())
    result = asyncio.run(ToolRunner([skill]).run({"id": "1", "name": "s"}, ctx))
    assert "BODY" in result.content
    assert calls == ["before", "after"]  # once around run(), not doubled


# --- frontmatter parsing ----------------------------------------------------


def test_parse_frontmatter_extracts_flat_pairs_and_strips_quotes():
    meta, body = _parse_frontmatter(
        "---\nname: my-skill\ndescription: 'A quoted one.'\n---\nBody here.\n"
    )
    assert meta == {"name": "my-skill", "description": "A quoted one."}
    assert body == "Body here.\n"


def test_parse_frontmatter_no_fence_returns_whole_text_as_body():
    meta, body = _parse_frontmatter("Just a body, no frontmatter.")
    assert meta == {}
    assert body == "Just a body, no frontmatter."


def test_parse_frontmatter_unterminated_fence_is_not_treated_as_meta():
    meta, _ = _parse_frontmatter("---\nname: x\nnever closes\n")
    assert meta == {}


def test_parse_frontmatter_handles_crlf():
    meta, body = _parse_frontmatter(
        "---\r\nname: x\r\ndescription: y\r\n---\r\nbody\r\n"
    )
    assert meta == {"name": "x", "description": "y"}
    assert body.strip() == "body"


def test_parse_frontmatter_keeps_colons_in_value():
    meta, _ = _parse_frontmatter("---\ndescription: a: b: c\n---\nx")
    assert meta["description"] == "a: b: c"


def test_parse_frontmatter_empty_block_and_last_key_wins():
    assert _parse_frontmatter("---\n---\nbody")[0] == {}
    assert _parse_frontmatter("---\nname: a\nname: b\n---\nx")[0] == {"name": "b"}


# --- MarkdownSkill / load_skills -------------------------------------------


def _write_skill(root, folder: str, text: str):
    d = root / folder
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(text)


def test_markdown_skill_reads_frontmatter_eagerly_and_body_lazily(tmp_path):
    _write_skill(
        tmp_path,
        "pirate",
        "---\nname: pirate-speak\ndescription: Talk like a pirate.\n---\nSay Arrr.\n",
    )
    skill = MarkdownSkill(tmp_path / "pirate" / "SKILL.md")
    assert skill.name == "pirate-speak"
    assert skill.description == "Talk like a pirate."
    assert skill.instructions() == "Say Arrr."


def test_markdown_skill_name_falls_back_to_directory(tmp_path):
    _write_skill(tmp_path, "fallback-name", "No frontmatter, just a body.")
    skill = MarkdownSkill(tmp_path / "fallback-name" / "SKILL.md")
    assert skill.name == "fallback-name"
    assert skill.description == ""


def test_markdown_skill_frontmatter_only_yields_empty_body(tmp_path):
    _write_skill(tmp_path, "meta-only", "---\nname: m\ndescription: D\n---\n")
    skill = MarkdownSkill(tmp_path / "meta-only" / "SKILL.md")
    assert skill.name == "m"
    assert skill.description == "D"
    assert skill.instructions() == ""


def test_markdown_skill_empty_file_falls_back_to_folder(tmp_path):
    _write_skill(tmp_path, "blank", "")
    skill = MarkdownSkill(tmp_path / "blank" / "SKILL.md")
    assert skill.name == "blank"
    assert skill.instructions() == ""


def test_markdown_skill_tolerates_bom(tmp_path):
    _write_skill(tmp_path, "bommed", "﻿---\nname: b\ndescription: D\n---\nbody")
    skill = MarkdownSkill(tmp_path / "bommed" / "SKILL.md")
    assert skill.name == "b"  # BOM did not hide the frontmatter
    assert skill.description == "D"


def test_load_skills_finds_all_sorted(tmp_path):
    _write_skill(tmp_path, "b", "---\nname: beta\ndescription: B\n---\nbody b")
    _write_skill(tmp_path, "a", "---\nname: alpha\ndescription: A\n---\nbody a")
    skills = load_skills(tmp_path)
    assert [s.name for s in skills] == ["alpha", "beta"]  # sorted by path: a/, b/


def test_load_skills_recurses_nested_dirs_sorted_by_path(tmp_path):
    _write_skill(tmp_path, "a", "---\nname: a\ndescription: A\n---\nbody a")
    _write_skill(tmp_path, "a/nested", "No frontmatter here.")
    _write_skill(tmp_path, "b", "---\nname: b\ndescription: B\n---\nbody b")
    skills = load_skills(tmp_path)
    # sorted by path: a/SKILL.md, a/nested/SKILL.md, b/SKILL.md
    assert [s.name for s in skills] == ["a", "nested", "b"]
    assert skills[1].description == ""


def test_load_skills_missing_dir_returns_empty(tmp_path):
    assert load_skills(tmp_path / "does-not-exist") == []


def test_load_skills_expands_tilde(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    _write_skill(tmp_path, "skills/greet", "---\nname: greet\ndescription: G\n---\nhi")
    assert [s.name for s in load_skills("~/skills")] == ["greet"]


def test_load_skills_skips_unreadable_file_and_keeps_the_rest(tmp_path):
    _write_skill(tmp_path, "good", "---\nname: good\ndescription: G\n---\nbody")
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "SKILL.md").write_bytes(b"\xff\xfe\x00not utf-8")
    with pytest.warns(UserWarning, match="skipping unreadable skill"):
        skills = load_skills(tmp_path)
    assert [s.name for s in skills] == ["good"]


# --- end to end: a plain .use(skill) is all it takes -----------------------


def test_skill_is_advertised_to_the_model_as_a_tool():
    """Progressive disclosure: the model sees the skill's name + description."""
    captured: dict[str, str] = {}

    class Spy(ScriptedModel):
        async def complete(self, history, ctx):
            captured.update({t.name: t.description for t in ctx.all(Tool)})
            return await super().complete(history, ctx)

    harness = (
        NexusAIHarness()
        .use(AgenticLoop())
        .use(Spy(Response(text="hi")))
        .use(AutoApprove())
        .use(_StubSkill("pd", "Pirate dialect."))
    )
    asyncio.run(harness.run("hi", session=Session()))
    asyncio.run(harness.stop())
    assert captured.get("pd") == "Pirate dialect."


def test_skill_loads_into_conversation_through_the_loop():
    """Model calls the skill by name; its instructions land as a tool message."""
    model = ScriptedModel(
        Response(tool_calls=[{"id": "1", "name": "pd", "arguments": {}}]),
        Response(text="Understood."),
    )
    harness = (
        NexusAIHarness()
        .use(AgenticLoop())
        .use(model)
        .use(AutoApprove())
        .use(_StubSkill("pd", "Pirate dialect.", body="Respond only as a pirate."))
    )
    result = asyncio.run(harness.run("be a pirate", session=Session()))
    asyncio.run(harness.stop())

    assert result.output == "Understood."
    tool_texts = [m.content for m in model.calls[1] if m.role == "tool"]
    assert any("Respond only as a pirate." in t for t in tool_texts)
