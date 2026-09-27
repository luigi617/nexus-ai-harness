# Skills

A **skill** is a named, on-demand instruction set the model loads only when it's
relevant — *progressive disclosure*. A skill *is a tool*: the model sees its
`name` and one-line `description` in its tool list, and when it calls the skill
the full `instructions` are returned into the conversation to steer the
following turns. The prompt stays small while deep, task-specific guidance is
available exactly when needed.

Because a skill is a tool, there's nothing special to wire — register it like
any other plugin.

## Wiring

```python
from harness import NexusAIHarness
from plugins import AgenticLoop, load_skills

harness = NexusAIHarness().use(AgenticLoop()).use(model)
harness.use(DeepCalc())                              # a code-authored skill
for skill in load_skills("~/.nexus-ai-harness/skills"):   # or SKILL.md files
    harness.use(skill)
```

`load_skills(directory)` returns skills you `.use()`; there is no separate skill
tool to register. The batteries-included `default_harness(model)` works the same
way — add skills to it with `.use(skill)`.

Since a skill is a tool, it flows through the permission layer by name (an
`AllowList`/`DenyList` can gate individual skills) and shares the tool
namespace, so give skills names that won't collide with your other tools.

## Authoring a skill

### From a `SKILL.md` file (no code)

Lay skills out one folder each, matching the Claude Code convention:

```
skills/
  pirate-speak/
    SKILL.md
```

```markdown
---
name: pirate-speak
description: Respond in exaggerated pirate dialect.
---
Respond only as a pirate. Begin every reply with "ARRR" and use nautical slang.
```

`load_skills(directory)` reads every `SKILL.md` under `directory` (recursively).
Frontmatter (`name`, `description`) is parsed eagerly so the tool listing is
cheap; the instruction body is read lazily, only when the skill is invoked. If
`name` is omitted it falls back to the folder name. A file that can't be read is
skipped with a warning rather than sinking the whole load.

Frontmatter is intentionally minimal: flat `key: value` pairs only (a leading
BOM is tolerated). Nested or multi-line YAML isn't parsed, and the body must not
begin with a `---` line.

### In code

```python
from protocols.skill import Skill

class DeepCalc(Skill):
    name = "deep-calc"
    description = "Careful step-by-step arithmetic."

    def instructions(self) -> str:      # may be def or async def
        return "Work through each step and show your work before answering."
```

Only `name`, `description`, and `instructions` are yours to supply — the `Skill`
base handles being called (it returns the instructions and emits the event) and
declares an empty argument schema, since invoking a skill *is* the request to
load it.

## Observability

Invoking a skill emits a `SkillInvoked(name)` event — fired before the body is
read, so even a failed lazy load is visible — so hooks and tracers can record
which skills a run used. The usual `ToolCallStarted` / `ToolCallCompleted`
events fire too, since a skill is a tool.
