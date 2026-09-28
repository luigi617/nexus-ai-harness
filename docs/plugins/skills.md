# Skills

A skill is a set of instructions the agent loads only when it needs them. The
agent sees each skill's name and one-line description; when it decides a skill is
relevant, the full instructions are pulled into the conversation to guide the
next steps. Your prompt stays small, and deep guidance is there when it's needed.

A skill works like any other tool, so you register it the same way.

## Adding skills

```python
from nexus_ai_harness.plugins import load_skills

harness.use(DeepCalc())                                   # a skill written in code
for skill in load_skills("~/.nexus-ai-harness/skills"):   # skills from SKILL.md files
    harness.use(skill)
```

Give skills names that don't clash with your other tools — they share the same
tool namespace and go through the same permission checks.

## Writing a skill

### As a `SKILL.md` file (no code)

Put each skill in its own folder:

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

`load_skills(directory)` finds every `SKILL.md` under `directory`. The `name` and
`description` come from the frontmatter (if `name` is omitted, the folder name is
used); the instructions are the body below it.

### In code

```python
from nexus_ai_harness.protocols.skill import Skill

class DeepCalc(Skill):
    name = "deep-calc"
    description = "Careful step-by-step arithmetic."

    def instructions(self) -> str:      # may be def or async def
        return "Work through each step and show your work before answering."
```

You only supply `name`, `description`, and `instructions`; the base class handles
the rest.
