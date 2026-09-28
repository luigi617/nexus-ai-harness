# Evaluators

An evaluator scores agent output against questions you define and returns typed
answers — a yes/no, a chosen option, or a rubric score — each with a confidence.
Use it to grade runs or gate decisions instead of parsing free-form model text.

## Ask three kinds of question

- **`noul`** — a yes/no question, scored 0–1.
- **`choice`** — pick one option from a set you describe.
- **`score`** — rate against an ordered rubric of levels.

## Usage

`JevEvaluator` calls TypeSafe's evaluation service. Set `TYPESAFE_API_KEY` in
`.env`, then ask questions about some text:

```python
from nexus_ai_harness.plugins.evaluators import JevEvaluator
from nexus_ai_harness.plugins.evaluators.jev import noul, choice

evaluator = JevEvaluator()
result = evaluator.evaluate(
    state=agent_answer,
    questions={
        "correct": noul("Is the answer factually correct?"),
        "tone": choice("What is the tone?", {"formal": "...", "casual": "..."}),
    },
    ctx=ctx,
)
```
