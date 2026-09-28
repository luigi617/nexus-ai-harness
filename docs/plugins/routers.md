# Routers

A router picks which model handles each turn when you've registered more than
one. Without a router, the agent always uses the single registered model.

```python
from nexus_ai_harness.plugins.models import AnthropicModel, OpenAIModel
from nexus_ai_harness.plugins.routers import LLMRouter

harness.use(AnthropicModel(model="...")).use(OpenAIModel(model="..."))
harness.use(LLMRouter())        # choose a model per turn
```

## Built-in routers

- **`LLMRouter`**: asks a model to choose the best model for the current turn.
- **`StickyRouter`**: wraps another router, decides once on the first turn, and
  reuses that choice for the rest of the run.

```python
from nexus_ai_harness.plugins.routers import LLMRouter, StickyRouter

harness.use(StickyRouter(LLMRouter()))
```
