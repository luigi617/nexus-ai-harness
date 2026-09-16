# Models

A `Model` (protocol: `protocols/model.py`) turns a message history into a
`Response`. It is identified for routing by `provider` and `name`.

<!-- TODO:
- The Model protocol: complete(history, ctx) -> Response | Awaitable[Response].
- BedrockModel: provider/name, PRICING and DESCRIPTIONS consts.
- Adding a new backend (implement complete, set provider/name).
-->
