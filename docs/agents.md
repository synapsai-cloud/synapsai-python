# Agents

`client.agents.run(...)` → `POST /v1/agent/{agent_id}/run`

Run a **persisted** SynapsAI agent and stream [AG-UI](https://docs.ag-ui.com/) SSE events.

Tools, MCP servers, knowledge bases, and the system prompt come from the stored agent on the server. Client-supplied `tools` are forwarded for protocol compatibility and do not widen server capabilities.

## Basic run

```python
from synapsai import SynapsAI

client = SynapsAI()

for event in client.agents.run(
    "agt_your_agent_id",
    messages=["What is in my knowledge base about billing?"],
):
    if event.type == "TEXT_MESSAGE_CONTENT" and event.delta:
        print(event.delta, end="", flush=True)
    elif event.type == "RUN_ERROR":
        print("\nError:", event.message)
    elif event.type == "RUN_FINISHED":
        print("\nDone.")
```

## Message formats

`messages` accepts:

- plain strings (treated as `user` messages with generated ids)
- dicts (`role` required; `content` required unless `assistant`)
- `AgentMessage` models

```python
from synapsai.types import AgentMessage

events = client.agents.run(
    "agt_your_agent_id",
    messages=[
        {"id": "m1", "role": "user", "content": "Summarize the latest doc"},
        AgentMessage(id="m2", role="user", content="Keep it short"),
    ],
    thread_id="thread-optional",
    run_id="run-optional",
    state={"max_steps": 12},  # optional overrides allowed by the server
    forwarded_props={"store": True},
)
```

If `thread_id` / `run_id` are omitted, UUIDs are generated.

## Event types

Events are parsed into `AgentEvent` (extra fields allowed). Common `type` values:

| Type | Meaning |
| --- | --- |
| `RUN_STARTED` | Run began |
| `TEXT_MESSAGE_START` / `TEXT_MESSAGE_CONTENT` / `TEXT_MESSAGE_END` | Assistant text stream (`delta`) |
| `TOOL_CALL_START` / `TOOL_CALL_ARGS` / `TOOL_CALL_END` | Tool invocation |
| `TOOL_CALL_RESULT` | Tool output (`content`) |
| `RUN_FINISHED` | Success |
| `RUN_ERROR` | Failure (`message`) |

## Async

```python
from synapsai import AsyncSynapsAI

async with AsyncSynapsAI() as client:
    async for event in client.agents.run("agt_…", messages=["Hello"]):
        print(event.type, event.delta)
```

## Notes

- Auth uses your normal API key against the inference host.
- Nested model calls from the agent go through `/v1/chat/completions` for billing and tracing.
- See also [Chat](chat.md) and [Vector stores](vector-stores.md) for the building blocks agents use.
