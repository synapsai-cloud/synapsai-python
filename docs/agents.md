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

## Send a list of messages

`messages` accepts a string, a list of strings, or a list of message objects (`role` required; `content` required unless `assistant`).

```python
for event in client.agents.run(
    "agt_your_agent_id",
    messages=[
        {"id": "m1", "role": "user", "content": "Summarize the latest doc"},
        {"id": "m2", "role": "assistant", "content": "The latest doc covers billing and API keys."},
        {"id": "m3", "role": "user", "content": "Keep it short"},
    ],
):
    if event.type == "TEXT_MESSAGE_CONTENT" and event.delta:
        print(event.delta, end="", flush=True)
```

If `thread_id` / `run_id` are omitted, UUIDs are generated. Pass `run_id` yourself when you need to resume. `state` may set `max_steps` and `compress_context` for that run.

Closing the connection does not stop the run. Resume it on the server that is still running it. When the run ends, the in-memory copy is dropped. A later fetch works only if **Store requests** was on for that agent or model.

## Resume, fetch, and delete

```python
run_id = "run-1"

for event in client.agents.runs.stream("agt_your_agent_id", run_id, after=0):
    print(event.seq, event.type)

snapshot = client.agents.runs.retrieve("agt_your_agent_id", run_id)
print(snapshot.status, len(snapshot.events))

client.agents.runs.delete("agt_your_agent_id", run_id)
client.agents.delete("agt_your_agent_id")
```

| Method | Endpoint |
| --- | --- |
| `agents.runs.retrieve` | `GET /v1/agent/{agent_id}/runs/{run_id}` |
| `agents.runs.stream` | `GET /v1/agent/{agent_id}/runs/{run_id}/stream?after=` |
| `agents.runs.delete` | `DELETE /v1/agent/{agent_id}/runs/{run_id}` |
| `agents.delete` | `DELETE /v1/agent/{agent_id}` |

`after` is the last `seq` you already applied. Each event on the snapshot includes `seq`. A finished run replays from the data store and the stream ends. With store requests off, that run is gone.

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

- Auth uses your API key.
- Nested model calls from the agent go through `/v1/chat/completions` for billing and tracing.
- See also [Chat](chat.md) and [Vector stores](vector-stores.md) for the building blocks agents use.
