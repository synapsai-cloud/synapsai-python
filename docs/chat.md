# Chat completions

`client.chat.completions`

OpenAI-compatible chat API with optional streaming and stored-completion CRUD.

## Create

`create(...)` → `POST /v1/chat/completions`

```python
from synapsai import SynapsAI

client = SynapsAI()

response = client.chat.completions.create(
    model="your-model-id",
    messages=[
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Explain vector stores in one sentence."},
    ],
    temperature=0.7,
    max_completion_tokens=256,
    store=True,
    metadata={"channel": "docs"},
)

print(response.id, response.choices[0].message.content)
```

### Streaming

```python
stream = client.chat.completions.create(
    model="your-model-id",
    messages=[{"role": "user", "content": "Count to five."}],
    stream=True,
)

for chunk in stream:
    delta = chunk.choices[0].delta
    if delta and delta.content:
        print(delta.content, end="", flush=True)
```

### Continue a stored conversation

```python
follow_up = client.chat.completions.create(
    model="your-model-id",
    messages=[{"role": "user", "content": "Make it shorter."}],
    previous_completion_id=response.id,
    store=True,
)
```

## Stored completions CRUD

When the model has data store enabled and completions are persisted (`store` omitted or `True`):

| Method | Endpoint |
| --- | --- |
| `list(model=?, after=?, limit=20, order="desc")` | `GET /v1/chat/completions` |
| `retrieve(completion_id)` | `GET /v1/chat/completions/{id}` |
| `update(completion_id, metadata={...})` | `POST /v1/chat/completions/{id}` |
| `delete(completion_id)` | `DELETE /v1/chat/completions/{id}` |

```python
page = client.chat.completions.list(model="your-model-id", limit=20)
stored = client.chat.completions.retrieve(response.id)
stored = client.chat.completions.update(response.id, metadata={"reviewed": "true"})
client.chat.completions.delete(response.id)
```

## Main `create` parameters

| Parameter | Default | Notes |
| --- | --- | --- |
| `model` | required | Deployed model id |
| `messages` | required | Chat message list |
| `temperature` | `1.0` | |
| `top_p` | `1.0` | |
| `n` | `1` | |
| `stream` | `False` | Yields `ChatCompletionChunk` when `True` |
| `stop` | `[]` | |
| `max_completion_tokens` | `128` | |
| `presence_penalty` / `frequency_penalty` | `0.0` | |
| `tools` / `tool_choice` | — | Function / tool calling |
| `response_format` | — | |
| `seed` | — | |
| `reasoning_effort` | — | `none` … `max` when supported |
| `store` | — | Persist when data store is enabled |
| `previous_completion_id` | — | Continue a stored chain |
| `metadata` | — | Stored with the completion |
| `**kwargs` | — | Forwarded to the API |

Returns `ChatCompletionResponse`, or an iterator of `ChatCompletionChunk` when streaming.

For the Responses API, see [Responses](responses.md).
