# Responses

`client.responses`

OpenAI-style Responses API for `POST /v1/responses`, plus retrieve/delete for stored responses.

## Supported methods

| Method | Endpoint |
| --- | --- |
| `create(...)` | `POST /v1/responses` |
| `retrieve(response_id)` | `GET /v1/responses/{response_id}` |
| `delete(response_id)` | `DELETE /v1/responses/{response_id}` |

**Not offered** (no SDK methods): cancel a response, list input items, or count input tokens.

## Create

```python
from synapsai import SynapsAI

client = SynapsAI()

response = client.responses.create(
    model="your-model-id",
    input="Explain RAG in one paragraph.",
    store=True,  # persist when the model has data store enabled
    metadata={"topic": "rag"},
)
print(response.id, response.output_text or response.output)
```

`input` may be a string, a list of message/content items, or a dict.

### Continue a stored conversation

```python
follow_up = client.responses.create(
    model="your-model-id",
    input="Give a shorter version.",
    previous_response_id=response.id,
    store=True,
)
```

### Streaming

```python
for event in client.responses.create(
    model="your-model-id",
    input="Count to five.",
    stream=True,
):
    if event.type == "response.output_text.delta" and event.delta:
        print(event.delta, end="", flush=True)
    elif event.type == "response.completed" and event.response:
        print("\n", event.response.id)
```

## Retrieve / delete stored responses

```python
stored = client.responses.retrieve(response.id)
client.responses.delete(response.id)
```

## Main `create` parameters

| Parameter | Notes |
| --- | --- |
| `model` | required |
| `input` | required |
| `stream` | default `False` |
| `store` | persist when data store is enabled on the model |
| `previous_response_id` | continue a stored chain |
| `metadata` | optional object stored with the response |
| `instructions` / `temperature` / `top_p` / `tools` / `tool_choice` | forwarded when supported |
| `**kwargs` | extra Responses fields forwarded to the API |

See also [Chat](chat.md) for the chat completions data-store CRUD API.
