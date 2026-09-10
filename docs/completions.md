# Completions

`client.completions.create(...)` → `POST /v1/completions`

Classic prompt completions (non-chat).

```python
from synapsai import SynapsAI

client = SynapsAI()

response = client.completions.create(
    model="your-model-id",
    prompt="Once upon a time",
    max_completion_tokens=64,
    temperature=0.8,
)
print(response.choices[0].text)
```

## Streaming

```python
for chunk in client.completions.create(
    model="your-model-id",
    prompt="Hello",
    stream=True,
):
    print(chunk.choices[0].text, end="", flush=True)
```

## Main parameters

| Parameter | Default |
| --- | --- |
| `model` | required |
| `prompt` | required |
| `temperature` | `1.0` |
| `top_p` | `1.0` |
| `n` | `1` |
| `stream` | `False` |
| `stop` | `[]` |
| `max_completion_tokens` | `128` |
| `presence_penalty` / `frequency_penalty` | `0.0` |
| `logit_bias` | — |

Prefer [Chat](chat.md) for new applications.
