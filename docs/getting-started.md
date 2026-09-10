# Getting started

## Install

```bash
pip install --upgrade synapsai-python
```

Requires Python 3.8+.

## Configure

Set your API key (required):

```bash
export SYNAPSAI_API_KEY="synapsai-..."
```

Optional:

| Variable | Default | Purpose |
| --- | --- | --- |
| `SYNAPSAI_API_KEY` | — | Bearer token |
| `SYNAPSAI_API_BASE` | `https://api.synapsai.cloud/v1` | Inference / vector stores / agents |
| `SYNAPSAI_UPLOAD_BASE` | `https://upload.synapsai.cloud/v1` | Model artifact uploads (CLI) |

## Create a client

```python
from synapsai import SynapsAI

client = SynapsAI()  # uses SYNAPSAI_API_KEY
```

Or explicitly:

```python
from synapsai import SynapsAI, DEFAULT_UPLOAD_BASE_URL

client = SynapsAI(
    api_key="synapsai-...",
    base_url="https://api.synapsai.cloud/v1",
    timeout=300.0,
    max_retries=1,
)

upload_client = SynapsAI(
    api_key="synapsai-...",
    base_url=DEFAULT_UPLOAD_BASE_URL,
    timeout=3600.0,
)
```

### Constructor options

| Argument | Description |
| --- | --- |
| `api_key` | API key; falls back to `SYNAPSAI_API_KEY` |
| `base_url` | API root including `/v1` |
| `timeout` | Request timeout in seconds (default `300`) |
| `max_retries` | Attempts on 429/5xx and network errors (default `1`) |
| `headers` | Extra headers merged into defaults |
| `httpx_client` | Custom `httpx.Client` / `AsyncClient` |

Missing `api_key` raises `AuthenticationError`.

## First request

```python
from synapsai import SynapsAI

client = SynapsAI()

for model in client.models.list().data:
    print(model.id, model.status)

response = client.chat.completions.create(
    model="your-model-id",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

## Sync vs async

- `SynapsAI` — blocking (`httpx.Client`)
- `AsyncSynapsAI` — awaitable (`httpx.AsyncClient`); see [Async client](async.md)

## Resources on the client

```text
client.chat.completions
client.responses
client.completions
client.embeddings
client.images
client.audio          # .speech / .transcriptions / .translations
client.videos
client.models
client.classifications
client.question_answering
client.feature_extraction
client.fill_mask
client.rerank
client.vector_stores  # .files
client.model_artifacts
client.agents
```

## Next steps

- [Chat](chat.md) · [Responses](responses.md) · [Vector stores](vector-stores.md) · [Model artifacts](model-artifacts.md) · [Agents](agents.md) · [CLI](cli.md)
- [Errors](errors.md)
