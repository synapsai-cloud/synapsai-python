# Errors

## Exception hierarchy

```text
SynapsAIError
├── APIError              # HTTP / stream failures (status_code, message)
│   ├── AuthenticationError
│   └── RateLimitError    # defined for callers; retries handle 429
├── ValidationError
├── TimeoutError
└── ConnectionError
```

Import from `synapsai.exceptions` or catch `APIError` for most API failures.

```python
from synapsai import SynapsAI
from synapsai.exceptions import APIError, AuthenticationError

try:
    client = SynapsAI()
    client.models.list()
except AuthenticationError as e:
    print("Missing API key:", e)
except APIError as e:
    print(e.status_code, e.message)
```

## Retries

The client retries:

- Network / transport errors (`httpx.RequestError`)
- Timeouts (`httpx.TimeoutException`)
- HTTP `429` and `5xx`

Backoff is exponential with jitter (capped at 30s). Configure with `max_retries` (minimum `1`).

## Resource-level errors

| Situation | Raised |
| --- | --- |
| Empty path for artifact / vector upload | `FileNotFoundError` |
| Vector store `file` and `file_id` both/neither | `ValueError` |
| Invalid agent message shape | `TypeError` / `ValueError` |
| Video poll timeout | `APIError` |
