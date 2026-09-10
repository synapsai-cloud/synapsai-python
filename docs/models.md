# Models

`client.models`

List and retrieve models visible to your API key.

```python
from synapsai import SynapsAI

client = SynapsAI()

page = client.models.list()
for model in page.data:
    print(model.id, getattr(model, "status", None))

detail = client.models.retrieve("your-model-id")
print(detail)
```

| Method | Endpoint |
| --- | --- |
| `list()` | `GET /v1/models` |
| `retrieve(model)` | `GET /v1/models/{model}` |
