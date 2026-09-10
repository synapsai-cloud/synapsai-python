# Feature extraction

`client.feature_extraction.create(...)` → `POST /v1/feature-extraction`

Extract dense features / embeddings from text inputs via a feature-extraction pipeline model.

```python
from synapsai import SynapsAI

client = SynapsAI()

response = client.feature_extraction.create(
    model="your-feature-model",
    inputs="SynapsAI Cloud",
    # or inputs=["a", "b"]
)
print(response)
```

| Parameter | Notes |
| --- | --- |
| `model` | required |
| `inputs` | `str` or `list[str]` |
| `**kwargs` | Forwarded to the API |

For image feature extraction, see [`client.images.feature_extraction`](images.md).
For OpenAI-style embeddings, see [Embeddings](embeddings.md).
