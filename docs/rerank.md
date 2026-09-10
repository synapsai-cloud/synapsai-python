# Rerank

`client.rerank.create(...)` → `POST /v1/rerank`

Reorder documents by relevance to a query.

```python
from synapsai import SynapsAI

client = SynapsAI()

response = client.rerank.create(
    model="your-rerank-model",
    query="vector database setup",
    documents=[
        "Install and configure a vector store",
        "How to fry eggs",
        "Embedding models for RAG",
    ],
    top_n=2,
)

for item in response.results:
    print(item.index, item.relevance_score)
```

| Parameter | Default |
| --- | --- |
| `model` | required |
| `query` | required |
| `documents` | required |
| `top_n` | — (all scored) |
| `max_tokens_per_doc` | `4096` |
