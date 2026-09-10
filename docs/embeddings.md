# Embeddings

`client.embeddings`

## Create embeddings

`create(...)` → `POST /v1/embeddings`

```python
from synapsai import SynapsAI

client = SynapsAI()

response = client.embeddings.create(
    model="your-embedding-model",
    input="SynapsAI Cloud",
    # input=["a", "b"] also works
)
print(len(response.data[0].embedding))
```

| Parameter | Default |
| --- | --- |
| `model` | required |
| `input` | required (`str` or list of strings) |
| `encoding_format` | `"float"` |

## Similarity

`similarity(...)` embeds the source and candidates, then computes cosine similarity locally.

```python
result = client.embeddings.similarity(
    model="your-embedding-model",
    source_sentence="how to reset a password",
    sentences=[
        "password recovery steps",
        "deploy a model",
    ],
    return_embeddings=False,
)
for item in result.data:
    print(item.score, item.sentence)
```
