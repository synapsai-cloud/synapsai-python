# Fill mask

`client.fill_mask.create(...)` → `POST /v1/fill-mask`

Masked language modeling (MLM).

```python
from synapsai import SynapsAI

client = SynapsAI()

response = client.fill_mask.create(
    model="your-mlm-model",
    inputs="Paris is the [MASK] of France.",
    top_k=5,
)
print(response)
```

| Parameter | Default |
| --- | --- |
| `model` | required |
| `inputs` | required (string with mask token) |
| `targets` | optional candidate tokens |
| `top_k` | `5` |
