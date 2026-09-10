# Vector stores

`client.vector_stores` and `client.vector_stores.files`

OpenAI-compatible vector store API backed by SynapsAI knowledge bases. Uses the normal API host (`https://api.synapsai.cloud/v1`), not the upload host.

Your API key needs vector-store permissions.

## Create, list, retrieve, update, delete

```python
from synapsai import SynapsAI

client = SynapsAI()

vs = client.vector_stores.create(
    name="Product docs",
    description="Public handbook",
)

page = client.vector_stores.list(limit=20, order="desc")
detail = client.vector_stores.retrieve(vs.id)

updated = client.vector_stores.update(
    vs.id,
    name="Product docs v2",
    description="Updated handbook",
)

client.vector_stores.delete(vs.id)
```

## Search

```python
results = client.vector_stores.search(
    vs.id,
    query="How do I rotate API keys?",
    max_num_results=10,
    # filters=..., ranking_options={"score_threshold": 0.2}
)
for item in results.data:
    print(item.score, item.filename, item.content)
```

`query` may be a string or a list of strings.

## Files

### Upload a file (multipart)

```python
file = client.vector_stores.files.create(
    vs.id,
    file="./guide.pdf",
    attributes={"source": "handbook"},
)
print(file.id, file.status)
```

`file` may be a path, a file object, or `(filename, bytes[, content_type])`.

### Attach an existing document by id

```python
file = client.vector_stores.files.create(
    vs.id,
    file_id="doc-abc123",
    attributes={"tag": "imported"},
)
```

Provide **either** `file` **or** `file_id`, not both.

### Upload many paths (files and directories)

```python
uploaded = client.vector_stores.files.upload_paths(
    vs.id,
    ["./docs", "./README.md"],
    attributes={"batch": "2026-03"},
)
```

Directories are walked recursively. Same helper the CLI uses.

### List / retrieve / update / delete / content

```python
files = client.vector_stores.files.list(vs.id, filter="completed", limit=50)
one = client.vector_stores.files.retrieve(vs.id, file.id)
one = client.vector_stores.files.update(vs.id, file.id, attributes={"lang": "en"})
page = client.vector_stores.files.content(vs.id, file.id)
client.vector_stores.files.delete(vs.id, file.id)
```

## CLI

```bash
synapsai upload-vector-store-files vs_abc123 ./docs ./readme.md \
  --attributes '{"source":"handbook"}'
```

See [CLI](cli.md).
