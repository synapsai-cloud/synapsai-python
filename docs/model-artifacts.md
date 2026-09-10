# Model artifacts

`client.model_artifacts`

Upload model weight/config files into an **existing** artifact on the infra upload API.

## Host

Use the upload base URL (not the inference API):

```python
from synapsai import SynapsAI, DEFAULT_UPLOAD_BASE_URL

client = SynapsAI(base_url=DEFAULT_UPLOAD_BASE_URL)
# DEFAULT_UPLOAD_BASE_URL == "https://upload.synapsai.cloud/v1"
```

Your API key must include upload permission.

## API surface (matches the upload routes)

| Method | HTTP |
| --- | --- |
| `create(display_name, pipeline)` | `POST /v1/model-artifacts` |
| `retrieve(artifact_id)` | `GET /v1/model-artifacts/{id}` |
| `start_upload(artifact_id)` | `POST /v1/model-artifacts/{id}/uploads` |
| `upload_file(artifact_id, upload_id, path, file)` | `PUT …/uploads/{upload_id}/files?path=` |
| `complete_upload(artifact_id, upload_id)` | `POST …/uploads/{upload_id}/complete` |
| `upload(path, *, artifact_id)` | Convenience: start → files → complete |

Artifacts are normally created in the SynapsAI console (or via `create`). File upload always targets an existing `artifact_id`.

## High-level upload

Accepts a **single file** or a **directory**.

```python
from synapsai import SynapsAI, DEFAULT_UPLOAD_BASE_URL

client = SynapsAI(base_url=DEFAULT_UPLOAD_BASE_URL, timeout=3600.0)

def on_progress(relative_path, index, total):
    print(f"[{index}/{total}] {relative_path}")

artifact = client.model_artifacts.upload(
    "./my-model",                 # or "./model.safetensors"
    artifact_id="artifact-demo-abc12345",
    on_progress=on_progress,
)
print(artifact.id, artifact.status, artifact.size_gb)
```

Directory uploads preserve relative paths (`config.json`, `tokenizer/vocab.json`, …). A single file is stored as its basename.

## Step-by-step (manual)

```python
session = client.model_artifacts.start_upload(artifact_id)

client.model_artifacts.upload_file(
    artifact_id,
    session.upload_id,
    path="model.safetensors",
    file="./model.safetensors",
)
client.model_artifacts.upload_file(
    artifact_id,
    session.upload_id,
    path="config.json",
    file="./config.json",
)

result = client.model_artifacts.complete_upload(artifact_id, session.upload_id)
print(result.artifact.status)
```

## Create an artifact record (optional)

```python
created = client.model_artifacts.create(
    display_name="My weights",
    pipeline="text-generation",
)
artifact_id = created.artifact.id
```

Pipeline is only used at create time, not during `upload()`.

## CLI

```bash
synapsai upload-model ./my-model --artifact-id artifact-demo-abc12345
synapsai upload-model ./model.safetensors --artifact-id artifact-demo-abc12345
```

Defaults to `https://upload.synapsai.cloud/v1`. Override with `--host` or `SYNAPSAI_UPLOAD_BASE`.

See [CLI](cli.md).
