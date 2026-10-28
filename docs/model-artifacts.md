# Model artifacts

`client.model_artifacts`

Upload model weight/config files into an **existing** artifact. Requests go to `https://api.synapsai.cloud/v1/model-artifacts`.

```python
from synapsai import SynapsAI

client = SynapsAI()
```

Your API key must include upload permission.

## API surface (matches the upload routes)

| Method | HTTP |
| --- | --- |
| `create(display_name, pipeline)` | `POST /v1/model-artifacts` |
| `retrieve(artifact_id)` | `GET /v1/model-artifacts/{id}` |
| `start_upload(artifact_id)` | `POST /v1/model-artifacts/{id}/uploads` |
| `upload_file_direct(...)` | `POST …/direct-uploads`, presigned part `PUT`s, `POST …/direct-uploads/complete` |
| `upload_file(...)` | `PUT …/uploads/{upload_id}/files?path=` (local storage only) |
| `complete_upload(artifact_id, upload_id)` | `POST …/uploads/{upload_id}/complete` |
| `upload(path, *, artifact_id)` | Start, then the route selected by `transfer`, then complete |

## Which upload route

`start_upload` is the check. Its response includes `transfer`. `upload()` reads that field, so callers do not pass a flag.

| `transfer` | When | Route |
| --- | --- | --- |
| `direct` | Default. Artifact storage is S3 or R2. | `upload_file_direct`. Parts go to the bucket. The weight file is not sent through the API. |
| `proxy` | On-prem / BYOC with `MODEL_ARTIFACT_STORAGE=local`. | `upload_file`. The API writes each file onto the host disk. |

Use `upload_file` only when `session.transfer == "proxy"`. On every other value, including a missing one, use `upload_file_direct`.

Browser uploads read the same field. For `direct`, the bucket CORS policy must allow `PUT` from the platform origin and expose `ETag`, which multipart completion needs. Local storage does not use that CORS policy; its bytes stay on `PUT …/files`.

Artifacts are normally created in the SynapsAI console (or via `create`). File upload always targets an existing `artifact_id`.

## High-level upload

Accepts a **single file** or a **directory**.

```python
from synapsai import SynapsAI

client = SynapsAI(timeout=3600.0)

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

if session.transfer == "proxy":
    # Local disk on an on-prem deployment.
    client.model_artifacts.upload_file(
        artifact_id,
        session.upload_id,
        path="model.safetensors",
        file="./model.safetensors",
    )
else:
    # Default: straight to the bucket.
    client.model_artifacts.upload_file_direct(
        artifact_id,
        session.upload_id,
        path="model.safetensors",
        file="./model.safetensors",
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

Uses `https://api.synapsai.cloud/v1`, or `SYNAPSAI_API_BASE` when that variable is set.

See [CLI](cli.md).
