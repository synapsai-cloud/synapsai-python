# CLI

The package installs a `synapsai` console script.

```bash
synapsai --help
synapsai --version
```

## Environment

| Variable | Used by |
| --- | --- |
| `SYNAPSAI_API_KEY` | All commands (or pass `--api-key`) |
| `SYNAPSAI_API_BASE` | API host when `--host` is omitted. Default `https://api.synapsai.cloud/v1` |

A host may be a hostname or a full URL. The CLI normalizes it to `https://…/v1`.

---

## `upload-model`

Upload a **file or directory** into an **existing** model artifact.

The first request, `POST /v1/model-artifacts/{id}/uploads`, returns `transfer`. The CLI follows that field. There is no storage flag.

| `transfer` | Deployment | Where the file bytes go |
| --- | --- | --- |
| `direct` (default) | S3 or R2 | Presigned parts straight to the bucket: `POST …/direct-uploads`, then `PUT` each part to object storage, then `POST …/direct-uploads/complete` |
| `proxy` | Local disk, on-prem / BYOC | `PUT /v1/model-artifacts/{id}/uploads/{upload_id}/files?path=…` on the API |

Both paths finish with `POST …/uploads/{upload_id}/complete`.

```bash
synapsai upload-model ./my-model --artifact-id artifact-demo-abc12345

synapsai upload-model ./model.safetensors \
  --artifact-id artifact-demo-abc12345 \
  --timeout 7200
```

| Option | Required | Default |
| --- | --- | --- |
| `PATH` | yes | file or directory |
| `--artifact-id` | yes | — |
| `--host` | no | `SYNAPSAI_API_BASE` or `https://api.synapsai.cloud/v1` |
| `--api-key` | no | `SYNAPSAI_API_KEY` |
| `--timeout` | no | `3600` |

Directory uploads preserve relative paths (`config.json`, `subdir/tokenizer.json`, …). A single file is stored under its basename.

---

## `upload-vector-store-files`

Upload one or more files/directories into a vector store (multipart ingest).

```bash
synapsai upload-vector-store-files vs_abc123 ./docs ./readme.md

synapsai upload-vector-store-files vs_abc123 ./data \
  --attributes '{"source":"handbook"}'
```

| Option | Required | Default |
| --- | --- | --- |
| `VECTOR_STORE_ID` | yes | — |
| `PATHS…` | yes | files and/or directories (recursive) |
| `--attributes` | no | JSON object applied to each file |
| `--host` | no | `SYNAPSAI_API_BASE` / API default |
| `--api-key` | no | `SYNAPSAI_API_KEY` |
| `--timeout` | no | `600` |

See also [Model artifacts](model-artifacts.md) and [Vector stores](vector-stores.md).
