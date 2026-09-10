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
| `SYNAPSAI_UPLOAD_BASE` | `upload-model` host default |
| `SYNAPSAI_API_BASE` | Vector store uploads when `--host` omitted |

Hosts may be given as `upload.synapsai.cloud` or a full URL; the CLI normalizes to `https://…/v1`.

---

## `upload-model`

Upload a **file or directory** into an **existing** model artifact.

Flow (matches the upload API):

1. `POST /v1/model-artifacts/{id}/uploads`
2. `PUT /v1/model-artifacts/{id}/uploads/{upload_id}/files?path=…` per file
3. `POST /v1/model-artifacts/{id}/uploads/{upload_id}/complete`

```bash
synapsai upload-model ./my-model --artifact-id artifact-demo-abc12345

synapsai upload-model ./model.safetensors \
  --artifact-id artifact-demo-abc12345 \
  --host upload.synapsai.cloud \
  --timeout 7200
```

| Option | Required | Default |
| --- | --- | --- |
| `PATH` | yes | file or directory |
| `--artifact-id` | yes | — |
| `--host` | no | `SYNAPSAI_UPLOAD_BASE` or `https://upload.synapsai.cloud/v1` |
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
