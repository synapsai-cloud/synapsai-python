# SynapsAI Python SDK Documentation

Official docs for [`synapsai-python`](https://github.com/synapsai-cloud/synapsai-python) — the SynapsAI Cloud Python client.

## Contents

| Guide | Description |
| --- | --- |
| [Getting started](getting-started.md) | Install, configure, and make your first request |
| [Errors](errors.md) | Exception types and retry behavior |
| [Async client](async.md) | `AsyncSynapsAI` usage |
| [CLI](cli.md) | `synapsai` upload commands |

### Inference & media

| Guide | Client path |
| --- | --- |
| [Chat](chat.md) | `client.chat.completions` |
| [Responses](responses.md) | `client.responses` |
| [Completions](completions.md) | `client.completions` |
| [Embeddings](embeddings.md) | `client.embeddings` |
| [Rerank](rerank.md) | `client.rerank` |
| [Images](images.md) | `client.images` |
| [Audio](audio.md) | `client.audio` |
| [Videos](videos.md) | `client.videos` |
| [Models](models.md) | `client.models` |

### Specialized pipelines

| Guide | Client path |
| --- | --- |
| [Classifications](classifications.md) | `client.classifications` |
| [Question answering](question-answering.md) | `client.question_answering` |
| [Feature extraction](feature-extraction.md) | `client.feature_extraction` |
| [Fill mask](fill-mask.md) | `client.fill_mask` |

### Storage, upload & agents

| Guide | Client path |
| --- | --- |
| [Vector stores](vector-stores.md) | `client.vector_stores` |
| [Model artifacts](model-artifacts.md) | `client.model_artifacts` |
| [Agents](agents.md) | `client.agents` |

## Quick links

- Default API base: `https://api.synapsai.cloud/v1`
- Default upload base: `https://upload.synapsai.cloud/v1`
- Package: `pip install synapsai-python`
