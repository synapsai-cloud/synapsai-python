# SynapsAI Python SDK

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)

The official SynapsAI Cloud AI inference Python SDK. Seamlessly integrate multi-modal AI capabilities into your applications.

---

## Installation

Install the latest version of the SDK:

```bash
pip install --upgrade synapsai-python
```

### Requirements

* Python >= 3.8
* `httpx` >= 0.23.0, < 1.0.0
* `pydantic` >= 2.0, < 3
* `typing-extensions` >= 4.5, < 5
* `Pillow` >= 9.5.0, < 11
* `numpy` >= 1.21.0, < 2

---

## Quick Start

### 1. Setup Environment Variables

You can configure the SDK using environment variables instead of hardcoding your credentials:

* `SYNAPSAI_API_KEY` - Your SynapsAI API key
* `SYNAPSAI_API_BASE` - Custom base URL (Defaults to `https://api.synapsai.cloud/v1`)

### 2. Initialization & Example Usage

Here is how to initialize the client and fetch a list of all deployed models on your account:

```python
import os
from synapsai import SynapsAI

# Initialize the client (picks up env variables automatically if arguments are omitted)
client = SynapsAI(
    api_key=os.environ.get("SYNAPSAI_API_KEY"),
)

# Fetch and view deployed models
models_page = client.models.list()
for model in models_page.data:
    print(f"Model ID: {model.id} | Status: {model.status}")

```

---

## Documentation

For advanced configurations, deep-dive usage guides, and comprehensive code examples for every resource, visit our [Examples Documentation](https://docs.synapsai.cloud/examples/introduction).

---

## Available Resources

The SDK provides native support for the following SynapsAI Cloud capabilities:

| Resource | Description |
| --- | --- |
| **`models`** | Model management and discovery |
| **`chat.completions`** | Chat completions and real-time streaming |
| **`completions`** | Classic text completions |
| **`embeddings`** | Text embeddings generation |
| **`images`** | Image generation and manipulation |
| **`audio`** | Speech-to-text and text-to-speech processing |
| **`videos`** | Video generation and processing |
| **`classifications`** | Text classification tasks |
| **`question_answering`** | Targeted question answering |
| **`feature_extraction`** | Feature extraction pipelines |
| **`fill_mask`** | Masked Language Modeling (MLM) |
| **`rerank`** | Text reranking for search optimization |

---

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.