# Copyright 2026 SynapsAI Technologies Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Command-line interface for SynapsAI uploads."""

from __future__ import annotations

import json
import os
from typing import List, Optional

import click

from .client import DEFAULT_UPLOAD_BASE_URL, SynapsAI
from .exceptions import APIError


def _resolve_api_key(api_key: Optional[str]) -> str:
    value = api_key or os.environ.get("SYNAPSAI_API_KEY")
    if not value:
        raise click.ClickException(
            "API key required. Pass --api-key or set SYNAPSAI_API_KEY."
        )
    return value


def _normalize_base_url(host: str) -> str:
    value = host.strip().rstrip("/")
    if not value.startswith("http://") and not value.startswith("https://"):
        value = f"https://{value}"
    if not value.endswith("/v1"):
        value = f"{value}/v1"
    return value


@click.group()
@click.version_option(version="0.1.0", prog_name="synapsai")
def main() -> None:
    """SynapsAI Cloud CLI."""


@main.command("upload-model")
@click.argument("path", type=click.Path(exists=True, file_okay=True, dir_okay=True))
@click.option("--artifact-id", required=True, help="Existing model artifact id to upload into.")
@click.option(
    "--host",
    default=None,
    help=f"Upload host (default: {DEFAULT_UPLOAD_BASE_URL} or SYNAPSAI_UPLOAD_BASE).",
)
@click.option("--api-key", default=None, help="API key (default: SYNAPSAI_API_KEY).")
@click.option("--timeout", default=3600.0, show_default=True, type=float, help="Request timeout seconds.")
def upload_model(
    path: str,
    artifact_id: str,
    host: Optional[str],
    api_key: Optional[str],
    timeout: float,
) -> None:
    """Upload model files into an existing artifact on the upload API.

    PATH may be a single file or a directory of model weights/config files.
    Starts an upload session, streams files, then completes ingest.
    """
    base_url = _normalize_base_url(
        host
        or os.environ.get("SYNAPSAI_UPLOAD_BASE")
        or DEFAULT_UPLOAD_BASE_URL
    )
    client = SynapsAI(api_key=_resolve_api_key(api_key), base_url=base_url, timeout=timeout)

    def on_progress(relative: str, index: int, total: int) -> None:
        click.echo(f"[{index}/{total}] uploading {relative}")

    try:
        artifact = client.model_artifacts.upload(
            path,
            artifact_id=artifact_id,
            on_progress=on_progress,
        )
    except (APIError, OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(
        json.dumps(
            {
                "id": artifact.id,
                "display_name": artifact.display_name,
                "pipeline": artifact.pipeline,
                "status": artifact.status,
                "size_gb": artifact.size_gb,
            },
            indent=2,
        )
    )


@main.command("upload-vector-store-files")
@click.argument("vector_store_id")
@click.argument("paths", nargs=-1, required=True, type=click.Path(exists=True))
@click.option(
    "--host",
    default=None,
    help="API host (default: SYNAPSAI_API_BASE or https://api.synapsai.cloud/v1).",
)
@click.option("--api-key", default=None, help="API key (default: SYNAPSAI_API_KEY).")
@click.option(
    "--attributes",
    default=None,
    help="Optional JSON object applied as metadata attributes on each upload.",
)
@click.option("--timeout", default=600.0, show_default=True, type=float, help="Request timeout seconds.")
def upload_vector_store_files(
    vector_store_id: str,
    paths: List[str],
    host: Optional[str],
    api_key: Optional[str],
    attributes: Optional[str],
    timeout: float,
) -> None:
    """Upload files or directories into a vector store.

    Each PATH may be an individual file or a directory (recursively uploaded).
    """
    parsed_attributes = None
    if attributes:
        try:
            parsed_attributes = json.loads(attributes)
        except json.JSONDecodeError as exc:
            raise click.ClickException("--attributes must be valid JSON") from exc
        if not isinstance(parsed_attributes, dict):
            raise click.ClickException("--attributes must be a JSON object")

    base_url = None
    if host:
        base_url = _normalize_base_url(host)

    client = SynapsAI(
        api_key=_resolve_api_key(api_key),
        base_url=base_url,
        timeout=timeout,
    )

    try:
        results = client.vector_stores.files.upload_paths(
            vector_store_id,
            list(paths),
            attributes=parsed_attributes,
        )
    except (APIError, OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    for item in results:
        click.echo(f"{item.id}\t{item.status}\t{item.usage_bytes}")

    click.echo(f"Uploaded {len(results)} file(s) to vector store {vector_store_id}")


if __name__ == "__main__":
    main()
