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

"""Model artifact upload resource handlers."""

from pathlib import Path
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Union

from ..types.model_artifacts import (
    ModelArtifact,
    ModelArtifactFileUpload,
    ModelArtifactResponse,
    ModelArtifactUploadSession,
)

if TYPE_CHECKING:
    from ..client import AsyncSynapsAI, SynapsAI

PathLike = Union[str, Path]
ProgressCallback = Callable[[str, int, int], None]


def _iter_upload_files(root: Path) -> List[Tuple[str, Path]]:
    """Return (relative posix path, absolute path) pairs under root."""
    if root.is_file():
        return [(root.name, root)]
    if not root.is_dir():
        raise FileNotFoundError(f"Path not found: {root}")

    items: List[Tuple[str, Path]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        items.append((relative, path))
    return items


class ModelArtifactsResource:
    """Upload files into an existing model artifact on the infra upload API."""

    def __init__(self, client: "SynapsAI"):
        self._client = client

    def create(self, display_name: str, pipeline: str, **kwargs) -> ModelArtifactResponse:
        """POST /v1/model-artifacts — create an empty artifact record."""
        payload = self._client._build_request(
            display_name=display_name,
            pipeline=pipeline,
            **kwargs,
        )
        response = self._client._post("model-artifacts", json_data=payload)
        return ModelArtifactResponse.model_validate(response.json())

    def retrieve(self, artifact_id: str) -> ModelArtifact:
        """GET /v1/model-artifacts/{artifact_id}."""
        response = self._client._get(f"model-artifacts/{artifact_id}")
        return ModelArtifact.model_validate(response.json())

    def start_upload(self, artifact_id: str) -> ModelArtifactUploadSession:
        """POST /v1/model-artifacts/{artifact_id}/uploads."""
        response = self._client._post(f"model-artifacts/{artifact_id}/uploads")
        return ModelArtifactUploadSession.model_validate(response.json())

    def upload_file(
        self,
        artifact_id: str,
        upload_id: str,
        path: str,
        file: PathLike,
    ) -> ModelArtifactFileUpload:
        """PUT /v1/model-artifacts/{artifact_id}/uploads/{upload_id}/files?path=..."""
        file_path = Path(file)
        with file_path.open("rb") as handle:
            response = self._client._put(
                f"model-artifacts/{artifact_id}/uploads/{upload_id}/files",
                content=handle,
                params={"path": path},
                headers={"Content-Type": "application/octet-stream"},
            )
        return ModelArtifactFileUpload.model_validate(response.json())

    def complete_upload(self, artifact_id: str, upload_id: str) -> ModelArtifactResponse:
        """POST /v1/model-artifacts/{artifact_id}/uploads/{upload_id}/complete."""
        response = self._client._post(
            f"model-artifacts/{artifact_id}/uploads/{upload_id}/complete"
        )
        return ModelArtifactResponse.model_validate(response.json())

    def upload(
        self,
        path: PathLike,
        *,
        artifact_id: str,
        on_progress: Optional[ProgressCallback] = None,
    ) -> ModelArtifact:
        """
        Upload files into an existing artifact.

        Starts an upload session, streams every file under ``path``
        (file or directory), then completes the session — matching
        ``/uploads`` → ``/files`` → ``/complete`` on the upload API.
        """
        root = Path(path).expanduser().resolve()
        files = _iter_upload_files(root)
        if not files:
            raise FileNotFoundError(f"No files found to upload under {root}")

        session = self.start_upload(artifact_id)
        total = len(files)
        for index, (relative, file_path) in enumerate(files, start=1):
            if on_progress is not None:
                on_progress(relative, index, total)
            self.upload_file(artifact_id, session.upload_id, relative, file_path)

        finished = self.complete_upload(artifact_id, session.upload_id)
        return finished.artifact


class AsyncModelArtifactsResource:
    """Async upload of files into an existing model artifact."""

    def __init__(self, client: "AsyncSynapsAI"):
        self._client = client

    async def create(self, display_name: str, pipeline: str, **kwargs) -> ModelArtifactResponse:
        """POST /v1/model-artifacts — create an empty artifact record."""
        payload = self._client._build_request(
            display_name=display_name,
            pipeline=pipeline,
            **kwargs,
        )
        response = await self._client._post("model-artifacts", json_data=payload)
        return ModelArtifactResponse.model_validate(response.json())

    async def retrieve(self, artifact_id: str) -> ModelArtifact:
        """GET /v1/model-artifacts/{artifact_id}."""
        response = await self._client._get(f"model-artifacts/{artifact_id}")
        return ModelArtifact.model_validate(response.json())

    async def start_upload(self, artifact_id: str) -> ModelArtifactUploadSession:
        """POST /v1/model-artifacts/{artifact_id}/uploads."""
        response = await self._client._post(f"model-artifacts/{artifact_id}/uploads")
        return ModelArtifactUploadSession.model_validate(response.json())

    async def upload_file(
        self,
        artifact_id: str,
        upload_id: str,
        path: str,
        file: PathLike,
    ) -> ModelArtifactFileUpload:
        """PUT /v1/model-artifacts/{artifact_id}/uploads/{upload_id}/files?path=..."""
        file_path = Path(file)
        with file_path.open("rb") as handle:
            response = await self._client._put(
                f"model-artifacts/{artifact_id}/uploads/{upload_id}/files",
                content=handle,
                params={"path": path},
                headers={"Content-Type": "application/octet-stream"},
            )
        return ModelArtifactFileUpload.model_validate(response.json())

    async def complete_upload(self, artifact_id: str, upload_id: str) -> ModelArtifactResponse:
        """POST /v1/model-artifacts/{artifact_id}/uploads/{upload_id}/complete."""
        response = await self._client._post(
            f"model-artifacts/{artifact_id}/uploads/{upload_id}/complete"
        )
        return ModelArtifactResponse.model_validate(response.json())

    async def upload(
        self,
        path: PathLike,
        *,
        artifact_id: str,
        on_progress: Optional[ProgressCallback] = None,
    ) -> ModelArtifact:
        """
        Upload files into an existing artifact.

        Starts an upload session, streams every file under ``path``
        (file or directory), then completes the session — matching
        ``/uploads`` → ``/files`` → ``/complete`` on the upload API.
        """
        root = Path(path).expanduser().resolve()
        files = _iter_upload_files(root)
        if not files:
            raise FileNotFoundError(f"No files found to upload under {root}")

        session = await self.start_upload(artifact_id)
        total = len(files)
        for index, (relative, file_path) in enumerate(files, start=1):
            if on_progress is not None:
                on_progress(relative, index, total)
            await self.upload_file(artifact_id, session.upload_id, relative, file_path)

        finished = await self.complete_upload(artifact_id, session.upload_id)
        return finished.artifact
