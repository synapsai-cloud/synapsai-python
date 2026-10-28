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

import os
from pathlib import Path
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Union

import httpx

from ..exceptions import APIError
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


class _FileSlice:
    """A read/seek window over an open file. httpx uses it to set Content-Length."""

    def __init__(self, handle, length: int):
        self._handle = handle
        self._start = handle.tell()
        self._length = int(length)
        self._remaining = self._length

    def read(self, n: int = -1) -> bytes:
        if self._remaining <= 0:
            return b""
        if n is None or n < 0 or n > self._remaining:
            n = self._remaining
        data = self._handle.read(n)
        if not data:
            return b""
        self._remaining -= len(data)
        return data

    def tell(self) -> int:
        return self._handle.tell() - self._start

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_END:
            target = self._length + offset
        elif whence == os.SEEK_CUR:
            target = self.tell() + offset
        elif whence == os.SEEK_SET:
            target = offset
        else:
            raise OSError("unsupported seek")
        if target < 0 or target > self._length:
            raise OSError("seek out of range")
        self._handle.seek(self._start + target)
        self._remaining = self._length - target
        return target


def _put_presigned(url: str, file_path: Path, offset: int, length: int, timeout) -> str:
    """PUT one part straight to object storage. Only this part is read from disk."""
    last_error: Optional[Exception] = None
    for _attempt in range(3):
        try:
            with file_path.open("rb") as handle:
                handle.seek(offset)
                body = _FileSlice(handle, length)
                with httpx.Client(timeout=timeout) as client:
                    response = client.put(url, content=body)
            if response.status_code >= 400:
                last_error = APIError(
                    f"HTTP {response.status_code} uploading to storage",
                    status_code=response.status_code,
                )
                continue
            etag = response.headers.get("etag") or response.headers.get("ETag")
            if not etag:
                last_error = APIError("Storage upload did not return an ETag")
                continue
            return etag
        except httpx.HTTPError as exc:
            last_error = APIError(str(exc))
    if last_error is None:
        last_error = APIError("Storage upload failed")
    raise last_error


class ModelArtifactsResource:
    """Upload files into an existing model artifact on the client API host."""

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

    def upload_file_direct(
        self,
        artifact_id: str,
        upload_id: str,
        path: str,
        file: PathLike,
    ) -> ModelArtifactFileUpload:
        """Upload one file in presigned parts directly to object storage."""
        file_path = Path(file)
        size = file_path.stat().st_size
        response = self._client._post(
            f"model-artifacts/{artifact_id}/uploads/{upload_id}/direct-uploads",
            json_data={"path": path, "size_bytes": size},
        )
        plan = response.json()
        parts_out: List[dict] = []
        try:
            for part in plan.get("parts") or []:
                etag = _put_presigned(
                    part["url"],
                    file_path,
                    int(part.get("offset") or 0),
                    int(part.get("size_bytes") or 0),
                    self._client.timeout,
                )
                if plan.get("mode") == "multipart":
                    parts_out.append({"part_number": part["part_number"], "etag": etag})
            finished = self._client._post(
                f"model-artifacts/{artifact_id}/uploads/{upload_id}/direct-uploads/complete",
                json_data={"path": plan.get("path") or path, "parts": parts_out},
            )
        except Exception:
            try:
                self._client._post(
                    f"model-artifacts/{artifact_id}/uploads/{upload_id}/direct-uploads/abort",
                    json_data={"path": plan.get("path") or path},
                )
            except Exception:
                pass
            raise
        return ModelArtifactFileUpload.model_validate(finished.json())

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

        Starts an upload session, sends every file under ``path``
        (file or directory), then completes the session.

        The session response selects the route. ``transfer=proxy`` is local
        disk on an on-prem deployment and uses ``PUT .../files``. Any other
        value, including ``direct``, writes presigned parts straight to the
        bucket. One part is read from disk at a time.
        """
        root = Path(path).expanduser().resolve()
        files = _iter_upload_files(root)
        if not files:
            raise FileNotFoundError(f"No files found to upload under {root}")

        session = self.start_upload(artifact_id)
        local_storage = session.transfer == "proxy"
        total = len(files)
        for index, (relative, file_path) in enumerate(files, start=1):
            if on_progress is not None:
                on_progress(relative, index, total)
            if local_storage:
                self.upload_file(artifact_id, session.upload_id, relative, file_path)
            else:
                self.upload_file_direct(artifact_id, session.upload_id, relative, file_path)

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

    async def upload_file_direct(
        self,
        artifact_id: str,
        upload_id: str,
        path: str,
        file: PathLike,
    ) -> ModelArtifactFileUpload:
        """Upload one file in presigned parts directly to object storage."""
        import asyncio

        file_path = Path(file)
        size = file_path.stat().st_size
        response = await self._client._post(
            f"model-artifacts/{artifact_id}/uploads/{upload_id}/direct-uploads",
            json_data={"path": path, "size_bytes": size},
        )
        plan = response.json()
        parts_out: List[dict] = []
        try:
            for part in plan.get("parts") or []:
                etag = await asyncio.to_thread(
                    _put_presigned,
                    part["url"],
                    file_path,
                    int(part.get("offset") or 0),
                    int(part.get("size_bytes") or 0),
                    self._client.timeout,
                )
                if plan.get("mode") == "multipart":
                    parts_out.append({"part_number": part["part_number"], "etag": etag})
            finished = await self._client._post(
                f"model-artifacts/{artifact_id}/uploads/{upload_id}/direct-uploads/complete",
                json_data={"path": plan.get("path") or path, "parts": parts_out},
            )
        except Exception:
            try:
                await self._client._post(
                    f"model-artifacts/{artifact_id}/uploads/{upload_id}/direct-uploads/abort",
                    json_data={"path": plan.get("path") or path},
                )
            except Exception:
                pass
            raise
        return ModelArtifactFileUpload.model_validate(finished.json())

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

        Starts an upload session, sends every file under ``path``
        (file or directory), then completes the session.

        The session response selects the route. ``transfer=proxy`` is local
        disk on an on-prem deployment and uses ``PUT .../files``. Any other
        value, including ``direct``, writes presigned parts straight to the
        bucket. One part is read from disk at a time.
        """
        root = Path(path).expanduser().resolve()
        files = _iter_upload_files(root)
        if not files:
            raise FileNotFoundError(f"No files found to upload under {root}")

        session = await self.start_upload(artifact_id)
        local_storage = session.transfer == "proxy"
        total = len(files)
        for index, (relative, file_path) in enumerate(files, start=1):
            if on_progress is not None:
                on_progress(relative, index, total)
            if local_storage:
                await self.upload_file(artifact_id, session.upload_id, relative, file_path)
            else:
                await self.upload_file_direct(artifact_id, session.upload_id, relative, file_path)

        finished = await self.complete_upload(artifact_id, session.upload_id)
        return finished.artifact
