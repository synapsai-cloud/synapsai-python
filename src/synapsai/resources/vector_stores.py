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

"""Vector store resource handlers."""

from __future__ import annotations

import json
import mimetypes
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, Dict, List, Literal, Optional, Union

from ..types.vector_stores import (
    VectorStore,
    VectorStoreDeleted,
    VectorStoreFile,
    VectorStoreFileContent,
    VectorStoreFileDeleted,
    VectorStoreFileList,
    VectorStoreList,
    VectorStoreSearchResults,
)

if TYPE_CHECKING:
    from ..client import AsyncSynapsAI, SynapsAI

PathLike = Union[str, Path]
FileInput = Union[PathLike, BinaryIO, tuple]


def _pagination_params(
    *,
    after: Optional[str] = None,
    before: Optional[str] = None,
    limit: Optional[int] = None,
    order: Optional[str] = None,
    status_filter: Optional[str] = None,
) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    if after is not None:
        params["after"] = after
    if before is not None:
        params["before"] = before
    if limit is not None:
        params["limit"] = limit
    if order is not None:
        params["order"] = order
    if status_filter is not None:
        params["filter"] = status_filter
    return params


def _open_upload_file(file: FileInput) -> tuple[str, Any, Optional[str], Optional[Any]]:
    """
    Normalize file input to (filename, fileobj_or_bytes, content_type, closer).

    closer is a file handle that must be closed by the caller when not None.
    """
    if isinstance(file, tuple):
        if len(file) == 2:
            filename, content = file
            content_type = mimetypes.guess_type(str(filename))[0] or "application/octet-stream"
            return str(filename), content, content_type, None
        if len(file) >= 3:
            filename, content, content_type = file[0], file[1], file[2]
            return str(filename), content, content_type, None
        raise ValueError("file tuple must be (filename, content[, content_type])")

    if hasattr(file, "read") and not isinstance(file, (str, Path)):
        filename = getattr(file, "name", "upload.bin")
        filename = Path(str(filename)).name or "upload.bin"
        content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        return filename, file, content_type, None

    path = Path(file).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    handle = path.open("rb")
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return path.name, handle, content_type, handle


def collect_vector_store_paths(paths: List[PathLike]) -> List[Path]:
    """Expand files and directories into a flat list of files."""
    files: List[Path] = []
    for raw in paths:
        path = Path(raw).expanduser().resolve()
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file():
                    files.append(child)
        else:
            raise FileNotFoundError(f"Path not found: {path}")
    return files


class VectorStoreFilesResource:
    """Files attached to a vector store."""

    def __init__(self, client: "SynapsAI"):
        self._client = client

    def create(
        self,
        vector_store_id: str,
        *,
        file: Optional[FileInput] = None,
        file_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> VectorStoreFile:
        endpoint = f"vector_stores/{vector_store_id}/files"
        if file is not None and file_id is not None:
            raise ValueError("Provide either file or file_id, not both")
        if file is None and file_id is None:
            raise ValueError("Either file or file_id is required")

        if file_id is not None:
            payload = self._client._build_request(file_id=file_id, attributes=attributes, **kwargs)
            response = self._client._post(endpoint, json_data=payload)
            return VectorStoreFile.model_validate(response.json())

        filename, content, content_type, closer = _open_upload_file(file)
        try:
            form_data = None
            if attributes is not None:
                form_data = {"attributes": json.dumps(attributes)}
            response = self._client._post(
                endpoint,
                data=form_data,
                files={"file": (filename, content, content_type)},
            )
            return VectorStoreFile.model_validate(response.json())
        finally:
            if closer is not None:
                closer.close()

    def list(
        self,
        vector_store_id: str,
        *,
        after: Optional[str] = None,
        before: Optional[str] = None,
        limit: int = 20,
        order: Literal["asc", "desc"] = "desc",
        filter: Optional[Literal["in_progress", "completed", "failed", "cancelled"]] = None,
    ) -> VectorStoreFileList:
        params = _pagination_params(
            after=after,
            before=before,
            limit=limit,
            order=order,
            status_filter=filter,
        )
        response = self._client._get(f"vector_stores/{vector_store_id}/files", params=params)
        return VectorStoreFileList.model_validate(response.json())

    def retrieve(self, vector_store_id: str, file_id: str) -> VectorStoreFile:
        response = self._client._get(f"vector_stores/{vector_store_id}/files/{file_id}")
        return VectorStoreFile.model_validate(response.json())

    def update(
        self,
        vector_store_id: str,
        file_id: str,
        *,
        attributes: Dict[str, Any],
        **kwargs,
    ) -> VectorStoreFile:
        payload = self._client._build_request(attributes=attributes, **kwargs)
        response = self._client._post(
            f"vector_stores/{vector_store_id}/files/{file_id}",
            json_data=payload,
        )
        return VectorStoreFile.model_validate(response.json())

    def delete(self, vector_store_id: str, file_id: str) -> VectorStoreFileDeleted:
        response = self._client._delete(f"vector_stores/{vector_store_id}/files/{file_id}")
        return VectorStoreFileDeleted.model_validate(response.json())

    def content(self, vector_store_id: str, file_id: str) -> VectorStoreFileContent:
        response = self._client._get(f"vector_stores/{vector_store_id}/files/{file_id}/content")
        return VectorStoreFileContent.model_validate(response.json())

    def upload_paths(
        self,
        vector_store_id: str,
        paths: List[PathLike],
        *,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> List[VectorStoreFile]:
        """Upload each file under the given paths (files or directories)."""
        results: List[VectorStoreFile] = []
        for file_path in collect_vector_store_paths(paths):
            results.append(
                self.create(
                    vector_store_id,
                    file=file_path,
                    attributes=attributes,
                )
            )
        return results


class VectorStoresResource:
    """OpenAI-compatible vector store resource."""

    def __init__(self, client: "SynapsAI"):
        self._client = client
        self.files = VectorStoreFilesResource(client)

    def create(
        self,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        **kwargs,
    ) -> VectorStore:
        payload = self._client._build_request(name=name, description=description, **kwargs)
        response = self._client._post("vector_stores", json_data=payload)
        return VectorStore.model_validate(response.json())

    def list(
        self,
        *,
        after: Optional[str] = None,
        before: Optional[str] = None,
        limit: int = 20,
        order: Literal["asc", "desc"] = "desc",
    ) -> VectorStoreList:
        params = _pagination_params(after=after, before=before, limit=limit, order=order)
        response = self._client._get("vector_stores", params=params)
        return VectorStoreList.model_validate(response.json())

    def retrieve(self, vector_store_id: str) -> VectorStore:
        response = self._client._get(f"vector_stores/{vector_store_id}")
        return VectorStore.model_validate(response.json())

    def update(
        self,
        vector_store_id: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        **kwargs,
    ) -> VectorStore:
        payload = self._client._build_request(name=name, description=description, **kwargs)
        response = self._client._post(f"vector_stores/{vector_store_id}", json_data=payload)
        return VectorStore.model_validate(response.json())

    def delete(self, vector_store_id: str) -> VectorStoreDeleted:
        response = self._client._delete(f"vector_stores/{vector_store_id}")
        return VectorStoreDeleted.model_validate(response.json())

    def search(
        self,
        vector_store_id: str,
        query: Union[str, List[str]],
        *,
        filters: Optional[Dict[str, Any]] = None,
        max_num_results: Optional[int] = None,
        ranking_options: Optional[Dict[str, Any]] = None,
        rewrite_query: Optional[bool] = None,
        **kwargs,
    ) -> VectorStoreSearchResults:
        payload = self._client._build_request(
            query=query,
            filters=filters,
            max_num_results=max_num_results,
            ranking_options=ranking_options,
            rewrite_query=rewrite_query,
            **kwargs,
        )
        response = self._client._post(f"vector_stores/{vector_store_id}/search", json_data=payload)
        return VectorStoreSearchResults.model_validate(response.json())


class AsyncVectorStoreFilesResource:
    """Async files attached to a vector store."""

    def __init__(self, client: "AsyncSynapsAI"):
        self._client = client

    async def create(
        self,
        vector_store_id: str,
        *,
        file: Optional[FileInput] = None,
        file_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> VectorStoreFile:
        endpoint = f"vector_stores/{vector_store_id}/files"
        if file is not None and file_id is not None:
            raise ValueError("Provide either file or file_id, not both")
        if file is None and file_id is None:
            raise ValueError("Either file or file_id is required")

        if file_id is not None:
            payload = self._client._build_request(file_id=file_id, attributes=attributes, **kwargs)
            response = await self._client._post(endpoint, json_data=payload)
            return VectorStoreFile.model_validate(response.json())

        filename, content, content_type, closer = _open_upload_file(file)
        try:
            form_data = None
            if attributes is not None:
                form_data = {"attributes": json.dumps(attributes)}
            response = await self._client._post(
                endpoint,
                data=form_data,
                files={"file": (filename, content, content_type)},
            )
            return VectorStoreFile.model_validate(response.json())
        finally:
            if closer is not None:
                closer.close()

    async def list(
        self,
        vector_store_id: str,
        *,
        after: Optional[str] = None,
        before: Optional[str] = None,
        limit: int = 20,
        order: Literal["asc", "desc"] = "desc",
        filter: Optional[Literal["in_progress", "completed", "failed", "cancelled"]] = None,
    ) -> VectorStoreFileList:
        params = _pagination_params(
            after=after,
            before=before,
            limit=limit,
            order=order,
            status_filter=filter,
        )
        response = await self._client._get(f"vector_stores/{vector_store_id}/files", params=params)
        return VectorStoreFileList.model_validate(response.json())

    async def retrieve(self, vector_store_id: str, file_id: str) -> VectorStoreFile:
        response = await self._client._get(f"vector_stores/{vector_store_id}/files/{file_id}")
        return VectorStoreFile.model_validate(response.json())

    async def update(
        self,
        vector_store_id: str,
        file_id: str,
        *,
        attributes: Dict[str, Any],
        **kwargs,
    ) -> VectorStoreFile:
        payload = self._client._build_request(attributes=attributes, **kwargs)
        response = await self._client._post(
            f"vector_stores/{vector_store_id}/files/{file_id}",
            json_data=payload,
        )
        return VectorStoreFile.model_validate(response.json())

    async def delete(self, vector_store_id: str, file_id: str) -> VectorStoreFileDeleted:
        response = await self._client._delete(f"vector_stores/{vector_store_id}/files/{file_id}")
        return VectorStoreFileDeleted.model_validate(response.json())

    async def content(self, vector_store_id: str, file_id: str) -> VectorStoreFileContent:
        response = await self._client._get(
            f"vector_stores/{vector_store_id}/files/{file_id}/content"
        )
        return VectorStoreFileContent.model_validate(response.json())

    async def upload_paths(
        self,
        vector_store_id: str,
        paths: List[PathLike],
        *,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> List[VectorStoreFile]:
        results: List[VectorStoreFile] = []
        for file_path in collect_vector_store_paths(paths):
            results.append(
                await self.create(
                    vector_store_id,
                    file=file_path,
                    attributes=attributes,
                )
            )
        return results


class AsyncVectorStoresResource:
    """Async OpenAI-compatible vector store resource."""

    def __init__(self, client: "AsyncSynapsAI"):
        self._client = client
        self.files = AsyncVectorStoreFilesResource(client)

    async def create(
        self,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        **kwargs,
    ) -> VectorStore:
        payload = self._client._build_request(name=name, description=description, **kwargs)
        response = await self._client._post("vector_stores", json_data=payload)
        return VectorStore.model_validate(response.json())

    async def list(
        self,
        *,
        after: Optional[str] = None,
        before: Optional[str] = None,
        limit: int = 20,
        order: Literal["asc", "desc"] = "desc",
    ) -> VectorStoreList:
        params = _pagination_params(after=after, before=before, limit=limit, order=order)
        response = await self._client._get("vector_stores", params=params)
        return VectorStoreList.model_validate(response.json())

    async def retrieve(self, vector_store_id: str) -> VectorStore:
        response = await self._client._get(f"vector_stores/{vector_store_id}")
        return VectorStore.model_validate(response.json())

    async def update(
        self,
        vector_store_id: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        **kwargs,
    ) -> VectorStore:
        payload = self._client._build_request(name=name, description=description, **kwargs)
        response = await self._client._post(f"vector_stores/{vector_store_id}", json_data=payload)
        return VectorStore.model_validate(response.json())

    async def delete(self, vector_store_id: str) -> VectorStoreDeleted:
        response = await self._client._delete(f"vector_stores/{vector_store_id}")
        return VectorStoreDeleted.model_validate(response.json())

    async def search(
        self,
        vector_store_id: str,
        query: Union[str, List[str]],
        *,
        filters: Optional[Dict[str, Any]] = None,
        max_num_results: Optional[int] = None,
        ranking_options: Optional[Dict[str, Any]] = None,
        rewrite_query: Optional[bool] = None,
        **kwargs,
    ) -> VectorStoreSearchResults:
        payload = self._client._build_request(
            query=query,
            filters=filters,
            max_num_results=max_num_results,
            ranking_options=ranking_options,
            rewrite_query=rewrite_query,
            **kwargs,
        )
        response = await self._client._post(
            f"vector_stores/{vector_store_id}/search",
            json_data=payload,
        )
        return VectorStoreSearchResults.model_validate(response.json())
