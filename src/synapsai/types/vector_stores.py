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

"""Vector store type definitions."""

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class VectorStoreFileCounts(BaseModel):
    model_config = ConfigDict(extra="allow")

    in_progress: int = 0
    completed: int = 0
    failed: int = 0
    cancelled: int = 0
    total: int = 0


class VectorStore(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    object: Literal["vector_store"] = "vector_store"
    created_at: int = 0
    name: Optional[str] = None
    description: Optional[str] = None
    file_counts: Optional[VectorStoreFileCounts] = None
    last_active_at: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    status: Optional[str] = None
    usage_bytes: int = 0
    expires_at: Optional[int] = None


class VectorStoreList(BaseModel):
    model_config = ConfigDict(extra="allow")

    object: Literal["list"] = "list"
    data: List[VectorStore] = Field(default_factory=list)
    first_id: Optional[str] = None
    last_id: Optional[str] = None
    has_more: bool = False


class VectorStoreDeleted(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    object: Literal["vector_store.deleted"] = "vector_store.deleted"
    deleted: bool = True


class VectorStoreSearchResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    file_id: Optional[str] = None
    filename: Optional[str] = None
    score: Optional[float] = None
    content: Optional[List[Dict[str, Any]]] = None
    attributes: Optional[Dict[str, Any]] = None


class VectorStoreSearchResults(BaseModel):
    model_config = ConfigDict(extra="allow")

    object: Literal["vector_store.search_results.page"] = "vector_store.search_results.page"
    search_query: Optional[Union[str, List[str]]] = None
    data: List[VectorStoreSearchResult] = Field(default_factory=list)
    has_more: bool = False
    next_page: Optional[str] = None


class VectorStoreFileError(BaseModel):
    model_config = ConfigDict(extra="allow")

    code: Optional[str] = None
    message: Optional[str] = None


class VectorStoreFile(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    object: Literal["vector_store.file"] = "vector_store.file"
    created_at: int = 0
    vector_store_id: str
    status: Optional[str] = None
    last_error: Optional[VectorStoreFileError] = None
    usage_bytes: int = 0
    attributes: Optional[Dict[str, Any]] = None
    chunking_strategy: Optional[Dict[str, Any]] = None


class VectorStoreFileList(BaseModel):
    model_config = ConfigDict(extra="allow")

    object: Literal["list"] = "list"
    data: List[VectorStoreFile] = Field(default_factory=list)
    first_id: Optional[str] = None
    last_id: Optional[str] = None
    has_more: bool = False


class VectorStoreFileDeleted(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    object: Literal["vector_store.file.deleted"] = "vector_store.file.deleted"
    deleted: bool = True


class VectorStoreFileContent(BaseModel):
    model_config = ConfigDict(extra="allow")

    object: Literal["vector_store.file_content.page"] = "vector_store.file_content.page"
    data: List[Dict[str, Any]] = Field(default_factory=list)
    has_more: bool = False
    next_page: Optional[str] = None
    file_id: Optional[str] = None
    filename: Optional[str] = None
    attributes: Optional[Dict[str, Any]] = None
