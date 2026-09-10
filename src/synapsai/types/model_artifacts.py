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

"""Model artifact upload type definitions."""

from typing import Optional

from pydantic import BaseModel, ConfigDict


class ModelArtifact(BaseModel):
    """Model artifact metadata returned by the upload API."""

    model_config = ConfigDict(extra="allow")

    id: str
    display_name: Optional[str] = None
    pipeline: Optional[str] = None
    status: Optional[str] = None
    size_gb: Optional[float] = None
    storage_uri: Optional[str] = None
    number_of_parameters: Optional[float] = None
    default_precision: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    error_message: Optional[str] = None


class ModelArtifactResponse(BaseModel):
    """Wrapped artifact create/complete response."""

    model_config = ConfigDict(extra="allow")

    artifact: ModelArtifact
    message: Optional[str] = None


class ModelArtifactUploadSession(BaseModel):
    """Upload session started for an artifact."""

    model_config = ConfigDict(extra="allow")

    upload_id: str
    message: Optional[str] = None
    artifact: Optional[ModelArtifact] = None


class ModelArtifactFileUpload(BaseModel):
    """Result of uploading one artifact file."""

    model_config = ConfigDict(extra="allow")

    message: Optional[str] = None
    path: str
    size_bytes: int
    bytes_copied: Optional[int] = None
    files_copied: Optional[int] = None
