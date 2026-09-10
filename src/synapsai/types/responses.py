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

"""OpenAI-compatible Responses API type definitions."""

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class ResponseUsage(BaseModel):
    model_config = ConfigDict(extra="allow")

    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class Response(BaseModel):
    """Stored or completed Responses API object."""

    model_config = ConfigDict(extra="allow")

    id: Optional[str] = None
    object: Literal["response"] = "response"
    created_at: Optional[int] = None
    model: Optional[str] = None
    status: Optional[str] = None
    error: Optional[Any] = None
    incomplete_details: Optional[Any] = None
    instructions: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None
    output: Optional[List[Any]] = None
    output_text: Optional[str] = None
    parallel_tool_calls: Optional[bool] = None
    previous_response_id: Optional[str] = None
    reasoning: Optional[Any] = None
    store: Optional[bool] = None
    temperature: Optional[float] = None
    tool_choice: Optional[Any] = None
    tools: Optional[List[Any]] = None
    top_p: Optional[float] = None
    truncation: Optional[Any] = None
    usage: Optional[ResponseUsage] = None
    user: Optional[str] = None


class ResponseDeleted(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    object: Literal["response.deleted"] = "response.deleted"
    deleted: bool = True


class ResponseStreamEvent(BaseModel):
    """SSE event from a streamed Responses API run."""

    model_config = ConfigDict(extra="allow")

    type: str
    response: Optional[Response] = None
    item_id: Optional[str] = None
    output_index: Optional[int] = None
    content_index: Optional[int] = None
    delta: Optional[Any] = None
    text: Optional[str] = None
    item: Optional[Any] = None
    sequence_number: Optional[int] = None


ResponseInput = Union[str, List[Any], Dict[str, Any]]
