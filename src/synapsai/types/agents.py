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

"""AG-UI agent run type definitions."""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class AgentMessage(BaseModel):
    """Message accepted by POST /v1/agent/{id}/run."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    id: str
    role: str
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = Field(default=None, alias="toolCallId")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(default=None, alias="toolCalls")


class AgentTool(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    description: str
    parameters: Optional[Any] = None


class AgentContext(BaseModel):
    model_config = ConfigDict(extra="allow")

    description: str
    value: str


class AgentRunInput(BaseModel):
    """Payload for running a persisted agent (AG-UI RunAgentInput)."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    thread_id: str = Field(alias="threadId")
    run_id: str = Field(alias="runId")
    parent_run_id: Optional[str] = Field(default=None, alias="parentRunId")
    state: Optional[Any] = None
    messages: List[AgentMessage] = Field(default_factory=list)
    tools: List[AgentTool] = Field(default_factory=list)
    context: List[AgentContext] = Field(default_factory=list)
    forwarded_props: Optional[Any] = Field(default=None, alias="forwardedProps")
    resume: Optional[Any] = None


class AgentEvent(BaseModel):
    """Parsed AG-UI SSE event from an agent run stream."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    type: str
    thread_id: Optional[str] = Field(default=None, alias="threadId")
    run_id: Optional[str] = Field(default=None, alias="runId")
    message_id: Optional[str] = Field(default=None, alias="messageId")
    role: Optional[str] = None
    delta: Optional[str] = None
    tool_call_id: Optional[str] = Field(default=None, alias="toolCallId")
    tool_call_name: Optional[str] = Field(default=None, alias="toolCallName")
    content: Optional[Any] = None
    message: Optional[str] = None
