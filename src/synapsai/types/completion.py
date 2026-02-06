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

"""
Chat completion type definitions
"""

from typing import Optional, Dict, Any, Union, List, Literal
from pydantic import BaseModel, Field
from enum import Enum

from .common import APIResponse, FinishReason, Usage


class ChatRole(str, Enum):
    """Chat message roles"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"


class FunctionCall(BaseModel):
    """Function call object"""
    name: str
    arguments: str


class ToolCall(BaseModel):
    """Tool call object"""
    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class ChatMessage(BaseModel):
    """Chat message object"""
    role: ChatRole
    content: Optional[str] = None
    name: Optional[str] = None
    function_call: Optional[FunctionCall] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None


class Function(BaseModel):
    """Function definition"""
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class Tool(BaseModel):
    """Tool definition"""
    type: Literal["function"] = "function"
    function: Function


class ResponseFormat(BaseModel):
    """Response format specification"""
    type: Literal["text", "json_object"] = "text"


class ChatCompletionRequest(BaseModel):
    """Chat completion request"""
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = Field(default=1.0, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    n: Optional[int] = Field(default=1, ge=1)
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = Field(default=None, gt=0)
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    logit_bias: Optional[Dict[str, float]] = None
    functions: Optional[List[Function]] = None
    function_call: Optional[Union[str, Dict[str, str]]] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    response_format: Optional[ResponseFormat] = None
    seed: Optional[int] = None

class CompletionRequest(BaseModel):
    """Completion request"""
    model: str
    prompt: str
    temperature: Optional[float] = Field(default=1.0, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    n: Optional[int] = Field(default=1, ge=1)
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = Field(default=None, gt=0)
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    logit_bias: Optional[Dict[str, float]] = None
    response_format: Optional[ResponseFormat] = None
    seed: Optional[int] = None

class Delta(BaseModel):
    """Delta object for streaming responses"""
    role: Optional[ChatRole] = None
    content: Optional[str] = None
    function_call: Optional[FunctionCall] = None
    tool_calls: Optional[List[ToolCall]] = None


class ChatCompletionChoice(BaseModel):
    """Choice object"""
    index: int
    message: Optional[ChatMessage] = None
    delta: Optional[Delta] = None
    logprobs: Optional[dict | list[dict]] = None
    finish_reason: Optional[FinishReason] = None

class CompletionChoice(BaseModel):
    """Choice object"""
    index: int
    text: str
    logprobs: Optional[dict | list[dict]] = None
    finish_reason: Optional[FinishReason] = None

class ChatCompletionResponse(APIResponse):
    """Chat completion response"""
    model: str
    object: Literal["chat.completion"] = "chat.completion"
    choices: List[ChatCompletionChoice]
    system_fingerprint: Optional[str] = None
    usage: Optional[Usage] = None

class CompletionResponse(APIResponse):
    """Completion response"""
    model: str
    object: Literal["completion"] = "completion"
    choices: List[CompletionChoice]
    system_fingerprint: Optional[str] = None
    usage: Optional[Usage] = None

class ChatCompletionChunk(APIResponse):
    """Chat completion chunk for streaming"""
    model: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    choices: List[ChatCompletionChoice]
    system_fingerprint: Optional[str] = None 
    usage: Optional[Usage] = None
    
class CompletionChunk(APIResponse):
    """Completion chunk for streaming"""
    model: str
    object: Literal["completion.chunk"] = "completion.chunk"
    choices: List[CompletionChoice]
    system_fingerprint: Optional[str] = None 
    usage: Optional[Usage] = None

