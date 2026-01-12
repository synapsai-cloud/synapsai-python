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
Common type definitions used across SynapsAI API
"""

from typing import Optional, Dict, Any, Union, List
from pydantic import BaseModel, Field
from enum import Enum


class APIResponse(BaseModel):
    """Base response model for all API responses"""
    object: str
    created: Optional[int] = None
    id: Optional[str] = None


class Error(BaseModel):
    """Error object"""
    message: str
    type: Optional[str] = None
    param: Optional[str] = None
    code: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error response model"""
    error: Error


class FinishReason(str, Enum):
    """Reason why the generation finished"""
    STOP = "stop"
    LENGTH = "length"
    FUNCTION_CALL = "function_call"
    TOOL_CALLS = "tool_calls"
    CONTENT_FILTER = "content_filter" 

class Usage(BaseModel):
    """Token usage information"""
    prompt_tokens: int
    completion_tokens: Optional[int] = None
    total_tokens: int