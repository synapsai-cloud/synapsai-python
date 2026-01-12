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
Exception classes for SynapsAI client library
"""

from typing import Optional, Dict, Any


class SynapsAIError(Exception):
    """Base exception class for SynapsAI errors"""
    pass


class APIError(SynapsAIError):
    """Exception raised for API errors"""
    
    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_data: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.response_data = response_data


class AuthenticationError(APIError):
    """Exception raised for authentication errors"""
    pass


class RateLimitError(APIError):
    """Exception raised when rate limit is exceeded"""
    pass


class ValidationError(SynapsAIError):
    """Exception raised for validation errors"""
    pass


class TimeoutError(SynapsAIError):
    """Exception raised for timeout errors"""
    pass


class ConnectionError(SynapsAIError):
    """Exception raised for connection errors"""
    pass 