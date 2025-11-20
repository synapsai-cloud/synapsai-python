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