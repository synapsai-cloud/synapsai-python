"""
SynapsAI - OpenAI Compatible API Client Library

A scalable and maintainable Python client library for interacting with SynapsAI services,
compatible with OpenAI API patterns.
"""

from .client import SynapsAI, AsyncSynapsAI, DEFAULT_UPLOAD_BASE_URL
from .types import *
from .resources import *

__version__ = "0.1.1"
__all__ = [
    "SynapsAI",
    "AsyncSynapsAI",
    "DEFAULT_UPLOAD_BASE_URL",
]
