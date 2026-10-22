"""
SynapsAI - OpenAI Compatible API Client Library

A scalable and maintainable Python client library for interacting with SynapsAI services,
compatible with OpenAI API patterns.
"""

from .client import SynapsAI, AsyncSynapsAI
from .types import *
from .resources import *

__version__ = "0.1.5"
__all__ = [
    "SynapsAI",
    "AsyncSynapsAI",
]
