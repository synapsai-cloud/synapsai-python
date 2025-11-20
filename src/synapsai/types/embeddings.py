"""
Embeddings type definitions
"""

from typing import Optional, List, Literal, Union
from pydantic import BaseModel, Field

from .common import APIResponse, Usage


class Embedding(BaseModel):
    """Embedding object"""
    object: Literal["embedding"] = "embedding"
    embedding: Union[List[float], str]
    index: int = Field(default=0)


class EmbeddingRequest(BaseModel):
    """Embedding request"""
    model: str
    input: Union[str, List[str], List[int], List[List[int]]]
    encoding_format: Optional[Literal["float", "base64"]] = "float"
    dimensions: Optional[int] = Field(default=None, gt=0)


class EmbeddingResponse(APIResponse):
    """Embedding response"""
    object: Literal["list"] = "list"
    data: List[Embedding]
    model: str
    usage: Optional[Usage] = None


class SimilarityRequest(BaseModel):
    """Similarity request for sentence/embedding similarity"""
    model: str
    source_sentence: str
    sentences: List[str]
    return_embeddings: Optional[bool] = False
    encoding_format: Optional[Literal["float", "base64"]] = "float"

class SimilarityResult(BaseModel):
    """Single similarity result for a sentence"""
    object: Literal["similarity"] = "similarity"
    similarity: float
    embedding: Optional[Union[List[float], str]] = Field(default=None)
    index: int = Field(default=0)

class SimilarityResponse(APIResponse):
    """Similarity response object"""
    object: Literal["list"] = "list"
    data: List[SimilarityResult]
    model: str
    usage: Optional[Usage] = None