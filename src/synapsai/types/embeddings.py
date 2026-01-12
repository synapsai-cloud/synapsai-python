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