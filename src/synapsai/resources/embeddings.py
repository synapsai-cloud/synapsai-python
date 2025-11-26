"""
Embeddings resource handlers
"""

from typing import Union, List, TYPE_CHECKING, Literal, Optional

from ..types.embeddings import (
    EmbeddingResponse,
    SimilarityResponse,
)
from ..exceptions import APIError
import math

if TYPE_CHECKING:
    from ..client import SynapsAI, AsyncSynapsAI


def _dot_product(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _vector_norm(a: List[float]) -> float:
    return math.sqrt(sum(x * x for x in a))


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    denom = _vector_norm(a) * _vector_norm(b)
    if denom == 0:
        return 0.0
    return _dot_product(a, b) / denom


class EmbeddingsResource:
    """Embeddings resource handler"""
    
    def __init__(self, client: "SynapsAI"):
        self._client = client
    
    def create(
        self,
        model: str, 
        input: Union[str, List[str], List[int], List[List[int]]],
        encoding_format: Optional[Literal["float", "base64"]] = "float",
        **kwargs
    ) -> EmbeddingResponse:
        """Create embeddings for the given input"""
        
        # Build request
        request_data = self._client._build_request(
            model=model,
            input=input,
            encoding_format=encoding_format,
            **kwargs
        )
        
        endpoint = "embeddings"
        
        # Make request
        response = self._client._post(endpoint, json_data=request_data)
        response_data = response.json()
        return EmbeddingResponse.model_validate(response_data)

    def similarity(
        self,
        model: str,
        source_sentence: str,
        sentences: List[str],
        return_embeddings: Optional[bool] = False,
        encoding_format: Optional[Literal["float", "base64"]] = "float",
        **kwargs
    ) -> SimilarityResponse:
        """Calculate similarity between a source sentence and other sentences. It basically is a wrapper around the create function and returns the similarity score."""
        # Validate inputs
        if not sentences:
            raise APIError("`sentences` must be a non-empty list")

        # Build inputs: first element is source_sentence followed by other sentences
        inputs = [source_sentence] + sentences

        # Call create to get embeddings
        emb_response = self.create(
            model=model,
            input=inputs,
            encoding_format=encoding_format,
            **kwargs,
        )

        # Validate response
        if not emb_response or not getattr(emb_response, "data", None):
            raise APIError("Failed to obtain embeddings")

        # First embedding is for source
        source_emb = emb_response.data[0].embedding

        results = []
        for item in emb_response.data[1:]:
            sim = _cosine_similarity(source_emb, item.embedding)
            result_obj = {
                "object": "similarity",
                "similarity": float(sim),
                "embedding": item.embedding if return_embeddings else None,
                "index": int(item.index - 1),
            }
            results.append(result_obj)

        response_obj = {
            "object": "list",
            "data": results,
            "model": emb_response.model,
            "usage": getattr(emb_response, "usage", None),
        }

        return SimilarityResponse.model_validate(response_obj)

class AsyncEmbeddingsResource:
    """Async embeddings resource handler"""
    
    def __init__(self, client: "AsyncSynapsAI"):
        self._client = client
    
    async def create(
        self,
        model: str,
        input: Union[str, List[str], List[int], List[List[int]]],
        encoding_format: Optional[Literal["float", "base64"]] = "float",
        **kwargs
    ) -> EmbeddingResponse:
        """Create embeddings for the given input asynchronously"""
        
        # Build request
        request_data = self._client._build_request(
            model=model,
            input=input,
            encoding_format=encoding_format,
            **kwargs
        )
        
        endpoint = "embeddings"
        
        # Make request
        response = await self._client._post(endpoint, json_data=request_data)
        response_data = response.json()
        return EmbeddingResponse.model_validate(response_data) 

    async def similarity(
        self,
        model: str,
        source_sentence: str,
        sentences: List[str],
        return_embeddings: Optional[bool] = False,
        encoding_format: Optional[Literal["float", "base64"]] = "float",
        **kwargs
    ) -> SimilarityResponse:
        """Calculate similarity between a source sentence and other sentences. It basically is a wrapper around the create function and returns the similarity score."""
        # Validate inputs
        if not sentences:
            raise APIError("`sentences` must be a non-empty list")

        inputs = [source_sentence] + sentences

        emb_response = await self.create(
            model=model,
            input=inputs,
            encoding_format=encoding_format,
            **kwargs,
        )

        if not emb_response or not getattr(emb_response, "data", None):
            raise APIError("Failed to obtain embeddings")

        source_emb = emb_response.data[0].embedding

        results = []
        for item in emb_response.data[1:]:
            sim = _cosine_similarity(source_emb, item.embedding)
            result_obj = {
                "object": "similarity",
                "similarity": float(sim),
                "embedding": item.embedding if return_embeddings else None,
                "index": int(item.index - 1),
            }
            results.append(result_obj)

        response_obj = {
            "object": "list",
            "data": results,
            "model": emb_response.model,
            "usage": getattr(emb_response, "usage", None),
        }

        return SimilarityResponse.model_validate(response_obj)