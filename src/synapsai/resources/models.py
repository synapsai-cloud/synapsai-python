"""
Models resource handlers
"""

from typing import TYPE_CHECKING

from ..types.models import (
    ModelResponse, ModelsResponse
)

if TYPE_CHECKING:
    from ..client import SynapsAI, AsyncSynapsAI


class ModelsResource:
    """Models resource handler"""
    
    def __init__(self, client: "SynapsAI"):
        self._client = client
    
    def list(self) -> ModelsResponse:
        """Get available models."""
        
        # Make request
        endpoint = "models"
        
        response = self._client._get(endpoint)
        response_data = response.json()
        return ModelsResponse.model_validate(response_data)

    def retrieve(self, model: str) -> ModelResponse:
        """Retrieve a model."""

        # Make request
        endpoint = f"models/{model}"

        response = self._client._get(endpoint)
        response_data = response.json()
        return ModelResponse.model_validate(response_data)

class AsyncModelsResource:
    """Async images resource handler"""
    
    def __init__(self, client: "AsyncSynapsAI"):
        self._client = client
    
    async def list(self) -> ModelsResponse:
        """Get available models."""
        
        # Make request
        endpoint = "models"
        
        response = await self._client._get(endpoint)
        response_data = response.json()
        return ModelsResponse.model_validate(response_data)

    async def retrieve(self, model: str) -> ModelResponse:
        """Retrieve a model."""

        # Make request
        endpoint = f"models/{model}"

        response = await self._client._get(endpoint)
        response_data = response.json()
        return ModelResponse.model_validate(response_data)
    