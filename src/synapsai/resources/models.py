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
Models resource handlers
"""

from typing import TYPE_CHECKING

from ..types.models import (
    Model, Models
)

if TYPE_CHECKING:
    from ..client import SynapsAI, AsyncSynapsAI


class ModelsResource:
    """Models resource handler"""
    
    def __init__(self, client: "SynapsAI"):
        self._client = client
    
    def list(self) -> Models:
        """Get available models."""
        
        # Make request
        endpoint = "models"
        
        response = self._client._get(endpoint)
        response_data = response.json()
        return Models.model_validate(response_data)

    def retrieve(self, model: str) -> Model:
        """Retrieve a model."""

        # Make request
        endpoint = f"models/{model}"

        response = self._client._get(endpoint)
        response_data = response.json()
        return Model.model_validate(response_data)

class AsyncModelsResource:
    """Async images resource handler"""
    
    def __init__(self, client: "AsyncSynapsAI"):
        self._client = client
    
    async def list(self) -> Models:
        """Get available models."""
        
        # Make request
        endpoint = "models"
        
        response = await self._client._get(endpoint)
        response_data = response.json()
        return Models.model_validate(response_data)

    async def retrieve(self, model: str) -> Model:
        """Retrieve a model."""

        # Make request
        endpoint = f"models/{model}"

        response = await self._client._get(endpoint)
        response_data = response.json()
        return Model.model_validate(response_data)
    