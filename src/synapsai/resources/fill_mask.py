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
Fill mask resource handlers
"""

from typing import TYPE_CHECKING, Optional, Union

from ..types.fill_mask import FillMaskResponse

if TYPE_CHECKING:
    from ..client import SynapsAI, AsyncSynapsAI

class FillMaskResource:
    """Fill mask resource handler"""
    
    def __init__(self, client: "SynapsAI"):
        self._client = client
    
    def create(
        self,
        model: str,
        inputs: Union[str, list[str]],
        targets: Optional[Union[str, list[str]]] = None,
        top_k: int = 5,
        **kwargs
    ) -> FillMaskResponse:
        """Generate images from text prompts"""
        
        # Build request
        request_data = self._client._build_request(
            model=model,
            inputs=inputs,
            targets=targets,
            top_k=top_k,
            **kwargs
        )
        
        # Make request
        endpoint = "fill-mask"
        
        response = self._client._post(endpoint, json_data=request_data)
        response_data = response.json()
        return FillMaskResponse.model_validate(response_data)

class AsyncFillMaskResource:
    """Fill mask resource handler"""
    
    def __init__(self, client: "AsyncSynapsAI"):
        self._client = client
    
    async def create(
        self,
        model: str,
        inputs: Union[str, list[str]],
        targets: Optional[Union[str, list[str]]] = None,
        top_k: int = 5,
        **kwargs
    ) -> FillMaskResponse:
        """Generate images from text prompts"""
        
        # Build request
        request_data = self._client._build_request(
            model=model,
            inputs=inputs,
            targets=targets,
            top_k=top_k,
            **kwargs
        )
        
        # Make request
        endpoint = "fill-mask"
        
        response = await self._client._post(endpoint, json_data=request_data)
        response_data = response.json()
        return FillMaskResponse.model_validate(response_data)