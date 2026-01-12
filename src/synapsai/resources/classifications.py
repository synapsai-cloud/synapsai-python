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
Classification resource handlers
"""

from typing import TYPE_CHECKING, Optional, Union, List
from ..types.classifications import (
    AudioClassificationResponse,
    ImageClassificationResponse,
    TextClassificationResponse,
    TokenClassificationResponse,
    VideoClassificationResponse,
    VideoClassificationResponse,
    ZeroShotAudioClassificationResponse,
    ZeroShotClassificationResponse,
    ZeroShotImageClassificationResponse,
    ZeroShotObjectDetectionResponse,
)
from ..exceptions import APIError
from PIL import Image
import numpy as np


if TYPE_CHECKING:
    from ..client import SynapsAI, AsyncSynapsAI



class ZeroShotClassificationsResource:
    """Zero-Shot Classification resource handler"""
    
    def __init__(self, client: "SynapsAI"):
        self._client = client
    
    def audio(
        self,
        model: str,
        audios,
        candidate_labels: list[str],
        hypothesis_template: Optional[str] = None,
    ) -> ZeroShotAudioClassificationResponse:
        """Assign labels to the audio(s) passed as inputs (zero-shot)."""

        request_data = self._client._build_request(
            model=model,
            audios=audios,
            candidate_labels=candidate_labels,
            hypothesis_template=hypothesis_template,
        )

        endpoint = "classifications/zero-shot/audio"
        response = self._client._post(endpoint, json_data=request_data)
        return ZeroShotAudioClassificationResponse.model_validate(response.json())

    def text(
        self,
        model: str,
        sequences: Union[str, List[str]],
        candidate_labels: Union[str, List[str]],
        hypothesis_template: Optional[str] = None,
        multi_label: Optional[bool] = None,
    ) -> ZeroShotClassificationResponse:
        """Zero-shot text classification."""

        request_data = self._client._build_request(
            model=model,
            sequences=sequences,
            candidate_labels=candidate_labels,
            hypothesis_template=hypothesis_template,
            multi_label=multi_label,
        )

        endpoint = "classifications/zero-shot"
        response = self._client._post(endpoint, json_data=request_data)
        return ZeroShotClassificationResponse.model_validate(response.json())

    def image(
        self,
        model: str,
        image: Union[str, List[str], Image.Image, List[Image.Image]],
        candidate_labels: list[str],
        hypothesis_template: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> ZeroShotImageClassificationResponse:
        """Zero-shot image classification."""

        request_data = self._client._build_request(
            model=model,
            image=image,
            candidate_labels=candidate_labels,
            hypothesis_template=hypothesis_template,
            timeout=timeout,
        )

        endpoint = "classifications/zero-shot/image"
        response = self._client._post(endpoint, json_data=request_data)
        return ZeroShotImageClassificationResponse.model_validate(response.json())

    def object(
        self,
        model: str,
        box,
    ) -> ZeroShotObjectDetectionResponse:
        """Zero-shot object detection utility endpoint."""

        request_data = self._client._build_request(
            model=model,
            box=box,
        )

        endpoint = "classifications/zero-shot/object"
        response = self._client._post(endpoint, json_data=request_data)
        return ZeroShotObjectDetectionResponse.model_validate(response.json())


class ClassificationsResource:
    """Classification resource handler"""
    
    def __init__(self, client: "SynapsAI"):
        self._client = client
        self.zero_shot = ZeroShotClassificationsResource(client)
    
    
    def audio(
        self,
        model: str,
        inputs: Union[np.ndarray, bytes, dict], 
        top_k: Optional[int] = None, 
        function_to_apply: Optional[str] = None,
    ) -> AudioClassificationResponse:
        """Classify the sequence(s) given as inputs. See the [`AutomaticSpeechRecognitionPipeline`] documentation for more information."""
        
        # Build request
        request_data = self._client._build_request(
            model=model,
            inputs=inputs,
            top_k=top_k,
            function_to_apply=function_to_apply
        )

        endpoint = "classifications/audio"
        
        # Make request
        response = self._client._post(endpoint, json_data=request_data)
        response_data = response.json()
        return AudioClassificationResponse(**response_data)
    
    def image(
        self,
        model: str,
        inputs: Union[str, list[str], Image.Image, list[Image.Image]], 
        function_to_apply: Optional[str] = None, 
        top_k: Optional[int] = None, 
        timeout: Optional[float] = None,
    ) -> ImageClassificationResponse:
        """Assign labels to the image(s) passed as inputs."""
        
        # Build request
        request_data = self._client._build_request(
            model=model,
            inputs=inputs,
            function_to_apply=function_to_apply,
            top_k=top_k,
            timeout=timeout
        )
        
        endpoint = "classifications/image"
        
        # Make request
        response = self._client._post(endpoint, json_data=request_data)
        response_data = response.json()
        return ImageClassificationResponse.model_validate(response_data)
    
    def text(
        self,
        model: str,
        inputs: Union[str, list[str], dict[str], list[dict[str]]], 
        top_k: Optional[int] = None, 
        function_to_apply: Optional[str] = None,
    ) -> TextClassificationResponse:
        """Classify the text(s) given as inputs."""
        
        # Build request
        request_data = self._client._build_request(
            model=model,
            inputs=inputs,
            top_k=top_k,
            function_to_apply=function_to_apply
        )

        endpoint = "classifications/text"
        
        # Make request
        response = self._client._post(endpoint, json_data=request_data)
        response_data = response.json()
        return TextClassificationResponse.model_validate(response_data)
    
    def token(
        self,
        model: str,
        inputs: Union[str, List[str]],
    ) -> TokenClassificationResponse:
        """Classify each token of the text(s) given as inputs."""
        
        # Build request
        request_data = self._client._build_request(
            model=model,
            inputs=inputs
        )

        endpoint = "classifications/token"
        
        # Make request
        response = self._client._post(endpoint, json_data=request_data)
        response_data = response.json()
        return TokenClassificationResponse.model_validate(response_data)
    
    def video(
        self,
        model: str,
        inputs: Union[str, list[str]], 
        top_k: Optional[int] = None, 
        num_frames: Optional[int] = None, 
        frame_sampling_rate: Optional[int] = None, 
        function_to_apply: Optional[str] = None,
    ) -> VideoClassificationResponse:
        """Assign labels to the video(s) passed as inputs."""
        
        # Build request
        request_data = self._client._build_request(
            model=model,
            inputs=inputs,
            top_k=top_k,
            num_frames=num_frames,
            frame_sampling_rate=frame_sampling_rate,
            function_to_apply=function_to_apply
        )

        endpoint = "classifications/video"

        # Make request
        response = self._client._post(endpoint, json_data=request_data)
        response_data = response.json()
        return VideoClassificationResponse.model_validate(response_data)
  


class AsyncZeroShotClassificationsResource:
    """Async Zero-Shot Classification resource handler"""
    
    def __init__(self, client: "AsyncSynapsAI"):
        self._client = client
    async def audio(
        self,
        model: str,
        audios,
        candidate_labels: list[str],
        hypothesis_template: Optional[str] = None,
    ) -> ZeroShotAudioClassificationResponse:
        """Assign labels to the audio(s) passed as inputs (zero-shot)."""

        request_data = self._client._build_request(
            model=model,
            audios=audios,
            candidate_labels=candidate_labels,
            hypothesis_template=hypothesis_template,
        )

        endpoint = "classifications/zero-shot/audio"
        response = await self._client._post(endpoint, json_data=request_data)
        return ZeroShotAudioClassificationResponse.model_validate(response.json())

    async def text(
        self,
        model: str,
        sequences: Union[str, List[str]],
        candidate_labels: Union[str, List[str]],
        hypothesis_template: Optional[str] = None,
        multi_label: Optional[bool] = None,
    ) -> ZeroShotClassificationResponse:
        """Zero-shot text classification."""

        request_data = self._client._build_request(
            model=model,
            sequences=sequences,
            candidate_labels=candidate_labels,
            hypothesis_template=hypothesis_template,
            multi_label=multi_label,
        )

        endpoint = "classifications/zero-shot"
        response = await self._client._post(endpoint, json_data=request_data)
        return ZeroShotClassificationResponse.model_validate(response.json())

    async def image(
        self,
        model: str,
        image: Union[str, List[str], Image.Image, List[Image.Image]],
        candidate_labels: list[str],
        hypothesis_template: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> ZeroShotImageClassificationResponse:
        """Zero-shot image classification."""

        request_data = self._client._build_request(
            model=model,
            image=image,
            candidate_labels=candidate_labels,
            hypothesis_template=hypothesis_template,
            timeout=timeout,
        )

        endpoint = "classifications/zero-shot/image"
        response = await self._client._post(endpoint, json_data=request_data)
        return ZeroShotImageClassificationResponse.model_validate(response.json())

    async def object(
        self,
        model: str,
        box,
    ) -> ZeroShotObjectDetectionResponse:
        """Zero-shot object detection utility endpoint."""

        request_data = self._client._build_request(
            model=model,
            box=box,
        )

        endpoint = "classifications/zero-shot/object"
        response = await self._client._post(endpoint, json_data=request_data)
        return ZeroShotObjectDetectionResponse.model_validate(response.json())



class AsyncClassificationsResource:
    """Async Classification resource handler"""
    
    def __init__(self, client: "AsyncSynapsAI"):
        self._client = client
        self.zero_shot = AsyncZeroShotClassificationsResource(client)


    async def audio(
        self,
        model: str,
        inputs: Union[np.ndarray, bytes, dict], 
        top_k: Optional[int] = None, 
        function_to_apply: Optional[str] = None,
    ) -> AudioClassificationResponse:
        """Classify the sequence(s) given as inputs. See the [`AutomaticSpeechRecognitionPipeline`] documentation for more information."""
        
        # Build request
        request_data = self._client._build_request(
            model=model,
            inputs=inputs,
            top_k=top_k,
            function_to_apply=function_to_apply
        )
        
        endpoint = "classifications/audio"
        
        # Make request
        response = await self._client._post(endpoint, json_data=request_data)
        response_data = response.json()
        return AudioClassificationResponse.model_validate(response_data)
    
    async def image(
        self,
        model: str,
        inputs: Union[str, list[str], Image.Image, list[Image.Image]], 
        function_to_apply: Optional[str] = None, 
        top_k: Optional[int] = None, 
        timeout: Optional[float] = None,
    ) -> ImageClassificationResponse:
        """Assign labels to the image(s) passed as inputs."""
        
        # Build request
        request_data = self._client._build_request(
            model=model,
            inputs=inputs,
            function_to_apply=function_to_apply,
            top_k=top_k,
            timeout=timeout
        )

        endpoint = "classifications/image"

        # Make request
        response = await self._client._post(endpoint, json_data=request_data)
        response_data = response.json()
        return ImageClassificationResponse.model_validate(response_data)
    
    async def text(
        self,
        model: str,
        inputs: Union[str, list[str], dict[str], list[dict[str]]], 
        top_k: Optional[int] = None, 
        function_to_apply: Optional[str] = None,
    ) -> TextClassificationResponse:
        """Classify the text(s) given as inputs."""
        
        # Build request
        request_data = self._client._build_request(
            model=model,
            inputs=inputs,
            top_k=top_k,
            function_to_apply=function_to_apply
        )

        endpoint = "classifications/text"

        # Make request
        response = await self._client._post(endpoint, json_data=request_data)
        response_data = response.json()
        return TextClassificationResponse.model_validate(response_data)
    
    async def token(
        self,
        model: str,
        inputs: Union[str, List[str]],
    ) -> TokenClassificationResponse:
        """Classify each token of the text(s) given as inputs."""
        
        # Build request
        request_data = self._client._build_request(
            model=model,
            inputs=inputs
        )

        endpoint = "classifications/token"
        
        # Make request
        response = await self._client._post(endpoint, json_data=request_data)
        response_data = response.json()
        return TokenClassificationResponse.model_validate(response_data)
    
    async def video(
        self,
        model: str,
        inputs: Union[str, list[str]], 
        top_k: Optional[int] = None, 
        num_frames: Optional[int] = None, 
        frame_sampling_rate: Optional[int] = None, 
        function_to_apply: Optional[str] = None,
    ) -> VideoClassificationResponse:
        """Assign labels to the video(s) passed as inputs."""
        
        # Build request
        request_data = self._client._build_request(
            model=model,
            inputs=inputs,
            top_k=top_k,
            num_frames=num_frames,
            frame_sampling_rate=frame_sampling_rate,
            function_to_apply=function_to_apply
        )

        endpoint = "classifications/video"

        # Make request
        response = await self._client._post(endpoint, json_data=request_data)
        response_data = response.json()
        return VideoClassificationResponse.model_validate(response_data)
