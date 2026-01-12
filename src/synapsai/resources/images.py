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
Images resource handlers
"""

from typing import TYPE_CHECKING, Optional, Dict
from PIL import Image
import base64
import os

from ..types.images import (
    ImageGenerateRequest,
    ImageGenerateResponse,
    ImageEditRequest,
    ImageEditResponse,
    ImageAnalysisRequest,
    ImageAnalysisResponse,
    ImageSource,
    DepthEstimationRequest,
    DepthEstimationResponse,
    ObjectDetectionRequest,
    ObjectDetectionResponse,
    ImageFeatureExtractionRequest,
    ImageFeatureExtractionResponse,
)
from ..exceptions import APIError

if TYPE_CHECKING:
    from ..client import SynapsAI, AsyncSynapsAI

def process_image_input(image: ImageSource):
    """Process image input (file path, bytes, base64, URL, or lists of those)"""

    if image is None:
        return None

    # Handle batched inputs
    if isinstance(image, list):
        return [process_image_input(img) for img in image]

    # Single input handling
    if isinstance(image, str):
        # URL
        if image.startswith(("http://", "https://")):
            return image

        # File path
        if os.path.isfile(image):
            with open(image, "rb") as f:
                image_bytes = f.read()
            return base64.b64encode(image_bytes).decode("utf-8")

        # Assume base64
        return image

    if isinstance(image, bytes):
        return base64.b64encode(image).decode("utf-8")

    if isinstance(image, Image.Image):
        return base64.b64encode(image.tobytes()).decode("utf-8")

    raise ValueError(
        "Image must be a file path, bytes, PIL.Image, URL, base64 string, or a list of these"
    )

class ImagesResource:
    """Images resource handler"""
    
    def __init__(self, client: "SynapsAI"):
        self._client = client
    
    def generate(
        self,
        model: str,
        prompt: str,
        n: int = 1,
        quality: str = "standard",
        response_format: str = "url",
        size: str = "1024x1024",
        style: str = "vivid",
        **kwargs
    ) -> ImageGenerateResponse:
        """Generate images from text prompts"""
        
        # Build request
        request_data = self._client._build_request(
            model=model,
            prompt=prompt,
            n=n,
            quality=quality,
            response_format=response_format,
            size=size,
            style=style,
            **kwargs
        )
        
        # Make request
        endpoint = "images/generations"
        
        response = self._client._post(endpoint, json_data=request_data)
        response_data = response.json()
        return ImageGenerateResponse.model_validate(response_data)
    
    def edit(
        self,
        image: ImageSource,
        model: str,
        prompt: str,
        mask: ImageSource = None,
        n: int = 1,
        size: str = "1024x1024",
        response_format: str = "url",
        **kwargs
    ) -> ImageEditResponse:
        """Edit images with prompts"""
        
        # Handle image input (file path, bytes, or base64)
        image_data = process_image_input(image)
        mask_data = process_image_input(mask) if mask else None
        
        # Build request
        request_data = self._client._build_request(
            model=model,
            image=image_data,
            mask=mask_data,
            prompt=prompt,
            n=n,
            size=size,
            response_format=response_format,
            **kwargs
        )
        
        # Make request
        endpoint = "images/edits"
        
        response = self._client._post(endpoint, json_data=request_data)
        response_data = response.json()
        return ImageEditResponse.model_validate(response_data)
    
    def to_text(
        self,
        model: str,
        inputs: ImageSource,
        max_new_tokens: int = 300,
        generate_kwargs: Optional[Dict] = None,
        **kwargs
    ) -> ImageAnalysisResponse:
        """Analyze images and extract information"""
        
        # Handle image input
        image_data = process_image_input(inputs)
        
        # Build request
        request_data = self._client._build_request(
            model=model,
            inputs=image_data,
            max_new_tokens=max_new_tokens,
            generate_kwargs=generate_kwargs,
            **kwargs
        )
        
        # Make request
        endpoint = "images/to-text"
        
        response = self._client._post(endpoint, json_data=request_data)
        response_data = response.json()
        return ImageAnalysisResponse.model_validate(response_data)

    def feature_extraction(
        self,
        model: str,
        images: ImageSource,
        **kwargs
    ) -> ImageFeatureExtractionResponse:
        """Extract features from images"""
        
        # Handle image input
        image_data = process_image_input(images)
        
        # Build request
        request_data = self._client._build_request(
            model=model,
            images=image_data,
            **kwargs
        )
        
        # Make request
        endpoint = "images/feature-extraction"
        
        response = self._client._post(endpoint, json_data=request_data)
        response_data = response.json()
        return ImageFeatureExtractionResponse.model_validate(response_data)

    def segmentation(
        self,
        model: str,
        inputs: ImageSource,
        subtask: str = 'panoptic',
        threshold: float = 0.9,
        mask_threshold: float = 0.5,
        overlap_mask_area_threshold: float = 0.5,
        **kwargs
    ) -> ImageAnalysisResponse:
        """Analyze images and extract information."""
        
        # Handle image input
        image_data = process_image_input(inputs)
        
        # Build request
        request_data = self._client._build_request(
            model=model,
            inputs=image_data,
            subtask=subtask,
            threshold=threshold,
            mask_threshold=mask_threshold,
            overlap_mask_area_threshold=overlap_mask_area_threshold,
            **kwargs
        )
        
        # Make request
        endpoint = "images/segmentation"
        
        response = self._client._post(endpoint, json_data=request_data)
        response_data = response.json()
        return ImageAnalysisResponse.model_validate(response_data)

    def depth_estimation(
        self,
        model: str,
        inputs: ImageSource,
        parameters: Optional[Dict] = None,
        **kwargs
    ) -> DepthEstimationResponse:
        """Analyze images and extract information."""
        
        # Handle image input
        image_data = process_image_input(inputs)
        
        # Build request
        request_data = self._client._build_request(
            model=model,
            inputs=image_data,
            parameters=parameters,
            **kwargs
        )
        
        # Make request
        endpoint = "images/depth-estimation"
        
        response = self._client._post(endpoint, json_data=request_data)
        response_data = response.json()
        return DepthEstimationResponse.model_validate(response_data)

    def object_detection(
        self,
        model: str,
        inputs: ImageSource,
        threshold: float = 0.5,
        **kwargs
    ) -> ObjectDetectionResponse:
        """Analyze images and extract information."""
        
        # Handle image input
        image_data = process_image_input(inputs)
        
        # Build request
        request_data = self._client._build_request(
            model=model,
            inputs=image_data,
            threshold=threshold,
            **kwargs
        )
        
        # Make request
        endpoint = "images/object-detection"
        
        response = self._client._post(endpoint, json_data=request_data)
        response_data = response.json()
        return ObjectDetectionResponse.model_validate(response_data)

class AsyncImagesResource:
    """Async images resource handler"""
    
    def __init__(self, client: "AsyncSynapsAI"):
        self._client = client
    
    async def generate(
        self,
        model: str,
        prompt: str,
        n: int = 1,
        quality: str = "standard",
        response_format: str = "url",
        size: str = "1024x1024",
        style: str = "vivid",
        **kwargs
    ) -> ImageGenerateResponse:
        """Generate images from text prompts asynchronously"""
        
        # Build request
        request_data = self._client._build_request(
            model=model,
            prompt=prompt,
            n=n,
            quality=quality,
            response_format=response_format,
            size=size,
            style=style,
            **kwargs
        )
        
        # Make request
        endpoint = "images/generations"
        
        response = await self._client._post(endpoint, json_data=request_data)
        response_data = response.json()
        return ImageGenerateResponse.model_validate(response_data)
    
    async def edit(
        self,
        image: ImageSource,
        model: str,
        prompt: str,
        mask: ImageSource = None,
        n: int = 1,
        size: str = "1024x1024",
        response_format: str = "url",
        **kwargs
    ) -> ImageEditResponse:
        """Edit images with prompts asynchronously"""
        
        # Handle image input
        image_data = process_image_input(image)
        mask_data = process_image_input(mask) if mask else None
        
        # Build request
        request_data = self._client._build_request(
            model=model,
            image=image_data,
            mask=mask_data,
            prompt=prompt,
            n=n,
            size=size,
            response_format=response_format,
            **kwargs
        )
        
        # Make request
        endpoint = "images/edits"
        
        response = await self._client._post(endpoint, json_data=request_data)
        response_data = response.json()
        return ImageEditResponse.model_validate(response_data)
    
    async def to_text(
        self,
        inputs: ImageSource,
        model: str,
        max_new_tokens: int = 300,
        generate_kwargs: Optional[Dict] = None,
        **kwargs
    ) -> ImageAnalysisResponse:
        """Analyze images and extract information asynchronously (custom endpoint)"""
        
        # Handle image input
        inputs_data = process_image_input(inputs)
        
        # Build request
        request_data = self._client._build_request(
            model=model,
            inputs=inputs_data,
            max_new_tokens=max_new_tokens,
            generate_kwargs=generate_kwargs,
            **kwargs
        )
        
        # Make request
        endpoint = "/images/to-text"
        
        response = await self._client._post(endpoint, json_data=request_data)
        response_data = response.json()
        return ImageAnalysisResponse.model_validate(response_data)

    async def feature_extraction(
        self,
        model: str,
        images: ImageSource,
        **kwargs
    ) -> ImageFeatureExtractionResponse:
        """Extract features from images"""
        
        # Handle image input
        image_data = process_image_input(images)
        
        # Build request
        request_data = self._client._build_request(
            model=model,
            images=image_data,
            **kwargs
        )
        
        # Make request
        endpoint = "images/feature-extraction"
        
        response = await self._client._post(endpoint, json_data=request_data)
        response_data = response.json()
        return ImageFeatureExtractionResponse.model_validate(response_data)

    async def segmentation(
        self,
        model: str,
        inputs: ImageSource,
        subtask: str = 'panoptic',
        threshold: float = 0.9,
        mask_threshold: float = 0.5,
        overlap_mask_area_threshold: float = 0.5,
        **kwargs
    ) -> ImageAnalysisResponse:
        """Analyze images and extract information."""
        
        # Handle image input
        image_data = process_image_input(inputs)
        
        # Build request
        request_data = self._client._build_request(
            model=model,
            inputs=image_data,
            subtask=subtask,
            threshold=threshold,
            mask_threshold=mask_threshold,
            overlap_mask_area_threshold=overlap_mask_area_threshold,
            **kwargs
        )
        
        # Make request
        endpoint = "images/segmentation"
        
        response = await self._client._post(endpoint, json_data=request_data)
        response_data = response.json()
        return ImageAnalysisResponse.model_validate(response_data)

    async def depth_estimation(
        self,
        model: str,
        inputs: ImageSource,
        parameters: Optional[Dict] = None,
        **kwargs
    ) -> DepthEstimationResponse:
        """Analyze images and extract information."""
        
        # Handle image input
        image_data = process_image_input(inputs)
        
        # Build request
        request_data = self._client._build_request(
            model=model,
            inputs=image_data,
            parameters=parameters,
            **kwargs
        )
        
        # Make request
        endpoint = "images/depth-estimation"
        
        response = await self._client._post(endpoint, json_data=request_data)
        response_data = response.json()
        return DepthEstimationResponse.model_validate(response_data)

    async def object_detection(
        self,
        model: str,
        inputs: ImageSource,
        threshold: float = 0.5,
        **kwargs
    ) -> ObjectDetectionResponse:
        """Analyze images and extract information."""
        
        # Handle image input
        image_data = process_image_input(inputs)
        
        # Build request
        request_data = self._client._build_request(
            model=model,
            inputs=image_data,
            threshold=threshold,
            **kwargs
        )
        
        # Make request
        endpoint = "images/object-detection"
        
        response = await self._client._post(endpoint, json_data=request_data)
        response_data = response.json()
        return ObjectDetectionResponse.model_validate(response_data)