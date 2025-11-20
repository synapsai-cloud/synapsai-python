"""
Images resource handlers
"""

from typing import TYPE_CHECKING, Optional, Dict
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
)
from ..exceptions import APIError

if TYPE_CHECKING:
    from ..client import SynapsAI, AsyncSynapsAI


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
        endpoint = "/images/generations"
        
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
        image_data = self._process_image_input(image)
        mask_data = self._process_image_input(mask) if mask else None
        
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
        endpoint = "/images/edits"
        
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
        """Analyze images and extract information (custom endpoint)"""
        
        # Handle image input
        image_data = self._process_image_input(inputs)
        
        # Build request
        request_data = self._client._build_request(
            model=model,
            inputs=image_data,
            max_new_tokens=max_new_tokens,
            generate_kwargs=generate_kwargs,
            **kwargs
        )
        
        # Make request
        endpoint = "/images/to-text"
        
        response = self._client._post(endpoint, json_data=request_data)
        response_data = response.json()
        return ImageAnalysisResponse.model_validate(response_data)
    
    def _process_image_input(self, image: ImageSource):
        """Process image input (file path, bytes, or base64)"""
        if image is None:
            return None
            
        if isinstance(image, str):
            # Check if it's a URL
            if image.startswith(("http://", "https://")):
                return image
            # Check if it's a file path
            if os.path.isfile(image):
                with open(image, "rb") as f:
                    image_bytes = f.read()
                return base64.b64encode(image_bytes).decode("utf-8")
            # Assume it's already base64
            return image
        elif isinstance(image, bytes):
            return base64.b64encode(image).decode("utf-8")
        else:
            raise ValueError("Image must be a file path, bytes, URL, or base64 string")


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
        endpoint = "/images/generations"
        
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
        image_data = self._process_image_input(image)
        mask_data = self._process_image_input(mask) if mask else None
        
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
        endpoint = "/images/edits"
        
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
        inputs_data = self._process_image_input(inputs)
        
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
    
    def _process_image_input(self, image: ImageSource):
        """Process image input (file path, bytes, or base64)"""
        if image is None:
            return None
            
        if isinstance(image, str):
            # Check if it's a URL
            if image.startswith(("http://", "https://")):
                return image
            # Check if it's a file path
            if os.path.isfile(image):
                with open(image, "rb") as f:
                    image_bytes = f.read()
                return base64.b64encode(image_bytes).decode("utf-8")
            # Assume it's already base64
            return image
        elif isinstance(image, bytes):
            return base64.b64encode(image).decode("utf-8")
        else:
            raise ValueError("Image must be a file path, bytes, URL, or base64 string") 