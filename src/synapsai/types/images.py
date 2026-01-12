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
Image generation, editing, and analysis type definitions
"""

from typing import Optional, Dict, Any, Union, List, Literal
from pydantic import BaseModel, Field
from enum import Enum

from .common import APIResponse, Usage

ImageSource = Union[str, bytes, List[str], List[bytes]]

class ImageSize(str, Enum):
    """Image size options"""
    SIZE_256x256 = "256x256"
    SIZE_512x512 = "512x512"
    SIZE_1024x1024 = "1024x1024"
    SIZE_1792x1024 = "1792x1024"
    SIZE_1024x1792 = "1024x1792"


class ImageQuality(str, Enum):
    """Image quality options"""
    STANDARD = "standard"
    HD = "hd"


class ImageStyle(str, Enum):
    """Image style options"""
    VIVID = "vivid"
    NATURAL = "natural"


class ResponseFormat(str, Enum):
    """Image response format"""
    URL = "url"
    B64_JSON = "b64_json"


class Image(BaseModel):
    """Image object"""
    url: Optional[str] = None
    b64_json: Optional[str] = None
    image_bytes: Optional[bytes] = None
    revised_prompt: Optional[str] = None

class ImageGenerateRequest(BaseModel):
    """Image generation request"""
    model: str
    prompt: str = Field(max_length=4000)
    n: Optional[int] = Field(default=1, ge=1, le=10)
    quality: Optional[ImageQuality] = ImageQuality.STANDARD
    response_format: Optional[ResponseFormat] = ResponseFormat.URL
    size: Optional[ImageSize] = ImageSize.SIZE_1024x1024
    style: Optional[ImageStyle] = ImageStyle.VIVID

class ImageGenerateResponse(APIResponse):
    """Image generation response"""
    object: Literal["list"] = "list"
    data: List[Image]
    usage: Optional[Usage] = None

class ImageEditRequest(BaseModel):
    """Image edit request"""
    model: str
    image: ImageSource = Field(description="The image to edit (base64 encoded, URL, or file)")
    mask: Optional[ImageSource] = Field(default=None, description="Mask image (base64 encoded, URL, or file)")
    prompt: str = Field(max_length=1000)
    n: Optional[int] = Field(default=1, ge=1, le=10)
    size: Optional[ImageSize] = ImageSize.SIZE_1024x1024
    response_format: Optional[ResponseFormat] = ResponseFormat.URL

class ImageEditResponse(APIResponse):
    """Image edit response"""
    object: Literal["list"] = "list"
    data: List[Image]
    usage: Optional[Usage] = None

class ImageAnalysisRequest(BaseModel):
    """Image analysis request"""
    model: str
    image: ImageSource = Field(description="The image to analyze (base64 encoded, URL, or file)")
    fields: Optional[List[str]] = Field(
        default=None,
        description="Specific fields to extract (e.g., ['objects', 'text', 'colors', 'mood'])"
    )
    detail: Optional[Literal["low", "high", "auto"]] = "auto"
    max_tokens: Optional[int] = Field(default=300, gt=0)

class ImageAnalysisResponse(APIResponse):
    """Image analysis response"""
    object: Literal["list"] = "list"
    model: str
    data: List[Dict[str, Any]]
    usage: Optional[Usage] = None

class ImageSegmentationRequest(BaseModel):
    """Image segmentation request"""
    model: str
    image: ImageSource = Field(description="The image to analyze (base64 encoded, URL, or file)")
    subtask: str = 'panoptic'
    threshold: float = 0.9
    mask_threshold: float = 0.5
    overlap_mask_area_threshold: float = 0.5

class ImageSegmentationResponse(APIResponse):
    """Image segmentation response"""
    object: Literal["list"] = "list"
    model: str
    data: List[Dict[str, Any]]
    usage: Optional[Usage] = None

class ImageFeatureExtractionRequest(BaseModel):
    """Image feature extraction request"""
    model: str
    images: ImageSource = Field(description="The image to extract features from (base64 encoded, URL, or file)")

class ImageFeatureExtractionResponse(APIResponse):
    """Image feature extraction response"""
    object: Literal["list"] = "list"
    model: str
    data: List[Dict[str, Any]]
    usage: Optional[Usage] = None

class ObjectDetectionRequest(BaseModel):
    """Object detection request"""
    model: str
    inputs: ImageSource = Field(description="The image to analyze (base64 encoded, URL, or file)")
    threshold: float = 0.5

class ObjectDetectionResponse(APIResponse):
    """Object detection response"""
    object: Literal["list"] = "list"
    model: str
    data: List[Dict[str, Any]]
    usage: Optional[Usage] = None

class DepthEstimationRequest(BaseModel):
    """Depth estimation request"""
    model: str
    inputs: ImageSource = Field(description="The image to analyze (base64 encoded, URL, or file)")
    parameters: Optional[Dict] = None

class DepthEstimationResponse(APIResponse):
    """Depth estimation response"""
    object: Literal["list"] = "list"
    model: str
    data: List[Dict[str, Any]]
    usage: Optional[Usage] = None

class ImageFeatureExtractionRequest(BaseModel):
    """Feature extraction request"""
    model: str
    images: ImageSource = Field(description="The image to extract features from (base64 encoded, URL, or file)")

class ImageFeatureExtractionResponse(APIResponse):
    """Feature extraction response"""
    object: Literal["list"] = "list"
    model: str
    data: List[Dict[str, Any]]
    usage: Optional[Usage] = None