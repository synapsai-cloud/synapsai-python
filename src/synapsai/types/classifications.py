"""
Classification type definitions
"""

from typing import Optional, Dict, Any, Union, List, Literal
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum
from numpy import ndarray
from PIL.Image import Image as PilImage

from .common import APIResponse, Usage


class AudioClassificationData(BaseModel):
    """Data model for audio classification result"""
    label: str = Field(..., description="The label predicted for the audio")
    score: float = Field(..., description="The probability/score for the label")


class AudioClassificationRequest(BaseModel):
    """Request model for audio classification"""
    model: str
    inputs: Union[ndarray, bytes, Dict[str, Any]] = Field(..., description="The audio data to classify. Can be raw waveform, bytes from an audio file, or a dict with sampling rate and raw audio.")
    top_k: Optional[int] = Field(default=None, description="The number of top labels that will be returned by the pipeline.")
    function_to_apply: Optional[Literal['sigmoid','softmax','none']] = Field(default=None, description="The function to apply to the model outputs to retrieve scores. Valid: 'sigmoid', 'softmax', 'none'.")
    model_config = ConfigDict(arbitrary_types_allowed=True)


class AudioClassificationResponse(APIResponse):
    """Audio classification response"""
    object: Literal["list"] = "list"
    data: List[AudioClassificationData]
    usage: Optional[Usage] = None


class ImageClassificationData(BaseModel):
    """Data model for image classification result"""
    label: str = Field(..., description="The label identified by the model")
    score: float = Field(..., description="The score attributed by the model for that label")


class ImageClassificationRequest(BaseModel):
    """Request model for image classification"""
    model: str
    inputs: Union[str, List[str], PilImage, List[PilImage]] = Field(..., description="The image or list of images to classify. Can be a URL, base64, file path or PIL Image.")
    function_to_apply: Optional[Literal['sigmoid','softmax','none']] = Field(default='default', description="The function to apply to the model outputs in order to retrieve the scores.")
    top_k: Optional[int] = Field(default=5, description="The number of top labels that will be returned by the pipeline.")
    timeout: Optional[float] = Field(default=None, description="Maximum time in seconds to wait for fetching images from the web. If None, no timeout is set.")
    model_config = ConfigDict(arbitrary_types_allowed=True)

class ImageClassificationResponse(APIResponse):
    """Image classification response"""
    object: Literal["list"] = "list"
    data: List[ImageClassificationData]
    usage: Optional[Usage] = None


class TextClassificationData(BaseModel):
    """Data model for text classification result"""
    label: str = Field(..., description="The label predicted")
    score: float = Field(..., description="The corresponding probability")


class TextClassificationRequest(BaseModel):
    """Request model for text classification"""
    model: str
    inputs: Union[str, List[str], Dict[str, Any], List[Dict[str, Any]]] = Field(..., description="One or several texts to classify. To use text pairs, send a dict with {'text', 'text_pair'} or a list of such dicts.")
    top_k: Optional[int] = Field(default=1, description="How many results to return.")
    function_to_apply: Optional[Literal['sigmoid','softmax','none']] = Field(default='default', description="The function to apply to the model outputs in order to retrieve the scores.")


class TextClassificationResponse(APIResponse):
    """Text classification response"""
    object: Literal["list"] = "list"
    data: List[TextClassificationData]
    usage: Optional[Usage] = None


class TokenClassificationData(BaseModel):
    """Data model for token classification result"""
    word: str = Field(..., description="The token/word")
    score: float = Field(..., description="The score for this token")
    entity: str = Field(..., description="The entity label for the token")
    index: int = Field(..., description="Token index")
    start: int = Field(..., description="Start char offset in the original text")
    end: int = Field(..., description="End char offset in the original text")


class TokenClassificationRequest(BaseModel):
    """Request model for token classification"""
    model: str
    inputs: Union[str, List[str]] = Field(..., description="One or several texts (or one list of texts) for token classification.")


class TokenClassificationResponse(APIResponse):
    """Token classification response"""
    object: Literal["list"] = "list"
    data: List[TokenClassificationData]
    usage: Optional[Usage] = None


class VideoClassificationData(BaseModel):
    """Data model for video classification result"""
    label: str = Field(..., description="The label identified by the model")
    score: float = Field(..., description="The score attributed by the model for that label")


class VideoClassificationRequest(BaseModel):
    """Request model for video classification"""
    model: str
    inputs: Union[str, List[str]] = Field(..., description="A http link to a video or a local path to a video. Accepts single video or a batch.")
    top_k: Optional[int] = Field(default=5, description="The number of top labels that will be returned by the pipeline.")
    num_frames: Optional[int] = Field(default=16, description="The number of frames sampled from the video to run the classification on.")
    frame_sampling_rate: Optional[int] = Field(default=1, description="The sampling rate used to select frames from the video.")
    function_to_apply: Optional[Literal['sigmoid','softmax','none']] = Field(default='softmax', description="The function to apply to the model output. Valid: 'softmax', 'sigmoid', 'none'.")
    model_config = ConfigDict(arbitrary_types_allowed=True)

class VideoClassificationResponse(APIResponse):
    """Video classification response"""
    object: Literal["list"] = "list"
    data: List[VideoClassificationData]
    usage: Optional[Usage] = None



# ================================
# Zero-Shot Classifications
# ================================

class ZeroShotAudioClassificationData(BaseModel):
    """Data model for zero-shot audio classification result"""
    label: str = Field(..., description="The label predicted for the audio")
    score: float = Field(..., description="The corresponding probability")


class ZeroShotAudioClassificationRequest(BaseModel):
    """Request model for zero-shot audio classification"""
    model: str
    audios: Union[ndarray, List[ndarray]] = Field(..., description="An audio loaded in numpy. Accepts a single array or a list of arrays.")
    candidate_labels: List[str] = Field(..., description="The candidate labels for this audio.")
    hypothesis_template: Optional[str] = Field(default="This is a sound of {}", description="Template used with candidate labels.")
    model_config = ConfigDict(arbitrary_types_allowed=True)


class ZeroShotAudioClassificationResponse(APIResponse):
    """Zero-shot audio classification response"""
    object: Literal["list"] = "list"
    data: List[ZeroShotAudioClassificationData]
    usage: Optional[Usage] = None


class ZeroShotClassificationData(BaseModel):
    """Data model for zero-shot text classification result"""
    sequence: str = Field(..., description="The input sequence")
    labels: List[str] = Field(..., description="Candidate labels")
    scores: List[float] = Field(..., description="Scores aligned with labels")


class ZeroShotClassificationRequest(BaseModel):
    """Request model for zero-shot text classification"""
    model: str
    sequences: Union[str, List[str]] = Field(..., description="The sequence(s) to classify")
    candidate_labels: Union[str, List[str]] = Field(..., description="Possible class labels")
    hypothesis_template: Optional[str] = Field(default="This example is {}.", description="Template used with candidate labels")
    multi_label: Optional[bool] = Field(default=False, description="Whether multiple labels can be true")


class ZeroShotClassificationResponse(APIResponse):
    """Zero-shot text classification response"""
    object: Literal["list"] = "list"
    data: List[ZeroShotClassificationData]
    usage: Optional[Usage] = None


class ZeroShotImageClassificationData(BaseModel):
    """Data model for zero-shot image classification result"""
    label: str = Field(..., description="The label identified by the model")
    score: float = Field(..., description="The corresponding probability")


class ZeroShotImageClassificationRequest(BaseModel):
    """Request model for zero-shot image classification"""
    model: str
    image: Union[str, List[str], PilImage, List[PilImage]] = Field(..., description="An image or list of images to classify")
    candidate_labels: List[str] = Field(..., description="The candidate labels for this image")
    hypothesis_template: Optional[str] = Field(default="This is a photo of {}", description="Template used with candidate labels")
    timeout: Optional[float] = Field(default=None, description="Timeout for fetching images from the web")
    model_config = ConfigDict(arbitrary_types_allowed=True)


class ZeroShotImageClassificationResponse(APIResponse):
    """Zero-shot image classification response"""
    object: Literal["list"] = "list"
    data: List[ZeroShotImageClassificationData]
    usage: Optional[Usage] = None


class ZeroShotObjectDetectionData(BaseModel):
    """Data model for zero-shot object detection result"""
    bbox: Dict[str, int] = Field(..., description="Bounding box in corners format")


class ZeroShotObjectDetectionRequest(BaseModel):
    """Request model for zero-shot object detection"""
    model: str
    box: Any = Field(..., description="Tensor or list containing the coordinates in corners format")


class ZeroShotObjectDetectionResponse(APIResponse):
    """Zero-shot object detection response"""
    object: Literal["list"] = "list"
    data: List[ZeroShotObjectDetectionData]
    usage: Optional[Usage] = None

