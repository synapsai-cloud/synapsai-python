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
Video generation type definitions
"""

from enum import Enum
from typing import Literal, Optional, Union
from PIL import Image

from pydantic import BaseModel, Field

from .common import APIResponse


VideoInputReference = Union[str, bytes]


class VideoStatus(str, Enum):
    """Video generation lifecycle states"""

    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class VideoContentVariant(str, Enum):
    """Downloadable video asset variants"""

    VIDEO = "video"
    THUMBNAIL = "thumbnail"
    SPRITESHEET = "spritesheet"


class VideoCreateError(BaseModel):
    """Video generation error details"""

    code: Optional[str] = None
    message: Optional[str] = None


class VideoCreateRequest(BaseModel):
    """Video creation request"""

    model: str
    prompt: str = Field(description="Text prompt that describes the video to generate")
    input_reference: Optional[VideoInputReference] = Field(
        default=None,
        description="Optional reference video or asset as base64, URL, bytes, or file path",
    )
    seconds: int = 4
    size: str = "720x1280"
    num_inference_steps: int = 25
    guidance_scale: float = 5.0
    fps: int = 16


class Video(APIResponse):
    """Structured information describing a generated video job"""

    id: str
    object: Literal["video"] = "video"
    created_at: int
    completed_at: Optional[int] = None
    error: Optional[VideoCreateError] = None
    expires_at: Optional[int] = None
    model: str
    prompt: Optional[str] = None
    seconds: int
    size: str
    status: VideoStatus


class VideoDeleteResponse(APIResponse):
    """Confirmation payload returned after deleting a video"""

    id: str
    deleted: bool
    object: Literal["video.deleted"] = "video.deleted"
