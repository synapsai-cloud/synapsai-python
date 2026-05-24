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
Audio speech and transcription type definitions
"""

from typing import Optional, List, Literal, Union
from pydantic import BaseModel, Field
from enum import Enum

from .common import APIResponse, Usage


class AudioFormat(str, Enum):
    """Audio format options"""
    MP3 = "mp3"
    OPUS = "opus"
    AAC = "aac"
    FLAC = "flac"
    WAV = "wav"
    PCM = "pcm"


class AudioSpeechRequest(BaseModel):
    """Audio speech request"""
    model: str
    input: str = Field(max_length=4096)
    response_format: Optional[AudioFormat] = AudioFormat.MP3
    speed: Optional[float] = Field(default=1.0, ge=0.25, le=4.0)
    stream: Optional[bool] = False


class AudioSpeechResponse(BaseModel):
    """Audio speech response"""
    content: bytes = Field(description="Audio content as bytes")
    content_type: str = Field(description="MIME type of the audio")


class TimestampGranularity(str, Enum):
    """Timestamp granularity options"""
    WORD = "word"
    SEGMENT = "segment"
    CHAR = "char"


class Word(BaseModel):
    """Word-level timestamp"""
    word: str
    start: float
    end: float


class Segment(BaseModel):
    """Segment-level transcript"""
    id: int
    start: float
    end: float
    text: str
    temperature: float


class Char(BaseModel):
    """Character-level timestamp"""
    char: str
    start: float
    end: float


class AudioTranscriptionRequest(BaseModel):
    """Audio transcription request"""
    model: str
    file: str = Field(description="Audio file to transcribe (base64 encoded or file path)")
    language: Optional[str] = Field(default=None, description="ISO-639-1 language code")
    prompt: Optional[str] = Field(default=None, max_length=244)
    response_format: Literal["json", "text", "str", "verbose_json", "vtt"] = "json"
    temperature: Optional[float] = Field(default=0.0, ge=0.0, le=1.0)
    seed: Optional[int] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    n: Optional[int] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    max_completion_tokens: Optional[int] = None
    to_language: Optional[str] = None
    repetition_penalty: Optional[float] = None
    timestamp_granularities: Optional[List[TimestampGranularity]] = Field(
        default=[TimestampGranularity.SEGMENT]
    )
    stream: Optional[bool] = False


class Transcription(BaseModel):
    """Transcription object"""
    text: str


class AudioTranscriptionResponse(APIResponse):
    """Audio transcription response"""
    object: Literal["transcription"] = "transcription"
    text: str
    language: Optional[str] = None
    duration: Optional[float] = None
    words: Optional[List[Word]] = None
    segments: Optional[List[Segment]] = None
    chars: Optional[List[Char]] = None
    usage: Optional[dict] = None


class AudioTranslationRequest(BaseModel):
    """Audio translation request"""
    model: str
    file: str = Field(description="Audio file to translate (base64 encoded or file path)")
    language: Optional[str] = Field(default=None, description="ISO-639-1 language code")
    prompt: Optional[str] = Field(default=None, max_length=244)
    response_format: Literal["json", "text", "str", "verbose_json", "vtt"] = "json"
    temperature: Optional[float] = Field(default=0.0, ge=0.0, le=1.0)
    seed: Optional[int] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    n: Optional[int] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    max_completion_tokens: Optional[int] = None
    to_language: Optional[str] = None
    repetition_penalty: Optional[float] = None
    stream: Optional[bool] = False


class AudioTranslationResponse(APIResponse):
    """Audio translation response"""
    object: Literal["translation"] = "translation"
    text: str
    language: Optional[str] = None
    duration: Optional[float] = None
    segments: Optional[List[Segment]] = None
    usage: Optional[dict] = None


class Delta(BaseModel):
    """Delta object for streaming responses"""
    content: Optional[str] = None

class ChatCompletionChoice(BaseModel):
    """Choice object"""
    delta: Optional[Delta] = None
    logprobs: Optional[dict | list[dict]] = None

class ChatCompletionResponse(APIResponse):
    """Chat completion response"""
    model: str
    object: Literal["chat.completion"] = "chat.completion"
    choices: List[ChatCompletionChoice]
    system_fingerprint: Optional[str] = None
    usage: Optional[Usage] = None

class AudioTranscriptionChunk(APIResponse):
    """Chat completion chunk for streaming"""
    model: str
    object: Literal["transcription.chunk"] = "transcription.chunk"
    choices: List[ChatCompletionChoice]
    usage: Optional[Usage] = None

class AudioTranslationChunk(AudioTranscriptionChunk):
    """Streaming translation chunk"""
    object: Literal["translation.chunk"] = "translation.chunk"