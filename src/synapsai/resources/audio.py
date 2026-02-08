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
Audio resource handlers
"""

from typing import Union, Iterator, AsyncIterator, List, TYPE_CHECKING, Literal
from pathlib import Path

from ..types.audio import (
    AudioSpeechResponse,
    AudioTranscriptionResponse,
    Voice,
    AudioFormat,
    TimestampGranularity,
)
from ..processing import process_audio_input

if TYPE_CHECKING:
    from ..client import SynapsAI, AsyncSynapsAI


class _SpeechStreamingResponse:
    """Synchronous streaming response wrapper for speech audio."""

    def __init__(self, response):
        self._response = response

    def __enter__(self) -> "_SpeechStreamingResponse":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        close = getattr(self._response, "close", None)
        if callable(close):
            close()

    def __iter__(self) -> Iterator[bytes]:
        for chunk in self._response.iter_bytes(chunk_size=8192):
            if chunk:
                yield chunk

    def stream_to_file(self, file_path) -> None:
        """Stream audio to a file path or file-like object."""
        # Accept both Path-like and str for convenience
        if hasattr(file_path, "write"):
            # File-like object
            for chunk in self:
                file_path.write(chunk)
        else:
            path = Path(file_path)
            with path.open("wb") as f:
                for chunk in self:
                    f.write(chunk)


class _SpeechWithStreamingResponse:
    """Namespace that mirrors openai.audio.speech.with_streaming_response."""

    def __init__(self, speech_resource: "SpeechResource"):
        self._speech_resource = speech_resource

    def create(
        self,
        model: str,
        input: str,
        voice: Union[str, "Voice", None] = None,
        response_format: Union[str, AudioFormat] = "mp3",
        speed: float = 1.0,
        **kwargs,
    ) -> _SpeechStreamingResponse:
        """Create a streaming speech response."""

        # Build request (re-use SpeechResource's internal logic)
        request_data = self._speech_resource._client._build_request(
            model=model,
            input=input,
            voice=voice,
            response_format=response_format,
            speed=speed,
            stream_format="audio",
            **kwargs,
        )

        endpoint = "audio/speech"

        response = self._speech_resource._client._post(
            endpoint,
            json_data=request_data
        )

        return _SpeechStreamingResponse(response)


class SpeechResource:
    """Speech synthesis resource"""
    
    def __init__(self, client: "SynapsAI"):
        self._client = client
        self.with_streaming_response = _SpeechWithStreamingResponse(self)


class TranscriptionsResource:
    """Audio transcription resource"""
    
    def __init__(self, client: "SynapsAI"):
        self._client = client
    
    def create(
        self,
        model: str,
        file,
        language: str = None,
        prompt: str = None,
        response_format: Literal["json"] = "json",
        temperature: float = 0.0,
        timestamp_granularities: List[Union[str, TimestampGranularity]] = None,
        **kwargs
    ) -> AudioTranscriptionResponse:
        """Transcribe audio to text"""
        
        # Handle file input
        file_data = process_audio_input(file)
        
        # Build request
        request_data = self._client._build_request(
            model=model,
            file=file_data,
            language=language,
            prompt=prompt,
            response_format=response_format,
            temperature=temperature,
            timestamp_granularities=timestamp_granularities,
            **kwargs
        )

        endpoint = "audio/transcriptions"
        
        # Make request
        response = self._client._post(endpoint, json_data=request_data)
        response_data = response.json()
        return AudioTranscriptionResponse(**response_data)


class AudioResource:
    """Audio resource handler"""
    
    def __init__(self, client: "SynapsAI"):
        self._client = client
        self.speech = SpeechResource(client)
        self.transcriptions = TranscriptionsResource(client)


class AsyncSpeechResource:
    """Async speech synthesis resource"""
    
    def __init__(self, client: "AsyncSynapsAI"):
        self._client = client
    
    async def create(
        self,
        model: str,
        input: str,
        voice: Union[str, Voice],
        response_format: Union[str, AudioFormat] = "mp3",
        speed: float = 1.0,
        stream: bool = False,
        **kwargs
    ) -> Union[AudioSpeechResponse, AsyncIterator[bytes]]:
        """Generate speech from text asynchronously"""
        
        # Build request
        request_data = self._client._build_request(
            model=model,
            input=input,
            voice=voice,
            response_format=response_format,
            speed=speed,
            stream=stream,
            **kwargs
        )
        
        endpoint = "audio/speech"
        
        # Make request
        response = await self._client._post(
            endpoint,
            json_data=request_data,
            stream=stream
        )
        
        if stream:
            return self._stream_audio(response)
        else:
            content_type = response.headers.get("content-type", "audio/mpeg")
            return AudioSpeechResponse(
                content=response.content,
                content_type=content_type
            )
    
    async def _stream_audio(self, response) -> AsyncIterator[bytes]:
        """Stream audio chunks asynchronously"""
        async for chunk in response.aiter_bytes(chunk_size=8192):
            if chunk:
                yield chunk


class AsyncTranscriptionsResource:
    """Async audio transcription resource"""
    
    def __init__(self, client: "AsyncSynapsAI"):
        self._client = client
    
    async def create(
        self,
        model: str,
        file,
        language: str = None,
        prompt: str = None,
        response_format: Literal["json"] = "json",
        temperature: float = 0.0,
        timestamp_granularities: List[Union[str, TimestampGranularity]] = None,
        **kwargs
    ) -> AudioTranscriptionResponse:
        """Transcribe audio to text asynchronously"""
        
        # Handle file input
        file_data = process_audio_input(file)
        
        # Build request
        request_data = self._client._build_request(
            model=model,
            file=file_data,
            language=language,
            prompt=prompt,
            response_format=response_format,
            temperature=temperature,
            timestamp_granularities=timestamp_granularities,
            **kwargs
        )

        endpoint = "audio/transcriptions"
        
        # Make request
        response = await self._client._post(endpoint, json_data=request_data)
        response_data = response.json()
        return AudioTranscriptionResponse(**response_data)

class AsyncAudioResource:
    """Async audio resource handler"""
    
    def __init__(self, client: "AsyncSynapsAI"):
        self._client = client
        self.speech = AsyncSpeechResource(client)
        self.transcriptions = AsyncTranscriptionsResource(client) 