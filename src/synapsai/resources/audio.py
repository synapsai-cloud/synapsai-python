"""
Audio resource handlers
"""

from typing import Union, Iterator, AsyncIterator, List, TYPE_CHECKING, Literal
import base64
import os

from ..types.audio import (
    AudioSpeechResponse,
    AudioTranscriptionResponse,
    Voice,
    AudioFormat,
    TimestampGranularity,
)
from ..exceptions import APIError

if TYPE_CHECKING:
    from ..client import SynapsAI, AsyncSynapsAI


class SpeechResource:
    """Speech synthesis resource"""
    
    def __init__(self, client: "SynapsAI"):
        self._client = client
    
    def create(
        self,
        model: str,
        input: str,
        voice: Union[str, Voice],
        response_format: Union[str, AudioFormat] = "mp3",
        speed: float = 1.0,
        stream: bool = False,
        **kwargs
    ) -> Union[AudioSpeechResponse, Iterator[bytes]]:
        """Generate speech from text"""
        
        # Build request
        request_data = self._client._build_request(
            endpoint="/audio/speech",
            model=model,
            input=input,
            voice=voice,
            response_format=response_format,
            speed=speed,
            stream=stream,
            **kwargs
        )
        
        # Make request
        response = self._client._post(
            "/inference",
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
    
    def _stream_audio(self, response) -> Iterator[bytes]:
        """Stream audio chunks"""
        for chunk in response.iter_bytes(chunk_size=8192):
            if chunk:
                yield chunk


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
        file_data = self._process_audio_input(file)
        
        # Build request
        request_data = self._client._build_request(
            endpoint="/audio/transcriptions",
            model=model,
            file=file_data,
            language=language,
            prompt=prompt,
            response_format=response_format,
            temperature=temperature,
            timestamp_granularities=timestamp_granularities or ["segment"],
            **kwargs
        )
        
        # Make request
        response = self._client._post("/inference", json_data=request_data)
        response_data = response.json()
        return AudioTranscriptionResponse(**response_data)
    
    def _process_audio_input(self, file):
        """Process audio file input"""
        if isinstance(file, str):
            # Check if it's a file path
            if os.path.isfile(file):
                with open(file, "rb") as f:
                    audio_bytes = f.read()
                return base64.b64encode(audio_bytes).decode("utf-8")
            # Assume it's already base64
            return file
        elif isinstance(file, bytes):
            return base64.b64encode(file).decode("utf-8")
        else:
            raise ValueError("File must be a file path, bytes, or base64 string")


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
            endpoint="/audio/speech",
            model=model,
            input=input,
            voice=voice,
            response_format=response_format,
            speed=speed,
            stream=stream,
            **kwargs
        )
        
        # Make request
        response = await self._client._post(
            "/inference",
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
        file_data = self._process_audio_input(file)
        
        # Build request
        request_data = self._client._build_request(
            endpoint="/audio/transcriptions",
            model=model,
            file=file_data,
            language=language,
            prompt=prompt,
            response_format=response_format,
            temperature=temperature,
            timestamp_granularities=timestamp_granularities or ["segment"],
            **kwargs
        )
        
        # Make request
        response = await self._client._post("/inference", json_data=request_data)
        response_data = response.json()
        return AudioTranscriptionResponse(**response_data)
    
    def _process_audio_input(self, file):
        """Process audio file input"""
        if isinstance(file, str):
            # Check if it's a file path
            if os.path.isfile(file):
                with open(file, "rb") as f:
                    audio_bytes = f.read()
                return base64.b64encode(audio_bytes).decode("utf-8")
            # Assume it's already base64
            return file
        elif isinstance(file, bytes):
            return base64.b64encode(file).decode("utf-8")
        else:
            raise ValueError("File must be a file path, bytes, or base64 string")


class AsyncAudioResource:
    """Async audio resource handler"""
    
    def __init__(self, client: "AsyncSynapsAI"):
        self._client = client
        self.speech = AsyncSpeechResource(client)
        self.transcriptions = AsyncTranscriptionsResource(client) 