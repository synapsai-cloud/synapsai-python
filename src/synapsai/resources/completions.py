"""
Chat completion resource handlers
"""

from typing import Union, Iterator, AsyncIterator, TYPE_CHECKING, Optional, Dict
import json

from ..types.completion import (
    CompletionResponse,
    CompletionChunk,
)
from ..exceptions import APIError

if TYPE_CHECKING:
    from ..client import SynapsAI, AsyncSynapsAI


class CompletionsResource:
    """Chat completions resource"""
    
    def __init__(self, client: "SynapsAI"):
        self._client = client
    
    def create(
        self,
        model: str,
        prompt: str,
        temperature: float = 1.0,
        top_p: float = 1.0,
        n: int = 1,
        stream: bool = False,
        stop = [],
        max_completion_tokens = 128,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        logit_bias: Optional[Dict[int, float]] = None,
        **kwargs
    ) -> Union[CompletionResponse, Iterator[CompletionChunk]]:
        """Create a chat completion"""
        
        # Build request
        request_data = self._client._build_request(
            model=model,
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            n=n,
            stream=stream,
            stop=stop,
            max_completion_tokens=max_completion_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            **kwargs
        )
        
        endpoint = "/completions"
        
        if stream:
            return self._stream_completions(endpoint, request_data)
        else:
            response = self._client._post(endpoint, request_data)
            return CompletionResponse.model_validate(response.json())

    def _stream_completions(self, endpoint, request_data) -> Iterator[CompletionChunk]:
        for chunk_data in self._client._stream_response(endpoint, request_data):
            try:              
                yield CompletionChunk(**chunk_data)
            except Exception as e:
                print(f"Warning: Error creating CompletionChunk from data: {e}")
                print(f"Chunk data: {chunk_data}")
                continue




class AsyncCompletionsResource:
    """Async chat completions resource"""
    
    def __init__(self, client: "AsyncSynapsAI"):
        self._client = client
    
    async def create(
        self,
        model: str,
        prompt: str,
        temperature: float = 1.0,
        top_p: float = 1.0,
        n: int = 1,
        stream: bool = False,
        stop = [],
        max_completion_tokens = 128,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        logit_bias = None,
        **kwargs
    ) -> Union[CompletionResponse, AsyncIterator[CompletionChunk]]:
        """Create a chat completion asynchronously"""
        
        # Build request
        request_data = self._client._build_request(
            model=model,
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            n=n,
            stream=stream,
            stop=stop,
            max_completion_tokens=max_completion_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            **kwargs
        )

        endpoint = "/completions"
        
        # Make request
        if stream:
            return self._stream_completions(endpoint, request_data)
        else:
            response = await self._client._post(endpoint, request_data)
            return CompletionResponse.model_validate(response.json())
    
    async def _stream_completions(self, endpoint, request_data) -> AsyncIterator[CompletionChunk]:
        async for chunk_data in self._client._stream_response(endpoint, request_data):
            try:
                yield CompletionChunk(**chunk_data)
            except Exception as e:
                print(f"Warning: Error creating CompletionChunk from data: {e}")
                print(f"Chunk data: {chunk_data}")
                continue


