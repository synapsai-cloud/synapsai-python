"""
Chat completion resource handlers
"""

from typing import Union, Iterator, AsyncIterator, TYPE_CHECKING

from ..types.completion import (
    ChatCompletionResponse,
    ChatCompletionChunk,
)
from ..exceptions import APIError

if TYPE_CHECKING:
    from ..client import SynapsAI, AsyncSynapsAI


class ChatCompletionsResource:
    """Chat completions resource"""
    
    def __init__(self, client: "SynapsAI"):
        self._client = client
    
    def create(
        self,
        model: str,
        messages: list,
        temperature: float = 1.0,
        top_p: float = 1.0,
        n: int = 1,
        stream: bool = False,
        stop = None,
        max_tokens = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        logit_bias = None,
        functions = None,
        function_call = None,
        tools = None,
        tool_choice = None,
        response_format = None,
        seed = None,
        **kwargs
    ) -> Union[ChatCompletionResponse, Iterator[ChatCompletionChunk]]:
        """Create a chat completion"""
        
        # Build request
        request_data = self._client._build_request(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            n=n,
            stream=stream,
            stop=stop,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            functions=functions,
            function_call=function_call,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            seed=seed,
            **kwargs
        )

        endpoint = "chat/completions"
        
        if stream:
            return self._stream_completions(endpoint, request_data)
        else:
            response = self._client._post(endpoint, request_data)
            return ChatCompletionResponse.model_validate(response.json())

    def _stream_completions(self, endpoint, request_data) -> Iterator[ChatCompletionChunk]:
        for chunk_data in self._client._stream_response(endpoint, request_data):
            try:                
                yield ChatCompletionChunk(**chunk_data)
            except Exception as e:
                print(f"Warning: Error creating CompletionChunk from data: {e}")
                print(f"Chunk data: {chunk_data}")
                continue


class ChatResource:
    """Chat resource handler"""
    
    def __init__(self, client: "SynapsAI"):
        self._client = client
        self.completions = ChatCompletionsResource(client)


class AsyncChatCompletionsResource:
    """Async chat completions resource"""
    
    def __init__(self, client: "AsyncSynapsAI"):
        self._client = client
    
    async def create(
        self,
        model: str,
        messages: list,
        temperature: float = 1.0,
        top_p: float = 1.0,
        n: int = 1,
        stream: bool = False,
        stop = None,
        max_tokens = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        logit_bias = None,
        functions = None,
        function_call = None,
        tools = None,
        tool_choice = None,
        response_format = None,
        seed = None,
        **kwargs
    ) -> Union[ChatCompletionResponse, AsyncIterator[ChatCompletionChunk]]:
        """Create a chat completion asynchronously"""
        
        # Build request
        request_data = self._client._build_request(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            n=n,
            stream=stream,
            stop=stop,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            functions=functions,
            function_call=function_call,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            seed=seed,
            **kwargs
        )
    
        endpoint = "chat/completions"

        if stream:
            return self._stream_completions(endpoint, request_data)
        else:
            response = await self._client._post(endpoint, request_data)
            return ChatCompletionResponse.model_validate(response.json())
    
    async def _stream_completions(self, endpoint, request_data) -> AsyncIterator[ChatCompletionChunk]:
        """Stream chat completion chunks asynchronously"""
        async for chunk_data in self._client._stream_response(endpoint, request_data):
            try:
                yield ChatCompletionChunk(**chunk_data)
            except Exception as e:
                print(f"Warning: Error creating ChatCompletionChunk from data: {e}")
                print(f"Chunk data: {chunk_data}")
                continue


class AsyncChatResource:
    """Async chat resource handler"""
    
    def __init__(self, client: "AsyncSynapsAI"):
        self._client = client
        self.completions = AsyncChatCompletionsResource(client) 