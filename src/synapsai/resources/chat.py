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
Chat completion resource handlers
"""

from typing import Any, Dict, Iterator, AsyncIterator, List, Literal, Optional, TYPE_CHECKING, Union

from ..types.completion import (
    ChatCompletionChunk,
    ChatCompletionDeleted,
    ChatCompletionList,
    ChatCompletionResponse,
)
from ..logging import get_logger
from ..exceptions import APIError

if TYPE_CHECKING:
    from ..client import SynapsAI, AsyncSynapsAI

logger = get_logger(__name__)


class ChatCompletionsResource:
    """Chat completions resource (create + stored CRUD)."""

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
        stop=None,
        max_completion_tokens=128,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        logit_bias=None,
        functions=None,
        function_call=None,
        tools=None,
        tool_choice=None,
        response_format=None,
        seed=None,
        reasoning_effort: Optional[
            Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]
        ] = None,
        store: Optional[bool] = None,
        previous_completion_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Union[ChatCompletionResponse, Iterator[ChatCompletionChunk]]:
        """
        Create a chat completion.

        When the model has data store enabled, omit ``store`` or set ``store=True``
        to persist the completion. Use ``previous_completion_id`` to continue a
        stored conversation chain.
        """
        request_data = self._client._build_request(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            n=n,
            stream=stream,
            stop=stop if stop is not None else [],
            max_completion_tokens=max_completion_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            functions=functions,
            function_call=function_call,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            seed=seed,
            reasoning_effort=reasoning_effort,
            store=store,
            previous_completion_id=previous_completion_id,
            metadata=metadata,
            **kwargs,
        )

        endpoint = "chat/completions"

        if stream:
            return self._stream_completions(endpoint, request_data)
        response = self._client._post(endpoint, json_data=request_data)
        return ChatCompletionResponse.model_validate(response.json())

    def list(
        self,
        *,
        model: Optional[str] = None,
        after: Optional[str] = None,
        limit: int = 20,
        order: Literal["asc", "desc"] = "desc",
    ) -> ChatCompletionList:
        """GET /v1/chat/completions — list stored chat completions."""
        params: Dict[str, Any] = {"limit": limit, "order": order}
        if model is not None:
            params["model"] = model
        if after is not None:
            params["after"] = after
        response = self._client._get("chat/completions", params=params)
        return ChatCompletionList.model_validate(response.json())

    def retrieve(self, completion_id: str) -> ChatCompletionResponse:
        """GET /v1/chat/completions/{completion_id}."""
        response = self._client._get(f"chat/completions/{completion_id}")
        return ChatCompletionResponse.model_validate(response.json())

    def update(
        self,
        completion_id: str,
        *,
        metadata: Dict[str, Any],
    ) -> ChatCompletionResponse:
        """POST /v1/chat/completions/{completion_id} — update stored metadata."""
        response = self._client._post(
            f"chat/completions/{completion_id}",
            json_data={"metadata": metadata},
        )
        return ChatCompletionResponse.model_validate(response.json())

    def delete(self, completion_id: str) -> ChatCompletionDeleted:
        """DELETE /v1/chat/completions/{completion_id}."""
        response = self._client._delete(f"chat/completions/{completion_id}")
        return ChatCompletionDeleted.model_validate(response.json())

    def _stream_completions(self, endpoint, request_data) -> Iterator[ChatCompletionChunk]:
        for chunk_data in self._client._stream_response(endpoint, json_data=request_data):
            try:
                error = chunk_data.get("error")
                if error:
                    raise APIError(error)
                yield ChatCompletionChunk.model_validate(chunk_data)
            except APIError:
                raise
            except Exception:
                logger.warning(
                    "Failed to parse ChatCompletionChunk",
                    exc_info=True,
                    extra={"endpoint": endpoint},
                )
                continue


class ChatResource:
    """Chat resource handler"""

    def __init__(self, client: "SynapsAI"):
        self._client = client
        self.completions = ChatCompletionsResource(client)


class AsyncChatCompletionsResource:
    """Async chat completions resource (create + stored CRUD)."""

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
        stop=None,
        max_completion_tokens=128,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        logit_bias=None,
        functions=None,
        function_call=None,
        tools=None,
        tool_choice=None,
        response_format=None,
        seed=None,
        reasoning_effort: Optional[
            Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]
        ] = None,
        store: Optional[bool] = None,
        previous_completion_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Union[ChatCompletionResponse, AsyncIterator[ChatCompletionChunk]]:
        request_data = self._client._build_request(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            n=n,
            stream=stream,
            stop=stop if stop is not None else [],
            max_completion_tokens=max_completion_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            functions=functions,
            function_call=function_call,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            seed=seed,
            reasoning_effort=reasoning_effort,
            store=store,
            previous_completion_id=previous_completion_id,
            metadata=metadata,
            **kwargs,
        )

        endpoint = "chat/completions"

        if stream:
            return self._stream_completions(endpoint, request_data)
        response = await self._client._post(endpoint, json_data=request_data)
        return ChatCompletionResponse.model_validate(response.json())

    async def list(
        self,
        *,
        model: Optional[str] = None,
        after: Optional[str] = None,
        limit: int = 20,
        order: Literal["asc", "desc"] = "desc",
    ) -> ChatCompletionList:
        params: Dict[str, Any] = {"limit": limit, "order": order}
        if model is not None:
            params["model"] = model
        if after is not None:
            params["after"] = after
        response = await self._client._get("chat/completions", params=params)
        return ChatCompletionList.model_validate(response.json())

    async def retrieve(self, completion_id: str) -> ChatCompletionResponse:
        response = await self._client._get(f"chat/completions/{completion_id}")
        return ChatCompletionResponse.model_validate(response.json())

    async def update(
        self,
        completion_id: str,
        *,
        metadata: Dict[str, Any],
    ) -> ChatCompletionResponse:
        response = await self._client._post(
            f"chat/completions/{completion_id}",
            json_data={"metadata": metadata},
        )
        return ChatCompletionResponse.model_validate(response.json())

    async def delete(self, completion_id: str) -> ChatCompletionDeleted:
        response = await self._client._delete(f"chat/completions/{completion_id}")
        return ChatCompletionDeleted.model_validate(response.json())

    async def _stream_completions(self, endpoint, request_data) -> AsyncIterator[ChatCompletionChunk]:
        async for chunk_data in self._client._stream_response(endpoint, json_data=request_data):
            try:
                error = chunk_data.get("error") if isinstance(chunk_data, dict) else None
                if error:
                    raise APIError(error)
                yield ChatCompletionChunk.model_validate(chunk_data)
            except APIError:
                raise
            except Exception:
                logger.warning(
                    "Failed to parse ChatCompletionChunk",
                    exc_info=True,
                    extra={"endpoint": endpoint},
                )
                continue


class AsyncChatResource:
    """Async chat resource handler"""

    def __init__(self, client: "AsyncSynapsAI"):
        self._client = client
        self.completions = AsyncChatCompletionsResource(client)
