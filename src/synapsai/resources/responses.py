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

"""Responses API resource handlers."""

from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, Iterator, List, Optional, Union

from ..exceptions import APIError
from ..logging import get_logger
from ..types.responses import Response, ResponseDeleted, ResponseInput, ResponseStreamEvent

if TYPE_CHECKING:
    from ..client import AsyncSynapsAI, SynapsAI

logger = get_logger(__name__)


class ResponsesResource:
    """
    Responses API (`/v1/responses`).

    Supported: ``create``, ``retrieve``, ``delete``.
    Not offered: cancel, list input items, or count input tokens.
    """

    def __init__(self, client: "SynapsAI"):
        self._client = client

    def create(
        self,
        *,
        model: str,
        input: ResponseInput,
        stream: bool = False,
        store: Optional[bool] = None,
        previous_response_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        instructions: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        tools: Optional[List[Any]] = None,
        tool_choice: Optional[Any] = None,
        **kwargs,
    ) -> Union[Response, Iterator[ResponseStreamEvent]]:
        """
        Create a model response (``POST /v1/responses``).

        Set ``store=True`` (when the model enables data store) to persist the
        result for later ``retrieve`` / ``previous_response_id`` chaining.
        """
        request_data = self._client._build_request(
            model=model,
            input=input,
            stream=stream,
            store=store,
            previous_response_id=previous_response_id,
            metadata=metadata,
            instructions=instructions,
            temperature=temperature,
            top_p=top_p,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs,
        )
        if stream:
            return self._stream(request_data)
        response = self._client._post("responses", json_data=request_data)
        return Response.model_validate(response.json())

    def retrieve(self, response_id: str) -> Response:
        """GET /v1/responses/{response_id} — fetch a stored response."""
        response = self._client._get(f"responses/{response_id}")
        return Response.model_validate(response.json())

    def delete(self, response_id: str) -> ResponseDeleted:
        """DELETE /v1/responses/{response_id} — delete a stored response."""
        response = self._client._delete(f"responses/{response_id}")
        return ResponseDeleted.model_validate(response.json())

    def _stream(self, request_data: Dict[str, Any]) -> Iterator[ResponseStreamEvent]:
        for event_data in self._client._stream_response("responses", json_data=request_data):
            try:
                error = event_data.get("error") if isinstance(event_data, dict) else None
                if error and not event_data.get("type"):
                    raise APIError(error if isinstance(error, str) else str(error))
                yield ResponseStreamEvent.model_validate(event_data)
            except APIError:
                raise
            except Exception:
                logger.warning(
                    "Failed to parse ResponseStreamEvent",
                    exc_info=True,
                    extra={"endpoint": "responses"},
                )
                continue


class AsyncResponsesResource:
    """
    Async Responses API.

    Supported: ``create``, ``retrieve``, ``delete``.
    Not offered: cancel, list input items, or count input tokens.
    """

    def __init__(self, client: "AsyncSynapsAI"):
        self._client = client

    async def create(
        self,
        *,
        model: str,
        input: ResponseInput,
        stream: bool = False,
        store: Optional[bool] = None,
        previous_response_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        instructions: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        tools: Optional[List[Any]] = None,
        tool_choice: Optional[Any] = None,
        **kwargs,
    ) -> Union[Response, AsyncIterator[ResponseStreamEvent]]:
        """Create a model response (``POST /v1/responses``)."""
        request_data = self._client._build_request(
            model=model,
            input=input,
            stream=stream,
            store=store,
            previous_response_id=previous_response_id,
            metadata=metadata,
            instructions=instructions,
            temperature=temperature,
            top_p=top_p,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs,
        )
        if stream:
            return self._stream(request_data)
        response = await self._client._post("responses", json_data=request_data)
        return Response.model_validate(response.json())

    async def retrieve(self, response_id: str) -> Response:
        response = await self._client._get(f"responses/{response_id}")
        return Response.model_validate(response.json())

    async def delete(self, response_id: str) -> ResponseDeleted:
        response = await self._client._delete(f"responses/{response_id}")
        return ResponseDeleted.model_validate(response.json())

    async def _stream(self, request_data: Dict[str, Any]) -> AsyncIterator[ResponseStreamEvent]:
        async for event_data in self._client._stream_response("responses", json_data=request_data):
            try:
                error = event_data.get("error") if isinstance(event_data, dict) else None
                if error and not event_data.get("type"):
                    raise APIError(error if isinstance(error, str) else str(error))
                yield ResponseStreamEvent.model_validate(event_data)
            except APIError:
                raise
            except Exception:
                logger.warning(
                    "Failed to parse ResponseStreamEvent",
                    exc_info=True,
                    extra={"endpoint": "responses"},
                )
                continue
