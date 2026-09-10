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

"""Agent run resource handlers (AG-UI protocol)."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, Iterator, List, Optional, Union

from ..exceptions import APIError
from ..logging import get_logger
from ..types.agents import AgentEvent, AgentMessage

if TYPE_CHECKING:
    from ..client import AsyncSynapsAI, SynapsAI

logger = get_logger(__name__)

MessageInput = Union[AgentMessage, Dict[str, Any], str]


def _normalize_messages(messages: List[MessageInput]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for message in messages:
        if isinstance(message, str):
            normalized.append(
                {
                    "id": str(uuid.uuid4()),
                    "role": "user",
                    "content": message,
                }
            )
            continue
        if isinstance(message, AgentMessage):
            payload = message.model_dump(by_alias=True, exclude_none=True)
        elif isinstance(message, dict):
            payload = AgentMessage.model_validate(message).model_dump(
                by_alias=True,
                exclude_none=True,
            )
        else:
            raise TypeError(f"Unsupported message type: {type(message)!r}")

        payload.setdefault("id", str(uuid.uuid4()))
        if "role" not in payload:
            raise ValueError("Each message must include a role")
        if "content" not in payload and payload.get("role") != "assistant":
            raise ValueError("Each message must include content")
        normalized.append(payload)
    return normalized


def _build_run_payload(
    *,
    thread_id: Optional[str],
    run_id: Optional[str],
    messages: List[MessageInput],
    tools: Optional[List[Any]] = None,
    context: Optional[List[Any]] = None,
    state: Optional[Any] = None,
    forwarded_props: Optional[Any] = None,
    parent_run_id: Optional[str] = None,
    resume: Optional[Any] = None,
    **kwargs,
) -> Dict[str, Any]:
    payload = {
        "threadId": thread_id or str(uuid.uuid4()),
        "runId": run_id or str(uuid.uuid4()),
        "messages": _normalize_messages(messages),
        "tools": tools or [],
        "context": context or [],
        "forwardedProps": forwarded_props if forwarded_props is not None else {},
    }
    if parent_run_id is not None:
        payload["parentRunId"] = parent_run_id
    if state is not None:
        payload["state"] = state
    if resume is not None:
        payload["resume"] = resume
    payload.update({k: v for k, v in kwargs.items() if v is not None})
    return payload


class AgentsResource:
    """Run persisted SynapsAI agents via the AG-UI SSE endpoint."""

    def __init__(self, client: "SynapsAI"):
        self._client = client

    def run(
        self,
        agent_id: str,
        *,
        messages: List[MessageInput],
        thread_id: Optional[str] = None,
        run_id: Optional[str] = None,
        tools: Optional[List[Any]] = None,
        context: Optional[List[Any]] = None,
        state: Optional[Any] = None,
        forwarded_props: Optional[Any] = None,
        parent_run_id: Optional[str] = None,
        resume: Optional[Any] = None,
        **kwargs,
    ) -> Iterator[AgentEvent]:
        """
        Stream AG-UI events from ``POST /v1/agent/{agent_id}/run``.

        Tools, MCP servers, and the system prompt always come from the persisted
        agent on the server; client-supplied ``tools`` are forwarded for AG-UI
        compatibility but do not widen server capabilities.
        """
        payload = _build_run_payload(
            thread_id=thread_id,
            run_id=run_id,
            messages=messages,
            tools=tools,
            context=context,
            state=state,
            forwarded_props=forwarded_props,
            parent_run_id=parent_run_id,
            resume=resume,
            **kwargs,
        )
        for event_data in self._client._stream_response(f"agent/{agent_id}/run", json_data=payload):
            try:
                error = event_data.get("error") if isinstance(event_data, dict) else None
                if error and not event_data.get("type"):
                    raise APIError(error if isinstance(error, str) else str(error))
                yield AgentEvent.model_validate(event_data)
            except APIError:
                raise
            except Exception:
                logger.warning(
                    "Failed to parse agent event",
                    exc_info=True,
                    extra={"agent_id": agent_id},
                )
                continue


class AsyncAgentsResource:
    """Async run of persisted SynapsAI agents via the AG-UI SSE endpoint."""

    def __init__(self, client: "AsyncSynapsAI"):
        self._client = client

    async def run(
        self,
        agent_id: str,
        *,
        messages: List[MessageInput],
        thread_id: Optional[str] = None,
        run_id: Optional[str] = None,
        tools: Optional[List[Any]] = None,
        context: Optional[List[Any]] = None,
        state: Optional[Any] = None,
        forwarded_props: Optional[Any] = None,
        parent_run_id: Optional[str] = None,
        resume: Optional[Any] = None,
        **kwargs,
    ) -> AsyncIterator[AgentEvent]:
        payload = _build_run_payload(
            thread_id=thread_id,
            run_id=run_id,
            messages=messages,
            tools=tools,
            context=context,
            state=state,
            forwarded_props=forwarded_props,
            parent_run_id=parent_run_id,
            resume=resume,
            **kwargs,
        )
        async for event_data in self._client._stream_response(
            f"agent/{agent_id}/run",
            json_data=payload,
        ):
            try:
                error = event_data.get("error") if isinstance(event_data, dict) else None
                if error and not event_data.get("type"):
                    raise APIError(error if isinstance(error, str) else str(error))
                yield AgentEvent.model_validate(event_data)
            except APIError:
                raise
            except Exception:
                logger.warning(
                    "Failed to parse agent event",
                    exc_info=True,
                    extra={"agent_id": agent_id},
                )
                continue
