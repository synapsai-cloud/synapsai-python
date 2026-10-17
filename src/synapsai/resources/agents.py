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
from ..types.agents import AgentEvent, AgentMessage, AgentRun, DeletedAgent, DeletedAgentRun

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


def _parse_agent_event(event_data: Any, *, agent_id: str) -> Optional[AgentEvent]:
    try:
        error = event_data.get("error") if isinstance(event_data, dict) else None
        if error and not event_data.get("type"):
            raise APIError(error if isinstance(error, str) else str(error))
        return AgentEvent.model_validate(event_data)
    except APIError:
        raise
    except Exception:
        logger.warning(
            "Failed to parse agent event",
            exc_info=True,
            extra={"agent_id": agent_id},
        )
        return None


def _resume_params(after: int) -> Dict[str, int]:
    if after < 0:
        raise ValueError("after must be >= 0")
    return {"after": after}


class AgentRunsResource:
    """Get, resume, and delete one agent run."""

    def __init__(self, client: "SynapsAI"):
        self._client = client

    def retrieve(self, agent_id: str, run_id: str) -> AgentRun:
        """Return status and events for ``GET /v1/agent/{agent_id}/runs/{run_id}``."""
        response = self._client._get(f"agent/{agent_id}/runs/{run_id}")
        return AgentRun.model_validate(response.json())

    def delete(self, agent_id: str, run_id: str) -> DeletedAgentRun:
        """Stop a live run and delete it. ``DELETE /v1/agent/{agent_id}/runs/{run_id}``."""
        response = self._client._delete(f"agent/{agent_id}/runs/{run_id}")
        return DeletedAgentRun.model_validate(response.json())

    def stream(
        self,
        agent_id: str,
        run_id: str,
        *,
        after: int = 0,
    ) -> Iterator[AgentEvent]:
        """
        Resume ``GET /v1/agent/{agent_id}/runs/{run_id}/stream``.

        ``after`` is the last ``seq`` already applied. A finished stored run
        replays and the iterator ends. With store requests off, a finished run
        is no longer available.
        """
        for event_data in self._client._stream_response(
            f"agent/{agent_id}/runs/{run_id}/stream",
            params=_resume_params(after),
            method="GET",
        ):
            event = _parse_agent_event(event_data, agent_id=agent_id)
            if event is not None:
                yield event


class AgentsResource:
    """Run persisted SynapsAI agents via the AG-UI SSE endpoint."""

    def __init__(self, client: "SynapsAI"):
        self._client = client
        self.runs = AgentRunsResource(client)

    def delete(self, agent_id: str) -> DeletedAgent:
        """Delete a saved agent and its runs. ``DELETE /v1/agent/{agent_id}``."""
        response = self._client._delete(f"agent/{agent_id}")
        return DeletedAgent.model_validate(response.json())

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
            event = _parse_agent_event(event_data, agent_id=agent_id)
            if event is not None:
                yield event


class AsyncAgentRunsResource:
    """Async get, resume, and delete for one agent run."""

    def __init__(self, client: "AsyncSynapsAI"):
        self._client = client

    async def retrieve(self, agent_id: str, run_id: str) -> AgentRun:
        response = await self._client._get(f"agent/{agent_id}/runs/{run_id}")
        return AgentRun.model_validate(response.json())

    async def delete(self, agent_id: str, run_id: str) -> DeletedAgentRun:
        response = await self._client._delete(f"agent/{agent_id}/runs/{run_id}")
        return DeletedAgentRun.model_validate(response.json())

    async def stream(
        self,
        agent_id: str,
        run_id: str,
        *,
        after: int = 0,
    ) -> AsyncIterator[AgentEvent]:
        async for event_data in self._client._stream_response(
            f"agent/{agent_id}/runs/{run_id}/stream",
            params=_resume_params(after),
            method="GET",
        ):
            event = _parse_agent_event(event_data, agent_id=agent_id)
            if event is not None:
                yield event


class AsyncAgentsResource:
    """Async run of persisted SynapsAI agents via the AG-UI SSE endpoint."""

    def __init__(self, client: "AsyncSynapsAI"):
        self._client = client
        self.runs = AsyncAgentRunsResource(client)

    async def delete(self, agent_id: str) -> DeletedAgent:
        response = await self._client._delete(f"agent/{agent_id}")
        return DeletedAgent.model_validate(response.json())

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
            event = _parse_agent_event(event_data, agent_id=agent_id)
            if event is not None:
                yield event
