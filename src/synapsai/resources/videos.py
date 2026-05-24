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
Video resource handlers
"""

import asyncio
import base64
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, AsyncIterator, Iterator, Optional, Union, Tuple
from urllib.parse import urlencode

from ..processing import process_image_input

from ..exceptions import APIError
from ..types.videos import (
    Video,
    VideoContentVariant,
    VideoDeleteResponse,
    VideoInputReference,
    VideoStatus,
)

if TYPE_CHECKING:
    from ..client import AsyncSynapsAI, SynapsAI


class VideoContentResponse:
    """Streaming binary response for video content downloads."""

    def __init__(self, response, content_type: str):
        self._response = response
        self.content_type = content_type

    def iter_bytes(self, chunk_size: int = 8192) -> Iterator[bytes]:
        for chunk in self._response.iter_bytes(chunk_size=chunk_size):
            if chunk:
                yield chunk

    def write_to_file(self, file_path) -> None:
        if hasattr(file_path, "write"):
            for chunk in self.iter_bytes():
                file_path.write(chunk)
        else:
            with Path(file_path).open("wb") as f:
                for chunk in self.iter_bytes():
                    f.write(chunk)

    def read(self) -> bytes:
        return b"".join(self.iter_bytes())

    def __enter__(self) -> "VideoContentResponse":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        close = getattr(self._response, "close", None)
        if callable(close):
            close()


class AsyncVideoContentResponse:
    """Async streaming binary response for video content downloads."""

    def __init__(self, response, content_type: str):
        self._response = response
        self.content_type = content_type

    async def iter_bytes(self, chunk_size: int = 8192) -> AsyncIterator[bytes]:
        async for chunk in self._response.aiter_bytes(chunk_size=chunk_size):
            if chunk:
                yield chunk

    async def write_to_file(self, file_path) -> None:
        if hasattr(file_path, "write"):
            async for chunk in self.iter_bytes():
                file_path.write(chunk)
        else:
            with Path(file_path).open("wb") as f:
                async for chunk in self.iter_bytes():
                    f.write(chunk)

    async def read(self) -> bytes:
        return b"".join([chunk async for chunk in self.iter_bytes()])

    async def __aenter__(self) -> "AsyncVideoContentResponse":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        aclose = getattr(self._response, "aclose", None)
        if callable(aclose):
            await aclose()


class VideosResource:
    """Video generation resource"""

    TERMINAL_STATUSES = {VideoStatus.COMPLETED.value, VideoStatus.FAILED.value}

    def __init__(self, client: "SynapsAI"):
        self._client = client

    def _build_content_endpoint(
        self,
        video_id: str,
        variant: Optional[Union[str, VideoContentVariant]] = None,
    ) -> str:
        endpoint = f"videos/{video_id}/content"
        if variant is None:
            return endpoint
        query = urlencode({"variant": getattr(variant, "value", variant)})
        return f"{endpoint}?{query}"

    def create(
        self,
        model: str,
        prompt: str,
        input_reference: Optional[VideoInputReference] = None,
        seconds: int = 4,
        size: Optional[Union[str, Tuple[int, int]]] = "1280x720",
        fps: int = 16,
        num_inference_steps: int = 25,
        guidance_scale: float = 5.0,
        **kwargs,
    ) -> Video:
        request_data = self._client._build_request(
            prompt=prompt,
            input_reference=process_image_input(input_reference),
            model=model,
            seconds=seconds,
            size=size,
            fps=fps,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            **kwargs,
        )
        response = self._client._post("videos", json_data=request_data)
        return Video.model_validate(response.json())

    def retrieve(self, video_id: str) -> Video:
        response = self._client._get(f"videos/{video_id}")
        return Video.model_validate(response.json())

    def delete(self, video_id: str) -> VideoDeleteResponse:
        response = self._client._delete(f"videos/{video_id}")
        return VideoDeleteResponse.model_validate(response.json())

    def download_content(
        self,
        video_id: str,
        variant: Optional[Union[str, VideoContentVariant]] = None,
    ) -> VideoContentResponse:
        endpoint = self._build_content_endpoint(video_id=video_id, variant=variant)
        response = self._client._get_stream(endpoint)
        content_type = response.headers.get("content-type", "application/octet-stream")
        return VideoContentResponse(response, content_type=content_type)

    def create_and_poll(
        self,
        model: str,
        prompt: str,
        input_reference: Optional[VideoInputReference] = None,
        seconds: int = 4,
        size: Optional[Union[str, Tuple[int, int]]] = "1280x720",
        poll_interval: float = 2.0,
        timeout: Optional[float] = None,
        fps: int = 16,
        num_inference_steps: int = 25,
        guidance_scale: float = 5.0,
        **kwargs,
    ) -> Video:
        video = self.create(
            prompt=prompt,
            input_reference=input_reference,
            model=model,
            seconds=seconds,
            size=size,
            fps=fps,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            **kwargs,
        )
        started_at = time.monotonic()

        while getattr(video.status, "value", video.status) not in self.TERMINAL_STATUSES:
            if timeout is not None and time.monotonic() - started_at >= timeout:
                raise APIError(f"Timed out while waiting for video '{video.id}' to complete")
            time.sleep(poll_interval)
            video = self.retrieve(video.id)

        return video


class AsyncVideosResource:
    """Async video generation resource"""

    TERMINAL_STATUSES = {VideoStatus.COMPLETED.value, VideoStatus.FAILED.value}

    def __init__(self, client: "AsyncSynapsAI"):
        self._client = client

    def _build_content_endpoint(
        self,
        video_id: str,
        variant: Optional[Union[str, VideoContentVariant]] = None,
    ) -> str:
        endpoint = f"videos/{video_id}/content"
        if variant is None:
            return endpoint
        query = urlencode({"variant": getattr(variant, "value", variant)})
        return f"{endpoint}?{query}"

    async def create(
        self,
        model: str,
        prompt: str,
        input_reference: Optional[VideoInputReference] = None,
        seconds: Optional[int] = 4,
        size: Optional[Union[str, Tuple[int, int]]] = "1280x720",
        fps: int = 16,
        num_inference_steps: int = 25,
        guidance_scale: float = 5.0,
        **kwargs,
    ) -> Video:
        request_data = self._client._build_request(
            prompt=prompt,
            input_reference=process_image_input(input_reference),
            model=model,
            seconds=seconds,
            size=size,
            fps=fps,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            **kwargs,
        )
        response = await self._client._post("videos", json_data=request_data)
        return Video.model_validate(response.json())

    async def retrieve(self, video_id: str) -> Video:
        response = await self._client._get(f"videos/{video_id}")
        return Video.model_validate(response.json())

    async def delete(self, video_id: str) -> VideoDeleteResponse:
        response = await self._client._delete(f"videos/{video_id}")
        return VideoDeleteResponse.model_validate(response.json())

    async def download_content(
        self,
        video_id: str,
        variant: Optional[Union[str, VideoContentVariant]] = None,
    ) -> AsyncVideoContentResponse:
        endpoint = self._build_content_endpoint(video_id=video_id, variant=variant)
        response = await self._client._get_stream(endpoint)
        content_type = response.headers.get("content-type", "application/octet-stream")
        return AsyncVideoContentResponse(response, content_type=content_type)

    async def create_and_poll(
        self,
        model: str,
        prompt: str,
        input_reference: Optional[VideoInputReference] = None,
        seconds: Optional[int] = 4,
        size: Optional[Union[str, Tuple[int, int]]] = "1280x720",
        fps: int = 16,
        num_inference_steps: int = 25,
        guidance_scale: float = 5.0,
        poll_interval: float = 2.0,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Video:
        video = await self.create(
            prompt=prompt,
            input_reference=input_reference,
            model=model,
            seconds=seconds,
            size=size,
            fps=fps,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            **kwargs,
        )
        started_at = time.monotonic()

        while getattr(video.status, "value", video.status) not in self.TERMINAL_STATUSES:
            if timeout is not None and time.monotonic() - started_at >= timeout:
                raise APIError(f"Timed out while waiting for video '{video.id}' to complete")
            await asyncio.sleep(poll_interval)
            video = await self.retrieve(video.id)

        return video