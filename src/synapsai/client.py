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
Main client classes for SynapsAI API
"""

import httpx
import json
import os
import time
import random
import asyncio
from typing import Optional, Dict, Any, Union, AsyncIterator, Iterator

from .resources import (
    ChatResource,
    AsyncChatResource,
    ImagesResource,
    AsyncImagesResource,
    VideosResource,
    AsyncVideosResource,
    EmbeddingsResource,
    AsyncEmbeddingsResource,
    AudioResource,
    AsyncAudioResource,
    CompletionsResource,
    AsyncCompletionsResource,
    ClassificationsResource,
    AsyncClassificationsResource,
    QuestionAnsweringResource,
    AsyncQuestionAnsweringResource,
    ModelsResource,
    AsyncModelsResource,
    FeatureExtractionResource,
    AsyncFeatureExtractionResource,
    FillMaskResource,
    AsyncFillMaskResource,
    RerankResource,
    AsyncRerankResource,
    VectorStoresResource,
    AsyncVectorStoresResource,
    ModelArtifactsResource,
    AsyncModelArtifactsResource,
    AgentsResource,
    AsyncAgentsResource,
    ResponsesResource,
    AsyncResponsesResource,
)
from .exceptions import APIError, AuthenticationError
from .utils import build_url
from .logging import get_logger

logger = get_logger(__name__)

DEFAULT_UPLOAD_BASE_URL = "https://upload.synapsai.cloud/v1"


class BaseClient:
    """Base client with common functionality"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 300.0,
        max_retries: int = 1,
        headers: Optional[Dict[str, str]] = None,
        httpx_client: Optional[httpx.Client] = None,
    ):
        """
        Initialize the client with the provided arguments.

        Args:
            api_key: API key for authentication.
            base_url: Base URL for the API.
            timeout: Timeout for requests (seconds or httpx.Timeout).
            max_retries: Maximum number of retries for requests (>=1).
            headers: Additional headers to include in requests.
            httpx_client: Custom HTTP client to use for requests.
        """
        if api_key is None:
            api_key = os.environ.get("SYNAPSAI_API_KEY")

        if api_key is None:
            raise AuthenticationError(
                "No API key provided. You can set your API key in an environment variable `SYNAPSAI_API_KEY`, or you can pass it as an argument `SynapsAI(api_key=...)`."
            )

        if base_url is None:
            base_url = os.environ.get("SYNAPSAI_API_BASE")

        if base_url is None:
            base_url = "https://api.synapsai.cloud/v1"

        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        # ensure sensible value
        self.max_retries = max(1, int(max_retries))

        # Set up default headers
        self._headers = {
            "Content-Type": "application/json",
            "User-Agent": "synapsai-python/1.0.0",
            "Authorization": f"Bearer {api_key}",
        }

        if headers:
            self._headers.update(headers)

        self._client = httpx_client

    def _build_request(
        self,
        **kwargs
    ) -> Dict[str, Any]:
        """Build request payload for SynapsAI API"""
        # Filter out None values
        filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}

        return {
            **filtered_kwargs
        }

    def _handle_error_response(self, response: httpx.Response) -> None:
        # Ensure content is loaded for streamed responses
        try:
            # If _content is missing, this will populate it
            if not hasattr(response, "_content"):
                response.read()
        except Exception:
            # If this fails, we still fall back to a generic message below
            pass
    
        try:
            error_data = response.json()
            # Support the infra server structured error format:
            if isinstance(error_data, dict) and "error" in error_data:
                err = error_data["error"]
                message = err.get("message") or err.get("error") or str(err)

                # keep original status semantics
                raise APIError(message, status_code=response.status_code)
            else:
                # try to extract message from common patterns
                if isinstance(error_data, dict):
                    message = error_data.get("error", {}).get("message") or error_data.get("message") or str(error_data)
                else:
                    message = str(error_data)
        except Exception:
            message = f"HTTP {response.status_code}: {response.text}"
        raise APIError(message, status_code=response.status_code)

    def _should_retry(self, method: str, response: Optional[httpx.Response], exc: Optional[BaseException], attempt: int) -> bool:
        """
        Decide whether to retry a request.

        Retry on:
          - network/transport errors (httpx.RequestError)
          - timeout errors (httpx.TimeoutException)
          - HTTP 429 (rate limit) and 5xx server errors

        For safety, we limit number of retries via self.max_retries externally.
        """
        # If exception is present and is a network/timeout error -> retry
        if exc is not None:
            if isinstance(exc, (httpx.RequestError, httpx.TimeoutException)):
                return True
            return False

        if response is not None:
            # Retry on 429 or 5xx
            if response.status_code == 429:
                return True
            if 500 <= response.status_code < 600:
                return True

        return False

    def _backoff_delay(self, attempt: int) -> float:
        """
        Exponential backoff with jitter.

        attempt is 0-based. We use base 0.5s.
        """
        base = 0.5
        delay = base * (2 ** attempt)
        # add jitter up to 0.5s
        delay += random.uniform(0, 0.5)
        # Cap delay to a sensible maximum (e.g., 30s)
        return min(delay, 30.0)


class SynapsAI(BaseClient):
    """Synchronous SynapsAI client"""

    chat: ChatResource
    images: ImagesResource
    videos: VideosResource
    embeddings: EmbeddingsResource
    audio: AudioResource
    completions: CompletionsResource
    classifications: ClassificationsResource
    question_answering: QuestionAnsweringResource
    models: ModelsResource
    feature_extraction: FeatureExtractionResource
    fill_mask: FillMaskResource
    rerank: RerankResource
    vector_stores: VectorStoresResource
    model_artifacts: ModelArtifactsResource
    agents: AgentsResource
    responses: ResponsesResource

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(api_key=api_key, **kwargs)
        if self._client is None:
            self._client = httpx.Client(
                timeout=self.timeout,
                headers=self._headers,
                http2=True,
                verify=False,
            )

        # Initialize resource handlers
        self.chat = ChatResource(self)
        self.images = ImagesResource(self)
        self.videos = VideosResource(self)
        self.embeddings = EmbeddingsResource(self)
        self.audio = AudioResource(self)
        self.completions = CompletionsResource(self)
        self.classifications = ClassificationsResource(self)
        self.question_answering = QuestionAnsweringResource(self)
        self.models = ModelsResource(self)
        self.feature_extraction = FeatureExtractionResource(self)
        self.fill_mask = FillMaskResource(self)
        self.rerank = RerankResource(self)
        self.vector_stores = VectorStoresResource(self)
        self.model_artifacts = ModelArtifactsResource(self)
        self.agents = AgentsResource(self)
        self.responses = ResponsesResource(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Close the HTTP client"""
        self._client.close()

    def _prepare_request_kwargs(
        self,
        method: str,
        endpoint: str,
        json_data: Optional[Dict[str, Any]] = None,
        data: Optional[Any] = None,
        files: Optional[Any] = None,
        content: Optional[Any] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, Any]] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """Build httpx request kwargs, clearing Content-Type for multipart/raw bodies."""
        url = build_url(self.base_url, endpoint)
        kwargs: Dict[str, Any] = {
            "method": method,
            "url": url,
            "timeout": self.timeout,
        }
        if json_data is not None:
            kwargs["json"] = json_data
        if data is not None:
            kwargs["data"] = data
        if files is not None:
            kwargs["files"] = files
        if content is not None:
            kwargs["content"] = content
        if params is not None:
            kwargs["params"] = params
        if stream:
            kwargs["stream"] = True

        request_headers = dict(headers or {})
        if files is not None or content is not None:
            # Let httpx set multipart/raw content-type; drop JSON default.
            request_headers.setdefault("Content-Type", None)
        if request_headers:
            kwargs["headers"] = request_headers
        return kwargs

    def _request(
        self,
        method: str,
        endpoint: str,
        json_data: Optional[Dict[str, Any]] = None,
        data: Optional[Any] = None,
        files: Optional[Any] = None,
        content: Optional[Any] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, Any]] = None,
        stream: bool = False,
    ) -> httpx.Response:
        """
        Make a request to the API with retries.

        Retries on network errors, timeouts, and server-side errors (429, 5xx).
        """
        base_kwargs = self._prepare_request_kwargs(
            method=method,
            endpoint=endpoint,
            json_data=json_data,
            data=data,
            files=files,
            content=content,
            params=params,
            headers=headers,
            stream=stream,
        )

        attempt = 0
        last_exc: Optional[BaseException] = None
        last_response: Optional[httpx.Response] = None

        while attempt < self.max_retries:
            try:
                response = self._client.request(**base_kwargs)
                last_response = response

                # If server-side error or rate-limit, decide if retry
                if response.status_code >= 400:
                    if self._should_retry(method, response, None, attempt) and attempt < (self.max_retries - 1):
                        # close body and wait before retrying
                        try:
                            response.read()
                        except Exception:
                            pass
                        delay = self._backoff_delay(attempt)
                        time.sleep(delay)
                        attempt += 1
                        continue
                    else:
                        # Non-retryable or out of retries -> raise
                        self._handle_error_response(response)
                # success
                return response

            except (httpx.RequestError, httpx.TimeoutException) as e:
                last_exc = e
                # decide if we should retry
                if self._should_retry(method, None, e, attempt) and attempt < (self.max_retries - 1):
                    delay = self._backoff_delay(attempt)
                    time.sleep(delay)
                    attempt += 1
                    continue
                # no more retries
                raise APIError(str(e))

        # If we exit loop without returning, raise last known problem
        if last_response is not None:
            self._handle_error_response(last_response)
        if last_exc is not None:
            raise APIError(str(last_exc))
        raise APIError("Unknown error during request")

    def _post(
        self,
        endpoint: str,
        json_data: Optional[Dict[str, Any]] = None,
        data: Optional[Any] = None,
        files: Optional[Any] = None,
        content: Optional[Any] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, Any]] = None,
    ) -> httpx.Response:
        """Make a POST request"""
        return self._request(
            "POST",
            endpoint,
            json_data=json_data,
            data=data,
            files=files,
            content=content,
            params=params,
            headers=headers,
        )

    def _put(
        self,
        endpoint: str,
        json_data: Optional[Dict[str, Any]] = None,
        data: Optional[Any] = None,
        files: Optional[Any] = None,
        content: Optional[Any] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, Any]] = None,
    ) -> httpx.Response:
        """Make a PUT request"""
        return self._request(
            "PUT",
            endpoint,
            json_data=json_data,
            data=data,
            files=files,
            content=content,
            params=params,
            headers=headers,
        )

    def _get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> httpx.Response:
        """Make a GET request"""
        return self._request("GET", endpoint, params=params)

    def _delete(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> httpx.Response:
        """Make a DELETE request"""
        return self._request("DELETE", endpoint, params=params)

    def _stream_response(
        self,
        endpoint: str,
        json_data: Optional[Dict[str, Any]] = None,
        data: Optional[Any] = None,
        files: Optional[Any] = None,
        content: Optional[Any] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Dict[str, Any]]:
        """
        Stream response data from a POST endpoint.

        The attempt to establish the stream will be retried using the same backoff rules.
        Once the stream is established, streaming errors are raised as-is.
        """
        base_kwargs = self._prepare_request_kwargs(
            method="POST",
            endpoint=endpoint,
            json_data=json_data,
            data=data,
            files=files,
            content=content,
            params=params,
            headers=headers,
        )

        attempt = 0
        while attempt < self.max_retries:
            try:
                with self._client.stream(**base_kwargs) as response:
                    if response.status_code >= 400:
                        # decide whether to retry establishing stream
                        if self._should_retry("POST", response, None, attempt) and attempt < (self.max_retries - 1):
                            try:
                                response.read()
                            except Exception:
                                pass
                            delay = self._backoff_delay(attempt)
                            time.sleep(delay)
                            attempt += 1
                            continue
                        else:
                            self._handle_error_response(response)

                    # stream established, iterate lines and yield
                    for line in response.iter_lines():
                        line = line.strip()
                        if not line:
                            continue
                        if line.startswith("data:"):
                            data_line = line[5:]
                            if data_line.startswith(" "):
                                data_line = data_line[1:]
                            if data_line == "[DONE]":
                                return
                            try:
                                yield json.loads(data_line)
                            except json.JSONDecodeError as e:
                                # Log the malformed data for debugging
                                logger.warning(f"Received malformed JSON data: {data_line[:100]}...")
                                logger.warning(f"JSON decode error: {e}")
                                continue
                    # If stream ends naturally, return
                    return

            except (httpx.RequestError, httpx.TimeoutException) as e:
                # network issue while establishing stream, maybe retry
                if attempt < (self.max_retries - 1):
                    delay = self._backoff_delay(attempt)
                    time.sleep(delay)
                    attempt += 1
                    continue
                raise APIError(str(e))

        raise APIError("Failed to establish stream after retries")

    def _get_stream(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> httpx.Response:
        """Make a streaming GET request for binary content (e.g. video download)."""
        url = build_url(self.base_url, endpoint)
        request = self._client.build_request("GET", url, timeout=self.timeout, params=params)
        response = self._client.send(request, stream=True)
        if response.status_code >= 400:
            response.read()
            self._handle_error_response(response)
        return response


class AsyncSynapsAI(BaseClient):
    """Asynchronous SynapsAI client"""
    chat: AsyncChatResource
    images: AsyncImagesResource
    videos: AsyncVideosResource
    embeddings: AsyncEmbeddingsResource
    audio: AsyncAudioResource
    completions: AsyncCompletionsResource
    classifications: AsyncClassificationsResource
    question_answering: AsyncQuestionAnsweringResource
    models: AsyncModelsResource
    feature_extraction: AsyncFeatureExtractionResource
    fill_mask: AsyncFillMaskResource
    rerank: AsyncRerankResource
    vector_stores: AsyncVectorStoresResource
    model_artifacts: AsyncModelArtifactsResource
    agents: AsyncAgentsResource
    responses: AsyncResponsesResource

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(api_key=api_key, **kwargs)
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                headers=self._headers,
                http2=True,
            )

        # Initialize async resource handlers
        self.chat = AsyncChatResource(self)
        self.images = AsyncImagesResource(self)
        self.videos = AsyncVideosResource(self)
        self.embeddings = AsyncEmbeddingsResource(self)
        self.audio = AsyncAudioResource(self)
        self.completions = AsyncCompletionsResource(self)
        self.classifications = AsyncClassificationsResource(self)
        self.question_answering = AsyncQuestionAnsweringResource(self)
        self.models = AsyncModelsResource(self)
        self.feature_extraction = AsyncFeatureExtractionResource(self)
        self.fill_mask = AsyncFillMaskResource(self)
        self.rerank = AsyncRerankResource(self)
        self.vector_stores = AsyncVectorStoresResource(self)
        self.model_artifacts = AsyncModelArtifactsResource(self)
        self.agents = AsyncAgentsResource(self)
        self.responses = AsyncResponsesResource(self)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self):
        """Close the HTTP client"""
        await self._client.aclose()

    async def _request(
        self,
        method: str,
        endpoint: str,
        json_data: Optional[Dict[str, Any]] = None,
        data: Optional[Any] = None,
        files: Optional[Any] = None,
        content: Optional[Any] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, Any]] = None,
    ) -> httpx.Response:
        """Make an async request to the API with retries."""
        base_kwargs = self._prepare_request_kwargs(
            method=method,
            endpoint=endpoint,
            json_data=json_data,
            data=data,
            files=files,
            content=content,
            params=params,
            headers=headers,
        )

        attempt = 0
        last_exc: Optional[BaseException] = None
        last_response: Optional[httpx.Response] = None

        while attempt < self.max_retries:
            try:
                response = await self._client.request(**base_kwargs)
                last_response = response

                if response.status_code >= 400:
                    if self._should_retry(method, response, None, attempt) and attempt < (self.max_retries - 1):
                        try:
                            await response.aread()
                        except Exception:
                            pass
                        delay = self._backoff_delay(attempt)
                        await asyncio.sleep(delay)
                        attempt += 1
                        continue
                    else:
                        self._handle_error_response(response)
                return response

            except (httpx.RequestError, httpx.TimeoutException) as e:
                last_exc = e
                if self._should_retry(method, None, e, attempt) and attempt < (self.max_retries - 1):
                    delay = self._backoff_delay(attempt)
                    await asyncio.sleep(delay)
                    attempt += 1
                    continue
                raise APIError(str(e))

        # If we exit loop without returning, raise last known problem
        if last_response is not None:
            self._handle_error_response(last_response)
        if last_exc is not None:
            raise APIError(str(last_exc))
        raise APIError("Unknown error during request")

    async def _post(
        self,
        endpoint: str,
        json_data: Optional[Dict[str, Any]] = None,
        data: Optional[Any] = None,
        files: Optional[Any] = None,
        content: Optional[Any] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, Any]] = None,
    ) -> httpx.Response:
        """Make an async POST request"""
        return await self._request(
            "POST",
            endpoint,
            json_data=json_data,
            data=data,
            files=files,
            content=content,
            params=params,
            headers=headers,
        )

    async def _put(
        self,
        endpoint: str,
        json_data: Optional[Dict[str, Any]] = None,
        data: Optional[Any] = None,
        files: Optional[Any] = None,
        content: Optional[Any] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, Any]] = None,
    ) -> httpx.Response:
        """Make an async PUT request"""
        return await self._request(
            "PUT",
            endpoint,
            json_data=json_data,
            data=data,
            files=files,
            content=content,
            params=params,
            headers=headers,
        )

    async def _get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> httpx.Response:
        """Make a GET request"""
        return await self._request("GET", endpoint=endpoint, params=params)

    async def _delete(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> httpx.Response:
        """Make a DELETE request"""
        return await self._request("DELETE", endpoint=endpoint, params=params)

    async def _stream_response(
        self,
        endpoint: str,
        json_data: Optional[Dict[str, Any]] = None,
        data: Optional[Any] = None,
        files: Optional[Any] = None,
        content: Optional[Any] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream response data from a POST endpoint (async).

        The attempt to establish the stream will be retried using the same backoff rules.
        Once the stream is established, streaming errors are raised as-is.
        """
        base_kwargs = self._prepare_request_kwargs(
            method="POST",
            endpoint=endpoint,
            json_data=json_data,
            data=data,
            files=files,
            content=content,
            params=params,
            headers=headers,
        )

        attempt = 0
        while attempt < self.max_retries:
            try:
                async with self._client.stream(**base_kwargs) as response:
                    if response.status_code >= 400:
                        if self._should_retry("POST", response, None, attempt) and attempt < (self.max_retries - 1):
                            try:
                                await response.aread()
                            except Exception:
                                pass
                            delay = self._backoff_delay(attempt)
                            await asyncio.sleep(delay)
                            attempt += 1
                            continue
                        else:
                            self._handle_error_response(response)

                    async for raw_line in response.aiter_lines():
                        line = raw_line.strip()
                        if not line:
                            continue
                        if line.startswith("data:"):
                            data_line = line[5:]
                            if data_line.startswith(" "):
                                data_line = data_line[1:]
                            if data_line == "[DONE]":
                                return
                            try:
                                yield json.loads(data_line)
                            except json.JSONDecodeError:
                                # skip malformed line
                                continue
                    # stream ended normally
                    return

            except (httpx.RequestError, httpx.TimeoutException) as e:
                if attempt < (self.max_retries - 1):
                    delay = self._backoff_delay(attempt)
                    await asyncio.sleep(delay)
                    attempt += 1
                    continue
                raise APIError(str(e))

        raise APIError("Failed to establish stream after retries")

    async def _get_stream(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> httpx.Response:
        """Make an async streaming GET request for binary content (e.g. video download)."""
        url = build_url(self.base_url, endpoint)
        request = self._client.build_request("GET", url, timeout=self.timeout, params=params)
        response = await self._client.send(request, stream=True)
        if response.status_code >= 400:
            await response.aread()
            self._handle_error_response(response)
        return response