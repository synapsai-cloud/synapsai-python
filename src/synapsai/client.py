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
from urllib.parse import urljoin

from .resources import (
    ChatResource,
    AsyncChatResource,
    ImagesResource,
    AsyncImagesResource,
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
    AsyncModelsResource
)
from .exceptions import APIError, AuthenticationError
from .utils import build_url

class BaseClient:
    """Base client with common functionality"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 300.0,
        max_retries: int = 3,
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

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(api_key=api_key, **kwargs)
        if self._client is None:
            self._client = httpx.Client(
                timeout=self.timeout,
                headers=self._headers,
            )

        # Initialize resource handlers
        self.chat = ChatResource(self)
        self.images = ImagesResource(self)
        self.embeddings = EmbeddingsResource(self)
        self.audio = AudioResource(self)
        self.completions = CompletionsResource(self)
        self.classifications = ClassificationsResource(self)
        self.question_answering = QuestionAnsweringResource(self)
        self.models = ModelsResource(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Close the HTTP client"""
        self._client.close()

    def _request(
        self,
        method: str,
        endpoint: str,
        json_data: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
        stream: bool = False,
    ) -> httpx.Response:
        """
        Make a request to the API with retries.

        Retries on network errors, timeouts, and server-side errors (429, 5xx).
        """
        url = build_url(self.base_url, endpoint)

        # prepare kwargs used for every attempt
        base_kwargs = {
            "method": method,
            "url": url,
            "timeout": self.timeout,
        }

        if json_data:
            base_kwargs["json"] = json_data
        if data:
            base_kwargs["data"] = data
        if files:
            base_kwargs["files"] = files
        if stream:
            # note: for streaming we still attempt to (re)establish the stream on failures opening it
            base_kwargs["stream"] = True

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
        data: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
    ) -> httpx.Response:
        """Make a POST request"""
        return self._request("POST", endpoint, json_data, data, files)

    def _get(self, endpoint: str) -> httpx.Response:
        """Make a GET request"""
        return self._request("GET", endpoint)

    def _stream_response(
        self,
        endpoint: str,
        json_data: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Dict[str, Any]]:
        """
        Stream response data from a POST endpoint.

        The attempt to establish the stream will be retried using the same backoff rules.
        Once the stream is established, streaming errors are raised as-is.
        """
        url = self.base_url + endpoint
        base_kwargs = {
            "method": "POST",
            "url": url,
            "timeout": self.timeout,
            "json": json_data,
            "data": data,
            "files": files,
        }

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
                        if line.startswith("data: "):
                            data_line = line[6:]  # Remove "data: " prefix
                            if data_line == "[DONE]":
                                return
                            try:
                                yield json.loads(data_line)
                            except json.JSONDecodeError as e:
                                # Log the malformed data for debugging
                                print(f"Warning: Received malformed JSON data: {data_line[:100]}...")
                                print(f"JSON decode error: {e}")
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


class AsyncSynapsAI(BaseClient):
    """Asynchronous SynapsAI client"""

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(api_key=api_key, **kwargs)
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                headers=self._headers,
            )

        # Initialize async resource handlers
        self.chat = AsyncChatResource(self)
        self.images = AsyncImagesResource(self)
        self.embeddings = AsyncEmbeddingsResource(self)
        self.audio = AsyncAudioResource(self)
        self.completions = AsyncCompletionsResource(self)
        self.classifications = AsyncClassificationsResource(self)
        self.question_answering = AsyncQuestionAnsweringResource(self)
        self.models = AsyncModelsResource(self)

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
        data: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
    ) -> httpx.Response:
        """Make an async request to the API with retries."""
        url = build_url(self.base_url, endpoint)

        base_kwargs: Dict[str, Any] = {
            "method": method,
            "url": url,
            "timeout": self.timeout,
        }

        if json_data:
            base_kwargs["json"] = json_data
        if data:
            base_kwargs["data"] = data
        if files:
            base_kwargs["files"] = files

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
        data: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
    ) -> httpx.Response:
        """Make an async POST request"""
        return await self._request("POST", endpoint, json_data, data, files)

    async def _get(self, endpoint: str) -> httpx.Response:
        """Make a GET request"""
        return await self._request("GET", endpoint=endpoint)

    async def _stream_response(
        self,
        endpoint: str,
        json_data: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream response data from a POST endpoint (async).

        The attempt to establish the stream will be retried using the same backoff rules.
        Once the stream is established, streaming errors are raised as-is.
        """
        url = build_url(self.base_url, endpoint)

        base_kwargs = {
            "method": "POST",
            "url": url,
            "timeout": self.timeout,
            "json": json_data,
            "data": data,
            "files": files,
        }

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
                        if line.startswith("data: "):
                            data_line = line[6:]  # Remove "data: " prefix
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
