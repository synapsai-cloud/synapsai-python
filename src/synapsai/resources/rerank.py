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
Text ranking resource handlers
"""

from typing import TYPE_CHECKING, List

from ..types.rerank import RerankResponse

if TYPE_CHECKING:
    from ..client import SynapsAI, AsyncSynapsAI


class RerankResource:
    """Text ranking resource handler"""

    def __init__(self, client: "SynapsAI"):
        self._client = client

    def create(
        self,
        model: str,
        query: str,
        documents: List[str],
        top_n: int | None = None,
        max_tokens_per_doc: int = 4096,
        **kwargs,
    ) -> RerankResponse:
        """Rank documents against a query using a text-ranking model."""

        request_data = self._client._build_request(
            model=model,
            query=query,
            documents=documents,
            top_n=top_n,
            max_tokens_per_doc=max_tokens_per_doc,
            **kwargs,
        )

        endpoint = "rerank"

        response = self._client._post(endpoint, json_data=request_data)
        response_data = response.json()
        return RerankResponse.model_validate(response_data)


class AsyncRerankResource:
    """Async text ranking resource handler"""

    def __init__(self, client: "AsyncSynapsAI"):
        self._client = client

    async def create(
        self,
        model: str,
        query: str,
        documents: List[str],
        top_n: int | None = None,
        max_tokens_per_doc: int = 4096,
        **kwargs,
    ) -> RerankResponse:
        """Rank documents against a query using a text-ranking model."""

        request_data = self._client._build_request(
            model=model,
            query=query,
            documents=documents,
            top_n=top_n,
            max_tokens_per_doc=max_tokens_per_doc,
            **kwargs,
        )

        endpoint = "rerank"

        response = await self._client._post(endpoint, json_data=request_data)
        response_data = response.json()
        return RerankResponse.model_validate(response_data)
