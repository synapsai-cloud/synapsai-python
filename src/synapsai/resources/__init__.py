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
Resource handlers for SynapsAI client library
"""

from .chat import ChatResource, AsyncChatResource
from .images import ImagesResource, AsyncImagesResource
from .videos import VideosResource, AsyncVideosResource
from .embeddings import EmbeddingsResource, AsyncEmbeddingsResource
from .audio import AudioResource, AsyncAudioResource, TranslationsResource, AsyncTranslationsResource
from .completions import CompletionsResource, AsyncCompletionsResource
from .classifications import ClassificationsResource, AsyncClassificationsResource
from .question_answering import QuestionAnsweringResource, AsyncQuestionAnsweringResource
from .models import ModelsResource, AsyncModelsResource
from .feature_extraction import FeatureExtractionResource, AsyncFeatureExtractionResource
from .fill_mask import FillMaskResource, AsyncFillMaskResource
from .rerank import RerankResource, AsyncRerankResource

__all__ = [
    "ChatResource",
    "AsyncChatResource",
    "ImagesResource",
    "AsyncImagesResource",
    "VideosResource",
    "AsyncVideosResource",
    "EmbeddingsResource",
    "AsyncEmbeddingsResource",
    "AudioResource",
    "AsyncAudioResource",
    "TranslationsResource",
    "AsyncTranslationsResource",
    "CompletionsResource",
    "AsyncCompletionsResource",
    "ClassificationsResource",
    "AsyncClassificationsResource",
    "QuestionAnsweringResource",
    "AsyncQuestionAnsweringResource",
    "ModelsResource",
    "AsyncModelsResource",
    "FeatureExtractionResource",
    "AsyncFeatureExtractionResource",
    "FillMaskResource",
    "AsyncFillMaskResource",
    "RerankResource",
    "AsyncRerankResource",
]