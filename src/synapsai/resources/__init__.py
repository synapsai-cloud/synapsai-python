"""
Resource handlers for SynapsAI client library
"""

from .chat import ChatResource, AsyncChatResource
from .images import ImagesResource, AsyncImagesResource
from .embeddings import EmbeddingsResource, AsyncEmbeddingsResource
from .audio import AudioResource, AsyncAudioResource
from .completions import CompletionsResource, AsyncCompletionsResource
from .classifications import ClassificationsResource, AsyncClassificationsResource
from .question_answering import QuestionAnsweringResource, AsyncQuestionAnsweringResource
from .models import ModelsResource, AsyncModelsResource

__all__ = [
    "ChatResource",
    "AsyncChatResource",
    "ImagesResource",
    "AsyncImagesResource",
    "EmbeddingsResource",
    "AsyncEmbeddingsResource",
    "AudioResource",
    "AsyncAudioResource",
    "CompletionsResource",
    "AsyncCompletionsResource",
    "ClassificationsResource",
    "AsyncClassificationsResource",
    "QuestionAnsweringResource",
    "AsyncQuestionAnsweringResource",
    "ModelsResource",
    "AsyncModelsResource",
]