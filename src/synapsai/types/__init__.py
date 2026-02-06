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
Type definitions for SynapsAI client library
"""

from .completion import *
from .images import *
from .embeddings import *
from .audio import *
from .classifications import *
from .question_answering import *
from .common import *
from .models import *
from .feature_extraction import *
from .fill_mask import *

__all__ = [
    # Models types
    "Model",
    "Models",
    
    # Chat types
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "ChatCompletionChunk",
    "ChatMessage",
    "ChatRole",
    "ChatCompletionChoice",
    "CompletionChoice",
    "Delta",
    "FunctionCall",
    "Tool",
    "ToolCall",
    
    # Image types
    "ImageGenerateRequest",
    "ImageGenerateResponse",
    "ImageEditRequest",
    "ImageEditResponse",
    "ImageAnalysisRequest",
    "ImageAnalysisResponse",
    "ImageSize",
    "ImageQuality",
    "ImageStyle",
    "Image",
    "ImageSource",
    
    # Embedding types
    "EmbeddingRequest",
    "EmbeddingResponse",
    "Embedding",
    "SimilarityResponse",
    "SimilarityResult",
    
    # Audio types
    "AudioSpeechRequest",
    "AudioSpeechResponse",
    "AudioTranscriptionRequest",
    "AudioTranscriptionResponse",
    "AudioFormat",
    "Voice",
    "Transcription",
    "Usage",

    # Classification types
    "AudioClassificationRequest",
    "AudioClassificationResponse",
    "ImageClassificationRequest",
    "ImageClassificationResponse",
    "TextClassificationRequest",
    "TextClassificationResponse",
    "TokenClassificationRequest",
    "TokenClassificationResponse",
    "VideoClassificationRequest",
    "VideoClassificationResponse",
    "ZeroShotAudioClassificationRequest",
    "ZeroShotAudioClassificationResponse",
    "ZeroShotClassificationRequest",
    "ZeroShotClassificationResponse",
    "ZeroShotImageClassificationRequest",
    "ZeroShotImageClassificationResponse",
    "ZeroShotObjectDetectionRequest",
    "ZeroShotObjectDetectionResponse",
    
    # Question Answering types
    "DocumentQuestionAnsweringRequest",
    "DocumentQuestionAnsweringResponse",
    "QuestionAnsweringRequest",
    "QuestionAnsweringResponse",
    "TableQuestionAnsweringRequest",
    "TableQuestionAnsweringResponse",
    "VisualQuestionAnsweringRequest",
    "VisualQuestionAnsweringResponse",

    # Feature extraction types
    "FeatureExtractionRequest",
    "FeatureExtractionResponse",

    # Fill mask types
    "FillMaskRequest",
    "FillMaskResponse",
    
    # Common types
    "APIResponse",
    "Error",
    "ErrorResponse",
] 