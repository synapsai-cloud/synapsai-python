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

__all__ = [
    # Models types
    "ModelResponse",
    "ModelsResponse",
    
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
    
    # Common types
    "APIResponse",
    "Error",
    "ErrorResponse",
] 