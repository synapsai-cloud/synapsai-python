"""
Question Answering type definitions
"""

from typing import Optional, Dict, Any, Union, List, Tuple, Literal
from pydantic import BaseModel, Field, ConfigDict

from .common import APIResponse, Usage


class QAAnswer(BaseModel):
    """Generic QA answer (text QA)"""
    score: float = Field(..., description="Answer confidence score")
    start: int = Field(..., description="Start character index in the context")
    end: int = Field(..., description="End character index in the context")
    answer: str = Field(..., description="Extracted answer text")


class DocumentQAAnswer(QAAnswer):
    """Document QA answer extends generic QA with word alignment indices"""
    words: Optional[List[int]] = Field(default=None, description="Indices of words aligned to the answer, if available")


class DocumentQuestionAnsweringRequest(BaseModel):
    """Request model for document question answering"""
    model: str
    image: Union[str, Any] = Field(..., description="Image input (URL, base64, file path, or PIL.Image)")
    question: str = Field(..., description="Question to ask about the document")
    word_boxes: Optional[List[Tuple[str, Tuple[float, float, float, float]]]] = Field(default=None, description="List of (word, bbox) with bbox normalized 0..1000")
    top_k: Optional[int] = Field(default=1, description="Number of answers to return")
    doc_stride: Optional[int] = Field(default=128, description="Overlap size when splitting long documents")
    max_answer_len: Optional[int] = Field(default=15, description="Maximum answer length")
    max_seq_len: Optional[int] = Field(default=384, description="Maximum total token length per chunk")
    max_question_len: Optional[int] = Field(default=64, description="Maximum tokenized question length")
    handle_impossible_answer: Optional[bool] = Field(default=False, description="Allow 'no answer' outcomes")
    lang: Optional[str] = Field(default=None, description="OCR language when applicable")
    tesseract_config: Optional[str] = Field(default=None, description="Extra OCR flags")
    timeout: Optional[float] = Field(default=None, description="Fetch timeout for remote images")
    model_config = ConfigDict(arbitrary_types_allowed=True)


class DocumentQuestionAnsweringResponse(APIResponse):
    object: Literal["list"] = "list"
    data: Union[DocumentQAAnswer, List[DocumentQAAnswer]]
    usage: Optional[Usage] = None


class QuestionAnsweringRequest(BaseModel):
    """Request model for extractive text question answering"""
    model: str
    question: Union[str, List[str]] = Field(..., description="Question(s) to answer")
    context: Union[str, List[str]] = Field(..., description="Context(s) associated with the question(s)")
    top_k: Optional[int] = Field(default=1, description="Number of answers to return")
    doc_stride: Optional[int] = Field(default=128, description="Overlap size when splitting long contexts")
    max_answer_len: Optional[int] = Field(default=15, description="Maximum answer length")
    max_seq_len: Optional[int] = Field(default=384, description="Maximum total token length per chunk")
    max_question_len: Optional[int] = Field(default=64, description="Maximum tokenized question length")
    handle_impossible_answer: Optional[bool] = Field(default=False, description="Allow 'no answer' outcomes")
    align_to_words: Optional[bool] = Field(default=True, description="Align answers to word boundaries where possible")


class QuestionAnsweringResponse(APIResponse):
    object: Literal["list"] = "list"
    data: Union[QAAnswer, List[QAAnswer]]
    usage: Optional[Usage] = None


class TableQAAnswer(BaseModel):
    """Table QA answer"""
    answer: str = Field(..., description="Answer string")
    coordinates: Optional[List[Tuple[int, int]]] = Field(default=None, description="Cell coordinates contributing to the answer")
    cells: Optional[List[str]] = Field(default=None, description="Cell values contributing to the answer")
    aggregator: Optional[str] = Field(default=None, description="Aggregation used, if any")


class TableQuestionAnsweringRequest(BaseModel):
    """Request model for table question answering"""
    model: str
    table: Dict[str, Any] = Field(..., description="Table data as a JSON-serializable dict (converted to DataFrame server-side)")
    query: Union[str, List[str]] = Field(..., description="Query or queries over the table")
    sequential: Optional[bool] = Field(default=False, description="Run sequentially (required for conversational models)")
    padding: Optional[Union[bool, str]] = Field(default=False, description="Padding strategy")
    truncation: Optional[Union[bool, str]] = Field(default=False, description="Truncation strategy")


class TableQuestionAnsweringResponse(APIResponse):
    object: Literal["list"] = "list"
    data: Union[TableQAAnswer, List[TableQAAnswer]]
    usage: Optional[Usage] = None


class VQAResult(BaseModel):
    """Visual QA single result"""
    label: str = Field(..., description="Predicted label/answer")
    score: float = Field(..., description="Confidence score")


class VisualQuestionAnsweringRequest(BaseModel):
    """Request model for visual question answering"""
    model: str
    image: Union[str, List[str], Any] = Field(..., description="Image input(s) (URL, base64, file path, or PIL.Image)")
    question: Union[str, List[str]] = Field(..., description="Question(s) about the image(s)")
    top_k: Optional[int] = Field(default=5, description="Number of answers to return")
    timeout: Optional[float] = Field(default=None, description="Fetch timeout for remote images")
    model_config = ConfigDict(arbitrary_types_allowed=True)


class VisualQuestionAnsweringResponse(APIResponse):
    object: Literal["list"] = "list"
    data: List[VQAResult]
    usage: Optional[Usage] = None


