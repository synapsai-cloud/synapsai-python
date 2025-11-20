"""
Question Answering resource handlers
"""

from typing import TYPE_CHECKING, Optional, Union, List, Dict

from ..types.question_answering import (
    DocumentQuestionAnsweringRequest,
    DocumentQuestionAnsweringResponse,
    QuestionAnsweringRequest,
    QuestionAnsweringResponse,
    TableQuestionAnsweringRequest,
    TableQuestionAnsweringResponse,
    VisualQuestionAnsweringRequest,
    VisualQuestionAnsweringResponse,
)

if TYPE_CHECKING:
    from ..client import SynapsAI, AsyncSynapsAI


class QuestionAnsweringResource:
    """Question Answering resource handler"""
    
    def __init__(self, client: "SynapsAI"):
        self._client = client
    
    def document(
        self,
        model: str,
        image,
        question: str,
        word_boxes: Optional[List] = None,
        top_k: Optional[int] = None,
        doc_stride: Optional[int] = None,
        max_answer_len: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        max_question_len: Optional[int] = None,
        handle_impossible_answer: Optional[bool] = None,
        lang: Optional[str] = None,
        tesseract_config: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> DocumentQuestionAnsweringResponse:
        request_data = self._client._build_request(
            model=model,
            image=image,
            question=question,
            word_boxes=word_boxes,
            top_k=top_k,
            doc_stride=doc_stride,
            max_answer_len=max_answer_len,
            max_seq_len=max_seq_len,
            max_question_len=max_question_len,
            handle_impossible_answer=handle_impossible_answer,
            lang=lang,
            tesseract_config=tesseract_config,
            timeout=timeout,
        )
        endpoint = "/question-answering/document"
        response = self._client._post(endpoint, json_data=request_data)
        return DocumentQuestionAnsweringResponse.model_validate(response.json())
    
    def text(
        self,
        model: str,
        question: Union[str, List[str]],
        context: Union[str, List[str]],
        top_k: Optional[int] = None,
        doc_stride: Optional[int] = None,
        max_answer_len: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        max_question_len: Optional[int] = None,
        handle_impossible_answer: Optional[bool] = None,
        align_to_words: Optional[bool] = None,
    ) -> QuestionAnsweringResponse:
        request_data = self._client._build_request(
            model=model,
            question=question,
            context=context,
            top_k=top_k,
            doc_stride=doc_stride,
            max_answer_len=max_answer_len,
            max_seq_len=max_seq_len,
            max_question_len=max_question_len,
            handle_impossible_answer=handle_impossible_answer,
            align_to_words=align_to_words,
        )
        endpoint = "/question-answering"
        response = self._client._post(endpoint, json_data=request_data)
        return QuestionAnsweringResponse.model_validate(response.json())
    
    def table(
        self,
        model: str,
        table: Dict,
        query: Union[str, List[str]],
        sequential: Optional[bool] = None,
        padding: Optional[Union[bool, str]] = None,
        truncation: Optional[Union[bool, str]] = None,
    ) -> TableQuestionAnsweringResponse:
        request_data = self._client._build_request(
            model=model,
            table=table,
            query=query,
            sequential=sequential,
            padding=padding,
            truncation=truncation,
        )
        endpoint = "/question-answering/table"
        response = self._client._post(endpoint, json_data=request_data)
        return TableQuestionAnsweringResponse.model_validate(response.json())
    
    def visual(
        self,
        model: str,
        image,
        question: Union[str, List[str]],
        top_k: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> VisualQuestionAnsweringResponse:
        request_data = self._client._build_request(
            model=model,
            image=image,
            question=question,
            top_k=top_k,
            timeout=timeout,
        )
        endpoint = "/question-answering/visual"
        response = self._client._post(endpoint, json_data=request_data)
        return VisualQuestionAnsweringResponse.model_validate(response.json())


class AsyncQuestionAnsweringResource:
    """Async Question Answering resource handler"""
    
    def __init__(self, client: "AsyncSynapsAI"):
        self._client = client

    async def document(
        self,
        model: str,
        image,
        question: str,
        word_boxes: Optional[List] = None,
        top_k: Optional[int] = None,
        doc_stride: Optional[int] = None,
        max_answer_len: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        max_question_len: Optional[int] = None,
        handle_impossible_answer: Optional[bool] = None,
        lang: Optional[str] = None,
        tesseract_config: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> DocumentQuestionAnsweringResponse:
        request_data = self._client._build_request(
            model=model,
            image=image,
            question=question,
            word_boxes=word_boxes,
            top_k=top_k,
            doc_stride=doc_stride,
            max_answer_len=max_answer_len,
            max_seq_len=max_seq_len,
            max_question_len=max_question_len,
            handle_impossible_answer=handle_impossible_answer,
            lang=lang,
            tesseract_config=tesseract_config,
            timeout=timeout,
        )
        endpoint = "/question-answering/document"
        response = await self._client._post(endpoint, json_data=request_data)
        return DocumentQuestionAnsweringResponse.model_validate(response.json())

    async def text(
        self,
        model: str,
        question: Union[str, List[str]],
        context: Union[str, List[str]],
        top_k: Optional[int] = None,
        doc_stride: Optional[int] = None,
        max_answer_len: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        max_question_len: Optional[int] = None,
        handle_impossible_answer: Optional[bool] = None,
        align_to_words: Optional[bool] = None,
    ) -> QuestionAnsweringResponse:
        request_data = self._client._build_request(
            model=model,
            question=question,
            context=context,
            top_k=top_k,
            doc_stride=doc_stride,
            max_answer_len=max_answer_len,
            max_seq_len=max_seq_len,
            max_question_len=max_question_len,
            handle_impossible_answer=handle_impossible_answer,
            align_to_words=align_to_words,
        )
        endpoint = "/question-answering"
        response = await self._client._post(endpoint, json_data=request_data)
        return QuestionAnsweringResponse.model_validate(response.json())

    async def table(
        self,
        model: str,
        table: Dict,
        query: Union[str, List[str]],
        sequential: Optional[bool] = None,
        padding: Optional[Union[bool, str]] = None,
        truncation: Optional[Union[bool, str]] = None,
    ) -> TableQuestionAnsweringResponse:
        request_data = self._client._build_request(
            model=model,
            table=table,
            query=query,
            sequential=sequential,
            padding=padding,
            truncation=truncation,
        )
        endpoint = "/question-answering/table"
        response = await self._client._post(endpoint, json_data=request_data)
        return TableQuestionAnsweringResponse.model_validate(response.json())

    async def visual(
        self,
        model: str,
        image,
        question: Union[str, List[str]],
        top_k: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> VisualQuestionAnsweringResponse:
        request_data = self._client._build_request(
            model=model,
            image=image,
            question=question,
            top_k=top_k,
            timeout=timeout,
        )
        endpoint = "/question-answering/visual"
        response = await self._client._post(endpoint, json_data=request_data)
        return VisualQuestionAnsweringResponse.model_validate(response.json())


