from typing import Literal, List
from .common import APIResponse
from pydantic import BaseModel, ConfigDict

class ModelResponse(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str
    status: Literal["building", "idle", "ready", "failed"]


class ModelsResponse(BaseModel):
    object: Literal["list"] = "list"
    data: List[ModelResponse]
