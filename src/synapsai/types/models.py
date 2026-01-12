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

from typing import Literal, List
from pydantic import BaseModel

class Model(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str
    status: Literal["building", "sleeping", "ready", "failed"]


class Models(BaseModel):
    object: Literal["list"] = "list"
    data: List[Model]
