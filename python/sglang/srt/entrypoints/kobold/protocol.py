# Copyright 2026 SGLang Team
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
# ==============================================================================
"""KoboldCpp-compatible request/response shapes (#335).

Fields are declared even when this adapter cannot honour them. That is
deliberate and is the #335/#710 lesson: an undeclared field is dropped by
pydantic's default ``extra="ignore"`` with no trace, whereas a declared one
can be REFUSED BY NAME. The refusal lives in ``serving.KoboldServing``, not
here -- this module only describes the wire.
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel


class KoboldGenerateRequest(BaseModel):
    """``POST /api/v1/generate`` -- KoboldCpp's raw-prompt generation."""

    prompt: str
    # -- honoured, mapped onto the OpenAI completion path ------------------
    max_length: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stop_sequence: Optional[List[str]] = None
    #: Kobold sends this per request; SGLang fixes it at boot. Inert.
    max_context_length: Optional[int] = None
    #: Kobold's "suppress prompt echo" flag. SGLang never echoes. Inert.
    quiet: Optional[bool] = None
    n: Optional[int] = None

    # -- declared so they can be refused by NAME rather than dropped -------
    rep_pen: Optional[float] = None
    rep_pen_range: Optional[int] = None
    rep_pen_slope: Optional[float] = None
    top_a: Optional[float] = None
    typical: Optional[float] = None
    tfs: Optional[float] = None
    sampler_order: Optional[List[int]] = None
    sampler_seed: Optional[int] = None
    mirostat: Optional[int] = None
    mirostat_tau: Optional[float] = None
    mirostat_eta: Optional[float] = None
    grammar: Optional[str] = None
    logit_bias: Optional[Dict[str, float]] = None
    banned_tokens: Optional[List[str]] = None
    dynatemp_range: Optional[float] = None
    smoothing_factor: Optional[float] = None
    memory: Optional[str] = None
    images: Optional[List[str]] = None


class KoboldGenerateResult(BaseModel):
    text: str


class KoboldGenerateResponse(BaseModel):
    """Kobold clients read ``results[0].text`` and nothing else."""

    results: List[KoboldGenerateResult]


class KoboldStringResult(BaseModel):
    """The ``{"result": ...}`` envelope ``/api/v1/model`` and
    ``/api/extra/version`` both use."""

    result: Union[str, Dict[str, Any]]
