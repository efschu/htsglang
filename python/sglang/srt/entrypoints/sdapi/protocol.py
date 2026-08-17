# SPDX-License-Identifier: Apache-2.0
"""Request/response shapes for the AUTOMATIC1111 ``/sdapi/v1`` surface (#335).

Implemented against AUTOMATIC1111 ``stable-diffusion-webui``'s own API models
(``modules/api/models.py``, the ``StableDiffusionTxt2ImgProcessingAPI`` /
``TextToImageResponse`` pair) as stock clients send and expect them. No version
is asserted here: that project versions the app, not the API, and claiming a
number nobody can check against would be worse than naming the shape.

ONLY THE FIELDS STOCK CLIENTS ACTUALLY SEND are declared, and every one of them
is either mapped or refused by name in ``serving.py``. A field declared here and
then ignored would be the #710 tool-arg-loss family -- the same defect the
Ollama front was fixed for in this very task (9b5a72f826): a value the caller
supplied that vanishes between the request and the sampler, with a plausible
wrong image as the result.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Txt2ImgRequest(BaseModel):
    """``POST /sdapi/v1/txt2img``.

    Defaults match A1111's, so "the client did not set it" and "the client set
    it to the default" are the same request -- which is what lets the refusal
    rule below fire only on values that would actually change the image.
    """

    prompt: str = ""
    negative_prompt: str = ""
    #: A1111 renders ``batch_size * n_iter`` images in total.
    batch_size: int = 1
    n_iter: int = 1
    width: int = 512
    height: int = 512
    #: Diffusion controls. None of these exists in the OpenAI images protocol
    #: this surface composes over; each is refused by name when moved off its
    #: default rather than dropped. See ``serving.py``.
    steps: int = 50
    cfg_scale: float = 7.0
    sampler_name: Optional[str] = None
    sampler_index: Optional[str] = None
    seed: int = -1
    subseed: int = -1
    denoising_strength: Optional[float] = None
    restore_faces: bool = False
    tiling: bool = False
    styles: List[str] = Field(default_factory=list)
    override_settings: Dict[str, Any] = Field(default_factory=dict)
    #: A1111 accepts these and stock clients send them; they are inert here
    #: because this surface has no script engine, and saying so is the point.
    script_name: Optional[str] = None
    script_args: List[Any] = Field(default_factory=list)
    alwayson_scripts: Dict[str, Any] = Field(default_factory=dict)
    send_images: bool = True
    save_images: bool = False


class Img2ImgRequest(Txt2ImgRequest):
    """``POST /sdapi/v1/img2img``. Declared so the refusal can be specific."""

    init_images: List[str] = Field(default_factory=list)
    mask: Optional[str] = None
    resize_mode: int = 0
    inpainting_fill: int = 0
    inpaint_full_res: bool = True


class Txt2ImgResponse(BaseModel):
    """``TextToImageResponse``: base64 images, echoed parameters, info string.

    ``info`` is a JSON-encoded STRING in A1111, not an object. Stock clients
    ``json.loads`` it, so emitting an object here would break them at a point
    far from the cause -- the same failure shape the Ollama ``format`` refusal
    exists to prevent.
    """

    images: List[str]
    parameters: Dict[str, Any]
    info: str
