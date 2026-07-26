# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
"""Which multimodal requests a given attention lane can actually serve.

Task #186. The Triton backend routes every extend through `_forward_extend_dcp`
when `dcp_size > 1` (`triton_backend.py:2146`), and that function refuses a
`custom_mask` outright: the mask's row stride is the GLOBAL prefix length, so it
stops describing the rows the kernel walks once the prefix is owner-sharded.

Exactly two model families install such a mask -- Gemma-3 and Gemma-4 -- and
they do it from inside `forward`, out of band, for the bidirectional attention
their image soft tokens need. Without an early check the request dies mid-batch
with a `NotImplementedError` from four frames deep in the attention backend,
which is neither actionable for the user nor survivable for the batch.

This module answers the question once, cheaply, from config alone, so both the
admission path and the boot warmup can ask it.
"""

from typing import Optional

__all__ = [
    "BIDIRECTIONAL_IMAGE_MASK_ARCHS",
    "image_requests_unsupported_reason",
    "model_installs_bidirectional_image_mask",
    "triton_dcp_extend_lane_active",
]

# Architectures whose sglang model class defines (or inherits)
# `prepare_attn_masks`, i.e. installs a bidirectional image mask on
# `forward_metadata.custom_mask`. Kept as a literal instead of resolved through
# ModelRegistry on purpose: `resolve_model_cls` imports EVERY model module, and
# this predicate runs in the tokenizer process where that cost is not
# affordable. `test_gemma4_bidirectional_mask.py` scans the model files and
# fails if this tuple ever drifts from the source of truth.
BIDIRECTIONAL_IMAGE_MASK_ARCHS = (
    "Gemma3ForConditionalGeneration",
    "Gemma4ForConditionalGeneration",
    "Gemma4UnifiedForConditionalGeneration",
)


def triton_dcp_extend_lane_active(server_args) -> bool:
    """Config-level mirror of `TritonAttnBackend.forward_extend`'s dispatch.

    The dispatch itself tests nothing but `self.dcp_size > 1`; the uneven-plan
    and token-sharding flags are folded into that value in the constructor. The
    backend selector is added here because this runs outside the model process,
    where `get_attn_backend()` does not exist -- and it reads the PREFILL
    backend, since extend is what dispatches, and `--prefill-attention-backend`
    can be set on its own.
    """
    prefill_backend, _ = server_args.get_attention_backends()
    return prefill_backend == "triton" and server_args.dcp_size > 1


def model_installs_bidirectional_image_mask(model_config) -> bool:
    hf_config = getattr(model_config, "hf_config", None)
    architectures = getattr(hf_config, "architectures", None)
    if not architectures:
        return False
    return any(arch in BIDIRECTIONAL_IMAGE_MASK_ARCHS for arch in architectures)


def image_requests_unsupported_reason(server_args, model_config) -> Optional[str]:
    """The user-facing refusal text, or None when image requests are fine.

    Deliberately keyed on the model as well as the lane: a VL model that does
    not install a bidirectional mask (Qwen-VL and friends) runs on this lane
    today, and a blanket image refusal would break it.
    """
    if not triton_dcp_extend_lane_active(server_args):
        return None
    if not model_installs_bidirectional_image_mask(model_config):
        return None
    return (
        "Image inputs are not supported by the Triton attention backend under "
        f"decode context parallelism (--dcp-size {server_args.dcp_size}). This "
        "model attends bidirectionally within each image-token block, and the "
        "mask that expresses it is indexed by the GLOBAL prefix length, which "
        "stops describing the rows the kernel walks once the prefix is "
        "owner-sharded across DCP ranks. Text-only requests are served "
        "normally on this lane. For image requests, use --attention-backend "
        "flashinfer, or --dcp-size 1."
    )
