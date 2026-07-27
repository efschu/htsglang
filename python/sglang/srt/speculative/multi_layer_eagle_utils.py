# Copyright 2023-2024 SGLang Team
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

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Tuple

import torch

from sglang.kernels.ops.speculative.multi_layer_eagle import (
    rotate_input_ids,
    rotate_input_ids_kernel,
)

if TYPE_CHECKING:
    from sglang.srt.speculative.eagle_info import EagleDraftInput

logger = logging.getLogger(__name__)

__all__ = [
    "rotate_input_ids",
    "rotate_input_ids_kernel",
    "adapt_draft_columns",
    "adapt_draft_state_width",
]


def adapt_draft_columns(
    columns: Optional[torch.Tensor],
    target_width: int,
    dim: int = 1,
) -> Optional[torch.Tensor]:
    """Slice or pad *columns* along *dim* so it holds exactly *target_width* entries.

    ADAPTIVE DRAFT LENGTH ON MULTI-LAYER EAGLE (#138). Multi-layer EAGLE runs no
    model at draft time: the chain columns consumed by round N+1 were produced at
    the END of round N by k separate MTP layers. The adaptive k switch lands
    BETWEEN rounds (the controller is fed from the scheduler's batch-result
    processor, after forward_batch_generation returned), so the first round at a
    new k sees carried columns of the OLD width.

    - k_new < k_prev (downshift): take the first ``target_width`` columns.
      Semantically exact -- column i is MTP layer i's prediction for chain
      position i, so a prefix is precisely a shorter chain.
    - k_new > k_prev (upshift): repeat the LAST column to fill.
      Safe because speculative decoding is verified by the target model: a bad
      draft costs throughput, never correctness. Under
      --speculative-use-rejection-sampling the padded position carries a
      duplicate of the last position's token AND of its proposal distribution q,
      and that token really was drawn from that q, so the Leviathan accept test
      ``coin * q < p`` stays exact. Exactly one round per upshift is degraded;
      from the next round on the draft-extend has produced genuine k_new columns.

    Returns *columns* unchanged (same object) when the width already matches, so
    the steady state costs one integer comparison and no allocation.
    """
    if columns is None:
        return None
    width = columns.shape[dim]
    if width == target_width:
        return columns
    if target_width <= 0:
        raise ValueError(f"target_width must be >= 1, got {target_width}")
    if width == 0:
        raise ValueError(
            "cannot adapt a zero-width draft column tensor to "
            f"target_width={target_width}: there is no column to repeat"
        )
    if width > target_width:
        return columns.narrow(dim, 0, target_width).contiguous()
    pad_shape = [-1] * columns.dim()
    pad_shape[dim] = target_width - width
    pad = columns.narrow(dim, width - 1, 1).expand(pad_shape)
    return torch.cat((columns, pad), dim=dim).contiguous()


def adapt_draft_state_width(
    draft_input: "EagleDraftInput",
    speculative_num_steps: int,
    topk: int,
) -> bool:
    """Bring a carried ``EagleDraftInput``'s chain columns to the active k.

    Mutates ``topk_p`` / ``topk_index`` (``[bs, k * topk]``) and, when rejection
    sampling stashed them, ``draft_probs`` (``[bs, k, vocab]``). Returns True iff
    anything was resized (i.e. an adaptive k switch took effect this round).

    Rank-uniform by construction: the target width is derived from
    ``speculative_num_steps``, which every rank switched to on the same round
    (the k decision is a pure function of the rank-0-broadcast accept counts).
    """
    target_width = speculative_num_steps * topk
    topk_p = getattr(draft_input, "topk_p", None)
    if topk_p is None or topk_p.dim() < 2 or topk_p.shape[1] == target_width:
        return False

    old_width = topk_p.shape[1]
    draft_input.topk_p = adapt_draft_columns(topk_p, target_width)
    draft_input.topk_index = adapt_draft_columns(
        getattr(draft_input, "topk_index", None), target_width
    )
    # draft_probs is [bs, num_steps, vocab] -- one row per CHAIN STEP, so it is
    # resized to speculative_num_steps, not to k * topk.
    draft_input.draft_probs = adapt_draft_columns(
        getattr(draft_input, "draft_probs", None), speculative_num_steps
    )
    logger.debug(
        "multi-layer EAGLE adaptive: carried draft columns %d -> %d (k=%d, topk=%d)",
        old_width,
        target_width,
        speculative_num_steps,
        topk,
    )
    return True
