"""DFlash-family aux capture across pipeline stages.

The P-side DFlash draft-KV producer (weg2) needs the residual stream at a few
capture layers ([6, 20, 34, 48, 62] + 1) for EVERY token of a prefill chunk,
assembled on the LAST pipeline stage where the producer runs. Under PP the
capture layers are spread over the stages, so each stage captures at its own
layers and the tensors have to reach the last stage.

THE CARRY RIDES THE PIPELINE PROXY (weg2xsn261, 17.09.2026). The first form
of this module shipped every stage's captures to the last stage over the
typed channel as a SEPARATE point-to-point message, sent from inside the
model forward BEFORE the stage returned its ``PPProxyTensors``. On the metal
that is a deadlock by construction: PP0's ``send`` to PP2 blocks until PP2
posts a matching receive, PP2 sits in its ordinary proxy receive from PP1,
and PP1 waits for PP0's proxy that PP0 never returns. Measured on the first
P prefill of the DFLASH form after a flip (rid weg2-0-1, 4096-token chunk):
PP0 stack ``forward -> exchange_captured_aux -> send_typed_tensor_dict ->
torch send``, PP1/PP2 ``PP-RECV-OBJ awaiting_size`` for 240 s, watchdog.
The hermetic channel double was non-blocking, so the test could not see it.

Now every stage puts its captures INTO the proxy dict it already hands to
the next stage (``aux_layer_<id>`` entries), forwards whatever ``aux_layer_*``
entries it received, and the last stage assembles received + own in
layer-id order. No second message, no cross-stage receive, nothing that can
block outside the pipeline's own order. The proxy channel carries arbitrary
keys (``send_tensor_dict`` ships a key/shape metadata list), and the model
reads only the keys it names.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Mapping, Optional

import torch

logger = logging.getLogger(__name__)

#: The proxy-dict key prefix of a carried capture: ``aux_layer_<layer id>``.
AUX_KEY_PREFIX = "aux_layer_"


class PpAuxCaptureError(RuntimeError):
    """A stage's captures could not be carried or assembled."""


def _received_aux(received: Optional[Mapping[str, torch.Tensor]]) -> Dict[int, torch.Tensor]:
    out: Dict[int, torch.Tensor] = {}
    if not received:
        return out
    for key, value in received.items():
        if isinstance(key, str) and key.startswith(AUX_KEY_PREFIX):
            try:
                lid = int(key[len(AUX_KEY_PREFIX):])
            except ValueError as exc:
                raise PpAuxCaptureError(f"malformed aux carry key {key!r}") from exc
            out[lid] = value
    return out


def _merge(received: Dict[int, torch.Tensor], captured: Mapping[int, torch.Tensor],
           stage: int) -> Dict[int, torch.Tensor]:
    merged = dict(received)
    for lid, t in captured.items():
        lid = int(lid)
        if lid in merged:
            raise PpAuxCaptureError(
                f"capture layer {lid} arrived from an upstream stage but stage "
                f"{stage} captured it too -- two stages claim the same capture "
                "layer, the ownership map is inconsistent"
            )
        merged[lid] = t
    return merged


def carry_aux_forward(
    *,
    received: Optional[Mapping[str, torch.Tensor]],
    captured: Mapping[int, torch.Tensor],
    stage: int,
) -> Dict[str, torch.Tensor]:
    """The ``aux_layer_*`` entries a NON-last stage adds to its outgoing
    proxy dict: everything it received from upstream plus its own captures.
    Raises on a layer id that arrives twice."""
    merged = _merge(_received_aux(received), captured, stage)
    return {f"{AUX_KEY_PREFIX}{lid}": t for lid, t in merged.items()}


def assemble_aux_on_last_stage(
    *,
    received: Optional[Mapping[str, torch.Tensor]],
    captured: Mapping[int, torch.Tensor],
    stage: int,
) -> List[torch.Tensor]:
    """The layer-id-ordered capture list on the LAST stage: what the proxy
    carried in plus this stage's own captures."""
    merged = _merge(_received_aux(received), captured, stage)
    return [merged[k] for k in sorted(merged)]
