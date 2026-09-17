# DFlash-family aux hidden capture across pipeline stages (Weg 2 group P).
#
# WHY THIS EXISTS. A DFlash draft builds its context KV from the target's
# residual stream at a handful of capture layers (Qwen3.8-27B-DFlash2:
# [6, 20, 34, 48, 62] in capture-mark terms). Upstream captures them inside
# one forward on one rank ("DFLASH/DSPARK aux hidden capture requires PP=1",
# qwen3_5_text.py). Group P of the flip form runs the target as PP3 with a
# GAPPED layer ownership (#753 crossing wire), so the five capture layers sit
# on three ranks and only the last stage owns the LogitsProcessor that
# concatenates them. Without this module a P-side DFlash draft-KV producer
# would see the last stage's captures only -- a silently wrong context.
#
# WHAT IT DOES. Every stage captures at its OWN capture layers during the
# layer loop (the model marks every stage's layers, not just the last
# rank's). After its loop, a non-last stage ships its captured tensors to the
# last stage over the PP group's typed channel (kind ``AUX_CAPTURE_KIND``,
# demultiplexed like the #753 crossings, so it can never be mistaken for a
# crossing or a proxy). The last stage receives from every other stage in
# pp-rank order after its own loop and returns the tensors in LAYER-ID order
# -- the order the LogitsProcessor's concatenation and the draft's ``fc``
# were trained against. Ownership interleaving does not matter: the sort key
# is the layer id, never the stage.
#
# ORDERING / DEADLOCK. A stage sends only after its last owned layer, i.e.
# after its last crossing send. The last stage receives only after its last
# owned layer, which transitively required every crossing from every stage.
# So when the last stage posts the aux receive from stage s, stage s has
# either finished (and is blocked in its aux send) or will finish without
# needing anything from the last stage. The typed channel stashes any other
# kind that arrives first, so a crossing that lands while an aux receive is
# pending is kept, not dropped.
from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)

AUX_CAPTURE_KIND = "aux_capture"
_KEY_PREFIX = "aux_layer_"
_COUNT_KEY = "aux_count"


class PpAuxCaptureError(RuntimeError):
    """A stage's captures could not be assembled on the last stage."""


def _payload(captured: Dict[int, torch.Tensor], device) -> Dict[str, torch.Tensor]:
    payload: Dict[str, torch.Tensor] = {
        f"{_KEY_PREFIX}{int(lid)}": t for lid, t in captured.items()
    }
    # A stage without capture layers still announces itself, so the last
    # stage's receive count is the stage count and never a guess.
    payload[_COUNT_KEY] = torch.tensor(
        [len(captured)], dtype=torch.int32, device=device
    )
    return payload


def _parse(payload: Dict[str, torch.Tensor], src: int) -> Dict[int, torch.Tensor]:
    out: Dict[int, torch.Tensor] = {}
    for key, value in payload.items():
        if key.startswith(_KEY_PREFIX):
            out[int(key[len(_KEY_PREFIX) :])] = value
    declared = payload.get(_COUNT_KEY)
    if declared is not None and int(declared.reshape(-1)[0].item()) != len(out):
        raise PpAuxCaptureError(
            f"stage {src} declared {int(declared.reshape(-1)[0].item())} aux "
            f"tensor(s) but {len(out)} arrived"
        )
    return out


def exchange_captured_aux(
    *,
    captured: Dict[int, torch.Tensor],
    pp_group,
    send: Optional[Callable] = None,
    recv: Optional[Callable] = None,
) -> Optional[List[torch.Tensor]]:
    """Ship this stage's captures to the last stage; assemble them there.

    ``captured`` maps capture-layer id -> residual tensor ([T, hidden]) for
    the layers THIS stage owns. Returns the layer-id-ordered list on the
    last stage and ``None`` everywhere else. With one stage it returns the
    own captures in layer-id order and touches no channel.

    ``send``/``recv`` default to the typed channel and are injectable for
    hermetic tests: ``send(group, payload, dst, kind)`` and
    ``recv(group, kind, src=...) -> payload``.
    """
    world_size = int(pp_group.world_size)
    rank = int(pp_group.rank_in_group)
    if world_size <= 1:
        return [captured[k] for k in sorted(captured)]
    if send is None or recv is None:
        from sglang.srt.distributed.pp_typed_channel import (
            recv_typed_tensor_dict,
            send_typed_tensor_dict,
        )

        send = send or send_typed_tensor_dict
        recv = recv or recv_typed_tensor_dict
    last = world_size - 1
    if rank != last:
        device = next(iter(captured.values())).device if captured else "cpu"
        send(pp_group, _payload(captured, device), last, AUX_CAPTURE_KIND)
        return None
    merged: Dict[int, torch.Tensor] = dict(captured)
    for src in range(world_size - 1):
        payload = recv(pp_group, AUX_CAPTURE_KIND, src=src)
        if payload is None:
            raise PpAuxCaptureError(
                f"last stage {rank}: no aux-capture message arrived from stage {src}"
            )
        for lid, t in _parse(payload, src).items():
            if lid in merged:
                raise PpAuxCaptureError(
                    f"capture layer {lid} arrived from stage {src} but is "
                    "already present on the last stage -- two stages claim "
                    "the same capture layer, the ownership map is inconsistent"
                )
            merged[lid] = t
    return [merged[k] for k in sorted(merged)]
