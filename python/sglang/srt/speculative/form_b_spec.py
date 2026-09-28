# SPDX-License-Identifier: Apache-2.0
"""Form B's spec channel (F6 step 4; FORM-B-F6-ENTWURF-0928.md v2 §3, NF
objection 3).

Under Form B the draft runs on ONE rank, the lead (min of the weight ranks W),
but every rank takes part in the round: W in the draft's vocab gathers over
``model_tp``, and every rank -- the KV-only ranks K included -- in the target
verify's dcp collectives, whose row count is ``bs * (k + 1)``. With the adaptive
draft (fnFL2 H27) k changes from round to round, and it is decided on the lead
from rank-local measurements. So:

1. ``spec_k`` -- the lead publishes this round's k over ``dcp`` (ALL ranks)
   BEFORE the draft; every rank adopts it. No rank ever sizes a collective from
   its own k.
2. ``draft_block`` and ``accept`` -- the lead's broadcasts carry a FIXED form,
   padded to ``k_max``: their byte count depends on (bs, k_max) only, never on
   this round's k. A rank that is one round behind in its k bookkeeping then
   still posts the same-sized collective (a named wrong value downstream, not a
   size-mismatch hang inside the transport), and the collective sequence per
   communicator depends only on (forward_mode, has_prefix, layers, k) with k
   rank-uniform -- the F6 claim the trace test checks.

The channel is ``dcp`` (NF answer 1): it spans all ranks under Form B. The
scheduler group ``tp`` spans all ranks as well but carries control traffic
only (rank_role.guard_collective_subgroup).

Every function here is a no-op question (``form_b_spec_active()`` False) on a
boot without a Form B model_tp partition; the callers keep their classic
branch byte-identical.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch

#: Fill value of the padded tail. -1 is never a token id or an accept index a
#: consumer may read: every consumer reads the leading ``n`` entries only.
PAD = -1


class FormBSpecError(RuntimeError):
    """A Form B spec-channel invariant broke, named (W186)."""

    code = "W186 Weg2FormBSpecChannel"


def _refuse(text: str) -> FormBSpecError:
    return FormBSpecError(f"{FormBSpecError.code}: {text}")


# ---------------------------------------------------------------------------
# Who is Form B, who leads, which group
# ---------------------------------------------------------------------------
def form_b_spec_active() -> bool:
    """True iff a Form B model_tp partition is installed (F6)."""
    from sglang.srt.distributed import parallel_state as ps

    return ps._MODEL_TP is not None


def lead_rank_of(partition: Sequence[Sequence[int]]) -> int:
    """The lead from a model_tp partition: min of THE weight-rank group W (the
    one part with >= 2 members; every KV-only rank sits alone). Form B has
    |W| >= 2 by definition, so no such part means this is not Form B."""
    multi = [list(p) for p in partition if len(p) >= 2]
    if len(multi) != 1:
        raise _refuse(
            f"model_tp partition {[list(p) for p in partition]} has "
            f"{len(multi)} groups with >= 2 ranks; Form B has exactly one (W)."
        )
    return min(multi[0])


def form_b_lead() -> int:
    """Global rank of the Form B lead (draft solo, QSA index, vision lease)."""
    from sglang.srt.distributed import parallel_state as ps

    if ps._MODEL_TP_PARTITION is None:
        raise _refuse("no Form B model_tp partition is installed.")
    return lead_rank_of(ps._MODEL_TP_PARTITION)


def spec_group():
    """The spec channel: the dcp group, which under Form B spans ALL ranks."""
    from sglang.srt.distributed import parallel_state as ps

    group = ps.get_dcp_group_no_assert()
    tp = ps.get_tp_group()
    if group is None or tuple(group.ranks) != tuple(tp.ranks):
        raise _refuse(
            f"the spec channel is dcp over ALL ranks {list(tp.ranks)}, but the "
            f"dcp group is {None if group is None else list(group.ranks)}. "
            "Form B needs --dcp-size == --tp-size (every rank holds KV)."
        )
    return group


def _src_in(group, lead: Optional[int]) -> int:
    lead = form_b_lead() if lead is None else lead
    try:
        return list(group.ranks).index(lead)
    except ValueError:
        raise _refuse(
            f"the lead rank {lead} is not in the spec group {list(group.ranks)}."
        ) from None


# ---------------------------------------------------------------------------
# k_max: installed once at arm time by the spec worker
# ---------------------------------------------------------------------------
_K_MAX: Optional[int] = None


def set_spec_k_max(k_max: Optional[int]) -> None:
    """Install the largest chain length any round may use (the largest BUILT
    runtime state, or the static --speculative-num-steps). None = off."""
    global _K_MAX
    if k_max is not None and int(k_max) < 1:
        raise _refuse(f"k_max must be >= 1, got {k_max}.")
    _K_MAX = None if k_max is None else int(k_max)


def spec_k_max() -> int:
    if _K_MAX is None:
        raise _refuse(
            "k_max is not installed: the spec worker must call "
            "set_spec_k_max() when it arms under Form B."
        )
    return _K_MAX


# ---------------------------------------------------------------------------
# The three broadcasts
# ---------------------------------------------------------------------------
def _broadcast(group, buf: torch.Tensor, src: int) -> None:
    if group.world_size <= 1:
        return
    from sglang.srt.speculative.spec_utils import capture_safe_tp_broadcast

    capture_safe_tp_broadcast(group, (buf,), src=src)


def spec_k(
    k: Optional[int],
    *,
    device=None,
    group=None,
    lead: Optional[int] = None,
    k_max: Optional[int] = None,
) -> int:
    """This round's k, from the lead to every rank, BEFORE the draft.

    The lead passes its k, every other rank ``None``. One int64 over dcp and
    one host read of it: k decides which runtime state (graphs) replays, so it
    has to be on the host anyway. A received k outside [1, k_max] is a named
    stop -- adopting it would size the verify differently from the lead."""
    group = spec_group() if group is None else group
    k_max = spec_k_max() if k_max is None else int(k_max)
    src = _src_in(group, lead)
    is_src = group.rank_in_group == src
    if is_src and k is None:
        raise _refuse("the lead must pass its k to spec_k.")
    buf = torch.tensor(
        [int(k) if is_src else PAD], dtype=torch.int64,
        device=device if device is not None else "cpu",
    )
    _broadcast(group, buf, src)
    got = int(buf.item())
    if not 1 <= got <= k_max:
        raise _refuse(
            f"spec_k received k={got} from the lead (rank {group.ranks[src]}), "
            f"outside [1, k_max={k_max}]."
        )
    return got


def broadcast_padded(
    payload: Optional[torch.Tensor],
    shape: Tuple[int, ...],
    fixed_numel: int,
    *,
    dtype: torch.dtype,
    device,
    group=None,
    lead: Optional[int] = None,
) -> torch.Tensor:
    """Lead -> all, in a FIXED form of ``fixed_numel`` elements. The lead passes
    ``payload`` (``shape``), every other rank ``None``; every rank returns a
    tensor of ``shape`` holding the lead's values. The collective's size is
    ``fixed_numel`` whatever ``shape`` is this round."""
    group = spec_group() if group is None else group
    src = _src_in(group, lead)
    n = 1
    for d in shape:
        n *= int(d)
    if n > fixed_numel:
        raise _refuse(
            f"a spec payload of shape {tuple(shape)} ({n} elements) does not fit "
            f"the fixed form of {fixed_numel}; the padding bound (k_max) is wrong."
        )
    buf = torch.full((fixed_numel,), PAD, dtype=dtype, device=device)
    if group.rank_in_group == src:
        if payload is None:
            raise _refuse("the lead must pass the payload to broadcast_padded.")
        if tuple(payload.shape) != tuple(shape):
            raise _refuse(
                f"the lead's payload has shape {tuple(payload.shape)}, "
                f"declared {tuple(shape)}."
            )
        buf[:n].copy_(payload.reshape(-1))
    _broadcast(group, buf, src)
    return buf[:n].view(*shape)


def broadcast_padded_inplace(
    t: torch.Tensor, fixed_numel: int, *, group=None, lead: Optional[int] = None
) -> None:
    """Same fixed form, for a buffer every rank already holds (the packed accept
    payload of eagle_sample): the lead's values are written back into ``t``."""
    group = spec_group() if group is None else group
    is_src = group.rank_in_group == _src_in(group, lead)
    out = broadcast_padded(
        t if is_src else None, tuple(t.shape), fixed_numel,
        dtype=t.dtype, device=t.device, group=group, lead=lead,
    )
    if not is_src:
        t.copy_(out)


def draft_block_numel(bs: int, k_max: Optional[int] = None) -> int:
    """Fixed form of the EAGLE chain's draft-token broadcast: [bs, k_max]."""
    return int(bs) * (spec_k_max() if k_max is None else int(k_max))


def accept_numel(bs: int, k_max: Optional[int] = None) -> int:
    """Fixed form of eagle_sample's packed accept payload for a CHAIN
    (topk == 1): predict bs*(k+1) + accept_index bs*(k+1) + num_correct bs,
    at k = k_max."""
    k_max = spec_k_max() if k_max is None else int(k_max)
    return int(bs) * (2 * (k_max + 1) + 1)
