# SPDX-License-Identifier: Apache-2.0
"""L15-END-ANCHOR (desk 06.10., l15long3 ..._2b8ea5bf5a_1006_024639): a finished
DFLASH request on D leaves a Mamba anchor at its EXACT committed end.

THE MEASURED GAP. A LONG turn reaches D only as leg 2 (5 tokens extend + ~48
decode on P's prefix). The extra_buffer strategy anchors a finished request
only at a ``mamba_track_interval`` (256) grid point it crossed, so 33 of 35
long finishes left a tombstone (``#1469 RETAIN ... cache_len=None
value=False``), ``tips_of`` found nothing and L15 held nothing.

WHY NOT THE LIVE SLOT. D runs the overlap schedule: the finishing request
rides one more (discarded) verify that advances ``req.mamba_pool_idx`` by an
accept length the CPU never learns, so the live state at the insert sits at
an unknown position (batch_result_processor ``req.finished()`` skip, the
lazy-mode "stale prealloc from an overlap extra forward" note).

THE MECHANISM (no new slot, no new kernel). An armed request writes its
COMMITTED post-verify state into a ping-pong track slot on EVERY verify --
the same scatter the grid track uses (``update_mamba_state_after_mtp_verify``
with step = ``commit_lens - 1``, i.e. state position = prefix + commit_lens =
the post-verify ``kv_committed_len``). The two slots alternate PER ROUND AT
PLAN TIME, so the verify launched behind a round (the overlap extra forward
included) always writes the OTHER slot. When the round's result is processed
the request records (position, slot) explicitly: ``mamba_last_track_seqlen =
kv_committed_len`` and the keep index = the slot that round wrote. The radix
insert then donates exactly that slot at exactly that length.

Invariant (one batch in flight, the overlap loop's shape): at every planning
moment the keep slot holds the last PROCESSED round's state and the in-flight
round writes the other slot; the round planned next writes the keep slot, and
the CPU moves keep to the in-flight round's slot when that result is
processed -- before any insert of this request can read it.

PREREQUISITE, ALSO CLOSED HERE (upstream 44fd17b696, the hunk the fork did
not port): the fork's DFLASH verify never rebuilt ``batch.mamba_track_indices``
(``prepare_mamba_track_for_verify``). After any filter/merge they are None (no
track written while the scheduler still flips the ping-pong index and records
the grid point), on an unfiltered batch they are the extend's stale tensor.
Armed, the plan rebuilds them from the requests every verify.

Switch: ``SGLANG_WEG2_L15_END_ANCHOR`` (with the L15 master, default off =
today byte for byte). Refused by name where the invariant is not proven:
no extra_buffer / lazy / checkpoint interval / tree page != 1 / PP / a
non-DFLASH decode. Per request: no grammar, no streaming session, origin
length >= ``SGLANG_WEG2_L15_TREE_CAND_MIN_TOKENS`` (the tips L15 votes on).
"""

from __future__ import annotations

import logging
import os
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

END_ANCHOR_ENV = "SGLANG_WEG2_L15_END_ANCHOR"

#: per-request decision, taken once at the first armed plan (True/False)
ARMED_ATTR = "_weg2_ea_armed"
#: explicit keep index of the ping-pong buffer (the slot of the last processed round)
KEEP_ATTR = "_weg2_ea_keep"
#: the ping-pong buffer object the keep index belongs to (identity; a re-allocated
#: buffer after a retraction invalidates it)
BUF_ATTR = "_weg2_ea_buf"

#: plan entry for a request that is not armed this round
NOT_ARMED = -1
#: plan entry for an armed request whose track write could not be scheduled
NOT_WRITTEN = -2

_GATE: Optional[tuple] = None  # (on, reason)
_COUNTS: dict = {}


def _count(key: str) -> int:
    n = _COUNTS.get(key, 0) + 1
    _COUNTS[key] = n
    return n


def _log_due(n: int) -> bool:
    return n <= 16 or (n & (n - 1)) == 0


def counts() -> dict:
    return dict(_COUNTS)


def reset_for_tests() -> None:
    global _GATE
    _GATE = None
    _COUNTS.clear()


def switch_on(env=None) -> bool:
    env = os.environ if env is None else env
    from sglang.srt.weg2 import l15_plan

    on = str(env.get(END_ANCHOR_ENV, "") or "").strip().lower() in ("1", "true", "yes", "on")
    return on and l15_plan.master_on(env)


def _evaluate_gate(server_args) -> tuple:
    if not switch_on():
        return (False, "off")
    sa = server_args
    try:
        if not sa.enable_mamba_extra_buffer():
            return (False, "W-L15-EA-NO-EXTRA-BUFFER")
        if sa.enable_mamba_extra_buffer_lazy():
            return (False, "W-L15-EA-LAZY")
    except Exception:  # noqa: BLE001 - a stub without the strategy helpers
        return (False, "W-L15-EA-NO-EXTRA-BUFFER")
    if getattr(sa, "mamba_checkpoint_interval", None) is not None:
        return (False, "W-L15-EA-CKPT-INTERVAL")
    if int(getattr(sa, "page_size", 1) or 1) != 1:
        return (False, "W-L15-EA-PAGE")
    if int(getattr(sa, "pp_size", 1) or 1) != 1:
        return (False, "W-L15-EA-PP")
    algo = str(getattr(sa, "speculative_algorithm", None) or "").upper()
    if algo != "DFLASH":
        return (False, "W-L15-EA-NOT-DFLASH")
    return (True, "armed")


def gate(server_args=None) -> bool:
    """Process gate, evaluated once (env and server args are fixed per boot)."""
    global _GATE
    if _GATE is None:
        if server_args is None:
            from sglang.srt.runtime_context import get_server_args

            server_args = get_server_args()
        _GATE = _evaluate_gate(server_args)
        if _GATE[1] != "off":
            logger.info("L15-END-ANCHOR gate=%s reason=%s", "on" if _GATE[0] else "refused", _GATE[1])
    return _GATE[0]


def _min_tokens() -> int:
    from sglang.srt.weg2 import l15_tree_cand

    return l15_tree_cand.min_tokens()


def _decide(req, min_tokens: int) -> bool:
    d = getattr(req, ARMED_ATTR, None)
    if d is not None:
        return bool(d)
    ok = (
        getattr(req, "grammar", None) is None
        and getattr(req, "session", None) is None
        and len(getattr(req, "origin_input_ids", ()) or ()) >= min_tokens
    )
    setattr(req, ARMED_ATTR, ok)
    return ok


def keep_override(req) -> Optional[int]:
    """The explicit keep index of an armed request, or None (= the stock rule).
    Valid only for the ping-pong buffer it was recorded against."""
    k = getattr(req, KEEP_ATTR, None)
    if k is None:
        return None
    buf = getattr(req, "mamba_ping_pong_track_buffer", None)
    if buf is None or getattr(req, BUF_ATTR, None) is not buf:
        return None
    return int(k)


def plan_verify(batch, *, other_idx=None, rebuild=None, stock_keep=None) -> Optional[List[int]]:
    """Before a DFLASH TARGET_VERIFY: rebuild the track indices from the requests
    (upstream 44fd17b696), then, per armed request, take its write slot
    (``mamba_next_track_idx``, already in the rebuilt indices) and move the
    request's next slot to the other one. Returns the per-request plan (slot
    index written, or NOT_ARMED) and attaches it, with a device mask, to the
    batch. None when the gate is off (nothing touched)."""
    if not gate():
        return None
    pool = batch.req_to_token_pool
    if rebuild is None:
        from sglang.srt.speculative.spec_utils import prepare_mamba_track_for_verify as rebuild
    rebuild(batch)
    if other_idx is None:
        other_idx = pool.get_mamba_ping_pong_other_idx
    if stock_keep is None:
        stock_keep = pool.get_mamba_ping_pong_keep_idx
    min_tokens = _min_tokens()
    plan: List[int] = []
    for req in batch.reqs:
        buf = getattr(req, "mamba_ping_pong_track_buffer", None)
        nxt = getattr(req, "mamba_next_track_idx", None)
        if buf is None or nxt is None or not _decide(req, min_tokens):
            plan.append(NOT_ARMED)
            continue
        if keep_override(req) is None:
            # first armed round on this buffer: pin the stock keep (the slot of
            # the last processed write -- an extend track, or nothing) BEFORE
            # the toggle changes what the stock rule would answer
            setattr(req, KEEP_ATTR, int(stock_keep(req)))
            setattr(req, BUF_ATTR, buf)
        w = int(nxt)
        req.mamba_next_track_idx = other_idx(w)
        plan.append(w)
    batch.weg2_end_anchor = plan
    if any(p >= 0 for p in plan):
        import torch

        mask = torch.tensor([p >= 0 for p in plan], dtype=torch.bool)
        dev = getattr(batch, "device", None)
        if dev is not None and str(dev) != "cpu":
            mask = mask.pin_memory().to(device=dev, non_blocking=True)
        batch.weg2_end_anchor_mask = mask
    else:
        batch.weg2_end_anchor_mask = None
    return plan


def steps_to_track(batch, last_correct_step_indices, grid_steps):
    """Inside the post-verify commit: armed rows track their LAST COMMITTED step
    (state position = prefix + commit_lens), the others keep the grid answer.
    Returns the steps to hand to the scatter. If the plan exists but no track
    destinations do, the plan is downgraded to NOT_WRITTEN (the result then
    records no anchor)."""
    plan = getattr(batch, "weg2_end_anchor", None)
    mask = getattr(batch, "weg2_end_anchor_mask", None)
    if plan is None or mask is None:
        return grid_steps
    if getattr(batch, "mamba_track_indices", None) is None or grid_steps is None:
        batch.weg2_end_anchor = [NOT_WRITTEN if p >= 0 else p for p in plan]
        n = _count("not_written")
        if _log_due(n):
            logger.warning("L15-END-ANCHOR not_written n=%d: no track destinations at the commit", n)
        return grid_steps
    import torch

    return torch.where(mask, last_correct_step_indices.to(grid_steps.dtype), grid_steps)


def on_result(req, batch, i: int) -> bool:
    """Result processing of the round that ran ``batch`` (called before the
    finish insert). True = this request was armed in that round and its track
    state is settled here (the stock grid flip must not run)."""
    plan = getattr(batch, "weg2_end_anchor", None)
    if plan is None or i >= len(plan):
        return False
    w = plan[i]
    if w == NOT_ARMED:
        return False
    buf = getattr(req, "mamba_ping_pong_track_buffer", None)
    pos = int(getattr(req, "kv_committed_len", 0) or 0)
    seqlen = len(req.origin_input_ids) + len(req.output_ids)
    if w == NOT_WRITTEN or buf is None or pos != seqlen - 1 or pos <= 0:
        # no valid (state, position) pair from this round: no anchor; the slot
        # this round did (or did not) write is not the keep
        req.mamba_last_track_seqlen = None
        why = "not_written" if w == NOT_WRITTEN else ("no_buffer" if buf is None else "pos_mismatch")
        n = _count("decline_" + why)
        if _log_due(n):
            logger.warning(
                "L15-END-ANCHOR decline rid=%s why=%s pos=%d seqlen=%d n=%d",
                str(getattr(req, "rid", "?"))[:12], why, pos, seqlen, n,
            )
        return True
    req.mamba_last_track_seqlen = pos
    setattr(req, KEEP_ATTR, int(w))
    setattr(req, BUF_ATTR, buf)
    if req.finished():
        n = _count("finished")
        # fed tokens past the stop (a stop inside an accepted run): the anchor
        # is exact but sits past the chat's fork, the next turn may not reach it
        try:
            through = len(req.output_ids_through_stop)
        except Exception:  # noqa: BLE001 - stubs without the property
            through = len(req.output_ids)
        past_stop = max(0, len(req.output_ids) - 1 - through)
        m = _count("finished_past_stop") if past_stop else 0
        if _log_due(n) or (past_stop and _log_due(m)):
            # the instrument: anchor position vs. the request's token length
            logger.info(
                "L15-END-ANCHOR finished rid=%s anchor=%d token_ids_len=%d origin=%d out=%d "
                "through_stop=%d past_stop=%d keep_idx=%d slot=%d n=%d",
                str(getattr(req, "rid", "?"))[:12], pos, seqlen, len(req.origin_input_ids),
                len(req.output_ids), through, past_stop, int(w), int(buf[int(w)]), n,
            )
    return True


def clear_on_filter(batch) -> None:
    batch.weg2_end_anchor = None
    batch.weg2_end_anchor_mask = None


__all__ = [
    "END_ANCHOR_ENV", "ARMED_ATTR", "KEEP_ATTR", "BUF_ATTR", "NOT_ARMED", "NOT_WRITTEN",
    "switch_on", "gate", "plan_verify", "steps_to_track", "on_result", "keep_override",
    "clear_on_filter", "counts", "reset_for_tests",
]
