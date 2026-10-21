# SPDX-License-Identifier: Apache-2.0
"""H125f: the vision verdict rides the request chain -- every PP stage holds
an image request until PP0's verdict reaches it, and releases it in the SAME
pass (the pass that absorbs the verdict).

THE SPLIT (27B root of the rc12z30c death, 28.09. 20:50:53Z; same class in the
synchronous stage). PP0 alone decides about an image: it stages the tower in
``get_new_batch_prefill`` and holds a refused request out of its passes until
the origin abort lands (``vision_rank_pass``). The followers never run that
pass. Where they plan for themselves ('#631 ROW AUTHORITY DISABLED', the no-flip
PP=3 P group -- ``pp_row_carrier_present`` False), they admit the request in the
pass PP0 withholds it, block in the proxy receive for a frame PP0 does not owe,
and PP0's chain-send join expires 120 s later as '#973 RING COMMIT TIMEOUT'.
The abort cannot close it: PP0 forwarded the request in this pass's intake,
before its admission decided anything, so the abort reaches the followers one
pass late.

THE RULE (``pp_row_carrier_present``'s law, made independent of the carrier):
no rank admits an image request before a verdict about it has reached THAT
rank over the chain. PP0 puts the verdict on the wire at its next intake (the
origin hook that already carries the refusal aborts); every stage dispatches
the identical list before it plans, so all of them release -- or, for a
refusal, abort -- the request in the same pass. The price is one pass of
latency per image request, never a text request.

Wire object: :class:`PdFlipVisionVerdict` (``ok`` only -- a refusal travels as
the existing ``AbortReq`` with the W-code). No new collective, no switch: the
gate arms on every stage of a transient-vision P group with ``pp_size > 1``,
and a single stage keeps the old same-pass behaviour.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple

logger = logging.getLogger(__name__)

#: named line of the gate (log prefix only; the decision is the wire object)
W_GATE = "W102c PdFlipVisionVerdictGate"
_LOG_FIRST = 16


@dataclass
class PdFlipVisionVerdict:
    """PP0's verdict for one image request: its rows are attached on PP0,
    every stage may admit it in the pass that absorbs this object."""

    rid: str


def is_vision_req(req) -> bool:
    """An image/video request on ANY stage: every stage builds the request's
    ``multimodal_inputs`` from the same ``recv_req.mm_inputs``
    (``handle_generate_request``), so the predicate reads alike everywhere --
    unlike ``unstaged_items``, which only PP0's staging changes."""
    mm = getattr(req, "multimodal_inputs", None)
    return bool(getattr(mm, "mm_items", None))


def gate_armed(scheduler) -> bool:
    ps = getattr(scheduler, "ps", None)
    return int(getattr(ps, "pp_size", 1) or 1) > 1


def arm(scheduler) -> bool:
    """Arm the gate's state on this stage (PP0 and followers alike)."""
    on = gate_armed(scheduler)
    scheduler._pdflip_vision_gate = on
    scheduler._pdflip_vision_released = set()
    scheduler._pdflip_vision_verdict_sent = set()
    scheduler._pdflip_vision_origin_verdicts = []
    scheduler._pdflip_vision_gate_said = 0
    return on


def absorb(scheduler, verdict: PdFlipVisionVerdict) -> None:
    """Dispatcher handler on every stage: the request is released here."""
    released = getattr(scheduler, "_pdflip_vision_released", None)
    if released is None:
        released = scheduler._pdflip_vision_released = set()
    released.add(verdict.rid)
    n = getattr(scheduler, "_pdflip_vision_gate_said", 0)
    if n < _LOG_FIRST:
        scheduler._pdflip_vision_gate_said = n + 1
        ps = getattr(scheduler, "ps", None)
        logger.info("%s RELEASE rid=%s pp_rank=%s (verdict absorbed; every stage admits it "
                    "from this pass on)", W_GATE, verdict.rid, getattr(ps, "pp_rank", "?"))


def queue_verdicts(scheduler, wq: Sequence[Any], unstaged, refused) -> None:
    """PP0, after its staging: every image request whose rows are attached and
    that has no verdict on the wire yet gets one at the next intake."""
    sent = scheduler._pdflip_vision_verdict_sent
    released = scheduler._pdflip_vision_released
    out = scheduler._pdflip_vision_origin_verdicts
    for r in wq:
        if not is_vision_req(r) or r.rid in released or r.rid in sent or r.rid in refused:
            continue
        if unstaged(r):
            continue
        sent.add(r.rid)
        out.append(r.rid)
        logger.info("%s VERDICT rid=%s ok (on the wire at the next intake; PP0 holds it "
                    "until then like every follower)", W_GATE, r.rid)


def gate_held(scheduler, wq: Sequence[Any], already: Sequence[Any] = ()) -> List[Any]:
    """The image requests of this stage's queue without a released verdict,
    minus those already held by the caller."""
    if not getattr(scheduler, "_pdflip_vision_gate", False):
        return []
    released = scheduler._pdflip_vision_released
    skip = {id(r) for r in already}
    return [r for r in wq
            if id(r) not in skip and is_vision_req(r) and r.rid not in released]


def follower_pass(scheduler) -> List[Tuple[int, Any]]:
    """A follower's half, right before its admission: hold what PP0 has not
    released yet. Returns (index, req) for ``vision_unpark``."""
    if getattr(scheduler, "pdflip_dormant", False):
        return []
    wq = scheduler.waiting_queue
    held = gate_held(scheduler, wq)
    if not held:
        return []
    ids = {id(r) for r in held}
    parked = [(i, r) for i, r in enumerate(wq) if id(r) in ids]
    for i, _ in reversed(parked):
        wq.pop(i)
    return parked


#: P-COMMIT V (21.10., NF rc30 boot ..._1021_063252, P PP0 09:37:12-16, pdflip-144-1508): the S82 park ended
#: ``outcome=acks``, ``S82 PARK-ADMIT`` put the Admit (told=113728) on this pass's chain, the plan's vision pass
#: then said ``W102 PdFlipVisionStage SKIP reason=image-cached covered=113728 src=told image_end=113719`` --
#: no tower -- and queued the verdict "on the wire at the next intake". The next intake came only after the
#: output receive of the older 16235-token set: PP0 ``#1466 PASS-STALL ... commit_ms=4138``, the 5090 idle
#: ``gpu_gap_ms=4492.6`` before the 3036-token forward. Seven such drains in that boot (9.7 s of PP0 receive).
#: An image the ADMITTED told already covers needs nothing PP0 learns later in the pass: its verdict is known
#: when the Admit is formed, so it rides the SAME chain as the Admit and every stage releases the request in
#: the pass that absorbs both -- the same rule as before (no rank admits an image before PP0's verdict reached
#: it), one pass earlier, and only for the told-covered case. A staged image (tower, emb cache, pixel fetch)
#: still gets its verdict from the plan's vision pass at the next intake.
W_AT_ADMIT = "W102c PdFlipVisionVerdictGate VERDICT-AT-ADMIT"


def pp0_verdicts_at_admit(scheduler, rids: Sequence[str], skip_verdict) -> List[PdFlipVisionVerdict]:
    """PP0, ``pdflip_store_told.pp0_publish``, right after this pass's Admits are formed and BEFORE the chain
    send: the verdicts of the image requests admitted in this pass whose told covers every image position.
    ``skip_verdict(scheduler, req)`` is ``vision_rank_runner.stage_skip_verdict`` (the plan's own SKIP rule,
    injected so this module stays import-light). Each verdict is marked sent and released on PP0 here (PP0
    does not dispatch its own wire tail); the caller appends them to the chain. Never raises into the pass."""
    if not getattr(scheduler, "_pdflip_vision_gate", False) or getattr(scheduler, "pdflip_dormant", False):
        return []
    sent = getattr(scheduler, "_pdflip_vision_verdict_sent", None)
    released = getattr(scheduler, "_pdflip_vision_released", None)
    if sent is None or released is None:
        return []
    refused = getattr(scheduler, "_pdflip_vision_refused", None) or set()
    want = {str(r) for r in rids}
    out: List[PdFlipVisionVerdict] = []
    for req in list(getattr(scheduler, "waiting_queue", None) or ()):
        rid = str(getattr(req, "rid", ""))
        if rid not in want or rid in released or rid in sent or rid in refused:
            continue
        if not is_vision_req(req):
            continue
        try:
            verdict = skip_verdict(scheduler, req)
        except Exception as exc:  # noqa: BLE001 -- the plan's vision pass decides as before
            logger.warning("%s skipped rid=%s (%s: %s)", W_AT_ADMIT, rid, type(exc).__name__, exc)
            continue
        if not verdict or verdict[0] != "skip":
            continue
        # review A2/H1: PP0 released <=> the verdict is on the wire. Only non-raising container ops between the
        # append and the release (no absorb() call, no log line): whatever leaves this function released on PP0
        # is in ``out``, and the caller appends ``out`` unconditionally.
        item = PdFlipVisionVerdict(rid=rid)
        out.append(item)
        sent.add(rid)
        released.add(rid)
        _log_at_admit(scheduler, rid, verdict)
    return out


def _log_at_admit(scheduler, rid: str, verdict) -> None:
    """The VERDICT-AT-ADMIT line; a failing log never undoes the release (review A2)."""
    try:
        n = getattr(scheduler, "_pdflip_vision_at_admit_n", 0) + 1
        scheduler._pdflip_vision_at_admit_n = n
        logger.info("%s rid=%s covered=%s src=%s (n=%d): the admitted told covers every image position -- the "
                    "verdict rides THIS pass's chain beside the Admit, every stage admits it in this pass",
                    W_AT_ADMIT, rid, verdict[1] if len(verdict) > 1 else "?",
                    verdict[2] if len(verdict) > 2 else "?", n)
    except Exception:  # noqa: BLE001 -- an instrument
        pass


def take_verdicts(scheduler) -> List[PdFlipVisionVerdict]:
    out = getattr(scheduler, "_pdflip_vision_origin_verdicts", None)
    if not out:
        return []
    items = [PdFlipVisionVerdict(rid=rid) for rid in out]
    out.clear()
    return items
