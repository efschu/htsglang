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

Wire object: :class:`Weg2VisionVerdict` (``ok`` only -- a refusal travels as
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
W_GATE = "W102c Weg2VisionVerdictGate"
_LOG_FIRST = 16


@dataclass
class Weg2VisionVerdict:
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
    scheduler._weg2_vision_gate = on
    scheduler._weg2_vision_released = set()
    scheduler._weg2_vision_verdict_sent = set()
    scheduler._weg2_vision_origin_verdicts = []
    scheduler._weg2_vision_gate_said = 0
    return on


def absorb(scheduler, verdict: Weg2VisionVerdict) -> None:
    """Dispatcher handler on every stage: the request is released here."""
    released = getattr(scheduler, "_weg2_vision_released", None)
    if released is None:
        released = scheduler._weg2_vision_released = set()
    released.add(verdict.rid)
    n = getattr(scheduler, "_weg2_vision_gate_said", 0)
    if n < _LOG_FIRST:
        scheduler._weg2_vision_gate_said = n + 1
        ps = getattr(scheduler, "ps", None)
        logger.info("%s RELEASE rid=%s pp_rank=%s (verdict absorbed; every stage admits it "
                    "from this pass on)", W_GATE, verdict.rid, getattr(ps, "pp_rank", "?"))


def queue_verdicts(scheduler, wq: Sequence[Any], unstaged, refused) -> None:
    """PP0, after its staging: every image request whose rows are attached and
    that has no verdict on the wire yet gets one at the next intake."""
    sent = scheduler._weg2_vision_verdict_sent
    released = scheduler._weg2_vision_released
    out = scheduler._weg2_vision_origin_verdicts
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
    if not getattr(scheduler, "_weg2_vision_gate", False):
        return []
    released = scheduler._weg2_vision_released
    skip = {id(r) for r in already}
    return [r for r in wq
            if id(r) not in skip and is_vision_req(r) and r.rid not in released]


def follower_pass(scheduler) -> List[Tuple[int, Any]]:
    """A follower's half, right before its admission: hold what PP0 has not
    released yet. Returns (index, req) for ``vision_unpark``."""
    if getattr(scheduler, "weg2_dormant", False):
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


def take_verdicts(scheduler) -> List[Weg2VisionVerdict]:
    out = getattr(scheduler, "_weg2_vision_origin_verdicts", None)
    if not out:
        return []
    items = [Weg2VisionVerdict(rid=rid) for rid in out]
    out.clear()
    return items
