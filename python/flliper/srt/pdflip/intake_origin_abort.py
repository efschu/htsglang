"""W27 RID-SPLIT (NF nf9, 04.10.2026): PP0's intake-stall drop rides the request wire.

THE DEATH. Boot ``dkrnfint4h6ablbar1dauer10041052`` (image rc12z30y9nf9), 11:31:33Z, after 35 min
serving: ``#1233 W27 PP WIDTH DIVERGENCE REFUSED`` on PP1 -- PP0 sent 17 rows
(``pdflip-79-295`` [113408,113425)), PP1 had planned 4660 tokens for
``[pdflip-84-299 [20736,25379), pdflip-79-295 [113408,113425)]``. The chain, from the P log:

  * 11:31:29 PP0 ``PDFLIP-INTAKE-STALL rid=pdflip-84-299 ... dropped from this rank's waiting
    queue`` (``Scheduler._pdflip_intake_stall_observe``; weg2xsn288: ONLY PP0 refuses, a follower's
    queue is mutated only by the front's ``/abort_request``).
  * The front never sent that abort: the leg was a LEG1-EARLY future (posted at the D->P flip's
    begin), the P drain ended (``P-DRAIN prefilled=1 ... queue_at_exit=8``) without awaiting it,
    so ``_requeue_intake_stalled`` -- the only caller of the abort -- never ran (front log 11:33:38
    ``Task exception was never retrieved ... leg1 on P returned 503 ... PDFLIP-INTAKE-STALL
    rid=pdflip-84-299``).
  * 11:31:32/33 PP0 ``#queue-req: 2``, PP1 ``#queue-req: 3``; PP0 admitted ``[pdflip-79]``
    (fwd_ct=220, extend=17), PP1 ``[pdflip-84, pdflip-79]`` (fwd_ct=220, extend=4660) -> W27.

Without the #631 row authority (NF: ``p_row_authority`` off) every follower plans from ITS OWN
waiting queue, so a rank-local queue mutation on PP0 is a rank split by construction -- the
xsn288 rule only moved the window from the follower to the front's RPC latency (and here to
"forever"). The same class as #1158 (health probe dropped at rank 0, enqueued on the followers):
the cure is the same seam -- the origin puts its verdict on the request list it relays.

THE FIX. PP0's refusal notes ``(rid, message)`` here; the request origin (PP0, attn tp/cp 0)
appends one ``AbortReq`` per noted rid to the list of its NEXT intake (``origin_extra_reqs_hook``,
the PDFLIP VISION seam), which is TP-broadcast and chain-forwarded, so every stage drops the rid
before it plans that pass. PP0 itself no longer holds the rid -- its own copy of the abort finds
nothing and sends nothing (no second answer to the tokenizer). The front's later
``/abort_request``, if it comes, stays idempotent.

SCOPE. Only group P (``FLLIPER_PDFLIP_GROUP=P``), only ``pp_size > 1``, only the request origin,
and only after an intake-stall drop actually happened: every other form wires the hook exactly
as before (D, Dual, the flip path, TP-only P: unchanged).

MARKER (metal): ``PDFLIP-INTAKE-STALL ORIGIN-ABORT rid=... relayed`` on PP0 right after each
``PDFLIP-INTAKE-STALL rid=...`` line; target ``#1233 W27`` = 0 and equal ``#queue-req`` on
PP0/PP1/PP2 in the pass after it.
"""
from __future__ import annotations

import logging
from http import HTTPStatus
from typing import Any, Callable, List, Optional, Tuple

logger = logging.getLogger(__name__)

MARK = "PDFLIP-INTAKE-STALL ORIGIN-ABORT"


def relay_applies(*, group: Optional[str], pp_size: int, pp_rank: int,
                  attn_tp_rank: int, attn_cp_rank: int) -> bool:
    """True only on group P's request origin with a PP chain behind it."""
    return (
        str(group or "").strip().upper() == "P"
        and int(pp_size or 1) > 1
        and int(pp_rank or 0) == 0
        and int(attn_tp_rank or 0) == 0
        and int(attn_cp_rank or 0) == 0
    )


class IntakeOriginAborts:
    """The origin's pending intake-stall aborts, drained once per intake."""

    def __init__(self, *, relay: bool) -> None:
        self.relay = bool(relay)
        self._queue: List[Tuple[str, str]] = []
        self.noted = 0
        self.relayed = 0

    def note(self, rid: Any, message: str) -> bool:
        """PP0 dropped ``rid`` from its waiting queue: relay that drop."""
        if not self.relay:
            return False
        self._queue.append((str(rid), str(message)))
        self.noted += 1
        return True

    def take(self) -> List[Any]:
        """AbortReqs for the origin's next intake (empty list when none)."""
        if not self._queue:
            return []
        from flliper.srt.managers.io_struct import AbortReq

        out = [
            AbortReq(
                rid=rid,
                finished_reason={
                    "type": "abort",
                    "status_code": HTTPStatus.SERVICE_UNAVAILABLE,
                    "message": msg,
                },
            )
            for rid, msg in self._queue
        ]
        self._queue.clear()
        self.relayed += len(out)
        for a in out:
            logger.warning(
                "%s rid=%s relayed on the request wire (n=%d): every PP stage drops it in "
                "the same pass -- a follower plans from its own queue (no row authority), "
                "a rank-local drop is a W27 RID-SPLIT (nf9 pdflip-84-299)",
                MARK, a.rid, self.relayed,
            )
        return out


def origin_hook(
    collector: Optional[IntakeOriginAborts],
    other: Optional[Callable[[], List[Any]]] = None,
) -> Optional[Callable[[], List[Any]]]:
    """Compose the origin's extra-request hook: ``other`` (the vision verdicts
    and aborts, which must lead) first, then the intake-stall aborts. With no
    relaying collector the result IS ``other`` -- the hook every other form
    wired before."""
    if collector is None or not collector.relay:
        return other
    if other is None:
        return collector.take

    def _hook() -> List[Any]:
        out = list(other() or ())
        out.extend(collector.take())
        return out

    return _hook
