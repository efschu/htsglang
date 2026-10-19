"""S59 (19.10., NF rc15/rc14): the prefetch thread's store CLAIM in END coordinates.

HP1 made the Form A #580 registration vote and the completion MIN compare span
ENDS (start + count), because the host (TP0, attention bytes + mamba state)
and the expert workers (byteless shadow tree) start their spans at different
depths. The third reduce of the same read was left in LENGTHS: the claim
reduce in ``HiCacheController.prefetch_thread_func`` (scalar MIN of
``storage_hit_count``, or the packed ``[claim, -claim]`` vote). A worker then
took the host's page COUNT and applied it from its OWN start.

MEASURED, rc15 D, pdflip-26-100 (07:29:45): the host's span started at 17856
(its mamba anchor; the workers matched 0), the store held KV to 120640 and an
anchor at 119808 -- ``PDFLIP-CLAIM-SHORT asked=1609 kv=1606 claimed=1593``,
i.e. the host claimed 1593 pages = 101952 tokens from 17856 = END 119808. The
workers adopted 101952 from 0 = END 101952; the completion END MIN took the
group to 101952 (TP0 ``completed=101952 min_synced=84096``), the host's
admission vote followed (``X-GATE-TERMS head=101952 store=101888
floor=101952``), and D refused 18936 tokens with W31 -> W50 -> REROUTE_P/R6.
P then computed 1080 (``PDFLIP-USAGE-TRUE p_computed=1080``) -- exactly
120888 - 119808. Same shape, same 17856 (16832) offset, on 24-95, 26-102,
30-117, 48-336 (rc15) and 36-560, 47-708, 52-774, 56-800 (rc14).

THE FIX: every rank votes its claim as an END (its span start + its count) and
takes back ``group END - its own start``, clamped to its own claim (a rank
never reads pages its own store probe did not answer for: a page past the
host's END has no store bytes for these keys -- the "wrong key, wrong KV"
direction). The revoke threshold is compared against the HOST's length of the
group END (``group END - host start``, both group-agreed by the #580 vote), so
the revoke decision stays one verdict on every rank.

The frame travels by request id, registered by the tree before the operation
is queued (the ``pdflip_min_hit`` form) and consumed once by the prefetch
thread. No frame (every non-Form-A read) = the unchanged length reduce, byte
for byte. Group-uniform: the tree registers it from the #580 vote's reduced
``host_base`` and the rank's own ``span_base`` on every rank alike (the END
vote itself is armed group-uniformly, ``tp_match_floor.form_a_end_base``).
"""

from __future__ import annotations

from typing import NamedTuple, Optional

ATTR = "pdflip_claim_end_frame"
_CAP = 4096


class ClaimEndFrame(NamedTuple):
    #: this rank's absolute span start (``span_base`` of the #580 END vote)
    my_base: int
    #: the host's absolute span start (the #580 vote's reduced slot 6)
    host_base: int


def note_claim_end_frame(controller, request_id, my_base: Optional[int],
                         host_base: Optional[int]) -> None:
    """Register (both bases set) or clear (either None) ``request_id``'s frame."""
    if controller is None:
        return
    table = getattr(controller, ATTR, None)
    if my_base is None or host_base is None:
        if table:
            table.pop(request_id, None)
        return
    if table is None:
        table = {}
        try:
            setattr(controller, ATTR, table)
        except AttributeError:
            return
    table[request_id] = ClaimEndFrame(max(0, int(my_base)), max(0, int(host_base)))
    while len(table) > _CAP:
        table.pop(next(iter(table)))


def take_claim_end_frame(controller, operation) -> Optional[ClaimEndFrame]:
    """The frame registered for ``operation`` (consumed), or None."""
    table = getattr(controller, ATTR, None)
    if not table:
        return None
    return table.pop(getattr(operation, "request_id", None), None)


def local_claim_vote(frame: Optional[ClaimEndFrame], count: int) -> int:
    """What this rank puts into the claim MIN: its END under a frame, else
    its count (unchanged)."""
    if frame is None:
        return int(count)
    return int(frame.my_base) + int(count)


def adopt_group_claim(frame: Optional[ClaimEndFrame], reduced: int,
                      local_count: int) -> tuple:
    """``(own_count, threshold_count)`` from the reduced claim.

    Off a frame both are the reduced count (the old behaviour). Under a frame
    ``reduced`` is the group END: this rank reads ``END - my_base``, clamped
    to ``[0, local_count]``, and the revoke threshold compares the host's
    length ``END - host_base`` -- a reduced value minus a reduced value, the
    same number on every rank."""
    if frame is None:
        return int(reduced), int(reduced)
    end = int(reduced)
    own = max(0, min(int(local_count), end - int(frame.my_base)))
    return own, max(0, end - int(frame.host_base))


_N = [0]


def note_line(logger, rid, frame: ClaimEndFrame, local_count: int, group_end: int,
              own: int, host_len: int) -> None:
    """One named line where the frame changed what this rank reads (the old
    length MIN would have handed it ``host_len``); rate-limited, counted."""
    if frame is None or int(own) == int(host_len):
        return
    _N[0] += 1
    n = _N[0]
    if n <= 32 or n % 256 == 0:
        logger.info(
            "S59 CLAIM-END rid=%s my_base=%d host_base=%d local=%d group_end=%d own=%d "
            "host_len=%d (n=%d): the store claim was reduced as span ENDS; the length MIN "
            "would have read %d tokens from this rank's own start (END %d instead of %d)",
            rid, int(frame.my_base), int(frame.host_base), int(local_count), int(group_end),
            int(own), int(host_len), n, int(host_len), int(frame.my_base) + int(host_len),
            int(group_end),
        )
