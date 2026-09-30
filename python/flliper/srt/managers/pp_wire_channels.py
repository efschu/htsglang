"""The PP wire's CHANNEL TAXONOMY -- names, not machinery.

#1233 (WEG 2, S0). These names lived in ``phase_flip_counters``, which the
Weg-2 deletion surface lists as pure flip machinery. They are not: the
surviving PP loop uses ``CHAN_REQ`` / ``CHAN_DICT`` to name its two real
wires in ~48 places that have nothing to do with a layout change, and a
name-based sweep of the flip modules would have taken the pipeline's own
vocabulary with them (the §11.4 hazard, hit here for real -- ruff F821 x48).

So the taxonomy moves to a module that survives, and
``phase_flip_counters`` re-exports it until S7 deletes that file. ONE
DEFINITION, in the module whose lifetime matches the thing it names.

``CHAN_PASS`` and ``CHAN_SLOT`` are kept with their siblings deliberately
even though only the flip published them: they are values of the same
namespace, and splitting a namespace across two modules is how two
definitions start.
"""

import logging

logger = logging.getLogger(__name__)

LOG_PREFIX = "PP-WIRE-CHANNELS"



#: The request chain (point_to_point_pyobj), rank k -> k+1.
CHAN_REQ = "req"
#: The tensor-dict wire (proxy AND output share it), rank k -> k+1 mod n.
CHAN_DICT = "dict"
#: NOT A WIRE. The number of pp_loop_size slot iterations a rank has run
#: SINCE IT ARMED -- the instrument for #631 defect Q (the armed window has
#: no pass clock). It rides the counter machinery because that machinery is
#: already a cross-rank readable side channel with the right lifetime, so
#: one rank can print all three ranks' pass counts on one line instead of
#: three log streams having to be correlated by hand.
#:
#: Published ONLY while a flip is armed, and reset at each arm. A boot
#: without the flip, and every unarmed pass of a boot with it, writes
#: nothing at all.
CHAN_PASS = "pass"
#: NOT A WIRE. The MICROBATCH SLOT INDEX this rank is currently on, published
#: while a flip is armed -- the answer to defect Q rather than another
#: measurement of it.
#:
#: WHY THE SLOT INDEX IS THE QUANTITY THAT MATTERS. ``CHAN_PASS`` above
#: counts armed iterations and MEASURED the divergence (spreads of ~10787
#: iterations over a 5 s armed window, 2026-08-09 07:19:23Z). But the pass
#: count is not what the pipeline pairs on: ``mb_id`` is. Two ranks may spin
#: any number of parked iterations and stay correct so long as they RESUME
#: the pass loop on the same slot, because the proxy stamp, the ``mbs``
#: occupancy and the output pairing are all indexed by that slot and by
#: nothing else. So this gauge is what the falling-edge check reads, and
#: agreement on it is the invariant the armed window must preserve.
CHAN_SLOT = "slot"

#: #974: separates a wire's name from the MESSAGE KIND riding it, forming a
#: sub-channel of the same wire (``dict|output``). A separator rather than a
#: new axis in the file name because ``chan`` is already a free-form string
#: that names one countable stream: the sub-channel is one, so it needs no
#: new machinery, no new naming scheme, and no change to ``sweep`` (which
#: matches on the instance prefix and the ``role``/rank suffix, both of which
#: a sub-channel file carries unchanged).
KIND_SEP = "|"


def kind_channel(chan: str, kind: str) -> str:
    """The sub-channel of ``chan`` carrying only messages of ``kind``."""
    return f"{chan}{KIND_SEP}{kind}"


def kind_axis_covers(counters, chan: str, rank: int) -> bool:
    """May a reader trust the per-kind counters of ``rank`` on ``chan``?

    ONLY IF THE SENDER LABELS EVERYTHING IT POSTS, proved rather than
    assumed: the per-kind totals must account for every post the wire
    counter knows about. Anything less and the kind axis would read "no
    message of your kind is coming" for messages that simply were not
    labelled -- a false raise, which is the one direction this gate must
    never move in.

    FALSE IS ALWAYS THE SAFE ANSWER, and it is the answer for every case
    that is not provably covered: a counters object without the per-kind
    API at all (the stand-in holders across the #631/#757/#787/#789/#791/
    #795/#797/#798 test family carry only ``sent``/``attempted``/
    ``local_consumed``), a sender that has posted nothing yet, or a sender
    mixing labelled and unlabelled posts. In every one of them the caller
    falls back to the wire counter and behaves exactly as it did before
    #974 existed.

    The ``getattr`` is therefore not a default standing in for a value --
    it is the presence test itself, and its negative answer selects the
    unchanged code path rather than a guessed number.
    """
    total = getattr(counters, "kind_sent_total", None)
    if total is None:
        return False
    posted = counters.sent(chan, rank)
    if posted <= 0:
        # Nothing posted at all: both axes say the same thing, so there is
        # no reason to prefer the newer one. Keeps the "no upstream ever
        # scheduled anything" case (#789's original defect) byte-identical.
        return False
    return int(total(chan, rank)) >= int(posted)
