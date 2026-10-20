"""S64 S-CAP: name WHY a D store read ended short, instead of one label for all.

METAL (NF rc15 D, pdflip-72-527, 08:30:42-08:30:52, TP0): four reads of a
26592-token prompt whose KV and END anchor the store held (``PARK-DEMOTE
role=handoff pool=kv pages=415 l3_pages=415 complete=1``, X gate ``store=26560``)
delivered 26176, 26048, 25536, 25408, 25152 -- each shorter than the one
before, the deliverable unchanged (26560). The fallback said
``reason=store_held_no_growth`` ("no writer is filling the tail") and #248h
said "ended short on its own capacity". Neither was the cause: the TP0 span
started at its device match (22208, 26048, 25536, 25408), the workers'
spans 384..1152 tokens lower (21824, 25536, 24512, 24256), and the length
MIN of the claim reduce handed the workers TP0's page count from THEIR start;
the completion END MIN cut the group there -- exactly 26176 / 26048 / 25536 /
25408 (S59 class A, fixed in the base by CLAIM-END). What made each read
shorter was the device tree losing its tail between the reads (eviction),
not the store and not the arena.

THE NAMES (an instrument: no decision reads them; the store-short bound, the
fallback and the #248h vote are byte-for-byte what they were):

* ``settled_no_writer``  -- the request was settled at this wake (WT);
* ``capacity_regress``   -- delivered FELL below the previous read at the same
  deliverable: the reader lost what it held (device eviction / room), the
  store did not shrink;
* ``capacity_cut``       -- the read's recorded READ-DEPTH cause is a room
  cause (arena_full, l2/fill evicted before the reference);
* ``store_held_no_growth`` -- the store holds the span (probe hit covers it)
  and the re-read gained nothing, with no room cause recorded;
* ``no_writer_progress`` -- the store itself is short and nothing grew.
"""

from __future__ import annotations

from typing import Optional

CAPACITY_CUT_CAUSES = ("arena_full", "l2_evicted_before_ref", "fill_evicted_before_ref")

SETTLED = "settled_no_writer"
CAPACITY_REGRESS = "capacity_regress"
CAPACITY_CUT = "capacity_cut"
STORE_HELD = "store_held_no_growth"
NO_WRITER = "no_writer_progress"

PREV_DELIVERED_ATTR = "_s64_prev_delivered"
PREV_DELIVERABLE_ATTR = "_s64_prev_deliverable"
CUT_CAUSE_ATTR = "_s64_cut_cause"


def classify(delivered: int, deliverable: int, prev_delivered: Optional[int],
             prev_deliverable: Optional[int], cut_cause: Optional[str], held: bool,
             settled: bool) -> str:
    if settled:
        return SETTLED
    if (prev_delivered is not None and prev_deliverable is not None
            and int(delivered) < int(prev_delivered) and int(deliverable) == int(prev_deliverable)):
        return CAPACITY_REGRESS
    if cut_cause in CAPACITY_CUT_CAUSES:
        return CAPACITY_CUT
    if held:
        return STORE_HELD
    return NO_WRITER


def note_record(req, delivered: int, deliverable: int, cut_cause: Optional[str]) -> None:
    """Before the record overwrites the request's delivered/deliverable: keep
    the previous pair (the regress test) and this read's cut cause."""
    try:
        setattr(req, PREV_DELIVERED_ATTR, getattr(req, "_pdflip_store_delivered", None))
        setattr(req, PREV_DELIVERABLE_ATTR, getattr(req, "_pdflip_store_deliverable", None))
        setattr(req, CUT_CAUSE_ATTR, cut_cause)
    except Exception:  # noqa: BLE001 - an instrument
        pass


def of_request(req, held: bool, settled: bool) -> str:
    return classify(
        int(getattr(req, "_pdflip_store_delivered", 0) or 0),
        int(getattr(req, "_pdflip_store_deliverable", 0) or 0),
        getattr(req, PREV_DELIVERED_ATTR, None),
        getattr(req, PREV_DELIVERABLE_ATTR, None),
        getattr(req, CUT_CAUSE_ATTR, None),
        bool(held), bool(settled),
    )
