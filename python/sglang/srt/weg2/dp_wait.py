"""R28: one named line per request that waited for a D->P flip.

Release table row 28 ("long D->P waits under agent load") could only be read
by joining five line shapes of the front log per request: ``WEG2-ROUTE ...
BATCH queued (awake=D``, ``WEG2-FAIRNESS``, ``WEG2 MIN-DWELL``, ``WEG2-FLIP
begin``/``done`` and the ``WEG2-SERVED group=D`` lines of whatever D decoded in
between. Boot rc9i (dkrnfbar1agent09252237, front.log:536-614) shows why the
join is needed: weg2-4-5 queued at 22:47:14, waited 45.1 s for the fairness
bound (front.log:562), then 56.0 s in the flip's own drain for weg2-0-2, which
the admitter had handed to D 1.3 s before the bound fired (front.log:560), and
P woke 103.5 s after the arrival -- and no single line says so.

This module holds the pure part: the arrival snapshot, the decomposition of
the wait into its three consecutive intervals, and the line. The front takes
the snapshot where it queues a request while D is awake and prints the line
at ``WEG2-FLIP done`` of the D->P flip that ends the wait. No routing,
admission or flip decision reads anything here; it is an instrument only.

Decomposition (all wall seconds, consecutive, summing to ``wait_s``):

* ``hold_s``  arrival -> ``WEG2-FLIP begin``: the controller did not start the
  flip -- D still had work (a running decode, a hand-off, requests handed to
  D by the admitter after the arrival), MIN-DWELL held, or an earlier D->P
  flip returned without flipping (DRAIN WAITING / W1).
* ``drain_s`` flip begin (or the arrival, if later) -> end of ``drain(D)``: the
  flip waited for decodes already running on D (the fairness path begins the
  flip with ``outstanding>0`` and drains to the end, #1011).
* ``flip_s``  the rest, up to P awake: quiesce, the sleep/wake legs.

``hold_by`` names what the hold consisted of, from counters snapshotted at
arrival and read again at the end, so it costs no per-tick bookkeeping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional

#: counters the front already keeps (or gains with R28), read at arrival and
#: again when the wait ends; a rise names a component of the hold.
SNAP_KEYS = (
    "fairness_bound_hits",  # WEG2-FAIRNESS fired
    "min_dwell_holds",      # WEG2 MIN-DWELL src=D dst=P verdict=hold (R28)
    "d_admits",             # WEG2 D-ADMIT (R28, monotone over the boot)
    "weg2_drain_waiting",   # a D->P flip returned: DRAIN WAITING
    "W1_Weg2DrainRefused",  # a D->P flip returned: W1
)


@dataclass
class DpArrival:
    """What the front knew when the request joined the queue while D was awake."""

    t: float
    origin: str             # "long" | "batch" | "carrier" | "reroute"
    flipping: bool          # a D->P flip was already open at the arrival
    d_outstanding: int
    handoff: int
    ready_for_d: int
    snap: Dict[str, int]


def take(t: float, origin: str, flipping: bool, d_outstanding: int, handoff: int,
         ready_for_d: int, counters: Mapping[str, int]) -> DpArrival:
    return DpArrival(t=float(t), origin=str(origin), flipping=bool(flipping),
                     d_outstanding=int(d_outstanding), handoff=int(handoff),
                     ready_for_d=int(ready_for_d),
                     snap={k: int(counters.get(k, 0) or 0) for k in SNAP_KEYS})


def decompose(a: DpArrival, t_flip0: float, t_drain_end: Optional[float],
              t_awake: float) -> Dict[str, float]:
    """The three consecutive intervals; they sum to ``wait_s`` exactly."""
    wait = max(0.0, t_awake - a.t)
    hold = min(wait, max(0.0, t_flip0 - a.t))
    start = max(a.t, t_flip0)
    end_drain = t_drain_end if t_drain_end is not None else start
    drain = min(wait - hold, max(0.0, end_drain - start))
    flip = max(0.0, wait - hold - drain)
    return {"wait_s": wait, "hold_s": hold, "drain_s": drain, "flip_s": flip}


def hold_by(a: DpArrival, counters: Mapping[str, int]) -> str:
    """What the hold consisted of, '+'-joined; ``flip-open`` when the request
    arrived inside an open D->P flip (it had no hold of its own)."""
    if a.flipping:
        return "flip-open"
    rose = {k: int(counters.get(k, 0) or 0) - a.snap.get(k, 0) for k in SNAP_KEYS}
    parts = []
    if a.d_outstanding or a.handoff or a.ready_for_d or rose["d_admits"] > 0:
        parts.append("d-work")
    if rose["fairness_bound_hits"] > 0:
        parts.append("fairness")
    if rose["min_dwell_holds"] > 0:
        parts.append("min-dwell")
    if rose["weg2_drain_waiting"] > 0 or rose["W1_Weg2DrainRefused"] > 0:
        parts.append("flip-returned")
    return "+".join(parts) if parts else "none"


def dominant(dec: Mapping[str, float]) -> str:
    return max(("hold", "drain", "flip"), key=lambda k: dec[k + "_s"])


def line(rid: str, epoch: int, a: DpArrival, dec: Mapping[str, float], by: str,
         counters: Mapping[str, int], est_prompt: int, est_uncached: int,
         presence_span: int, span_known: bool) -> str:
    admitted = int(counters.get("d_admits", 0) or 0) - a.snap.get("d_admits", 0)
    return (
        "WEG2 DP-WAIT rid=%s epoch=%d wait_s=%.1f hold_s=%.1f drain_s=%.1f flip_s=%.1f "
        "dominant=%s hold_by=%s origin=%s at_arrival: d_outstanding=%d handoff=%d "
        "ready_for_d=%d; d_admitted_during_wait=%d est_prompt=%d uncached=%d "
        "presence_span=%d span_known=%s (R28: arrival while D awake -> P awake; "
        "hold = before WEG2-FLIP begin, drain = the flip waiting for D's running "
        "decodes, flip = the rest; the three sum to wait_s)"
        % (rid, int(epoch), dec["wait_s"], dec["hold_s"], dec["drain_s"], dec["flip_s"],
           dominant(dec), by, a.origin, a.d_outstanding, a.handoff, a.ready_for_d,
           max(0, admitted), int(est_prompt), int(est_uncached), int(presence_span),
           bool(span_known))
    )
