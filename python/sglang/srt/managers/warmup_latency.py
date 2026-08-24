# SPDX-License-Identifier: Apache-2.0
"""#856: what the no-carry flip actually costs a request, measured.

THE QUANTITY THIS TICKET IS ACCOUNTABLE FOR. The flip carries no KV. Its
residents are retracted and its prefix tree dropped, so the first requests
served after a cutover re-prefill from the hierarchical cache instead of from
device rows. That is the price paid for deleting the KV mover, the seam's
staging reserve, and the resident carry that made #825's tree reset crash.

It is a REAL price and it must not be assumed small. The design note says so
in as many words, and the user named it as the validation metric: not "rows
carried", but cutover-blocking time plus honest warm-up cost as SERVED-REQUEST
LATENCY.

NOTHING IN THE TREE MEASURED IT. The nearest prior art is
``regime_classifier.PhaseDwellGate.rounds_since_flip``, which is a GATE -- it
decides whether a flip is allowed to happen -- and carries no latency at all.
Searched: ``phase_flip_runtime`` for warm/latency/post-cutover, the seam
census (which times the seam, not what follows it), and the #605 flight
recorder (which records the seam's own peaks).

WHY BANDS AND NOT A MEAN. A mean over a phase hides exactly the shape being
asked about: the claim is that the cost is concentrated in the first rounds
after a cutover and decays as the cache warms. A mean cannot be wrong in a way
that shows that, and it cannot be right in a way that shows it either. The
bands are geometric because a warm-up that has not decayed by ~64 rounds is
not a warm-up, it is a regression.

THE STEADY BAND IS THE CONTROL. Every band is reported against it, so the
figure a reader takes away is a RATIO against this same instance's own
steady state -- not against another boot, another rig, or a remembered
number. An instrument that can only be read by comparing it to something
absent is the shape this build keeps removing.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

#: Upper bound (inclusive) of each warm band, in rounds since the cutover.
#: Anything past the last one is the steady state and is the control.
BANDS: Tuple[int, ...] = (1, 4, 16, 64)
STEADY = "steady"
NO_FLIP = "no-flip-yet"


def band_for(rounds_since_cutover: Optional[int]) -> str:
    """Which band a request completing now belongs to.

    ``None`` means no cutover has happened in this process, which is its own
    answer and must never be folded into the steady state: "this instance has
    not flipped" and "this instance has flipped and settled" are the two
    readings a warm-up ledger most needs to keep apart.
    """
    if rounds_since_cutover is None:
        return NO_FLIP
    n = int(rounds_since_cutover)
    if n < 0:
        return NO_FLIP
    for hi in BANDS:
        if n <= hi:
            return f"<={hi}"
    return STEADY


class WarmupLatencyLedger:
    """Served-request latency, tagged by rounds since the last cutover.

    Deliberately a plain accumulator over floats: no clock, no scheduler, no
    device. Everything it decides is a pure function of what it was told, so
    both the accounting and the verdict are falsifiable without a GPU -- the
    same split #852's estimator and #856's refill-bound phrase use, and for
    the same reason: a rule that can only be exercised on metal is one this
    corpus has repeatedly shipped inert.
    """

    def __init__(self) -> None:
        self._samples: Dict[str, List[float]] = {}
        self.cutovers = 0

    def note_cutover(self) -> None:
        self.cutovers += 1

    def note_request(
        self, rounds_since_cutover: Optional[int], latency_s: float
    ) -> None:
        """Record one COMPLETED request. Ignores an unreadable latency rather
        than poisoning a band with it -- this is an instrument on the serving
        path and may never be the thing that breaks a round."""
        try:
            v = float(latency_s)
        except (TypeError, ValueError):
            return
        if v < 0 or v != v:  # negative or NaN is not a measurement
            return
        self._samples.setdefault(band_for(rounds_since_cutover), []).append(v)

    def mean(self, band: str) -> Optional[float]:
        vals = self._samples.get(band)
        if not vals:
            return None
        return sum(vals) / len(vals)

    def count(self, band: str) -> int:
        return len(self._samples.get(band, ()))

    def warmup_ratio(self, band: str) -> Optional[float]:
        """This band's mean against the STEADY band's.

        ``None`` when either side is unmeasured -- and that is the honest
        answer, not 1.0. A ratio against a control that does not exist is the
        defaulted-measurement shape (#606): it reads as "no warm-up cost"
        while meaning "nothing was compared".
        """
        warm = self.mean(band)
        steady = self.mean(STEADY)
        if warm is None or steady is None or steady <= 0:
            return None
        return warm / steady

    def summary(self) -> str:
        """One line a window result can quote.

        Always says something, including that it has nothing to say: a silent
        instrument is indistinguishable from an absent one, which is the whole
        #851 defect class.
        """
        if not self._samples:
            return (
                f"#856 WARMUP LEDGER: no completed requests recorded "
                f"({self.cutovers} cutover(s)) -- nothing measured yet"
            )
        parts = []
        for band in [f"<={hi}" for hi in BANDS] + [STEADY, NO_FLIP]:
            n = self.count(band)
            if not n:
                continue
            m = self.mean(band)
            ratio = self.warmup_ratio(band)
            tail = "" if ratio is None else f" ({ratio:.2f}x steady)"
            parts.append(f"{band}: n={n} mean={m:.3f}s{tail}")
        return f"#856 WARMUP LEDGER after {self.cutovers} cutover(s) -- " + "; ".join(
            parts
        )
