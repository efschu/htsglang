"""HW-P1c 1003 (BAR1-WINDOW): the BAR1 group windows of an N-card launch, derived
from the card count and the MEASURED BAR1 of the cards instead of the 3-card
rig's constants.

User order 03.10. ~19:30Z: "unsere software muss mit beliebiger anzahl an
karten ... laufen". The windows ``--barlink-bar1-window-mib`` of group P
(``24,PP_0=96``; dual ``16,PP_0=64``) and group D (``16,TP_0=32,DCP_0=40``) were
tuned on metal for N = 3 on 256 MiB BARs (#1234 C1: "P 24+96 and D 16+32+40 =
208 of the 224 MiB usable per 3080, measured Used 224/256 including the RM
carve-out"). This module keeps THAT arithmetic and makes its inputs variable:

* **What a window is.** The size of the receive region a communicator exposes
  on its OWN card's BAR1 aperture (barlink_bar1.geometry); the peers map it, so
  the aperture a card spends is the SUM of the windows of the communicators
  that live on it -- independent of how many peers write into it. The usable
  aperture of a card is ``bar1_total - RESERVE`` (RESERVE = 32 MiB, the
  runtime's own ``barlink_matrix_transport.RESERVE_MIB_DEFAULT``): 224 on a
  3080, which the reference windows fill to 208.
* **What N changes.** A mesh/ring collective of R ranks needs ``2(R-1)`` slots
  of ``ceil(payload/R)`` (``barlink_bar1.window_requirement``) and the region
  holds ``6(R-1)`` slots, so the window that keeps the REFERENCE round count
  grows with ``(R-1)``: factor ``(R-1)/2`` against R = 3. Point-to-point
  windows (``PP_*``: the pipeline send of hidden states) do not depend on R.
  Below R = 3 the reference windows already carry more per slot than needed
  and are kept (byte-identical argv); above, the windows are scaled UP by that
  factor.
* **What the aperture allows.** The tightest card of the launch decides, and
  the aperture a card can spend is the BAR1 that is FREE before the launch
  (``nvmlDeviceGetBAR1MemoryInfo().bar1Free``: the baseline use of a display
  seat or a sibling process is already gone from it), not the BAR1 total; the
  total stands in only for a card whose free BAR1 NVML did not report. The
  runtime clips each window to ``bar1Free - RESERVE`` at the moment it maps it
  and every mapping costs :data:`OVERHEAD_MIB` of aperture on top of its size
  (measured), but the LAST window's overhead never has to fit under the
  reserve, so the constraint a card has to meet is::

      sum(windows) + OVERHEAD_MIB * (k - 1)  <=  free_before_launch - RESERVE_MIB

  (k = number of windows, P and D entries together). When the (scaled) windows
  do not meet it, every window is scaled DOWN proportionally to
  ``(usable - OVERHEAD_MIB * (k - 1)) / sum(windows)`` (more rounds per
  collective, the same behaviour the runtime shows when it grants less than
  requested) -- down to :data:`MIN_WINDOW_MIB` per window; below that the launch
  is refused BY NAME (``BAR1-WINDOW``) with the arithmetic, naming free and
  baseline BAR1. A card whose BAR1 NVML did not report cannot be checked: the
  reference windows stand, named UNCHECKED (the runtime gate
  ``Bar1WindowRefused`` still stands behind it). The reference rig (3 x 256 MiB
  BAR, free unknown or >= 254): 208 + 14 = 222 <= 254 - 32, the shipped windows.

``MIN_WINDOW_MIB`` and the factor are ESTIMATES (S): the measured anchors are
the reference rig's N = 3 / 256 MiB figures above and :data:`OVERHEAD_MIB`;
nothing else is measured.
PURE: stdlib only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

#: the shipped windows (launcher P_BARLINK_BAR1_WINDOW_MIB / argv_d / DUAL_P_...)
P_REFERENCE = "24,PP_0=96"
D_REFERENCE = "16,TP_0=32,DCP_0=40"
P_DUAL_REFERENCE = "16,PP_0=64"
#: runtime reserve (barlink_matrix_transport.RESERVE_MIB_DEFAULT)
RESERVE_MIB = 32
#: aperture one mapped window costs on top of its size (MiB, measured). Boot
#: dkr27bnvfp4duobar1duo10230534 (23.10.), RTX 5090 with a 256 MiB BAR1 and
#: 15 MiB baseline use: 256 - 15 - windows 24+96+16+32 (= 168 nominal) = 73 MiB
#: expected free, NVML showed 59 -> 14 MiB over 4 windows = 3.5 MiB per window.
#: Cross-check, reference rig (3080, 256 MiB, ~1 MiB baseline): 5 windows of
#: 208 MiB nominal, "measured Used 224/256 incl. RM carve-out". The overhead of
#: the LAST window never has to fit under the reserve, hence ``(k - 1)`` below.
OVERHEAD_MIB = 3.5
#: the card count the reference windows were measured on
REFERENCE_RANKS = 3
#: smallest window a communicator is given before the launch is refused (S)
MIN_WINDOW_MIB = 8
#: BARLINK_BAR1_MAX_RANKS
MAX_RANKS = 8


def parse_windows(spec: str) -> Tuple[int, Dict[str, int]]:
    """``"16,TP_0=32,DCP_0=40"`` -> ``(16, {"TP_0": 32, "DCP_0": 40})``; the bare
    number is the default window (0 = none stated)."""
    default = 0
    named: Dict[str, int] = {}
    for part in str(spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            k, _, v = part.partition("=")
            named[k.strip().upper()] = int(v)
        else:
            default = int(part)
    return default, named


def format_windows(default: int, named: Dict[str, int]) -> str:
    return ",".join(([str(default)] if default else []) + [f"{k}={v}" for k, v in named.items()])


def _is_p2p(key: str) -> bool:
    return key.upper().startswith("PP")


def _grow(value: int, key: str, ranks: int) -> int:
    """The window that keeps the reference round count at ``ranks`` ranks."""
    if _is_p2p(key) or ranks <= REFERENCE_RANKS:
        return int(value)
    return -(-int(value) * (ranks - 1) // (REFERENCE_RANKS - 1))


@dataclass(frozen=True)
class WindowPlan:
    ok: bool
    p: str
    d: str
    #: tightest card: BAR1 total / usable (MiB); None = no card reported its BAR1
    bar1_min_mib: Optional[int]
    usable_mib: Optional[int]
    demand_mib: int
    scale: float
    #: why not ok, or the note of a derived / unchecked plan
    why: str
    #: tightest card, when its FREE BAR1 (not the total) was the aperture: free
    #: BAR1 before the launch (MiB); None = the total stood in
    free_min_mib: Optional[int] = None
    #: mapping overhead the plan charged: OVERHEAD_MIB * (k - 1) (MiB)
    overhead_mib: float = 0.0

    def line(self) -> str:
        if self.usable_mib is None:
            aperture = "unknown aperture (UNCHECKED)"
        elif self.free_min_mib is not None:
            base = (f", baseline use {self.bar1_min_mib - self.free_min_mib} of BAR1 {self.bar1_min_mib}"
                    if self.bar1_min_mib is not None and self.bar1_min_mib != self.free_min_mib else "")
            aperture = (f"{self.usable_mib} MiB usable (tightest card: free BAR1 {self.free_min_mib}"
                        f"{base} - {RESERVE_MIB})")
        else:
            aperture = f"{self.usable_mib} MiB usable (tightest BAR1 {self.bar1_min_mib} - {RESERVE_MIB})"
        return (f"BAR1-WINDOW P='{self.p}' D='{self.d}' demand {self.demand_mib} MiB"
                + (f" + {self.overhead_mib:g} MiB mapping overhead" if self.overhead_mib else "")
                + " of " + aperture
                + (f", scale {self.scale:.2f}" if abs(self.scale - 1.0) > 1e-9 else "")
                + (f" -- {self.why}" if self.why else ""))


def plan(n_cards: int, bar1_mib: Sequence[Optional[int]], dual: bool = False,
         p_base: Optional[str] = None, d_base: Optional[str] = None,
         bar1_free_mib: Optional[Sequence[Optional[int]]] = None) -> WindowPlan:
    """The windows of an ``n_cards`` launch. ``bar1_mib``: BAR1 total of each
    card (None = not reported). ``bar1_free_mib``: the BAR1 FREE of each card
    before the launch (None / a None entry = not reported: the total stands in
    for that card). ``p_base`` / ``d_base``: the windows to start from (default:
    the shipped ones; a dual launch starts from the dual P).

    The constraint of a card: ``sum(windows) + OVERHEAD_MIB * (k - 1) <=
    aperture - RESERVE_MIB`` with ``aperture`` = its free BAR1 (total when free
    is unknown); the tightest card decides."""
    n = int(n_cards)
    p_spec = p_base if p_base is not None else (P_DUAL_REFERENCE if dual else P_REFERENCE)
    d_spec = d_base if d_base is not None else D_REFERENCE
    entries: List[Tuple[str, str, int]] = []   # (group, key, MiB)
    for grp, spec in (("P", p_spec), ("D", d_spec)):
        default, named = parse_windows(spec)
        if default:
            entries.append((grp, "", default))
        entries.extend((grp, k, v) for k, v in named.items())
    grown = [(g, k, _grow(v, k, n)) for g, k, v in entries]
    demand = sum(v for _, _, v in grown)
    k_windows = len(grown)
    overhead = OVERHEAD_MIB * max(k_windows - 1, 0)
    frees = list(bar1_free_mib) if bar1_free_mib is not None else []
    # (aperture, total, free) of every card that reported something
    cards_ap: List[Tuple[int, Optional[int], Optional[int]]] = []
    for i, t in enumerate(bar1_mib):
        f = frees[i] if i < len(frees) else None
        if f is None and t is None:
            continue
        ap = min(int(t), int(f)) if (t is not None and f is not None) else int(f if f is not None else t)
        cards_ap.append((ap, None if t is None else int(t), None if f is None else int(f)))
    if not cards_ap:
        return WindowPlan(True, p_spec, d_spec, None, None, sum(v for _, _, v in entries), 1.0,
                          "no card reported its BAR1: the shipped windows stand unchecked "
                          "(the runtime gate Bar1WindowRefused still applies)")
    tight, tight_total, tight_free = min(cards_ap, key=lambda x: x[0])
    usable = tight - RESERVE_MIB
    bar1_min = tight_total if tight_total is not None else tight
    free_used = tight_free if tight_free is not None and (tight_total is None or tight_free <= tight_total) else None
    fits = demand + overhead <= usable
    scale = 1.0 if fits else max(usable - overhead, 0) / float(demand)
    out: List[Tuple[str, str, int]] = []
    for g, k, v in grown:
        w = v if scale >= 1.0 else int(v * scale)
        out.append((g, k, w))
    low = [(g, k, w) for g, k, w in out if w < MIN_WINDOW_MIB]
    ap_txt = f"free BAR1 {free_used}" if free_used is not None else f"BAR1 {tight}"
    where = ap_txt + (f" of {tight_total} (baseline use {tight_total - free_used})"
                      if free_used is not None and tight_total is not None else "")
    if low:
        smallest = max(1, min(v for _, _, v in grown))
        need = -(-demand * MIN_WINDOW_MIB // smallest) + RESERVE_MIB + math.ceil(overhead)
        return WindowPlan(
            False, p_spec, d_spec, bar1_min, usable, demand, scale,
            f"{n} cards on a tightest card with {where}: the windows that keep the reference round count "
            f"({demand} MiB for {n} ranks, reference {sum(v for _, _, v in entries)} MiB for "
            f"{REFERENCE_RANKS}) plus {overhead:g} MiB mapping overhead ({k_windows} windows) do not fit the "
            f"{usable} MiB usable aperture ({ap_txt} - reserve {RESERVE_MIB}); scaled to "
            f"{scale:.2f} the window "
            f"{', '.join(f'{g}:{k or 'default'}={w}' for g, k, w in low)} falls under the "
            f"{MIN_WINDOW_MIB} MiB floor (estimate, S). Needs {need} MiB of "
            f"{'free ' if free_used is not None else ''}BAR1 at least (ReBAR / close the display "
            f"seat) on the tightest card, or fewer cards",
            free_min_mib=free_used, overhead_mib=overhead)
    res: Dict[str, Tuple[int, Dict[str, int]]] = {"P": (0, {}), "D": (0, {})}
    for g, k, w in out:
        d_, nm = res[g]
        if k:
            nm[k] = w
        else:
            d_ = w
        res[g] = (d_, nm)
    p_new, d_new = format_windows(*res["P"]), format_windows(*res["D"])
    why = ""
    if (p_new, d_new) != (p_spec, d_spec):
        why = (f"derived for {n} ranks on a tightest card with {where} from the reference windows "
               f"('{p_spec}' / '{d_spec}', measured N = {REFERENCE_RANKS}, 256 MiB BAR)")
    return WindowPlan(True, p_new, d_new, bar1_min, usable, demand, scale, why,
                      free_min_mib=free_used, overhead_mib=overhead)
