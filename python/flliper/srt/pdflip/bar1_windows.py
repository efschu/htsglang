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
* **What the aperture allows.** The tightest card of the launch decides. When
  the (scaled) windows do not sum into its usable aperture, every window is
  scaled DOWN proportionally (more rounds per collective, the same behaviour
  the runtime shows when it grants less than requested) -- down to
  :data:`MIN_WINDOW_MIB` per window; below that the launch is refused BY NAME
  (``BAR1-WINDOW``) with the arithmetic. A card whose BAR1 NVML did not report
  cannot be checked: the reference windows stand, named UNCHECKED (the runtime
  gate ``Bar1WindowRefused`` still stands behind it).

``MIN_WINDOW_MIB`` and the factor are ESTIMATES (S): the measured anchors are
the reference rig's N = 3 / 256 MiB figures above; nothing else is measured.
PURE: stdlib only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

#: the shipped windows (launcher P_BARLINK_BAR1_WINDOW_MIB / argv_d / DUAL_P_...)
P_REFERENCE = "24,PP_0=96"
D_REFERENCE = "16,TP_0=32,DCP_0=40"
P_DUAL_REFERENCE = "16,PP_0=64"
#: runtime reserve (barlink_matrix_transport.RESERVE_MIB_DEFAULT)
RESERVE_MIB = 32
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

    def line(self) -> str:
        return (f"BAR1-WINDOW P='{self.p}' D='{self.d}' demand {self.demand_mib} MiB of "
                + ("unknown aperture (UNCHECKED)" if self.usable_mib is None else
                   f"{self.usable_mib} MiB usable (tightest BAR1 {self.bar1_min_mib} - {RESERVE_MIB})")
                + (f", scale {self.scale:.2f}" if abs(self.scale - 1.0) > 1e-9 else "")
                + (f" -- {self.why}" if self.why else ""))


def plan(n_cards: int, bar1_mib: Sequence[Optional[int]], dual: bool = False,
         p_base: Optional[str] = None, d_base: Optional[str] = None) -> WindowPlan:
    """The windows of an ``n_cards`` launch. ``bar1_mib``: BAR1 total of each
    card (None = not reported). ``p_base`` / ``d_base``: the windows to start
    from (default: the shipped ones; a dual launch starts from the dual P)."""
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
    known = [int(b) for b in bar1_mib if b is not None]
    if not known:
        return WindowPlan(True, p_spec, d_spec, None, None, sum(v for _, _, v in entries), 1.0,
                          "no card reported its BAR1: the shipped windows stand unchecked "
                          "(the runtime gate Bar1WindowRefused still applies)")
    tight = min(known)
    usable = tight - RESERVE_MIB
    scale = 1.0 if demand <= usable else max(usable, 0) / float(demand)
    out: List[Tuple[str, str, int]] = []
    for g, k, v in grown:
        w = v if scale >= 1.0 else int(v * scale)
        out.append((g, k, w))
    low = [(g, k, w) for g, k, w in out if w < MIN_WINDOW_MIB]
    if low:
        smallest = max(1, min(v for _, _, v in grown))
        need_bar1 = -(-demand * MIN_WINDOW_MIB // smallest) + RESERVE_MIB
        return WindowPlan(
            False, p_spec, d_spec, tight, usable, demand, scale,
            f"{n} cards on a {tight} MiB BAR1: the windows that keep the reference round count "
            f"({demand} MiB for {n} ranks, reference {sum(v for _, _, v in entries)} MiB for "
            f"{REFERENCE_RANKS}) do not fit the {usable} MiB usable aperture "
            f"(BAR1 {tight} - reserve {RESERVE_MIB}); scaled to {scale:.2f} the window "
            f"{', '.join(f'{g}:{k or 'default'}={w}' for g, k, w in low)} falls under the "
            f"{MIN_WINDOW_MIB} MiB floor (estimate, S). Needs a BAR1 of at least "
            f"{need_bar1} MiB (ReBAR) on the tightest card, or fewer cards")
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
        why = (f"derived for {n} ranks on a {tight} MiB BAR1 from the reference windows "
               f"('{p_spec}' / '{d_spec}', measured N = {REFERENCE_RANKS}, 256 MiB BAR)")
    return WindowPlan(True, p_new, d_new, tight, usable, demand, scale, why)
