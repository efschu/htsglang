"""Input tokens per class: aus Cache / neu gerechnet (P, D) / Übergabe P->D.

User order 29.09. ~13:40Z: "aus Cache" counts only tokens whose KV existed BEFORE the request --
the prefix hit at admission.  When D, after the flip, reads from L2/L3 the KV that P has just
prefilled for the SAME request, that is the hand-over P->D: neither a cache hit nor a
recomputation.  It is its own class and never counts as cache.  Every prompt token lands in
exactly one of: cache, P-computed, hand-over; D-computed is D's real extra prefill work (the
rest D re-extends after the hand-over, or the whole uncached part of a D-direct request).

The distinguishing fact is the request id: the front runs leg 1 on P for a rid
(``Pending.leg1_ran``, front.py) before it posts leg 2 of the SAME rid to D.  So a D leg whose rid
had a P leg 1 is a P->D request and its ``cached_tokens`` is the hand-over; a D leg without one is
a D-direct request (small prefill under X) and its ``cached_tokens`` is a real prefix hit.  A KV
that an EARLIER request prefilled on P and handed over is real cache for a LATER rid -- it existed
before that rid's admission -- and the rule gets that right because it keys on the rid, not on
the tier the hit came from.

Two sources, one rule:

* IPC (preferred): ``state.json front.served_tokens`` with the row ``D_after_P`` (subset of ``D``,
  written by the front beside the ``D`` row when ``pending.leg1_ran``; agreed with NF 29.09.,
  RANKSTATS-S3-SCHEMA-0929 "Vorschlag DASHBOARD-GRAFIKEN").  :func:`from_served_tokens`.
* Log (Übergang, until that row is in the image): the front's ``WEG2-SERVED group=P leg=1 rid=``
  and ``group=D leg=2 rid=`` lines, paired by rid.  :func:`split_legs`.

``est_uncached`` / ``agent_span`` is NOT a source: it is the admission-time estimate (character
based, and #49 credits the prompt of a request still in flight as held), never a measured hit.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Set, Tuple

CLASSES = ("cache", "comp_p", "comp_d", "handoff")
LABELS = {"cache": "aus Cache", "comp_p": "neu gerechnet P", "comp_d": "neu gerechnet D",
          "handoff": "Übergabe P→D"}


def blank() -> Dict[str, int]:
    return {c: 0 for c in CLASSES}


def _i(x) -> int:
    try:
        return max(0, int(x or 0))
    except (TypeError, ValueError):
        return 0


def classify_leg(group: str, rid: Optional[str], prompt, cached, p_rids: Set[str]) -> Dict[str, int]:
    """One served leg -> its token classes.

    ``p_rids``: the rids that had a P leg 1 in this boot (the front serves leg 1 before leg 2 of the
    same rid, so by the time a D leg is classified its rid is in the set iff P prefilled it).
    """
    out = blank()
    pt, ct = _i(prompt), _i(cached)
    ct = min(ct, pt) if pt else ct
    if group == "P":
        out["cache"] += ct
        out["comp_p"] += max(0, pt - ct)
    elif group == "D":
        if rid is not None and rid in p_rids:
            out["handoff"] += ct
        else:
            out["cache"] += ct
        out["comp_d"] += max(0, pt - ct)
    return out


def split_legs(legs: Iterable[Tuple], p_rids: Optional[Set[str]] = None) -> Dict[str, int]:
    """Sum over served legs ``(t, group, leg, rid, prompt, cached, completion)``.

    ``p_rids`` is updated in place with the P leg-1 rids seen (pass the boot's set to carry it
    across calls; a fresh set is used when omitted, which is right when all legs are given)."""
    if p_rids is None:
        p_rids = set()
    legs = sorted(legs, key=lambda x: x[0])
    tot = blank()
    for t, group, leg, rid, prompt, cached, _comp in legs:
        if group == "P" and rid is not None:
            p_rids.add(rid)
        for k, v in classify_leg(group, rid, prompt, cached, p_rids).items():
            tot[k] += v
    return tot


def has_handoff_row(served_tokens) -> bool:
    return isinstance(served_tokens, dict) and isinstance(served_tokens.get("D_after_P"), dict)


def from_served_tokens(st: dict) -> Optional[Dict[str, int]]:
    """Cumulative classes from ``front.served_tokens`` (IPC).  None when the ``D_after_P`` row is
    missing -- without it D's cached share cannot be split, and a D total would count the
    hand-over as cache (the trap this module exists for)."""
    if not has_handoff_row(st):
        return None
    p = st.get("P") or {}
    d = st.get("D") or {}
    dp = st.get("D_after_P") or {}
    p_prompt, p_cached = _i(p.get("prompt")), _i(p.get("cached"))
    d_prompt, d_cached = _i(d.get("prompt")), _i(d.get("cached"))
    h = min(_i(dp.get("cached")), d_cached)
    return {"cache": p_cached + (d_cached - h),
            "comp_p": max(0, p_prompt - p_cached),
            "comp_d": max(0, d_prompt - d_cached),
            "handoff": h}


TIERS = ("device", "host", "storage", "unassigned")


def tiers_from_served_tokens(st: dict) -> Optional[Dict[str, int]]:
    """Per-tier split of the CACHE class: device, host (= L2), storage (= L3), and ``unassigned`` --
    the cached tokens of answers that carried no CachedTokensDetails (front 9266bdfb8d adds to a
    row's ``cached_tier`` only when the answer carried the detail; today only /generate does).
    The hand-over's tiers are taken out (D-direct = D - D_after_P).  None when there is nothing to
    split, or when D carries tiers but its D_after_P subset does not (the hand-over could then not
    be taken out of D's tiers)."""
    if not has_handoff_row(st):
        return None
    rows = {g: (st.get(g) or {}) for g in ("P", "D", "D_after_P")}
    tier = {g: r.get("cached_tier") if isinstance(r.get("cached_tier"), dict) else None for g, r in rows.items()}
    if not any(tier.values()):
        return None
    if tier["D"] is not None and tier["D_after_P"] is None and _i(rows["D_after_P"].get("cached")):
        return None
    z = {k: 0 for k in TIERS[:3]}
    t = {g: (v or z) for g, v in tier.items()}
    out = {k: _i(t["P"].get(k)) + max(0, _i(t["D"].get(k)) - _i(t["D_after_P"].get(k))) for k in TIERS[:3]}
    cache = (from_served_tokens(st) or {}).get("cache", 0)
    out["unassigned"] = max(0, cache - sum(out.values()))
    return out


def delta(prev: Optional[Dict[str, int]], cur: Dict[str, int]) -> Dict[str, int]:
    """Counter delta; a class that went backwards is a restart of the counter (the new value)."""
    if prev is None:
        return blank()
    out = {}
    for k in CLASSES:
        a, b = cur.get(k, 0), prev.get(k, 0)
        out[k] = a - b if a >= b else a
    return out


def hit_share(tot: Dict[str, int]) -> Optional[float]:
    """Cache share of the input tokens that entered a prefill decision: cache / (cache + P + D).
    The hand-over is not in the denominator -- it is the same tokens P computed, carried over."""
    den = tot.get("cache", 0) + tot.get("comp_p", 0) + tot.get("comp_d", 0)
    return (tot.get("cache", 0) / den) if den else None
