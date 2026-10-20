"""ZERO-ANCHOR 1015: two per-rid lines that give every "cache hit ignored"
case of the Prefill-Cache monitor a cause (an instrument -- it decides
nothing, every number it prints was already computed and then discarded).

THE GAP (NF h16p 226aad7105 + the 15.10. boots, replay of the corrected
monitor, report h98): 17 of 74 cases (P leg 1, store presence >= cached + 4096)
carried NO P-log marker, and the largest class (BELOW-ANCHOR, 38) was named
after the wrong term:

* ``#1028B FETCH CAP kv= claimed= lost= caps=`` -- the component cap of the
  store's answer (KV present, the QSA index or the mamba anchor not) -- carries
  NO rid and is rate limited (40 per process, then every 256th; PP0 printed 40
  in 6 of 16 boots). pdflip-5-52 (boot 133855, 108608 tokens): ``kv=1730
  claimed=27 caps={qsa_indexer: 47, mamba: 27}`` -- joinable only by the
  neighbouring ``#257 PROBE-HOLD rid=`` line. pdflip-40-179 (60352 tokens): the
  same line came as ``n=30`` with ``kv=865`` (the store's KV prefix itself
  ended at the fork) -- never joined to the rid.
* a claim the store answers SHORT of the asked span with no component cap
  (``final == kv``, so ``#1028B`` stays silent and ``#1035 R13`` only speaks
  at kv == 0): pdflip-0-6 (boot 122948) read 47 of 576 pages while the front's
  ``probe=fast`` presence named 576 -- no line at all.
* ``#257 PREFETCH BELOW-ANCHOR`` is rate limited too (16 per process) and
  prints ``read=`` / ``anchored=`` but not what the cut COST against what the
  short landing cost: over the 35 attributable BELOW-ANCHOR cases the cut to
  the deepest anchor inside the landed pages was 78784 tokens (4.1 %), the
  pages that never landed 1915520 (95.9 %). The label points at the anchor,
  the tokens are the landing's (``PDFLIP-READ-DEPTH cause=arena_full`` etc.).

THE LINES (rank-local, never rate limited, each at most once per read)::

    PDFLIP-CLAIM-SHORT rid=<rid> pp=<rank> asked=<pages> kv=<pages> claimed=<pages>
        lost=<pages> short=<pages> by=<pool,...|kv_prefix> tokens=<claimed*page>

  once per store probe whose claim ends >= ``min_pages`` short of the asked
  span (default 64 pages = the monitor's 4096-token gap) or whose KV prefix a
  component cut (lost > 0). ``by`` names the binding pool(s); ``kv_prefix``
  = the store's KV prefix itself ended there (no component cut).

    PDFLIP-ANCHOR-CUT rid=<rid> pp=<rank> read=<tokens> of=<tokens>
        anchored=<tokens> cut_to=<tokens> cut_loss=<tokens> unlanded=<tokens>
        landing=<read-depth cause|->

  once per #257 cut: ``cut_loss`` = read - cut (the KV landed above the
  deepest reachable anchor, given back), ``unlanded`` = of - read (the pages
  the short landing never brought). ``landing`` is the READ-DEPTH cause of the
  same operation (the walk's first unlanded page), so the monitor attributes
  the case to what cost the tokens.

Switch: ``FLLIPER_PDFLIP_CLAIM_LINES`` (default ON; ``0`` = no line, byte-equal
log otherwise)."""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_N = {"claim": 0, "cut": 0}


def enabled() -> bool:
    return os.environ.get("FLLIPER_PDFLIP_CLAIM_LINES", "1").strip().lower() not in (
        "0", "false", "no", "off", "")


def min_pages() -> int:
    try:
        return max(1, int(os.environ.get("FLLIPER_PDFLIP_CLAIM_LINE_MIN_PAGES", "64")))
    except ValueError:
        return 64


def _binding(hit_result, claimed: int) -> str:
    """The pool(s) that set the claim: every capping pool whose boundary IS
    the claim, plus every pool capped to exactly 0 (absent from the hit dict
    by construction -- ``zero_capped_pools``)."""
    names = []
    hits = getattr(hit_result, "extra_pool_hit_pages", None) or {}
    for name, v in hits.items():
        s = str(getattr(name, "value", name))
        if s == "kv":
            continue
        try:
            if int(v) == int(claimed):
                names.append(s)
        except (TypeError, ValueError):
            continue
    for s in getattr(hit_result, "zero_capped_pools", ()) or ():
        s = str(getattr(s, "value", s))
        if s not in names:
            names.append(s)
    return ",".join(sorted(names)) or "kv_prefix"


def claim_short(operation, hit_result, page_size: int, pp_rank) -> Optional[str]:
    """Emit ``PDFLIP-CLAIM-SHORT`` for the store probe of ``operation``;
    returns the ``by`` term (None = not short, nothing emitted). Never raises."""
    try:
        if not enabled() or hit_result is None:
            return None
        asked = int(getattr(hit_result, "keys_asked", 0) or 0)
        claimed = int(getattr(hit_result, "kv_hit_pages", 0) or 0)
        kv = int(getattr(hit_result, "kv_uncapped", 0) or 0)
        kv = max(kv, claimed)
        lost = kv - claimed
        short = asked - claimed
        if asked <= 0 or (lost <= 0 and short < min_pages()):
            return None
        by = _binding(hit_result, claimed) if lost > 0 else "kv_prefix"
        _N["claim"] += 1
        logger.warning(
            "PDFLIP-CLAIM-SHORT rid=%s pp=%s asked=%d kv=%d claimed=%d lost=%d short=%d by=%s "
            "tokens=%d (n=%d): the store answered this read short of its span -- lost = KV "
            "pages present but cut by a component (by= names it), kv_prefix = the store's KV "
            "itself ends there",
            getattr(operation, "request_id", "?"), pp_rank, asked, kv, claimed, lost, short, by,
            claimed * max(1, int(page_size)), _N["claim"],
        )
        return by
    except Exception:  # noqa: BLE001 - an instrument never breaks a probe
        return None


def reuse_gap(est: int, presence: int, presence_kv: int, cached: int) -> dict:
    """The type-A split (front ``P-PREFIX-REUSE``, no rid): one P leg's
    ``forgone = est - cached`` cut into what each term cost, in tokens.

    * ``read_short`` -- the anchored store presence the arrival probe saw
      lay above ``cached``: the read did not land it (the P-log lines of the
      same rid -- CLAIM-SHORT / READ-DEPTH / ANCHOR-CUT -- say why);
    * ``kv_above_anchor`` -- KV pages the store held above its deepest anchor
      (``kv_pages`` > ``pages`` of the probe): no state to resume there;
    * ``est_over_store`` -- the front's text-span estimate above everything
      the store held (an instrument term, not a recompute of stored KV).

    Boot 133855 epoch 103 (forgone 6912): pdflip-102-516 est 67840 presence
    65792 kv 67840 cached 65792 -> kv_above_anchor 2048; pdflip-102-517 est
    50176 presence 45312 kv 45824 cached 45312 -> kv_above_anchor 512,
    est_over_store 4352."""
    est, presence, cached = int(est), int(presence), int(cached)
    kv = max(int(presence_kv), presence)
    top = max(est, cached)
    read_short = max(0, min(presence, top) - cached)
    kv_above = max(0, min(kv, top) - max(presence, cached))
    over = max(0, est - max(kv, cached))
    return {"read_short": read_short, "kv_above_anchor": kv_above, "est_over_store": over}


def anchor_cut(req_id, pp_rank, read: int, of: int, anchored: int, cut: int,
               landing=None) -> Optional[str]:
    """Emit ``PDFLIP-ANCHOR-CUT`` for one #257 cut; returns the line's landing
    term (None = switched off). Never raises."""
    try:
        if not enabled():
            return None
        read, of, cut = int(read), int(of), int(cut)
        land = "-"
        if landing is not None:
            try:
                land = str(landing[1])
            except (TypeError, IndexError):
                land = str(landing)
        _N["cut"] += 1
        logger.warning(
            "PDFLIP-ANCHOR-CUT rid=%s pp=%s read=%d of=%d anchored=%d cut_to=%d cut_loss=%d "
            "unlanded=%d landing=%s (n=%d): cut_loss = KV landed above the deepest reachable "
            "anchor (given back), unlanded = pages the short read never brought -- the case "
            "costs cut_loss + unlanded, the anchor only cut_loss",
            req_id, pp_rank, read, of, int(anchored), cut, max(0, read - cut),
            max(0, of - read), land, _N["cut"],
        )
        return land
    except Exception:  # noqa: BLE001 - an instrument never breaks a reap
        return None
