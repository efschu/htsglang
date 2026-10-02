"""SA: the recurrent state a prefetch commits sits at the depth the claim was cut to.

A trailing-pages pool (the mamba anchor: one page, SWA: a window) carries the
state AFTER its last key's page. ``commit_hicache_transfer(PREFETCH)`` files
that state on the deepest node the insert produced -- the node at the CUT.
Before SA a short read never read a state, so the cut and the state could not
disagree. Since SA a short read reads its state at the deepest anchor inside
the landed pages and votes that page as its #257 anchor, so on a rank that
decides its own cut the two coincide by construction. This check makes the
coincidence a precondition instead of an assumption: a state whose last key
is not the cut page is never committed (its pool's hit count is zeroed, the
component releases the slot and the node stays stateless -- a re-prefill,
never a wrong state at a foreign depth, #767's direction).
"""

from __future__ import annotations

import logging
from typing import Iterable, List, Sequence

from sglang.srt.mem_cache.hicache_storage import PoolHitPolicy

logger = logging.getLogger(__name__)


def state_depth_mismatch(
    *,
    transfers: Iterable,
    hash_value: Sequence[str],
    cut_pages: int,
    hit_pages: dict,
) -> List[str]:
    """Names of the trailing-pages pools whose read state is NOT at
    ``cut_pages`` (their last key differs from ``hash_value[cut_pages - 1]``,
    or the cut is 0) while ``hit_pages`` says the state loaded."""
    out: List[str] = []
    for t in transfers:
        if t.hit_policy != PoolHitPolicy.TRAILING_PAGES:
            continue
        if int(hit_pages.get(t.name, 0) or 0) < 1:
            continue
        keys = t.keys or []
        at_cut = (
            cut_pages > 0
            and cut_pages <= len(hash_value)
            and len(keys) > 0
            and keys[-1] == hash_value[cut_pages - 1]
        )
        if not at_cut:
            out.append(t.name)
    return out


def refuse_foreign_depth_states(
    *, rid, transfers: Iterable, hash_value: Sequence[str], cut_pages: int, hit_pages: dict
) -> List[str]:
    """Zero the hit count of every pool :func:`state_depth_mismatch` names, so
    the component's commit releases that state instead of filing it at the
    cut. Returns the names (empty = every loaded state sits at the cut)."""
    bad = state_depth_mismatch(
        transfers=transfers, hash_value=hash_value, cut_pages=cut_pages, hit_pages=hit_pages
    )
    for name in bad:
        hit_pages[name] = 0
    if bad:
        logger.warning(
            "SA STATE-DEPTH REFUSED rid=%s pools=%s cut_pages=%d: the read state is not "
            "at the cut page -- released, the node stays stateless (re-prefill, never a "
            "state at a foreign depth)",
            rid, [str(n) for n in bad], int(cut_pages),
        )
    return bad
