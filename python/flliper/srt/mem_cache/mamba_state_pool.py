# SPDX-License-Identifier: Apache-2.0
"""Phase-aware mamba STATE pool resolution (#767 residual).

Under a phase-flip build the radix cache's ``req_to_token_pool`` is bound
once, to the PRIMARY PP stack's pool, and never rebound (slot NUMBERING is
deliberately single-authority: one allocator, aliased mapping tensors).
The mamba STATE tensors, however, are per-stack -- a request computing in
the TP phase writes its conv/ssm bytes into the TP stack's pool. Any
tree-side operation that touches state BYTES through the bound pool
therefore reads stale storage whenever the other stack is computing:
measured 2026-08-19 as checkpoint slots filled with a previous PP-phase
occupant's state (a kite prompt answered with a foreign river essay).

``active_mamba_state_pool`` is the one seam: bookkeeping stays on the
bound pool, state-byte operations resolve the pool of the phase that is
actually computing. Non-flip builds have no resolver installed and fall
back to the bound pool -- bit-for-bit the old behavior.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def active_mamba_state_pool(cache):
    """The mamba pool whose TENSORS hold the currently-computing phase's
    state bytes.

    ``cache`` is a radix cache. A phase-flip boot installs
    ``cache.phase_active_mamba_pool`` (see
    gdn_flip_mover.install_phase_aware_mamba_state_pool); without it the
    bound pool is the only pool and is returned unchanged.
    """
    resolver = getattr(cache, "phase_active_mamba_pool", None)
    if resolver is not None:
        return resolver()
    return cache.req_to_token_pool.mamba_pool


def is_phase_split_pool(cache) -> bool:
    """True when this boot has more than one mamba STATE pool.

    The resolver is installed exactly once, by
    ``gdn_flip_mover.install_phase_aware_mamba_state_pool``, and only on a
    phase-flip boot. Without it there is one pool, every anchor's bytes are
    trivially reachable, and every predicate below answers the old way.
    """
    return getattr(cache, "phase_active_mamba_pool", None) is not None


# ---------------------------------------------------------------------------
# #928: ANCHOR PROVENANCE.
#
# ``active_mamba_state_pool`` answers "who is computing NOW". A radix mamba
# anchor is a bare slot id, and the bytes behind that id were fixed at the
# moment it was donated -- by whichever stack was computing THEN. The two
# questions coincide only if the phase does not change between donate and
# resume, and under strict phase purity it always does: the prefill runs in
# PP, the request is retracted at the cutover, and the decode resumes the same
# anchor in TP. The resume then reads slot N of the TP pool while the state
# sits at slot N of the PP pool, and slot N of the TP pool holds whatever the
# TP stack last left there. Measured 2026-08-27 (#928): three identical sends
# at temperature 0, three different answers, one of them a bare EOS. The same
# failure is on record from the write side in this module's header, measured
# 2026-08-19 ("a kite prompt answered with a foreign river essay") -- #767
# fixed that half and scoped itself to it (see
# ``install_phase_aware_mamba_state_pool``'s docstring: "only the byte copies
# (checkpoint donation copy, int8 store)").
#
# WHY A LEDGER AND NOT A CROSS-POOL COPY. Moving the bytes at resume is not a
# local operation: the PP pool is layer-axis sharded and the TP pool is
# head-axis sharded, which is what ``gdn_flip_mover`` exists to translate, on
# a plan and a collective. A per-request copy at match time cannot do it. So
# the seam ANSWERS THE QUESTION and the caller refuses -- the shape
# ``retention_shrinks_protected`` already established for #824, and the rule
# ``MambaComponent.finalize_match_result`` already states for its neighbour
# case: "Reusing the KV prefix without the matching mamba state would be
# silently wrong, so the whole match is zeroed." A refusal costs a re-prefill;
# the silence cost a wrong answer. Carrying cached anchors ACROSS the cutover
# (the mover already does it for resident slots) is the capacity fix and is a
# separate posten -- it restores the hit, it is not what makes the answer
# right.
#
# THE LEDGER IS BOUNDED BY THE POOL. One entry per mamba slot id, overwritten
# on every insert, so a reused slot cannot carry a stale pool: the donate that
# reuses it records again before any match can read it.
# ---------------------------------------------------------------------------


def _anchor_pool_ledger(cache) -> dict:
    ledger = getattr(cache, "_mamba_anchor_pool", None)
    if ledger is None:
        ledger = {}
        cache._mamba_anchor_pool = ledger
    return ledger


def _slot_key(slot_ids: Any) -> Optional[int]:
    """The physical slot id an anchor value names, as a plain int.

    One D2H sync per INSERT and per MATCH -- both are per-request and already
    sit beside ``.item()`` calls on the same path (schedule_batch.py:3279).
    Never called per token or inside a forward.
    """
    reshape = getattr(slot_ids, "reshape", None)
    if reshape is None:
        return int(slot_ids)
    flat = reshape(-1)
    if flat.numel() == 0:
        return None
    return int(flat[0].item())


def note_anchor_bytes(cache, slot_ids: Any) -> None:
    """Record WHICH pool's tensors hold the bytes behind this anchor.

    Called where an anchor enters the tree, so the recorded pool is the one
    that was computing when the state was produced.
    """
    if not is_phase_split_pool(cache) or slot_ids is None:
        return
    key = _slot_key(slot_ids)
    if key is not None:
        _anchor_pool_ledger(cache)[key] = active_mamba_state_pool(cache)


def anchor_bytes_pool(cache, slot_ids: Any):
    """The pool recorded for this anchor, or ``None`` if nothing recorded it."""
    if slot_ids is None:
        return None
    key = _slot_key(slot_ids)
    if key is None:
        return None
    return _anchor_pool_ledger(cache).get(key)


def anchor_bytes_reachable(cache, slot_ids: Any) -> bool:
    """Can the phase that is computing NOW read this anchor's state bytes?

    Single-pool boots answer True unconditionally -- there is one pool, the
    question does not arise, and the behaviour is bit-for-bit the old one.

    On a phase-split boot an anchor with NO recorded pool answers False. That
    is deliberate and it is the conservative direction: an unrecorded anchor
    is one this seam cannot vouch for (a session-restore insert, or one
    planted before the ledger existed), and the cost of refusing it is a
    re-prefill while the cost of trusting it is the wrong answer.
    """
    if not is_phase_split_pool(cache):
        return True
    if slot_ids is None:
        return False
    written_to = anchor_bytes_pool(cache, slot_ids)
    if written_to is None:
        return False
    return written_to is active_mamba_state_pool(cache)


def anchor_provenance_verdict(cache, slot_ids: Any) -> str:
    """One word for the log line, so a boot answers this with one grep.

    ``single`` (no split), ``same`` (the computing pool wrote it),
    ``foreign`` (the other stack wrote it), ``unknown`` (nothing recorded it).
    """
    if not is_phase_split_pool(cache):
        return "single"
    written_to = anchor_bytes_pool(cache, slot_ids)
    if written_to is None:
        return "unknown"
    return "same" if written_to is active_mamba_state_pool(cache) else "foreign"
