"""Q-1220 DUAL NOT-NAMED GIVE-BACK: a dual P follower gives back the mamba
slot its admission match drew for a queued request PP0 did not name this
pass -- the give-back PP0's own refusal of that request already does.

METAL (27B NVFP4 dual y9d2, boot dkr27bnvfp4dual1mpsleepsharebar1fs10040413,
image 6c7c37327f, P PP1 death 04:18:20Z; the log cuts rids to 8 characters):

  P mamba pool: ``max_mamba_cache_size: 8`` on every PP rank.
  04:18:19  weg2-0-30 resumes at told=8192 and runs five chunks; weg2-0-35,
            weg2-0-37, weg2-0-6 wait behind it with told=8192 ('#1400 FOLLOWER
            SATISFIED LOCALLY ... local_prefix=8192' on PP1/PP2, nothing read,
            nothing pinned).
  PP0       ``mamba usage: 0.25`` on all six chunk passes; one mamba eviction
            for the chunks (node 58), PP0 decides weg2-0-35 ``resident=8192``.
  PP1/PP2   ``mamba usage: 0.25 -> 0.38 -> 0.50 -> 0.50 -> 0.62``, one step per
            '#1416e PACED-ADMIT ABSORBED' of a queued rid (n=4/5/6); a mamba
            eviction on EVERY chunk (nodes 58, 61, 43, 44 -- PP0 evicted neither
            43 nor 44 up to the log's end) -- the internal tombstones took the
            8192 grid anchor, whose state had no host copy.
  PP1       '#1040 EXTENT STATE-ALIGN rid=weg2-0-35 kv=4096 anchor_depth=4096
            device_len=0 key_match_depth=8192', 'MAMBA-HOST-RESUME ... depth=4096',
            then '#968 PREFIX MATERIALISATION SHORTFALL ... prefix_len=8192, this
            rank holds 0 ... adopted a GDN anchor at a depth the schedule does
            not name; served 4096' -> #1223 DEBUG-HOLD -> W17 Weg2GroupDead (P).

THE MECHANISM. Every rank runs the same admission loop over its waiting queue.
For a queued request with a told verdict, ``req.init_next_round_input`` matches
with ``cow_mamba`` and draws a COW resume slot speculatively
(``MambaComponent.finalize_match_result``, stamp
``mamba_slot_acquired_this_admission``). PP0 then offers it to the adder, the
adder refuses (the chunk took the budget) and the #991 revert gives the slot
back (``release_admission_acquired_mamba_slot`` site=admission_revert). A
follower never reaches the adder for it: PP0's row does not name the rid, the
loop takes the ``pp_not_named`` exit -- and that exit had no give-back. The
slot stayed with the queued request (the next match keeps an existing
``mamba_pool_idx``), one per queued told request, on the followers only. With 8
slots, three held slots made every chunk allocation of the running request
evict a tree anchor on PP1/PP2 that PP0 kept: the replicas parted, and the
prefix PP0 named was gone on PP1.

THE FIX. At the ``pp_not_named`` exit a dual-layout rank gives the slot back
through the one #991 helper -- exactly what PP0's refusal of the same request
does, so every PP rank holds the same slots. Only a slot THIS admission drew
(the stamp) is touched; a batch-owned or session-held slot never is.

Dual layout only (``dual_decode_join.dual_layout_rank``): the flip/INT8/NF
forms return before reading anything (the 'Flip unverändert' test).
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

MARKER = "Q-1220 DUAL NOT-NAMED GIVE-BACK"
SITE = "pp_not_named"

_N = [0]


def give_back_not_named(req, tree_cache, env=None) -> bool:
    """The ``pp_not_named`` exit on a dual-layout rank: give back the mamba
    slot this admission's match drew for ``req``. Returns True when a slot was
    returned; False (and nothing read or written) off the dual layout."""
    from sglang.srt.weg2.dual_decode_join import dual_layout_rank

    if not dual_layout_rank(env):
        return False
    if tree_cache is None or not getattr(req, "mamba_slot_acquired_this_admission", False):
        return False
    from sglang.srt.mem_cache.common import release_admission_acquired_mamba_slot

    slot = getattr(req, "mamba_pool_idx", None)
    try:
        slot_txt = int(slot)
    except Exception:  # noqa: BLE001 - a log field only
        slot_txt = slot
    released = bool(release_admission_acquired_mamba_slot(req, tree_cache, site=SITE))
    if released:
        _N[0] += 1
        n = _N[0]
        if n <= 16 or n % 256 == 0:
            logger.info(
                "%s rid=%s slot=%s n=%d (follower: PP0 did not name this queued "
                "request this pass; the COW slot its match drew goes back, as PP0's "
                "own refusal of it gives it back -- every PP rank holds the same "
                "mamba slots)",
                MARKER, str(getattr(req, "rid", "?")), slot_txt, n,
            )
    return released
