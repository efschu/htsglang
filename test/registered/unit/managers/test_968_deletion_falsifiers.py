"""#968 DELETION FALSIFIERS -- the tests that secure the compensation-layer cuts.

Companion to /spinning/gpu-arb/LOESCH_INVENTAR_968.md (F2a, 2026-09-01).
One test per deleted layer, in the direction that matters: each asserts the
DELETED behaviour stays deleted, and where the layer had a legitimate residue
(mid-admission quiescence), that the residue still holds. Written while the
Schicht-1/2 cut was in flight (uncommitted, F2b) and Schicht 3
(``take_agreed_reissue`` / ``_reissue_pending``) still existed -- the Schicht-3
tests are therefore RED-FIRST by construction and go green with that deletion.

THE DESIGN LAW THESE PIN (user, 2026-09-01): a flip behaves like a freshly
started server with a cache hit -- everything is nulled, then KV/GDN/draft
flow back through the NORMAL new-arrival path. Any predicate that holds a
flip hostage to in-layout prefill progress, and any group vote that lets one
re-admitted request compute uncovered while its peers wait, contradicts that
law (measured: boot_855_tiprevert1033, 37 arm/abandon cycles over 1114 s and
the cached=0 double-prefill motor, HANDOVER_F2_631ROW.md).
"""

from __future__ import annotations

import inspect

from sglang.srt.managers.phase_flip_runtime import chunk_blocks_quiescence


class _Chunked:
    """The one field the surviving predicate reads."""

    def __init__(self, req_pool_idx):
        self.req_pool_idx = req_pool_idx
        # Fields the DELETED strict clause used to read; present so a
        # resurrected clause would see a maximally "incomplete" chunk and
        # go red here rather than pass by AttributeError-default.
        self.extend_range = type("R", (), {"start": 0, "end": 3817})()
        self.fill_ids = list(range(36912))


# -- Schicht 1: the strict clause is gone ---------------------------------


def test_an_incomplete_chunk_at_a_settled_boundary_never_blocks_quiescence():
    """The 1114-s hold, inverted. boot_855_tiprevert1033: 33k tokens pending,
    chunk mid-request but BETWEEN chunks (committed KV, accounted
    extend_range) -- the strict clause held the flip on it for 37 cycles
    while the builder refused to run it. Under the cutover-full-reset design
    the flip discards nothing, so this must be a settled boundary."""
    req = _Chunked(req_pool_idx=7)  # admitted: has a pool row
    assert chunk_blocks_quiescence(req) is False


def test_mid_admission_still_blocks_quiescence():
    """The legitimate residue: chosen but no pool row yet -- KV has no home,
    no settled boundary exists. Clears within a round."""
    assert chunk_blocks_quiescence(_Chunked(req_pool_idx=None)) is True


def test_no_chunk_no_block():
    assert chunk_blocks_quiescence(None) is False


def test_the_strict_clause_did_not_grow_back():
    """The deletion is structural: the predicate takes ONE argument. A
    resurrected ``strict=``/``prefill_runnable_here=`` keyword is the layer
    growing back under the same name (the #1024-told-carrier misreading,
    Order-Amendment 2026-08-29)."""
    params = inspect.signature(chunk_blocks_quiescence).parameters
    assert list(params) == ["chunked_req"], params


# -- Schicht 2: one oracle, not two ---------------------------------------


def test_the_runnability_oracle_is_gone():
    """``prefill_runnable_in_current_layout`` answered the builder's question
    ("can prefill make progress here") from the raw #887 budget, while the
    builder's own gate additionally required fits_in_one_chunk + seam grant.
    Two oracles, one question, opposite answers -> hold without exit. The
    builder's gate (``phase_purity.prefill_blocked_here``) is the ONE
    surviving oracle."""
    import sglang.srt.managers.phase_flip_runtime as pfr

    assert not hasattr(pfr, "prefill_runnable_in_current_layout")


# -- Schicht 3: no single-rid vote, no owed ledger (RED until deleted) -----


def test_the_group_vote_is_gone():
    """``take_agreed_reissue`` elected ONE rid per round via a CPU
    all_reduce ON THE ADMISSION PATH (a path collective, recorded fatal by
    the Order) while every other re-admitted request computed uncovered --
    the cached=0 double-prefill motor. The upstream-equivalent hold is
    per-request: admission waits on ``check_prefetch_progress(req_id)``
    exactly as for any new arrival."""
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    assert not hasattr(UnifiedRadixCache, "take_agreed_reissue")


def test_the_owed_ledger_is_gone():
    """``_reissue_pending`` was second bookkeeping beside the upstream hold:
    written only where the stale insert was refused, read only by the vote.
    With the vote gone it must go too -- a ledger with no reader is exactly
    the rot the upstream-minimal law names."""
    import inspect as _i

    from sglang.srt.mem_cache import unified_radix_cache as urc

    src = _i.getsource(urc)
    assert "_reissue_pending" not in src
