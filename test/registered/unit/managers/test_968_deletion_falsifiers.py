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


# -- Slice B (c): the post-cutover fresh-fetch sweep ----------------------


def test_the_post_cutover_readmit_consumes_the_stash_once():
    """(c) The deferred requeue is a one-shot handoff: the release half
    STASHES (`_pending_seam_readmit = (released, n)`), the post-cutover half
    CONSUMES AND CLEARS. Two producers or a missing clear would re-admit a
    cutover's residents twice -- the double-prefill shape from the other
    side."""
    import inspect as _i

    from sglang.srt.managers.phase_flip_runtime import PhaseFlipRuntime

    assert hasattr(PhaseFlipRuntime, "_post_cutover_readmit")
    body = _i.getsource(PhaseFlipRuntime._post_cutover_readmit)
    assert "_pending_seam_readmit" in body
    assert "self._pending_seam_readmit = None" in body, "one-shot clear missing"
    import sglang.srt.managers.phase_flip_runtime as pfr

    src = _i.getsource(pfr)
    assert src.count("self._pending_seam_readmit = (") == 1, "exactly one stasher"


def test_the_sweep_instrument_exists():
    """(c) '#1066 POST-CUTOVER FRESH-FETCH' is the execution proof the
    speed-mode law demands: a deferred path with no log line is
    desk-written-never-executed waiting to happen."""
    import inspect as _i

    import sglang.srt.managers.phase_flip_runtime as pfr

    assert "#1066 POST-CUTOVER FRESH-FETCH" in _i.getsource(pfr)


# -- Slice B (d): PP0 holds on its prefetch verdict, followers never ------


def test_pp0_holds_and_followers_never_do():
    """(d) ONE verdict, at PP0: under the PP admission branch only
    ``pp_rank == 0`` consults the drained prefetch verdict and skips
    (``prefetch_pending_pp0``); followers stay credit-only
    (``pop_prefetch_loaded_tokens``) and never decide -- the row they
    receive before planning IS their verdict (Order: downstream verdiktfrei).
    Pins: the skip exists exactly once, its guard window names pp_rank == 0,
    and no second prefetch-pending skip variant gates followers."""
    import inspect as _i

    import sglang.srt.managers.scheduler as sched

    src = _i.getsource(sched)
    assert src.count('_note_skip("prefetch_pending_pp0"') == 1
    idx = src.index('_note_skip("prefetch_pending_pp0"')
    window = src[max(0, idx - 2000) : idx]
    assert "pp_rank == 0" in window, "the PP0 gate must guard the skip"
    # Scope the no-follower-verdict law to the PP arm: the ballot machinery
    # in the `elif enable_hicache_storage:` (non-PP) arm keeps its own
    # per-request hold -- that IS the upstream-equivalent and stays.
    pp_arm_start = src.index("if self.enable_hicache_storage and _pp_group:")
    pp_arm_end = src.index("elif self.enable_hicache_storage:", pp_arm_start)
    pp_arm = src[pp_arm_start:pp_arm_end]
    assert pp_arm.count('_note_skip("prefetch_pending') == 1, (
        "a second prefetch-pending skip inside the PP arm would put a "
        "verdict on a follower"
    )
    assert "prefetch_pending_pp0" in pp_arm
    assert "pop_prefetch_loaded_tokens" in pp_arm, (
        "the follower credit-only path must survive in the PP arm"
    )


# -- #1067: the armed park is the admission owner's verdict ----------------


def test_the_armed_park_is_pp0_only_in_the_pp_layout():
    """(#1067) A follower that parks under an armed pp_to_tp flip withholds
    the very pass that would drain the mb its upstream already launched --
    measured boot_855_1065umbau 06:26:07Z: PP1 received and reconciled pass
    31 (ROW-PROBE delivered=31), parked before building, PP2 stayed at 30,
    PP0 blocked 665 s in the output recv and the flip never assembled.
    Under the PP0-authority order followers execute what the row tells them
    and decide nothing: in the PP layout only PP0 parks (it stops SOURCING;
    in-flight mbs keep draining), in the TP layout the park stays
    group-wide (every rank is a replica of one decision). Pin: the park
    block's guard names the pending-direction exemption and the PP0 gate
    together, ahead of the shared quiescence predicate."""
    import inspect as _i

    import sglang.srt.managers.scheduler as sched

    src = _i.getsource(sched)
    idx = src.index("and not chunk_blocks_quiescence(self.chunked_req)")
    window = src[max(0, idx - 2600) : idx]
    assert "pending != PP_TO_TP" in window, "the PP-layout exemption is gone"
    assert "pp_rank == 0" in window, "the PP0-only gate is gone"
    # The exemption must be a disjunction (follower passes the park), not a
    # conjunction that would park everyone again.
    tail = window[window.index("pending != PP_TO_TP") :]
    assert "or" in tail.split("pp_rank == 0")[0], (
        "the direction exemption and the PP0 gate must be OR-joined"
    )


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


# -- Slice B (F2b, 2026-09-01): the seam re-admission takes the normal path --


def test_the_resident_requeue_left_the_release_path():
    """(a) RED-FIRST. Today `readmit_seam_residents` runs inside the RELEASE
    half of the cutover (phase_flip_runtime, before rebind), so the re-issued
    prefetch registers on the OUTGOING binding generation and #937 refuses
    its insert every time -- fetched, paid for, thrown away (the docstring of
    reissue_seam_prefetch names the sequencing as the defect). The cut moves
    the requeue BEHIND cutover_fn/rebind via a deferred carrier
    (`_pending_seam_readmit`), so the intake prefetch opens on the INCOMING
    binding. Structural pin: the deferred carrier exists, and the release
    module no longer calls the requeue directly."""
    import inspect as _i

    import sglang.srt.managers.phase_flip_runtime as pfr

    src = _i.getsource(pfr)
    assert "_pending_seam_readmit" in src, "deferred requeue carrier missing"
    # The release half must not requeue any more; the one surviving call runs
    # post-cutover, off the pending carrier.
    release_calls = src.count("readmit_seam_residents(")
    assert release_calls <= 1, (
        "readmit_seam_residents called %d times in phase_flip_runtime -- the "
        "release-path call grew back" % release_calls
    )


def test_retracted_equals_readmitted_survives_the_move():
    """(a) guard for the count law: RETRACTED MUST EQUAL READMITTED (W31).
    The move behind the rebind must not lose the reconciliation -- a dropped
    request is owned by nobody, which is the W31 defect verbatim."""
    import inspect as _i

    import sglang.srt.managers.phase_flip_runtime as pfr

    assert "readmitted != " in _i.getsource(pfr)


def test_the_outgoing_binding_reissue_shim_is_gone():
    """(b) RED-FIRST. `reissue_seam_prefetch` (#1025b) re-issued the fetches
    that the release-path requeue had burned on the outgoing binding -- a
    compensation for the sequencing defect, not a mechanism. With the requeue
    deferred behind the rebind (a), the first issue already lands on the
    incoming binding and the shim is objectless."""
    import sglang.srt.managers.phase_flip_runtime as pfr

    assert not hasattr(pfr, "reissue_seam_prefetch")


def test_the_owed_ledger_is_gone():
    """``_reissue_pending`` was second bookkeeping beside the upstream hold:
    written only where the stale insert was refused, read only by the vote.
    With the vote gone it must go too -- a ledger with no reader is exactly
    the rot the upstream-minimal law names."""
    import inspect as _i

    from sglang.srt.mem_cache import unified_radix_cache as urc

    src = _i.getsource(urc)
    assert "_reissue_pending" not in src
