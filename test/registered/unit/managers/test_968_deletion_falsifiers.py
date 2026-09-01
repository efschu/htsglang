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


# -- #1068: an instrument may never gate ----------------------------------


def test_the_fence_store_scan_is_time_budgeted():
    """(#1068) The #1063 fence snapshot walks every page stem through
    os.path.exists against a million-file store, ON THE CUTOVER'S NO-RETURN
    PATH. Measured boot_855_1067park: the warm scan took ~3 s over 221862
    stems at 06:47:10; the very next fence hit the re-prefill's fresh stems
    cold and never returned -- all three ranks' last line for 25 minutes was
    the FENCE-NODES header, no abandon could run (PP0 is the timeout
    carrier), and the deadman killed the boot at 07:15:37. The scan now
    carries a hard deadline and reports itself CAPPED (counts become a
    declared sample) instead of holding the seam."""
    import inspect as _i

    from sglang.srt.mem_cache.hicache_flip_writeback import _1063_record_fence

    body = _i.getsource(_1063_record_fence)
    assert "_scan_deadline" in body, "the hard time budget is gone"
    assert "SCAN CAPPED" in body, "the capped self-report is gone"
    # The budget must be checked INSIDE the walk, not once before it: a
    # single pre-check cannot stop a scan that goes cold mid-walk.
    assert body.index("_scan_deadline = ") < body.index("> _scan_deadline"), (
        "deadline set after its check -- the budget cannot bind"
    )


def test_the_capped_scan_returns_within_budget(monkeypatch):
    """(#1068) Behavioral arm: a store whose exists() is pathologically slow
    must not hold the fence past its budget. Every stem probe costs 50 ms;
    an unbudgeted walk over 4096 tracked stems would take minutes -- the
    capped walk must return in ~2 s and still record a fence snapshot."""
    import time as _t

    import sglang.srt.mem_cache.hicache_flip_writeback as fwb

    class _Node:
        def __init__(self, i):
            self.id = i

    class _Backend:
        pass

    class _CC:
        storage_backend = _Backend()

    class _Tree:
        cache_controller = _CC()

    def _slow_stems(backend, node):
        return [f"stem_{node.id}_{j}" for j in range(64)]

    def _slow_state(backend, stem):
        _t.sleep(0.05)
        return "readable"

    monkeypatch.setattr(fwb, "_1063_stems_for_node", _slow_stems)
    monkeypatch.setattr(fwb, "_1063_stem_state", _slow_state)
    start = _t.monotonic()
    fwb._1063_record_fence(_Tree(), [_Node(i) for i in range(256)])
    elapsed = _t.monotonic() - start
    assert elapsed < 10.0, (
        "the fence scan held the seam for %.1fs despite the budget" % elapsed
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


# -- #1069: cohort-in-service (falsifiers for commit 929525805e) -----------
#
# Metal evidence 1068cap (2026-09-01): 4 group flip pairs, the SAME
# 3-request cohort seam-re-prefilled from zero in every TP window
# (4 x 110710 tok), zero decode batches. Two arm faces of one blind spot:
# windows 1+4 armed on "pending prefill > 0 ... nothing decoding" while the
# cohort sat queued; windows 2+3 armed on "set has not shrunk for 51.8s"
# while a member's prefix was actively growing. The fix is ONE mechanism
# (cohort-in-service) at two sites; these tests pin both clauses and the
# phase-entry reset, so a revert of either half goes red by name.


class _ObserveStub:
    """The attribute surface ``observe_idle`` actually reads. Methods return
    'work exists, someone decodes' so the idle clock stays out of the way."""

    def __init__(self, phase, now, seam_cohort_pending_tokens):
        self.phase = phase
        self.now = now
        self.pending_prefill_tokens = 500
        self.running_bs = 2  # FLAT across observations: the bs-shrink clause
        self.seam_cohort_pending_tokens = seam_cohort_pending_tokens
        self.nothing_can_run = False
        self.ready_carriers = 0
        self.queue_nonempty = False

    def decode_work_bs(self):
        return 1

    def work_exists(self):
        return True


def test_1069_sinking_cohort_pending_is_bundle_progress():
    """The clock clause: with running_bs FLAT, a SINKING
    seam_cohort_pending_tokens must stamp last_bundle_progress_at; flat or
    RISING must not. This is the exact 1068cap failure inverted: windows 2+3
    fired 'set has not shrunk for 51.8s' at a member whose prefix was
    actively growing."""
    from sglang.srt.managers import phase_policy as pp

    state = pp.PhasePolicyState()
    # t=100: phase entry -- baseline, marker None, stamp comes from entry.
    pp.observe_idle(state, _ObserveStub("tp", 100.0, 8000))
    assert state.last_seam_cohort_pending == 8000
    # t=110: cohort pending flat -> NO progress stamp (clock stays at entry;
    # note the bs-shrink clause cannot fire either, running_bs is flat).
    pp.observe_idle(state, _ObserveStub("tp", 110.0, 8000))
    assert state.last_bundle_progress_at == 100.0, (
        "flat cohort pending stamped progress -- the clause lost its "
        "comparison and became a level read"
    )
    # t=120: cohort pending SINKS 8000 -> 4000 -> progress stamp at 120.
    pp.observe_idle(state, _ObserveStub("tp", 120.0, 4000))
    assert state.last_bundle_progress_at == 120.0, (
        "sinking seam_cohort_pending_tokens did not stamp bundle progress "
        "-- the #1069 clock clause is gone and the 51.8s false STALL arm "
        "(1068cap windows 2+3) is back"
    )
    # t=130: RISING 4000 -> 9000 (a new cutover's larger cohort) -> no stamp.
    pp.observe_idle(state, _ObserveStub("tp", 130.0, 9000))
    assert state.last_bundle_progress_at == 120.0, (
        "rising cohort pending stamped progress -- refill would now reset "
        "the stall clock forever, the un-drainable-bundle wedge"
    )


def test_1069_phase_entry_resets_the_cohort_marker():
    """A cutover ZEROES the cohort accounting; comparing across the phase
    boundary would credit that zeroing as service progress. The marker must
    restart with the phase (same law as last_running_bs / #833)."""
    from sglang.srt.managers import phase_policy as pp

    state = pp.PhasePolicyState()
    pp.observe_idle(state, _ObserveStub("tp", 100.0, 8000))
    # Phase change tp -> pp with a LOWER cohort value: the entry reset must
    # eat the comparison; the progress stamp must be the ENTRY stamp, and a
    # subsequent flat observation must not inherit the pre-flip baseline.
    pp.observe_idle(state, _ObserveStub("pp", 200.0, 0))
    assert state.last_bundle_progress_at == 200.0  # entry stamp, not "sink"
    pp.observe_idle(state, _ObserveStub("pp", 210.0, 0))
    assert state.last_bundle_progress_at == 200.0, (
        "the cohort marker survived the phase boundary -- a cutover's own "
        "zeroing now reads as service progress"
    )


def test_1069_dwell_holds_the_ppward_arm_and_is_bounded():
    """The arm clause: _decide_from_load must consult
    seam_cohort_dwell_active() on the pp-ward (tp_to_pp) path and refuse with
    the NAMED holding line -- the metal grepper 'seam cohort in service
    (#1069)'. And the predicate itself must stay BOUNDED: dwell lapses at
    SEAM_COHORT_DWELL_ROUNDS, so the hold can never become the W37-E wedge."""
    import inspect as _i

    from sglang.srt.managers import phase_policy as pp

    src = _i.getsource(pp._decide_from_load)
    assert "seam cohort in service (#1069)" in src, (
        "the named holding line left _decide_from_load -- the pp-ward arm "
        "no longer declares the cohort hold"
    )
    assert "seam_cohort_dwell_active()" in src, (
        "the arm no longer consults the dwell predicate -- windows 1+4 "
        "('nothing decoding' ~13s in, cohort still queued) are back"
    )
    # The bound, behaviorally: active inside the dwell, lapsed at the bound.
    class _S:
        seam_cohort_pending_bs = 2
        seam_cohort_stall_rounds = 0

    assert pp.PhasePolicyInputs.seam_cohort_dwell_active(_S()) is True
    _S.seam_cohort_stall_rounds = pp.SEAM_COHORT_DWELL_ROUNDS
    assert pp.PhasePolicyInputs.seam_cohort_dwell_active(_S()) is False, (
        "the dwell no longer lapses at SEAM_COHORT_DWELL_ROUNDS -- the hold "
        "is unbounded, which is the livelock this campaign has paid for"
    )
