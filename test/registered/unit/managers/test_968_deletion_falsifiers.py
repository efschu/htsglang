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


# -- (A-i)/(A-ii): the DEFER-armed chain wait (1068cap end-wedge) ----------
#
# RED-FIRST against 929525805e. Metal: 1068cap 07:34:09, PP1's DEFER armed
# `_pp_row_chain_owed` for rid 41fcb9cd -- a rid that had traveled the chain
# long before (READMIT-MATCH n=179/181 two seconds earlier) and became
# unlocatable through PP1's own #791/#797 void of the told=37147/local=0
# pass. The premise comment at the arm site ("PROVES a chain send is in
# flight") was false, the one-shot bypassed the counter-against-twin gate
# (:8622) that fixed row7, and pp_chain_receiver.recv() blocked forever:
# PP0 in the output recv, PP1 in a chain recv nobody owes, PP2 spinning on
# its occupant. These go green with F2b's (A-i)+(A-ii) cut.


def _defer_arm_slice():
    """The DEFER block of `_pp_proxy_frame_pending`, anchored on the
    `defer_rid` stats key (stable across the fix; the premise comment is
    not) up to the arm's `_trace("defer_rid")`."""
    import inspect as _i

    from sglang.srt.managers import scheduler_pp_mixin as mixin

    src = _i.getsource(mixin.SchedulerPPMixin._pp_proxy_frame_pending)
    start = src.index('"defer_rid"')
    end = src.index('_trace("defer_rid")')
    return src[start:end]


# (A-i) IS WITHDRAWN, AND THE WITHDRAWAL IS THE POINT (P1, 2026-09-01).
#
# The two tests that stood here demanded counter proof and a raise INSIDE the
# DEFER block. Both were correct about that block and both were the wrong
# instrument, for two independent reasons recorded rather than deleted:
#
#   1. F2b's own retraction: the counter proof is BLIND at that site -- the
#      rendezvous bumps `sent` only when the receiver ENTERS the recv, so
#      "balanced counters" cannot distinguish in-flight from never-posted.
#   2. The arc-breaker rule (user order 2026-09-01, "nicht wieder in den
#      Endlosbogen einbiegen"): the DEFER one-shot is a COMPENSATION for a
#      rank-local verdict. Hardening it is answer (a) -- a deletion
#      candidate, never a fix order. Its trigger was a rid whose ownership
#      moved under it because a mid-rank #797 void had swallowed the pass
#      (`told=37147/local=0`, 1068cap 07:34). #1071 deletes that void, so the
#      trigger is removed at its source instead of survived at its symptom.
#
# What replaces them asserts the DELETION, which is the thing that must not
# silently grow back.


def test_the_rank_local_shortfall_verdict_is_gone():
    """#1071 ZOMBIE TEST. `_pp_void_retracted_pass` let ONE rank decide the
    GROUP's pass ran nowhere, and told nobody upstream -- PP0's slot stayed
    set, the last rank's did not, and PP0 parked in `_do_recv` until the
    deadman (1068cap 07:34:09, 1069cohort 08:00:55). Its own docstring named
    the return trip that made it safe; #969 CUT V had already deleted that
    emitter. The verdict is deleted, not repaired."""
    from sglang.srt.managers import scheduler_pp_mixin as mixin

    assert not hasattr(mixin.SchedulerPPMixin, "_pp_void_retracted_pass"), (
        "the rank-local shortfall void grew back: a downstream rank may not "
        "void a pass PP0 launched, because the void reaches downstream ranks "
        "and never PP0 (RAENGE-NIE-UNEINS / #968 PP0 authority)"
    )


def test_an_unhonourable_told_is_loud_not_silent():
    """#1071, the other half. Deleting the void is only safe if the state it
    used to swallow is now NAMED. An unhonourable told is the ranks
    disagreeing about a pass; detection is a crash, never a clamp (a clamp
    would be rank-local geometry, i.e. #631) and never a wait."""
    import inspect as _i

    from sglang.srt.managers import scheduler_pp_mixin as mixin

    src = _i.getsource(mixin.SchedulerPPMixin._pp_assert_told_honourable)
    assert "raise" in src and "entries_retracted_by_rank" in src, (
        "the replacement for the deleted void does not stop on a detected "
        "shortfall -- silence here is the 1069cohort park again"
    )


def test_Aii_chain_recv_is_counter_bounded():
    """(A-ii) RED-FIRST. `PpChainReceiver.recv` blocks with
    `while not self.inbox: self._advance(block=True)` -- no counter check,
    no guard, unlike its sibling `consume_up_to` (max_messages guard,
    counter-against-wire comparison). 1068cap: PP1 sat in exactly this
    loop for 5+ minutes with nothing posted. The fix bounds recv the same
    way its sibling already is."""
    import inspect as _i

    from sglang.srt.managers import pp_chain_receiver as pcr

    src = _i.getsource(pcr.PpChainReceiver.recv)
    assert any(k in src for k in ("self.consumed", "sent", "max_messages")), (
        "recv() still blocks unbounded with no counter evidence -- a "
        "one-shot armed on a false premise parks the rank forever "
        "(1068cap PP1, 07:34:09 until the deadman)"
    )


# -- (C): the occupant-sleep node and the half-built void relay ------------
#
# RED-FIRST against ca0ee3acd4. This is the node BOTH measured stall
# instances have in common, and neither (A-i) nor (A-ii) touches it:
#
#   1068cap 07:34:09  PP1 in pp_chain_receiver.recv, PP2 spinning on its
#                     occupant, PP0 in the output recv (cur_batch SET).
#   1069cohort 08:00:55  roles SWAPPED -- PP1 in the occupant throttle
#                     (py-spy, scheduler_pp_mixin :4189), PP2 reading
#                     counters at :8622, PP0 identical to 1068cap.
#
# Different entries (1069cohort's ROW-DELIVER d41-d43 are clean: planned,
# no void, no DEFER, no #992), same halting member: the `else:` arm of the
# chain-receive gate sleeps 2 ms on "my slot is occupied and no upstream
# statement came", forever, instead of moving the in-flight pass or saying
# anything. It is a throttle where a horizon belongs.
#
# WHY THE NODE EXISTS AT ALL -- named here because it decides which of the
# two green paths below is the right one (UPSTREAM-MINIMAL, user order
# 2026-08-29). The occupant is not a race; it is the PRODUCT of a rank-local
# verdict. Metal, 1068cap 07:34:02-09: the #797 shortfall void ran on rank 1
# ONLY -- no void/retract line on PP0 or PP2, PP0 admitted fwd 174 three
# seconds later, PP2's occupant counter climbed. A rank-local
# state-changing decision leaves exactly this shape: PP0's slot stays set,
# downstream is nulled, and the middle rank owes a statement its own void
# swallowed. #969 CUT V then deleted the void-output EMITTER (the deletion
# is recorded verbatim in the comment at `_pp_send_output_to_next_stage`)
# on the premise "the batch IS the verdict, both sides ask mbs[slot]" --
# which holds for an ordinary pass and breaks for a MID-RANK void.
#
# So the compensation layer here is the void RELAY, and it is half-built:
# `pp_void_forward_payload` (the #801 relay invariant, in full prose) is
# computed into `self._pp_void_forward_payload` by the absorber and read by
# NOBODY (1 writer / 0 readers, devindex `where` at pin ca0ee3acd4 and grep
# on the branch, both), while `_PP_VOID_OUTPUT_KEY: True` is constructed at
# exactly one site -- inside the absorber that consumes it. A message no
# rank originates, relayed by nobody, absorbed by one.
#
# The falsifiers below are therefore written so BOTH exits go green: wire
# the relay (the O2 E-a/E-b form), or delete it whole and let PP0 own the
# verdict (the structural cut). What they forbid is the third state we are
# actually in -- half a relay and a silent sleep under it.


def _occupant_throttle_sources():
    """The `else:` arm of the chain-receive gate in `_event_loop_pp_body`
    (scheduler_pp_mixin :4180-4189 at ca0ee3acd4), plus ONE HOP: the source
    of every `self.<method>()` it calls, so a fix that puts the horizon in a
    helper still reads as green.

    Anchored structurally (ast: the `with record_function("recv_requests")`
    block, then the terminal `else` of the `_chain_gate` if-chain), not on
    comment text -- the comments at this node are being rewritten by the
    fix that makes these tests pass.
    """
    import ast as _ast
    import inspect as _i
    import textwrap as _t

    from sglang.srt.managers import scheduler_pp_mixin as mixin

    src = _t.dedent(_i.getsource(mixin.SchedulerPPMixin._event_loop_pp_body))
    tree = _ast.parse(src)
    lines = src.splitlines()

    gate = None
    for node in _ast.walk(tree):
        if isinstance(node, _ast.With) and "recv_requests" in lines[node.lineno - 1]:
            gate = node
            break
    assert gate is not None, (
        "the chain-receive gate block is gone from _event_loop_pp_body -- "
        "re-derive this anchor against the new loop before trusting a pass"
    )

    # The gate is the if-chain whose TEST reads the gate variable -- not the
    # row-authority guard that assigns it, which is the first `If` in the
    # block and has no else arm at all.
    branch = None
    for node in gate.body:
        if not isinstance(node, _ast.If):
            continue
        if "_chain_gate" not in (_ast.get_source_segment(src, node.test) or ""):
            continue
        cur = node
        while len(cur.orelse) == 1 and isinstance(cur.orelse[0], _ast.If):
            cur = cur.orelse[0]
        if cur.orelse:
            branch = cur.orelse
        break
    assert branch is not None, (
        "the gate no longer has a terminal else arm -- if the occupant case "
        "was folded away entirely, re-derive this test rather than pass it"
    )

    texts = [_ast.get_source_segment(src, stmt) or "" for stmt in branch]

    # one hop: helpers called from the branch count as part of the branch
    for stmt in branch:
        for call in _ast.walk(stmt):
            if not isinstance(call, _ast.Call):
                continue
            fn = call.func
            if (
                isinstance(fn, _ast.Attribute)
                and isinstance(fn.value, _ast.Name)
                and fn.value.id == "self"
            ):
                target = getattr(mixin.SchedulerPPMixin, fn.attr, None)
                if callable(target):
                    try:
                        texts.append(_i.getsource(target))
                    except (OSError, TypeError):
                        pass
    return "\n".join(texts)


def test_C_the_occupant_throttle_carries_a_verdict_horizon():
    """(C) RED-FIRST. The throttle arm today is two statements -- empty the
    request list, sleep 2 ms -- with no memory of how long the occupant has
    been unanswered. A rank that sleeps on "occupied slot, no upstream
    statement" with no horizon cannot distinguish the 2 ms of ordinary
    pipeline skew from the 12 minutes of 1069cohort. The fix gives the arm
    a horizon: a clock or a round count measured against a bound (whichever
    form -- this asserts the vocabulary of one, not a particular one).

    Honest bound of this assertion: it pins that SOME horizon term is
    consulted at or one hop below the node, not that the bound is correct.
    The bound's correctness is boot evidence, not desk evidence."""
    seg = _occupant_throttle_sources()
    horizon_terms = (
        "monotonic",
        "deadline",
        "horizon",
        "elapsed",
        "_since",
        "since_",
        "timeout",
        "stall",
        "budget",
        "rounds",
        "_ROUNDS",
        "perf_counter",
    )
    assert any(t in seg for t in horizon_terms), (
        "the occupant throttle still sleeps with no horizon: the arm cannot "
        "tell 2 ms of skew from 12 minutes of stall, which is exactly what "
        "both measured instances did (1068cap PP2, 1069cohort PP1 at :4189)"
    )


def test_C_an_occupant_past_the_horizon_is_loud():
    """(C) RED-FIRST, second half. An occupant that outlives its horizon is
    not a slow peer -- it is the ranks disagreeing about who owes the next
    statement for that slot, and RAENGE-NIE-UNEINS says detected
    disagreement is a crash/stop, never a compensating wait. The arm must
    carry a loud path (raise, or at minimum a named error line with rid,
    slot and counters) for the lapsed case.

    Note the asymmetry that makes this safe: taking the arm is legitimate
    and frequent; OUTLIVING the horizon on it never is."""
    seg = _occupant_throttle_sources()
    assert ("raise" in seg) or ("logger.error" in seg), (
        "no loud path at the occupant node: a slot occupied past any "
        "reasonable horizon degrades into a silent 2 ms spin instead of "
        "naming the rank-divergence it actually is"
    )


def _void_relay_census():
    """(emitter_sites_outside_the_absorber, payload_readers) for the #797
    void-output relay, by ast over the module -- the two halves that decide
    whether the relay is wired, deleted, or (today) half-built."""
    import ast as _ast
    import inspect as _i

    from sglang.srt.managers import scheduler_pp_mixin as mixin

    src = open(_i.getsourcefile(mixin)).read()
    tree = _ast.parse(src)

    def enclosing(lineno):
        best = None
        for n in _ast.walk(tree):
            if isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                if n.lineno <= lineno <= (n.end_lineno or n.lineno):
                    if best is None or n.lineno > best.lineno:
                        best = n
        return best.name if best else None

    emitters = []
    for n in _ast.walk(tree):
        if isinstance(n, _ast.Dict):
            for k in n.keys:
                if isinstance(k, _ast.Name) and k.id == "_PP_VOID_OUTPUT_KEY":
                    owner = enclosing(n.lineno)
                    if owner != "_pp_absorb_void_output":
                        emitters.append((n.lineno, owner))

    readers = [
        (n.lineno, enclosing(n.lineno))
        for n in _ast.walk(tree)
        if isinstance(n, _ast.Attribute)
        and n.attr == "_pp_void_forward_payload"
        and isinstance(n.ctx, _ast.Load)
    ]
    gone = not any(
        name in src
        for name in (
            "_PP_VOID_OUTPUT_KEY",
            "pp_void_forward_payload",
        )
    )
    return emitters, readers, gone


def test_C_the_void_relay_is_wired_or_deleted_but_never_half_built():
    """(C) RED-FIRST, third half -- and deliberately satisfiable BOTH ways.

    Measured at ca0ee3acd4: `_PP_VOID_OUTPUT_KEY: True` is constructed at
    exactly ONE site, inside `_pp_absorb_void_output`, i.e. only ever when
    re-forwarding a void that was already received; no rank originates one,
    because #969 CUT V deleted the emitter. And the re-forward payload is
    assigned to `self._pp_void_forward_payload` and read by nobody (1
    writer / 0 readers). `pp_void_forward_payload`'s own docstring states
    the #801 relay invariant it can no longer keep: "a non-last rank that
    took exactly one message off this wire for a ring generation must put
    exactly one back on it, void included".

    Half a relay is the worst of the three states: PP0 parks on an output
    the mid-rank void swallowed, and the successor throttles on an occupant
    nobody will speak for. Either exit closes it --

      WIRED   an originating emitter exists AND the forward payload is
              actually read/sent (the O2 E-a/E-b form), or
      DELETED the relay symbols are gone, PP0 owns the verdict, and the
              downstream ranks carry no void bookkeeping at all (the
              structural cut, and the default under UPSTREAM-MINIMAL:
              repair carries the burden of proof, deletion does not).
    """
    emitters, readers, gone = _void_relay_census()
    wired = bool(emitters) and bool(readers)
    assert wired or gone, (
        "the #797 void relay is half-built: originating emitters outside "
        f"the absorber = {emitters or 'NONE'}, readers of "
        f"_pp_void_forward_payload = {readers or 'NONE'}. A void nobody "
        "originates, nobody relays and one rank absorbs leaves PP0 parked "
        "and the successor spinning on an occupant (1068cap + 1069cohort)"
    )
