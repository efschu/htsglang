"""#631 Route A -- unit tests for the automatic phase policy.

Written RED FIRST. The three falsifiers the build spec names are the first
three sections; each is a test that PASSES only because the policy has the
property, and that would fail against an obvious wrong implementation:

  * thrash falsifier -- bursty arrivals below N must NOT flip,
  * drain falsifier  -- a drained prefill queue flips to the decode layout,
  * in-flight safety -- the policy never aborts a request; the only thing it
    can do is ARM, and the bounded park abandons the FLIP instead.

plus the resting-state rule: with the resting layout at PP, an idle server
returns there on its own, and a short prefill never forces a flip.

CPU-only; the policy is a pure function and needs no GPU.
"""

import pytest

from sglang.srt.managers.phase_policy import (
    PHASE_PP,
    PHASE_TP,
    PP_TO_TP,
    REST_DECODE,
    REST_PREFILL,
    TP_TO_PP,
    PhasePolicyConfig,
    PhasePolicyError,
    PhasePolicyInputs,
    PhasePolicyState,
    break_even_tokens,
    config_from_env,
    decide,
    note_flip_armed,
    observe_idle,
)

N = 30000  # a stand-in threshold; the real one is measured


def cfg(**kw):
    base = dict(
        enabled=True,
        flip_tokens=N,
        min_dwell_s=10.0,
        idle_dwell_s=20.0,
        rest_state=REST_PREFILL,
    )
    base.update(kw)
    return PhasePolicyConfig(**base)


def inp(phase, pending=0, running=0, now=1000.0):
    return PhasePolicyInputs(
        phase=phase,
        pending_prefill_tokens=pending,
        running_bs=running,
        now=now,
    )


# -- the break-even ------------------------------------------------------------


def test_break_even_matches_the_closed_form():
    # N = C / (1/X - 1/P). With C=3.2, X=5000, P=7245.5:
    #   1/5000 - 1/7245.5 = 2.0000e-4 - 1.38018e-4 = 6.1982e-5
    #   3.2 / 6.1982e-5 = 51628.5...
    got = break_even_tokens(3.2, 5000.0, 7245.5)
    assert got == pytest.approx(51629, abs=2)


def test_break_even_refuses_when_tp_is_not_slower():
    # The premise of Route A's prefill layout is that PP prefills FASTER.
    # If a measurement says otherwise there is no token count that repays
    # the flip, and inventing one would be a silent lie.
    with pytest.raises(PhasePolicyError, match="no break-even exists"):
        break_even_tokens(3.2, 8000.0, 7245.5)
    with pytest.raises(PhasePolicyError, match="no break-even exists"):
        break_even_tokens(3.2, 7245.5, 7245.5)


def test_break_even_refuses_unmeasured_inputs():
    with pytest.raises(PhasePolicyError):
        break_even_tokens(3.2, 0.0, 7245.5)
    with pytest.raises(PhasePolicyError):
        break_even_tokens(0.0, 5000.0, 7245.5)


# -- falsifier 1: thrash -------------------------------------------------------


def test_thrash_falsifier_bursty_arrivals_below_n_never_flip():
    """Bursty arrivals, each below N, must not move the server.

    This is the falsifier for a policy that flips on "there is prefill
    work" rather than on "there is enough prefill work to repay the flip".
    """
    c = cfg()
    st = PhasePolicyState(last_flip_at=0.0)
    t = 1000.0
    for burst in range(20):
        # A burst arrives, below the threshold, then drains, repeatedly.
        for pending in (N - 1, N // 2, N - 100, 0):
            i = inp(PHASE_TP, pending=pending, running=2, now=t)
            observe_idle(st, i)
            d = decide(c, st, i)
            assert d.direction is None, (
                f"burst {burst} pending={pending} flipped: {d.reason}"
            )
            t += 1.0
    assert st.flips_armed == 0


def test_min_dwell_blocks_a_second_flip_even_when_tokens_say_go():
    """N alone does not bound thrash; the dwell timer must.

    Falsifier for a policy that only checks the token threshold: here the
    token condition is satisfied, and the ONLY thing that may hold the flip
    back is the independent minimum-dwell timer.
    """
    c = cfg(min_dwell_s=10.0)
    st = PhasePolicyState()
    # A flip just happened at t=1000.
    note_flip_armed(st, decide(c, st, inp(PHASE_TP, pending=N + 1)), now=1000.0)
    # 2s later the token condition is loudly true, but dwell must refuse.
    d = decide(c, st, inp(PHASE_TP, pending=N * 10, running=1, now=1002.0))
    assert d.direction is None
    assert "min dwell" in d.reason
    # Past the dwell, the same condition is honoured.
    d = decide(c, st, inp(PHASE_TP, pending=N * 10, running=1, now=1011.0))
    assert d.direction == TP_TO_PP


def test_above_threshold_flips_to_the_prefill_layout():
    c = cfg()
    st = PhasePolicyState()
    d = decide(c, st, inp(PHASE_TP, pending=N + 1, running=1))
    assert d.direction == TP_TO_PP
    assert f"N={N}" in d.reason


def test_exactly_at_threshold_does_not_flip():
    # N is a break-even: AT it, flipping neither gains nor loses, so the
    # tie goes to not moving the server.
    c = cfg()
    st = PhasePolicyState()
    assert decide(c, st, inp(PHASE_TP, pending=N, running=1)).direction is None


# -- falsifier 2: drain --------------------------------------------------------


def test_drain_falsifier_prefill_queue_drained_flips_to_decode_layout():
    """When prefill drains and decode work remains, go to TP.

    Speculation and the decode CUDA graphs exist only in TP, so a server
    left decoding in PP is the bug this rule prevents.
    """
    c = cfg()
    st = PhasePolicyState()
    d = decide(c, st, inp(PHASE_PP, pending=0, running=3))
    assert d.direction == PP_TO_TP
    assert "3 req decoding" in d.reason


def test_drain_ignores_n_because_there_is_no_prefill_left_to_price():
    # N prices a prefill against a flip. With the queue empty there is no
    # prefill to price, so a huge N must not block the decode flip.
    c = cfg(flip_tokens=10**9)
    st = PhasePolicyState()
    assert decide(c, st, inp(PHASE_PP, pending=0, running=1)).direction == PP_TO_TP


def test_pp_with_a_worthwhile_prefill_backlog_stays_in_pp():
    """Above N, the backlog is worth finishing in the fast layout."""
    c = cfg()
    st = PhasePolicyState()
    d = decide(c, st, inp(PHASE_PP, pending=N + 1, running=1))
    assert d.direction is None
    assert "prefilling in pp" in d.reason


def test_batching_small_residual_prefill_does_not_pin_the_server_in_pp():
    """Continuous arrivals must not trap decode in the prefill layout.

    Falsifier for an ``== 0`` drain rule: under batching the queue may
    never reach exactly zero, and decoding in PP means decoding with no
    speculation and no decode CUDA graphs.
    """
    c = cfg()
    st = PhasePolicyState()
    d = decide(c, st, inp(PHASE_PP, pending=N - 1, running=2))
    assert d.direction == PP_TO_TP
    assert "decoding" in d.reason


# -- falsifier 3: in-flight safety --------------------------------------------


def test_policy_can_only_ever_arm_never_abort():
    """The policy's entire vocabulary is a direction or None.

    In-flight safety is structural: the policy has no way to express
    "abort". A flip that cannot reach quiescence is abandoned by the
    bounded park (PhaseFlipRuntime._abandon_parked_flip), which drops the
    FLIP and leaves the parked requests to run. This test pins the
    vocabulary so a future edit cannot widen it without being noticed.
    """
    c = cfg()
    st = PhasePolicyState()
    seen = set()
    for phase in (PHASE_PP, PHASE_TP):
        for pending in (0, 1, N - 1, N, N + 1, 10**7):
            for running in (0, 1, 4):
                d = decide(c, st, inp(phase, pending, running, now=5000.0))
                seen.add(d.direction)
                assert d.direction in (None, PP_TO_TP, TP_TO_PP)
                assert isinstance(d.reason, str) and d.reason
    assert seen <= {None, PP_TO_TP, TP_TO_PP}


def test_decide_is_pure_and_does_not_mutate_state():
    c = cfg()
    st = PhasePolicyState(last_flip_at=1.0, idle_since=2.0, flips_armed=3)
    before = (st.last_flip_at, st.idle_since, st.flips_armed, st.last_reason)
    decide(c, st, inp(PHASE_TP, pending=N * 5, running=1, now=9999.0))
    after = (st.last_flip_at, st.idle_since, st.flips_armed, st.last_reason)
    assert before == after, "decide must not mutate policy state"


# -- the resting state (user directive: rest in PP) ---------------------------


def test_idle_server_returns_to_the_prefill_layout_after_the_idle_dwell():
    c = cfg(idle_dwell_s=20.0)
    st = PhasePolicyState()
    # Server goes quiet in TP at t=1000.
    i = inp(PHASE_TP, pending=0, running=0, now=1000.0)
    observe_idle(st, i)
    assert decide(c, st, i).direction is None  # dwell not yet served

    i = inp(PHASE_TP, pending=0, running=0, now=1015.0)
    observe_idle(st, i)
    d = decide(c, st, i)
    assert d.direction is None and "idle dwell" in d.reason

    i = inp(PHASE_TP, pending=0, running=0, now=1021.0)
    observe_idle(st, i)
    d = decide(c, st, i)
    assert d.direction == TP_TO_PP
    assert "resting layout" in d.reason


def test_idle_clock_resets_when_work_arrives():
    """A brief gap must not accumulate toward the idle return."""
    c = cfg(idle_dwell_s=20.0)
    st = PhasePolicyState()
    observe_idle(st, inp(PHASE_TP, 0, 0, now=1000.0))
    assert st.idle_since == 1000.0
    # Work arrives at t=1010 -> the idle clock is cleared...
    observe_idle(st, inp(PHASE_TP, 0, 1, now=1010.0))
    assert st.idle_since is None
    # ...and restarts from the next quiet round, so at t=1025 the server
    # has only been idle 5s and must NOT flip.
    observe_idle(st, inp(PHASE_TP, 0, 0, now=1020.0))
    i = inp(PHASE_TP, 0, 0, now=1025.0)
    observe_idle(st, i)
    assert decide(c, st, i).direction is None


def test_at_rest_in_pp_an_idle_server_stays_put():
    """The resting layout must be a fixed point: no idle flip-flopping."""
    c = cfg()
    st = PhasePolicyState()
    for t in range(1000, 1200, 5):
        i = inp(PHASE_PP, pending=0, running=0, now=float(t))
        observe_idle(st, i)
        d = decide(c, st, i)
        assert d.direction is None, f"idle PP flipped at t={t}: {d.reason}"


def test_short_prefill_from_rest_needs_no_flip_at_all():
    """The point of resting in PP: a long prompt arriving at a quiet server
    prefills immediately, with no flip in its latency path."""
    c = cfg()
    st = PhasePolicyState()
    i = inp(PHASE_PP, pending=10**6, running=0, now=1000.0)
    observe_idle(st, i)
    assert decide(c, st, i).direction is None


def test_rest_state_decode_inverts_the_idle_return():
    c = cfg(rest_state=REST_DECODE)
    assert c.rest_phase == PHASE_TP
    st = PhasePolicyState()
    observe_idle(st, inp(PHASE_PP, 0, 0, now=1000.0))
    i = inp(PHASE_PP, 0, 0, now=1030.0)
    observe_idle(st, i)
    d = decide(c, st, i)
    assert d.direction == PP_TO_TP
    assert "resting layout" in d.reason


def test_rest_state_is_validated():
    with pytest.raises(PhasePolicyError, match="not a known resting state"):
        PhasePolicyConfig(enabled=False, flip_tokens=1, rest_state="sideways")


# -- disabled is inert ---------------------------------------------------------


def test_disabled_policy_never_decides_anything():
    c = PhasePolicyConfig(enabled=False)
    st = PhasePolicyState()
    for phase in (PHASE_PP, PHASE_TP):
        d = decide(c, st, inp(phase, pending=10**7, running=4, now=1.0))
        assert d.direction is None
        assert d.reason == "policy disabled"


def test_enabled_without_a_threshold_is_refused():
    with pytest.raises(PhasePolicyError, match="positive flip threshold"):
        PhasePolicyConfig(enabled=True, flip_tokens=0)


# -- configuration -------------------------------------------------------------


def test_config_from_env_derives_n_from_a_measured_throughput(monkeypatch):
    import sglang.srt.managers.phase_policy as pp

    monkeypatch.setattr(pp, "DEFAULT_TP_PREFILL_TOK_S", 5000.0)
    monkeypatch.delenv(pp.ENV_FLIP_TOKENS, raising=False)
    monkeypatch.delenv(pp.ENV_REST_STATE, raising=False)
    monkeypatch.delenv(pp.ENV_MIN_DWELL, raising=False)
    monkeypatch.delenv(pp.ENV_IDLE_DWELL, raising=False)
    c = pp.config_from_env(enabled=True)
    assert c.enabled
    assert c.flip_tokens == pytest.approx(51629, abs=2)
    assert c.rest_state == REST_PREFILL  # the default is the prefill layout


def test_config_from_env_explicit_threshold_wins(monkeypatch):
    import sglang.srt.managers.phase_policy as pp

    monkeypatch.setenv(pp.ENV_FLIP_TOKENS, "12345")
    monkeypatch.setenv(pp.ENV_MIN_DWELL, "3.5")
    monkeypatch.setenv(pp.ENV_IDLE_DWELL, "7.5")
    monkeypatch.setenv(pp.ENV_REST_STATE, REST_DECODE)
    c = pp.config_from_env(enabled=True)
    assert c.flip_tokens == 12345
    assert c.min_dwell_s == 3.5
    assert c.idle_dwell_s == 7.5
    assert c.rest_state == REST_DECODE


def test_config_from_env_refuses_enabled_without_any_threshold(monkeypatch):
    import sglang.srt.managers.phase_policy as pp

    monkeypatch.setattr(pp, "DEFAULT_TP_PREFILL_TOK_S", 0.0)
    monkeypatch.delenv(pp.ENV_FLIP_TOKENS, raising=False)
    with pytest.raises(PhasePolicyError, match="A threshold is a measurement"):
        pp.config_from_env(enabled=True)


def test_config_from_env_off_is_inert_without_a_threshold(monkeypatch):
    import sglang.srt.managers.phase_policy as pp

    monkeypatch.setattr(pp, "DEFAULT_TP_PREFILL_TOK_S", 0.0)
    monkeypatch.delenv(pp.ENV_FLIP_TOKENS, raising=False)
    c = pp.config_from_env(enabled=False)
    assert not c.enabled and c.flip_tokens == 0


def test_bad_env_number_is_named(monkeypatch):
    import sglang.srt.managers.phase_policy as pp

    monkeypatch.setenv(pp.ENV_MIN_DWELL, "soon")
    with pytest.raises(PhasePolicyError, match="MIN_DWELL"):
        pp.config_from_env(enabled=False)


# -- wiring: DELIVERY-BEFORE-BLOCK --------------------------------------------
#
# The arm rides the same chain a manual POST /phase_flip uses, because
# forwarding it is what WAKES the downstream stages out of their blocking
# chain recv. The invariant that makes that safe: no rank may enter the
# blocking flip reduction while it still owes, or has uncommitted, chain
# sends. Measured failures behind every assertion here are named in the
# docstrings -- all three were live wedges, not hypotheticals.


class _StubRuntime:
    def __init__(self, phase=PHASE_TP, pending=None):
        self.phase = phase
        self.pending = pending


class _StubBatch:
    def __init__(self, n):
        self._n = n

    def batch_size(self):
        return self._n


class _StubReq:
    def __init__(self, n):
        self.origin_input_ids = [0] * n


def _sched(cfg_kw=None, phase=PHASE_TP, pending=None, queue=(), running=0):
    from sglang.srt.managers.scheduler import Scheduler

    class S:
        pass

    s = S()
    s.phase_policy_cfg = cfg(**(cfg_kw or {}))
    s.phase_policy_state = PhasePolicyState()
    s.phase_flip_runtime = _StubRuntime(phase=phase, pending=pending)
    s.waiting_queue = list(queue)
    s.running_batch = _StubBatch(running)
    s.maybe_arm_phase_policy = Scheduler.maybe_arm_phase_policy.__get__(s, S)
    return s


def test_policy_emits_the_same_request_the_rpc_path_produces():
    """The automatic path differs from the proven manual one only in who
    originates the arm."""
    from sglang.srt.managers.io_struct import PhaseFlipReqInput

    s = _sched(queue=[_StubReq(N + 1000)], running=1)
    req = s.maybe_arm_phase_policy()
    assert isinstance(req, PhaseFlipReqInput)
    assert req.direction == TP_TO_PP
    assert req.source == "policy"  # evidence in the log that no human flipped
    assert req.internal is True  # must never be answered


def test_policy_emits_nothing_below_the_threshold():
    s = _sched(queue=[_StubReq(N - 1)], running=1)
    assert s.maybe_arm_phase_policy() is None


def test_policy_is_silent_before_the_flip_runtime_exists():
    s = _sched(queue=[_StubReq(N * 10)], running=1)
    s.phase_flip_runtime = None
    assert s.maybe_arm_phase_policy() is None


def test_policy_does_not_rearm_a_flip_already_pending():
    s = _sched(queue=[_StubReq(N * 10)], running=1, pending=TP_TO_PP)
    assert s.maybe_arm_phase_policy() is None


def test_disabled_policy_emits_nothing():
    s = _sched(queue=[_StubReq(N * 10)], running=1)
    s.phase_policy_cfg = PhasePolicyConfig(enabled=False)
    assert s.maybe_arm_phase_policy() is None


def _fwd_harness(is_last_rank=False, track_commits=False, armed=False):
    from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin

    sends = []
    processed = []
    commits = []
    services = []

    class _Grp:
        pass

    _Grp.is_last_rank = is_last_rank

    class S:
        pp_group = _Grp()
        send_req_work = None

        def pp_phase_flip_armed(self):
            # #631: the real predicate reads the runtime's _pending; the
            # harness pins the two branches directly.
            return armed

        def pp_flip_service(self):
            services.append(True)

        def _pp_commit_comm_work(self, work):
            # "post-send" means: committed AFTER this pass issued its own
            # forward, i.e. the targeted in-pass commit rather than the
            # ordinary top-of-pass commit of the PREVIOUS pass's handle.
            commits.append("post-send" if sends else "top-of-pass")

        def _pp_send_pyobj_to_next_stage(self, data, async_send=False):
            sends.append((async_send, list(data) if data else []))
            return ["work"]

        def process_input_requests(self, reqs):
            processed.append(list(reqs) if reqs else [])

    s = S()
    s.services = services
    fwd = SchedulerPPMixin._pp_forward_and_process_input_requests.__get__(s, S)
    if track_commits:
        return s, fwd, sends, processed, commits
    return s, fwd, sends, processed


def test_arm_is_armed_in_the_same_pass_it_arrives():
    """(i) SAME-PASS JOIN. Specimen: boot 12, 2026-08-08.

    Deferring the arm by one pass is a GUARANTEED miss: the downstream
    stage must re-enter the chain recv to reach the pass where it acts,
    and that recv blocks because upstream is already inside the
    reduction. All three ranks armed, cutovers=0, dead at 40 s.
    """
    from sglang.srt.managers.io_struct import PhaseFlipReqInput

    arm = PhaseFlipReqInput(direction=PP_TO_TP, source="policy", internal=True)
    s, fwd, sends, processed = _fwd_harness()
    fwd(["ordinary", arm])
    assert arm in processed[0], (
        "the arm was not acted on in the pass it arrived; a deferred arm "
        "needs another pass, and that pass begins with a recv that blocks"
    )
    assert "ordinary" in processed[0]


def test_no_blocking_commit_anywhere_in_the_armed_path():
    """BOOT-13 SPECIMEN. The design law, pinned at the source.

    Committing the arm-carrying send in-pass is corpse B': rank 0 blocked
    in _pp_commit_comm_work while ranks 1-2 sat in the HIDDEN-STATES
    exchange. "The peer is waiting for the arm" is false -- it may be in
    another channel entirely, so ANY blocking wait on the chain send can
    pair with a peer parked elsewhere.

    The arm forward is pumped non-blockingly by the armed poll loop
    instead. This test fails if a blocking commit is reintroduced for
    arm-carrying batches.
    """
    from sglang.srt.managers.io_struct import PhaseFlipReqInput

    arm = PhaseFlipReqInput(direction=PP_TO_TP, source="policy", internal=True)
    s, fwd, sends, processed, commits = _fwd_harness(track_commits=True)
    fwd([arm])
    assert sends[0][0] is True, "the forward must stay async"
    assert "post-send" not in commits, (
        "an arm-carrying send was committed in-pass; that is corpse B' -- "
        "this rank blocks on the chain send while a peer may be in the "
        "hidden-states channel"
    )


def test_ordinary_traffic_keeps_the_uncommitted_async_forward():
    """The targeted commit is for arm batches ONLY. Committing every
    forward would serialise the pipeline on each poll."""
    s, fwd, sends, processed, commits = _fwd_harness(track_commits=True)
    fwd(["ordinary"])
    assert sends[0][0] is True
    assert "post-send" not in commits, (
        "an ordinary batch was committed in-pass; that serialises the chain"
    )


def test_manual_flip_takes_the_same_non_blocking_path():
    """(d) The manual RPC goes through the same epoch machinery.

    Strictly safer, and it removes manual's latent at-idle deadlock:
    manual has only ever been exercised UNDER TRAFFIC, where the loop
    keeps cycling and delivery happens by accident rather than by
    construction.
    """
    from sglang.srt.managers.io_struct import PhaseFlipReqInput

    rpc = PhaseFlipReqInput(direction=PP_TO_TP)  # internal=False
    s, fwd, sends, processed, commits = _fwd_harness(track_commits=True)
    fwd([rpc])
    assert rpc in processed[0], "the manual arm must also be same-pass"
    assert "post-send" not in commits, (
        "the manual flip blocks on its chain send -- corpse B'"
    )


def test_last_stage_owes_no_forward_and_joins_directly():
    """(iii) The last rank has nobody to wake; it must not try to commit
    a send it never issued, and must still arm in-pass."""
    from sglang.srt.managers.io_struct import PhaseFlipReqInput

    arm = PhaseFlipReqInput(direction=PP_TO_TP, source="policy", internal=True)
    s, fwd, sends, processed, commits = _fwd_harness(
        is_last_rank=True, track_commits=True
    )
    fwd([arm])
    assert sends == [], "the last rank must not forward"
    assert arm in processed[0], "the last rank must still arm in-pass"


def test_only_the_zmq_intake_rank_injects():
    """Origin guard. A list is NOT the origin signal: in the PP phase
    every stage gets a list, so a list-based guard made each stage inject
    its own arm -- measured 1/2/3 arms on PP0/PP1/PP2 and a self-kill.
    """
    from contextlib import ExitStack
    from unittest import mock

    from sglang.srt.managers.io_struct import PhaseFlipReqInput
    from sglang.srt.managers.scheduler_components.request_receiver import (
        SchedulerRequestReceiver,
    )

    sentinel = PhaseFlipReqInput(
        direction=TP_TO_PP, source="policy", internal=True
    )

    def run(pp_rank, pulled):
        calls = []

        class _PS:
            pass

        _PS.pp_rank = pp_rank
        _PS.attn_tp_rank = 0
        _PS.attn_cp_rank = 0

        recv = SchedulerRequestReceiver(
            recv_from_tokenizer=None,
            recv_from_rpc=None,
            recv_skipper=None,
            input_blocker=None,
            mm_receiver=None,
            ps=_PS(),
            tp_group=None,
            tp_cpu_group=None,
            attn_tp_group=None,
            attn_tp_cpu_group=None,
            attn_cp_group=None,
            attn_cp_cpu_group=None,
            world_group=None,
            server_args=None,
            model_config=None,
            max_recv_per_poll=-1,
            stream_output=lambda *a, **kw: None,
            get_last_forward_mode=lambda: None,
            phase_policy_hook=lambda: (calls.append(1), sentinel)[1],
        )
        cls = SchedulerRequestReceiver
        with ExitStack() as st:
            for p in (
                mock.patch.object(cls, "_pull_raw_reqs", lambda self: pulled),
                mock.patch.object(
                    cls, "_broadcast_reqs_across_ranks", lambda self, r: r
                ),
                mock.patch.object(
                    cls, "unwrap_pickle_wrapper", lambda self, r: None
                ),
                mock.patch.object(cls, "_apply_mm_receiver", lambda self, r: r),
                mock.patch.object(
                    cls, "_finalize_shm_features", lambda self, r: None
                ),
            ):
                st.enter_context(p)
            return recv.recv_requests(), calls

    out, calls = run(0, ["existing"])
    assert out == ["existing", sentinel] and calls == [1]

    # A downstream stage gets a populated list off the wire and must add
    # nothing of its own.
    out, calls = run(1, ["from-upstream"])
    assert out == ["from-upstream"], "a downstream stage injected its own arm"
    assert calls == []


# -- (c) bounded join: WITHDRAWN, measured fatal ------------------------------


def test_the_flip_join_is_not_bounded_and_that_is_deliberate():
    """A bounded join that abandons from inside KILLS THE GROUP.

    Measured 2026-08-08: the bound fired (CollectiveTimeoutError) and the
    moment the rank walked away its peers saw "Connection closed by peer"
    from gloo and every rank died with "Fatal Python error: Aborted". A
    rank that has ENTERED an all_reduce owes that all_reduce. Any bound
    must therefore be applied BEFORE entry, or the reduction must become
    a non-blocking poll -- a different design, not a timeout.

    This test exists so nobody re-adds the timeout believing it is the
    obvious missing safety net. It is the opposite.
    """
    import inspect

    from sglang.srt.managers.phase_flip_runtime import PhaseFlipRuntime

    src = inspect.getsource(PhaseFlipRuntime.on_round)
    assert "_join_bounded" not in src, (
        "the flip join was bounded again; abandoning an entered gloo "
        "all_reduce aborts every rank in the group"
    )


# -- option 2(b): the pollable presence gate ----------------------------------
#
# THE DESIGN LAW: no rank may block on any channel while a peer may be in a
# different blocking channel. The gate is what makes entering the blocking
# reduction safe BY CONSTRUCTION -- every participant is provably at the
# entry, so none is anywhere else.


def _presence(tmpdir, n_ranks=3, rank=0):
    from sglang.srt.managers.phase_flip_presence import PhaseFlipPresence

    return PhaseFlipPresence(
        n_ranks=n_ranks, rank=rank, directory=str(tmpdir), instance="test"
    )


def test_presence_is_pollable_and_never_blocks(tmp_path):
    p0 = _presence(tmp_path, rank=0)
    assert p0.all_present(epoch=1) is False
    assert p0.missing(epoch=1) == [0, 1, 2]
    p0.announce(1)
    assert p0.observe(1) == {0}
    assert p0.all_present(1) is False


def test_gate_opens_only_when_every_rank_is_present(tmp_path):
    """ALL-READY GATE. Entering with a peer missing is the deadlock."""
    ranks = [_presence(tmp_path, rank=r) for r in range(3)]
    ranks[0].announce(1)
    ranks[1].announce(1)
    assert ranks[0].all_present(1) is False, (
        "the gate opened with rank 2 missing; that rank is elsewhere and "
        "the reduction would block on it"
    )
    ranks[2].announce(1)
    assert ranks[0].all_present(1) is True


def test_announce_is_idempotent(tmp_path):
    """The armed poll loop announces every iteration rather than tracking
    whether it already did."""
    p0 = _presence(tmp_path, rank=0)
    for _ in range(5):
        p0.announce(1)
    assert p0.observe(1) == {0}


def test_epochs_are_monotone_and_a_retraction_mints_a_new_one(tmp_path):
    """Flags are NEVER cleared. A flag observed for epoch E is a fact
    about E for ever, so a poll cannot be fooled by a racing writer and a
    writer never has to coordinate a clear."""
    ranks = [_presence(tmp_path, rank=r) for r in range(3)]
    for r in ranks:
        r.announce(1)
    assert ranks[0].all_present(1) is True
    # A retraction moves to epoch 2; epoch 1's flags must not count.
    assert ranks[0].all_present(2) is False
    assert ranks[0].missing(2) == [0, 1, 2]


def test_sweep_never_drops_the_epoch_in_flight(tmp_path):
    ranks = [_presence(tmp_path, rank=r) for r in range(3)]
    for r in ranks:
        r.announce(1)
        r.announce(2)
    ranks[0].sweep(keep_epoch=2)
    assert ranks[0].all_present(2) is True
    assert ranks[0].all_present(1) is False


def _runtime_stub(
    presence,
    deadline=60.0,
    pending="pp_to_tp",
    clock=None,
    drain_fn=None,
    owes_send_fn=None,
    service_fn=None,
    channels_empty_fn=None,
):
    from sglang.srt.managers.phase_flip_runtime import PhaseFlipRuntime

    class R:
        pass

    r = R()
    r._presence = presence
    r._pump_fn = None
    # #631 boot 18: (ii) keep consuming while armed, (i) announce only
    # once this rank owes no send. Default None/absent keeps the stub on
    # the old behaviour, so the pre-existing gate pins keep their meaning.
    r._drain_fn = drain_fn
    r._owes_send_fn = owes_send_fn
    # #631 G: the armed service turn and the flip-commit hygiene probe.
    # Default None keeps every pre-existing gate pin on its old meaning.
    r._service_fn = service_fn
    r._channels_empty_fn = channels_empty_fn
    r.presence_withheld_rounds = 0
    r.presence_withheld_channels = 0
    r.entry_channel_violations = 0
    r._last_withhold_log = None
    # #631 round-scoped entry evidence: the gate reads (epoch, round).
    r._entry_round = 0
    r._presence_wait_stamp = None
    r._presence_deadline_s = deadline
    r._presence_wait_started = None
    r._gate_open_epoch = None
    r._epoch = 1
    r._pending = pending
    r._armed_at = 0.0
    r._last_hold_reason = None
    r._phase = PHASE_PP
    r.presence_timeouts = 0
    r._clock = clock or (lambda: 0.0)
    r._sleep = lambda _s: None
    r._presence_poll_interval_s = 0.0
    for _n in ("_await_group_presence", "_abandon_no_quorum", "_commit_to_entering"):
        setattr(r, _n, getattr(PhaseFlipRuntime, _n).__get__(r, R))
    return r


def test_gate_holds_the_caller_until_the_group_arrives(tmp_path):
    r = _runtime_stub(_presence(tmp_path, rank=0))
    assert r._await_group_presence() is None, "the gate opened too early"
    # Peers arrive.
    _presence(tmp_path, rank=1).announce(1)
    _presence(tmp_path, rank=2).announce(1)
    assert r._await_group_presence() is True


def test_pre_entry_timeout_disarms_loudly_and_keeps_serving(tmp_path):
    """PRE-ENTRY BOUND, legal precisely because nothing was entered.

    Contrast the withdrawn (c): abandoning an ENTERED all_reduce closed
    the gloo pairs and aborted every rank. Abandoning a POLL costs
    nothing -- no peer is owed a collective.
    """
    now = {"t": 0.0}
    r = _runtime_stub(
        _presence(tmp_path, rank=0), deadline=10.0, clock=lambda: now["t"]
    )
    assert r._await_group_presence() is None
    now["t"] = 11.0
    out = r._await_group_presence()
    assert out is None, "abandonment must not raise into the event loop"
    assert r._pending is None, "the flip must be disarmed so it can be retried"
    assert r.presence_timeouts == 1


def test_gate_pumps_the_arm_forward_while_it_waits(tmp_path):
    """The pump is what delivers the arm without blocking -- the fix for
    corpse A (an async send is otherwise progressed only by the commit at
    the top of the NEXT pass, which never comes once this rank is armed
    and polling)."""
    pumped = []
    r = _runtime_stub(_presence(tmp_path, rank=0))
    r._pump_fn = lambda: pumped.append(1)
    r._await_group_presence()
    assert pumped, "the armed poll loop did not pump its arm forward"


def test_a_failing_pump_never_breaks_the_gate(tmp_path):
    r = _runtime_stub(_presence(tmp_path, rank=0))

    def boom():
        raise RuntimeError("transport hiccup")

    r._pump_fn = boom
    assert r._await_group_presence() is None  # no raise


def test_park_clock_is_rebased_when_the_group_assembles(tmp_path):
    """BOOT-14 SPECIMEN: two bounds must not race.

    The park deadline measures "armed but never quiescent" -- meaningful
    only once the group is assembled. Left measuring from the arm, it
    races the presence gate: a rank whose peers are slow to arrive
    abandons on the PARK deadline while they are still polling, the ranks
    then disagree around a gloo collective, and "Connection closed by
    peer" aborts every rank. Measured: all three announced presence, then
    abandoned at exactly 30.0 s and the group died.
    """
    now = {"t": 0.0}
    r = _runtime_stub(_presence(tmp_path, rank=0), clock=lambda: now["t"])
    r._armed_at = 0.0
    now["t"] = 25.0
    assert r._await_group_presence() is None  # peers not here yet
    assert r._armed_at == 0.0, "the park clock moved before assembly"

    _presence(tmp_path, rank=1).announce(1)
    _presence(tmp_path, rank=2).announce(1)
    now["t"] = 28.0
    assert r._await_group_presence() is True
    assert r._armed_at == 28.0, (
        "the park clock was not re-based on assembly; it would expire "
        "mid-quiescence and desync the group around the collective"
    )


def test_stale_markers_from_an_earlier_boot_never_open_the_gate(tmp_path):
    """BOOT-15 SPECIMEN: the gate opened on a previous boot's flags.

    The instance tag was os.getpid()//100000, which COLLIDES across
    consecutive boots (3163115 and 3180590 both give 31). Boot 15 read
    boot 14's markers, the gate opened "after 0.00s" before its peers had
    armed, and rank 0 entered the reduction alone -- the gate causing the
    exact failure it exists to prevent.
    """
    from sglang.srt.managers.phase_flip_presence import PhaseFlipPresence

    # An earlier boot left a full quorum behind.
    old = [
        PhaseFlipPresence(
            n_ranks=3, rank=r, directory=str(tmp_path), instance="boot-14"
        )
        for r in range(3)
    ]
    for o in old:
        o.announce(0)
    assert old[0].all_present(0) is True

    # A NEW boot must not see any of it.
    fresh = PhaseFlipPresence(
        n_ranks=3, rank=0, directory=str(tmp_path), instance="boot-15"
    )
    assert fresh.all_present(0) is False, (
        "the gate opened on an earlier boot's markers"
    )
    assert fresh.missing(0) == [0, 1, 2]


def test_the_instance_tag_is_identical_across_ranks_of_one_boot(monkeypatch, tmp_path):
    """The flags are a RENDEZVOUS: a per-process tag would give every
    rank a different quorum and none would ever assemble."""
    from sglang.srt.managers.phase_flip_presence import PhaseFlipPresence

    monkeypatch.setenv("SGLANG_PHASE_FLIP_INSTANCE", "boot-xyz")
    tags = {
        PhaseFlipPresence(n_ranks=3, rank=r, directory=str(tmp_path)).instance
        for r in range(3)
    }
    assert tags == {"boot-xyz"}, f"ranks disagreed on the rendezvous tag: {tags}"


def test_presence_wait_does_not_count_toward_the_park_deadline():
    """BOOT-16 SPECIMEN, re-pinned as BEHAVIOUR after the quiescent-announce
    inversion.

    The lesson stands: the park deadline asks "armed but never
    quiescent?", and time spent waiting for the GROUP to assemble must
    never be counted as a failure to quiesce. With the gate once evaluated
    after the expiry, a rank that waited out the presence poll was already
    flagged expired and the flip was abandoned the instant the gate opened
    -- logged as "armed for 0.0s". All three ranks abandoned unanimously
    and the cutover never happened.

    This used to be pinned as a SOURCE ORDER (gate before expiry). That
    proxy is now wrong, because announcing requires quiescence: expiry is
    computed first so a never-draining rank can still reach the reduction
    and make the abandonment group-agreed. The invariant it stood for is
    pinned directly instead, and it now holds by CONSTRUCTION -- the two
    waits are disjoint. A rank is either draining (park clock) or spinning
    at the gate (per-round presence clock), never both, because a
    non-quiescent rank returns to the pass loop without announcing.
    """
    from sglang.srt.managers.phase_flip_runtime import PhaseFlipRuntime

    class R:
        pass

    r = R()
    now = {"t": 0.0}
    r._clock = lambda: now["t"]
    r._park_deadline_s = 30.0
    r._armed_at = 0.0
    expired = PhaseFlipRuntime._park_expired.__get__(r, R)

    # A QUIESCENT rank is never expired, however long it then waits at the
    # gate. This is what stops assembly time from being read as a failure
    # to drain.
    now["t"] = 10_000.0
    assert expired(armed=1, ready=1) is False, (
        "a quiescent rank was flagged expired; time spent waiting for the "
        "group to assemble is being counted as a failure to quiesce, and "
        "the flip is abandoned as the gate opens (boot 16)"
    )

    # A rank that never drains DOES expire -- that is the deadline's whole
    # job, and it is what carries a group-agreed abandonment.
    assert expired(armed=1, ready=0) is True

    # And an unarmed rank is never expired.
    assert expired(armed=0, ready=0) is False


def test_park_clock_is_rebased_once_per_arm_not_once_per_round(tmp_path):
    """BOOT-17 SPECIMEN: a per-round re-base makes the park deadline
    unreachable.

    The gate opens every round once the group is present. Re-basing the
    park clock each time means a flip that can NEVER reach quiescence
    holds for ever with its requests parked -- measured as repeated
    "group present after 0.00s" on every rank, cutovers=0, abandoned=0,
    and a server answering nothing. The re-base is a per-ARM event.
    """
    now = {"t": 0.0}
    r = _runtime_stub(_presence(tmp_path, rank=0), clock=lambda: now["t"])
    _presence(tmp_path, rank=1).announce(1)
    _presence(tmp_path, rank=2).announce(1)
    r._gate_open_epoch = None

    now["t"] = 5.0
    assert r._await_group_presence() is True
    assert r._armed_at == 5.0

    # Later rounds re-open the gate; the clock must NOT move again, or
    # the park deadline can never expire.
    now["t"] = 40.0
    assert r._await_group_presence() is True
    assert r._armed_at == 5.0, (
        "the park clock was re-based on a later round; the park deadline "
        "becomes unreachable and a stuck flip parks requests for ever"
    )


# -- boot 18: the gate assembled and the group still wedged -------------------
#
# THE SPECIMEN. Rank 0 sat inside the consensus reduction while rank 1 was
# blocked on the ORDINARY top-of-pass commit of the previous pass's chain
# forward (scheduler_pp_mixin :1109 from :705), because rank 2 had armed
# and stopped consuming. That commit PRECEDES the gate, so no gate could
# ever cover it: rank 1 had already announced, gone back around the pass,
# and blocked before it could reach the reduction its own flag promised.
#
# Two clauses close it, and neither works alone:
#   (i)  announce only once this rank owes no send -- the flag must mean
#        "my chain is flushed", not "I am armed";
#   (ii) an armed rank keeps CONSUMING non-blockingly, or its upstream can
#        never flush and so can never satisfy (i).


def test_can_fail_presence_is_withheld_while_this_rank_owes_a_send(tmp_path):
    """CLAUSE (i). Announcing while a forward is outstanding is a LIE, and
    boot 18 is what the lie costs: the peers see a full quorum, enter the
    blocking reduction, and wait on a rank that is blocked at a channel
    operation upstream of the gate."""
    owes = {"v": True}
    presence = _presence(tmp_path, rank=0)
    r = _runtime_stub(presence, owes_send_fn=lambda: owes["v"])

    assert r._await_group_presence() is None
    assert presence.observe(1) == set(), (
        "this rank announced presence while it still owed a chain send. "
        "That is the boot-18 flag: the peers enter the reduction on a "
        "quorum that includes a rank still blocked in work.wait() at its "
        "top-of-pass commit, and the group wedges with the gate OPEN"
    )
    assert r.presence_withheld_rounds == 1

    # Once the pump has drained the handle, the flag becomes true and may
    # be raised. Withholding is safe because presence is monotone: a later
    # announce is simply a later fact.
    owes["v"] = False
    r._await_group_presence()
    assert presence.observe(1) == {0}


def test_can_fail_armed_round_consumes_the_chain_before_it_announces(tmp_path):
    """CLAUSE (ii), and the ORDER. An armed rank that stops reading makes
    its upstream block at the top-of-pass commit -- upstream of the gate,
    where no gate can reach it. Consuming must happen every armed round,
    and before the announce, so the rank is never merely waiting."""
    order = []
    presence = _presence(tmp_path, rank=0)

    def _drain():
        order.append("drain")

    def _owes():
        order.append("owes-probe")
        return False

    r = _runtime_stub(presence, drain_fn=_drain, owes_send_fn=_owes)
    r._await_group_presence()
    assert order and order[0] == "drain", (
        "the armed round did not consume the chain first; an armed rank "
        "that stops consuming blocks its upstream (boot 18)"
    )
    assert presence.observe(1) == {0}

    # And it keeps consuming on EVERY later round, not only the first --
    # the upstream keeps sending for as long as this rank is at the gate.
    r._await_group_presence()
    assert order.count("drain") == 2


def test_a_failing_drain_never_breaks_the_gate(tmp_path):
    """Draining is best effort. A drain that raises must not take the flip
    down -- the gate still has a bounded, loud abandonment of its own."""

    def _boom():
        raise RuntimeError("chain drain exploded")

    presence = _presence(tmp_path, rank=0)
    r = _runtime_stub(presence, drain_fn=_boom, owes_send_fn=lambda: False)
    assert r._await_group_presence() is None
    assert presence.observe(1) == {0}


def test_can_fail_a_rank_that_never_flushes_abandons_instead_of_wedging(tmp_path):
    """Withholding presence must not become a NEW unbounded wait. A rank
    that can never flush its forward has to give up loudly and leave
    serving alone -- pre-entry abandonment is free, because nothing was
    entered and no peer is owed anything."""
    now = {"t": 0.0}
    presence = _presence(tmp_path, rank=0)
    r = _runtime_stub(
        presence,
        deadline=30.0,
        clock=lambda: now["t"],
        owes_send_fn=lambda: True,
    )
    assert r._await_group_presence() is None
    now["t"] = 31.0
    assert r._await_group_presence() is None
    assert r.presence_timeouts == 1, (
        "a rank that never flushes waited past its pre-entry deadline "
        "without abandoning; withholding presence must stay bounded"
    )
    assert r._pending is None, "the flip must be disarmed after abandonment"
    assert presence.observe(1) == set()


def test_armed_forward_path_services_and_issues_no_new_forward():
    """THE BOOT-18 FIX, downstream half, as rebuilt for #631 G. While
    armed, the top-of-pass commit must not block unconditionally: it is
    replaced by the SERVICE TURN, which consumes what the upstream posted
    and reaps this rank's send only once the downstream's counter proves
    it consumed. No new forward is issued, because an armed rank admits no
    new work."""
    s, fwd, sends, processed, commits = _fwd_harness(
        track_commits=True, armed=True
    )
    fwd([])
    assert commits == [], (
        "the armed path performed a BLOCKING commit; that is the exact "
        "call (scheduler_pp_mixin :705 -> :1109) rank 1 was found "
        "blocked in while ranks 0 and 2 sat in the reduction"
    )
    assert s.services == [True], (
        "the armed path must still take its service turn -- that is what "
        "reaps the outstanding forward, and while it is unreaped this rank "
        "owes a send and withholds presence for ever"
    )
    assert sends == [], (
        "the armed path issued a new chain forward; an armed rank admits "
        "no new work, so it has nothing to forward, and a fresh unmatched "
        "send would be a new obligation nobody can satisfy"
    )


def test_unarmed_forward_path_is_unchanged():
    """BACKWARD COMPATIBILITY. Without an armed flip the path must keep
    its exact shape: the ordinary top-of-pass commit, then the async
    forward."""
    s, fwd, sends, processed, commits = _fwd_harness(
        track_commits=True, armed=False
    )
    fwd(["req"])
    assert commits == ["top-of-pass"]
    assert sends == [(True, ["req"])]
    assert s.services == []


# -- round-scoped entry evidence ----------------------------------------------
#
# THE RULE: evidence must have the same scope as the guarantee it licenses.
# The gate's guarantee is per ROUND; epoch-scoped flags made round N's
# quorum a standing authorisation for every later round, because flags are
# never cleared. Reproduced on metal 2026-08-08 23:12:38Z with all three
# stacks (evidence-631/wedge_20260808T231450Z_INSIDE_REDUCTION): ranks 0
# and 2 inside round N+1's reduction, rank 1 blocked at its top-of-pass
# commit between rounds, gate re-opened in 0.00s on stale flags.


def test_can_fail_a_completed_round_does_not_open_the_next_one(tmp_path):
    """THE REPRODUCTION, as a unit falsifier.

    Round N assembles and the gate opens. The rank then completes that
    reduction and moves to round N+1. The SAME flags are still on disk --
    they are never cleared -- and they must NOT open round N+1, because
    they are evidence about N and say nothing about where the peers are
    now. This is the exact cycle the metal wedge closed.
    """
    presence = _presence(tmp_path, rank=0)
    peers = [_presence(tmp_path, rank=r) for r in (1, 2)]
    r = _runtime_stub(presence)

    # Round 0: the whole group announces, the gate opens.
    for p in peers:
        p.announce(1, round_=0)
    assert r._await_group_presence() is True

    # The reduction completes; the rank advances to round 1. on_round does
    # this increment right after the collective returns.
    r._entry_round = 1

    assert r._await_group_presence() is not True, (
        "round 0's quorum opened round 1. Flags are never cleared, so an "
        "epoch-scoped read makes the gate a rubber stamp after its first "
        "use -- and the peers then enter a reduction that a rank stuck at "
        "its top-of-pass commit can never join (metal wedge 23:12:38Z)"
    )
    assert presence.observe(1, round_=0) == {0, 1, 2}, "round 0 stays a fact"
    assert presence.observe(1, round_=1) == {0}, "only this rank has reached round 1"

    # And round 1 opens on ITS OWN quorum, not before.
    for p in peers:
        p.announce(1, round_=1)
    assert r._await_group_presence() is True


def test_round_stamped_flags_stay_monotone_within_their_stamp(tmp_path):
    """Round-scoping must not cost the racing-writer safety that motivated
    monotonicity. Within a stamp a flag is still set once and never
    unset; a later round simply asks a NEW question."""
    presence = _presence(tmp_path, rank=0)
    for _ in range(5):
        presence.announce(1, round_=3)
    assert presence.observe(1, round_=3) == {0}
    # Earlier and later rounds are untouched by it.
    assert presence.observe(1, round_=2) == set()
    assert presence.observe(1, round_=4) == set()
    # And a retraction still mints a new EPOCH rather than clearing.
    assert presence.all_present(2, round_=3) is False


def test_pre_entry_bound_is_per_round_not_per_arm(tmp_path):
    """A fresh round is a fresh question and gets its own budget. Carrying
    the previous round's elapsed time forward would abandon a healthy
    round for time spent waiting in an earlier one."""
    now = {"t": 0.0}
    presence = _presence(tmp_path, rank=0)
    r = _runtime_stub(presence, deadline=30.0, clock=lambda: now["t"])

    # Round 0 waits almost the whole budget without assembling.
    assert r._await_group_presence() is None
    now["t"] = 29.0
    assert r._await_group_presence() is None
    assert r.presence_timeouts == 0

    # Move to round 1: the clock restarts, so this round is not condemned
    # by round 0's wait.
    r._entry_round = 1
    now["t"] = 40.0
    assert r._await_group_presence() is None
    assert r.presence_timeouts == 0, (
        "the new round inherited the previous round's elapsed time and "
        "abandoned immediately"
    )
    # It does still expire on its OWN budget.
    now["t"] = 71.0
    assert r._await_group_presence() is None
    assert r.presence_timeouts == 1


# -- quiescent-announce + bounded spin at the hook -----------------------------
#
# THE FLAG'S MEANING: "I am at the entry, quiescent, and owe nothing" --
# not "I was at the entry once". Round-scoping stops stale evidence ACROSS
# rounds; quiescent-announce stops it WITHIN a round.


class _StopHere(Exception):
    """Marks that control reached the reduction, without running one."""


def _onround_stub(presence, ready, clock=None, deadline=60.0, pending="pp_to_tp"):
    """A runtime whose on_round can be driven without a collective."""
    from sglang.srt.managers.phase_flip_runtime import PhaseFlipRuntime

    class R:
        pass

    r = R()
    r._round = 0
    r._interval = 1
    r._pending = pending
    r._ready_fn = ready
    r._presence = presence
    r._pump_fn = None
    r._drain_fn = None
    r._owes_send_fn = None
    # #631 G: absent by default, so this stub keeps exercising the gate
    # itself rather than the service loop layered on it.
    r._service_fn = None
    r._channels_empty_fn = None
    r.presence_withheld_rounds = 0
    r.presence_withheld_channels = 0
    r.entry_channel_violations = 0
    r._last_withhold_log = None
    r._entry_round = 0
    r._presence_wait_stamp = None
    r._presence_deadline_s = deadline
    r._presence_wait_started = None
    r._gate_open_epoch = None
    r._epoch = 1
    r._armed_at = 0.0
    r._park_deadline_s = 30.0
    r._last_hold_reason = None
    r._phase = PHASE_PP
    r.presence_timeouts = 0
    r._clock = clock or (lambda: 0.0)
    r._sleep = lambda _s: None
    r._presence_poll_interval_s = 0.0
    for name in (
        "_await_group_presence",
        "_spin_for_group_presence",
        "_abandon_no_quorum",
        "_park_expired",
        "_commit_to_entering",
    ):
        setattr(r, name, getattr(PhaseFlipRuntime, name).__get__(r, R))
    return r


def test_can_fail_a_non_quiescent_rank_does_not_announce(tmp_path):
    """THE INVERSION, and the 23:39Z three-stack specimen.

    A rank that is not yet drained must go back around the pass loop to
    drain -- and must NOT publish presence on the way. Announcing early is
    what made the flag mean "I was at the entry once": the rank published,
    returned to the loop, met its top-of-pass commit, and the rank that
    had already entered the reduction was no longer consuming, so it
    blocked there for ever.
    """
    from sglang.srt.managers.phase_flip_runtime import PhaseFlipRuntime

    presence = _presence(tmp_path, rank=0)
    now = {"t": 0.0}
    r = _onround_stub(
        presence, ready=lambda: False, clock=lambda: now["t"], deadline=5.0
    )
    # The clock advances only when something SLEEPS, i.e. only if this rank
    # wrongly entered the spin. That keeps the mutation (announce without
    # quiescence) terminating on the pre-entry bound and FAILING the
    # assertion below, rather than hanging the suite -- a pin that hangs
    # tells you nothing.
    def _advance(_s):
        now["t"] += 1.0

    r._sleep = _advance
    assert PhaseFlipRuntime.on_round(r, require_armed_and_parked=True) is None
    assert presence.observe(1, round_=0) == set(), (
        "a NON-QUIESCENT rank announced presence. Its peers may now enter "
        "the reduction on a quorum that includes a rank which still has to "
        "traverse a blocking top-of-pass commit to get there (23:39Z)"
    )


def test_a_quiescent_rank_announces_and_the_gate_opens_on_live_evidence(tmp_path):
    """At true idle every rank is quiescent at once, so the gate opens on
    evidence that is current rather than remembered."""
    from sglang.srt.managers.phase_flip_runtime import PhaseFlipRuntime

    presence = _presence(tmp_path, rank=0)
    for peer in (1, 2):
        _presence(tmp_path, rank=peer).announce(1, round_=0)
    r = _onround_stub(presence, ready=lambda: True)
    reached = {"collective": False}

    def _collective(payload):
        reached["collective"] = True
        raise _StopHere()

    r._collective_min = _collective
    r._fp = 0
    r._vec = (1,)
    r._n = 1
    r.desync_checks = 0
    # Reaching the collective at all means the gate opened on this round's
    # own evidence; the reduction itself is pinned elsewhere.
    try:
        PhaseFlipRuntime.on_round(r, require_armed_and_parked=True)
    except _StopHere:
        pass
    assert reached["collective"] is True, "the gate did not open"
    assert presence.observe(1, round_=0) == {0, 1, 2}


def test_can_fail_mid_drain_skew_abandons_within_the_bound_and_keeps_serving(
    tmp_path,
):
    """CASE (b) OF THE SAFETY ARGUMENT: a peer that is not yet quiescent.

    The spinners must NOT wait for ever. Their per-round pre-entry bound
    expires, they abandon LOUDLY, and -- this is the part that matters --
    the abandonment is non-fatal and leaves the rank disarmed and serving,
    so the loop resumes forwarding and the straggler can drain. A later
    epoch then retries with everyone genuinely quiescent.
    """
    from sglang.srt.managers.phase_flip_runtime import PhaseFlipRuntime

    now = {"t": 0.0}
    presence = _presence(tmp_path, rank=0)
    # Peers never arrive: they are still draining.
    r = _onround_stub(
        presence, ready=lambda: True, clock=lambda: now["t"], deadline=30.0
    )

    def _advance(_s):
        now["t"] += 1.0

    r._sleep = _advance

    assert PhaseFlipRuntime.on_round(r, require_armed_and_parked=True) is None
    assert r.presence_timeouts == 1, (
        "the spin did not end on its per-round bound; a mid-drain peer "
        "would hold the spinners for ever instead of yielding a bounded "
        "retry"
    )
    assert r._pending is None, "abandonment must disarm, so serving resumes"
    assert now["t"] >= 30.0, "it abandoned before its bound was actually spent"


def test_spin_does_not_return_to_the_caller_between_polls(tmp_path):
    """The announce-to-entry interval must contain no return to the pass
    loop -- that interval is where the top-of-pass commit lives."""
    from sglang.srt.managers.phase_flip_runtime import PhaseFlipRuntime

    presence = _presence(tmp_path, rank=0)
    polls = {"n": 0}
    r = _onround_stub(presence, ready=lambda: True)

    def _sleep_then_let_peers_arrive(_s):
        polls["n"] += 1
        if polls["n"] == 3:
            for peer in (1, 2):
                _presence(tmp_path, rank=peer).announce(1, round_=0)

    r._sleep = _sleep_then_let_peers_arrive
    assert r._spin_for_group_presence() is True
    assert polls["n"] == 3, (
        "the gate returned to its caller instead of spinning; the rank "
        "would meet its top-of-pass commit before it could enter"
    )


# -- H: publishable withdrawal, and the tie-break ------------------------------
#
# THE INVARIANT: a withdrawal is only effective if nobody committed on it.
# Any commit converts every committed-or-withdrawing rank into an enterer,
# so there is no interleaving in which one rank enters and another stays
# out. Monotonicity survives per marker -- presence, WITHDRAWN and ENTERING
# are each write-once with a single writer.


def test_can_fail_a_stale_flag_from_a_withdrawn_rank_does_not_form_a_quorum(
    tmp_path,
):
    """CORPSE H, measured 00:07:34Z. Rank 0 abandoned epoch 0 and re-armed
    at epoch 1 while ranks 1 and 2 formed a full epoch-0 quorum on rank 0's
    STALE presence flag and entered a reduction it would never join."""
    ranks = [_presence(tmp_path, rank=r) for r in range(3)]
    for r in ranks:
        r.announce(1, round_=0)
    assert ranks[1].quorum(1, round_=0) is True

    # Rank 0 hits its pre-entry bound and leaves. Its presence marker is
    # NOT cleared -- it never can be -- so only a published withdrawal can
    # stop the others counting it.
    ranks[0].declare_withdrawn(1, round_=0)
    assert ranks[1].quorum(1, round_=0) is False, (
        "a rank that has withdrawn still counted toward the quorum; its "
        "peers enter a reduction it will never join and the group dies on "
        "the collective timeout (00:09:39Z)"
    )
    assert ranks[1].all_present(1, round_=0) is True, "presence stays monotone"
    assert ranks[1].withdrawn(1, round_=0) == {0}


def test_tie_break_forces_the_withdrawer_in_when_a_peer_committed(tmp_path):
    """THE RACE, and the rule that closes it. A peer that has published
    ENTERING committed on this rank's presence and is on its way into a
    blocking reduction. This rank may no longer leave."""
    ranks = [_presence(tmp_path, rank=r) for r in range(3)]
    for r in ranks:
        r.announce(1, round_=0)

    ranks[1].declare_entering(1, round_=0)
    assert ranks[0].may_withdraw(1, round_=0) is False, (
        "a rank was allowed to withdraw after a peer had committed to "
        "entering on its presence -- that strands the peer"
    )

    # It follows through instead. Even having already published WITHDRAWN
    # (the write/re-check race), ENTERING wins and it stops counting as
    # withdrawn -- so the quorum re-forms rather than deadlocking.
    ranks[0].declare_withdrawn(1, round_=0)
    ranks[0].declare_entering(1, round_=0)
    assert ranks[0].withdrawn(1, round_=0) == set(), (
        "a rank carrying BOTH markers was treated as withdrawn; the "
        "invariant is that any commit converts it into an enterer"
    )
    assert ranks[1].quorum(1, round_=0) is True


def test_withdrawal_is_permitted_while_nobody_has_committed(tmp_path):
    ranks = [_presence(tmp_path, rank=r) for r in range(3)]
    ranks[0].announce(1, round_=0)
    assert ranks[0].may_withdraw(1, round_=0) is True
    # This rank's OWN entering marker must not block its own withdrawal.
    ranks[0].declare_entering(1, round_=0)
    assert ranks[0].may_withdraw(1, round_=0) is True


def test_withdrawal_is_scoped_to_its_own_round(tmp_path):
    """A withdrawal from round 0 must not condemn round 1 -- same scoping
    discipline as presence."""
    ranks = [_presence(tmp_path, rank=r) for r in range(3)]
    for r in ranks:
        r.announce(1, round_=0)
        r.announce(1, round_=1)
    ranks[0].declare_withdrawn(1, round_=0)
    assert ranks[1].quorum(1, round_=0) is False
    assert ranks[1].quorum(1, round_=1) is True


def test_can_fail_entering_rank_waits_out_a_withdrawal_then_enters(tmp_path):
    """The entering side of the tie-break: seeing a WITHDRAWN at the final
    re-check, the rank waits -- and that wait TERMINATES by construction,
    because its own ENTERING marker forces the withdrawer to follow
    through."""
    from sglang.srt.managers.phase_flip_runtime import PhaseFlipRuntime

    presence = _presence(tmp_path, rank=0)
    peers = [_presence(tmp_path, rank=r) for r in (1, 2)]
    for p in peers:
        p.announce(1, round_=0)
    # Rank 2 withdrew just before rank 0 committed.
    peers[1].declare_withdrawn(1, round_=0)

    r = _runtime_stub(presence)
    polls = {"n": 0}

    def _sleep(_s):
        polls["n"] += 1
        if polls["n"] == 2:
            # The tie-break fires: rank 2 sees rank 0 ENTERING and follows.
            peers[1].declare_entering(1, round_=0)

    r._sleep = _sleep
    r._commit_to_entering(1, 0)
    assert polls["n"] == 2, "the enterer did not wait for the withdrawal to resolve"
    assert presence.withdrawn(1, round_=0) == set()


# -- #631 G: the armed service loop, and flip-commit hygiene -------------------
#
# THE DEFECT (metal, 2026-08-09 00:06-00:09Z): a spinning rank stopped
# issuing its per-pass chain forward, and its downstream reached the hook
# ONLY by returning from the blocking recv that forward satisfied. The
# first rank to quiesce therefore blocked every rank behind it -- bounded
# by the pre-entry deadline, but NOT convergent, because the same rank
# drained first every epoch and the starvation reproduced identically.
#
# THE FIX: while armed, a rank SERVICES its channels each turn (consume
# what the upstream's counter accounts for, reap what the downstream's
# counter proves consumed) and reaches the hook by its own poll, so no
# rank's readiness depends on a peer's traffic.


def test_the_gate_takes_a_service_turn_on_every_iteration(tmp_path):
    """The service turn is what makes the spin a service loop rather than
    a starvation source. If it stops running, corpse G is back."""
    presence = _presence(tmp_path, rank=0)
    turns = []
    r = _runtime_stub(presence, service_fn=lambda: turns.append(1))
    r._await_group_presence()
    r._await_group_presence()
    assert turns == [1, 1], (
        "the armed gate skipped its service turn; a rank that stops "
        "consuming blocks its upstream at a point that PRECEDES the gate, "
        "where no gate can reach it"
    )


def test_a_failing_service_turn_never_breaks_the_gate(tmp_path):
    """Servicing is best effort; the gate must survive it failing."""
    presence = _presence(tmp_path, rank=0)

    def _boom():
        raise RuntimeError("wire fell over")

    r = _runtime_stub(presence, service_fn=_boom)
    assert r._await_group_presence() is None
    assert presence.observe(1, round_=0) == {0}, (
        "a failed service turn must not stop this rank announcing"
    )


def test_presence_is_withheld_while_a_channel_is_not_empty(tmp_path):
    """FLIP-COMMIT HYGIENE, the withholding half.

    Quiescent AND fully serviced implies every channel is empty. A rank
    that is not there yet withholds presence exactly as a rank that still
    owes a send does -- WITHHOLDING, not abandoning, because a message in
    flight is normally reaped by the next service turn and abandoning on
    it would be non-convergent.
    """
    presence = _presence(tmp_path, rank=0)
    state = {"why": "send_req_work is not reaped"}
    r = _runtime_stub(presence, channels_empty_fn=lambda: state["why"])

    assert r._await_group_presence() is None
    assert presence.observe(1, round_=0) == set(), (
        "a rank announced while a channel still held a message; the flag "
        "would then mean 'I was at the entry once', not 'I owe nothing'"
    )
    assert r.presence_withheld_channels == 1

    state["why"] = None
    assert r._await_group_presence() is None
    assert presence.observe(1, round_=0) == {0}, (
        "presence was not announced once the channels came up empty"
    )


def test_can_fail_a_non_empty_channel_at_entry_abandons_before_entering(tmp_path):
    """CAN-FAIL FOR THE ENTRY ASSERT, and the nastiest silent failure this
    change can introduce.

    The withholding check above proves nothing about the INSTANT a quorum
    forms: a peer's message can land in between. A half-consumed two-step
    point_to_point_pyobj message -- or an unreaped isend -- crossing the
    re-formation would misframe the post-flip stream silently, long after
    the flip is forgotten.

    The probe here reports CLEAN on the withholding check and DIRTY on the
    entry re-check, which is exactly that interleaving. The gate must
    refuse to enter and abandon PRE-ENTRY, where nothing is owed to
    anyone.
    """
    presence = _presence(tmp_path, rank=0)
    for peer in (1, 2):
        _presence(tmp_path, rank=peer).announce(1, round_=0)

    answers = [None, "request chain has 1 unconsumed message(s) from rank 2"]
    r = _runtime_stub(presence, channels_empty_fn=lambda: answers.pop(0))

    assert r._await_group_presence() is None, (
        "the gate ENTERED the reduction with a live channel; a message in "
        "flight across the re-formation misframes the post-flip stream"
    )
    assert r.entry_channel_violations == 1
    assert r._pending is None, "a pre-entry abandonment must disarm"
    assert presence.entering(1, round_=0) == set(), (
        "the rank published ENTERING and then did not enter -- that is the "
        "shape that strands a peer inside a gloo collective"
    )


def test_a_rank_that_can_never_empty_its_channels_abandons_instead_of_wedging(
    tmp_path,
):
    """Withholding MUST fall through to the pre-entry deadline.

    Returning early from the withhold would turn 'wait until clean' into a
    NEW unbounded wait -- the same shape as the wedge it exists to remove.
    Same lesson as the owes-a-send clause, which learned it the hard way.
    """
    presence = _presence(tmp_path, rank=0)
    now = {"t": 0.0}
    r = _runtime_stub(
        presence,
        deadline=30.0,
        clock=lambda: now["t"],
        channels_empty_fn=lambda: "send_output_work is not reaped",
    )
    for _ in range(4):
        assert r._await_group_presence() is None
        now["t"] += 12.0
    assert r._pending is None, (
        "a rank whose channels never empty waited past its pre-entry "
        "deadline without abandoning"
    )
    assert r.presence_timeouts == 1
