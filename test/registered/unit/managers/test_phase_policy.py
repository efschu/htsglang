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


def _fwd_harness(is_last_rank=False, track_commits=False):
    from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin

    sends = []
    processed = []
    commits = []

    class _Grp:
        pass

    _Grp.is_last_rank = is_last_rank

    class S:
        pp_group = _Grp()
        send_req_work = None

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


def test_arm_carrying_send_is_committed_before_the_join():
    """(ii) TARGETED COMMIT. Specimen: variant A, 2026-08-08.

    An async forward is progressed by _pp_commit_comm_work at the TOP of
    the NEXT pass -- which never comes if this rank arms and blocks in
    the reduction first. py-spy: rank 0 in bounded_collective, ranks 1-2
    in _pull_raw_reqs, 0 % GPU. So the arm-carrying handle must be
    committed in THIS pass, before the flip hook can join.
    """
    from sglang.srt.managers.io_struct import PhaseFlipReqInput

    arm = PhaseFlipReqInput(direction=PP_TO_TP, source="policy", internal=True)
    s, fwd, sends, processed, commits = _fwd_harness(track_commits=True)
    fwd([arm])
    assert sends[0][0] is True, (
        "the forward must stay ASYNC -- a blanket synchronous send is "
        "variant B and deadlocks against the hidden-states exchange"
    )
    assert commits, "the arm-carrying send was never committed"
    assert commits[-1] == "post-send", (
        "the arm-carrying handle was not committed in this pass; the peer "
        "cannot have the arm before this rank joins"
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


def test_manual_flip_takes_the_same_committed_path():
    """(d) The manual RPC gets the invariant too.

    Strictly safer, and it removes manual's latent at-idle deadlock:
    manual has only ever run UNDER TRAFFIC, where the loop keeps cycling
    and the commit happens by accident rather than by construction.
    """
    from sglang.srt.managers.io_struct import PhaseFlipReqInput

    rpc = PhaseFlipReqInput(direction=PP_TO_TP)  # internal=False
    s, fwd, sends, processed, commits = _fwd_harness(track_commits=True)
    fwd([rpc])
    assert rpc in processed[0], "the manual arm must also be same-pass"
    assert commits[-1] == "post-send", (
        "the manual flip skipped the targeted commit"
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
