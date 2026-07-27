"""Unit tests for the #236 spill budget (kv-session-offload).

CPU-only; no server, no GPU. Run:
  python -m pytest test/registered/unit/test_kv_spill_budget_unit.py -q
"""

import types

import pytest

from sglang.srt.managers.kv_session_offload import (
    BUDGET_ADMISSION_ORDER,
    GDN_STATE_MIN_ITEMSIZE,
    KVSessionOffloadManager,
    RestoreHysteresis,
    SpillBudgetConfig,
    SpillBudgetCounters,
    SpillCooldownRegistry,
    SpillRateBucket,
    SpillSlot,
    WaveBackController,
    budget_admission_violation,
    budget_episode_violation,
    gdn_token_equivalent,
    select_spill_victim,
)


# ---------------------------------------------------------------------------
# Config: default open, each regler individually armable
# ---------------------------------------------------------------------------


def _sa(**over):
    ns = types.SimpleNamespace()
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


def test_default_config_is_open():
    cfg = SpillBudgetConfig.from_server_args(_sa())
    assert not cfg.armed
    assert not cfg.needs_clock
    assert not cfg.has_volume
    # explicit: an all-default config binds NOTHING at admission
    assert (
        budget_admission_violation(
            cfg,
            n_open_slots=999,
            spill_tokens=10**9,
            phase="decode",
            session_tokens_after=10**9,
            prefill_tokens_after=10**9,
            decode_tokens_after=10**9,
            total_tokens_after=10**9,
            rate_ready=False,
        )
        is None
    )
    assert (
        budget_episode_violation(
            cfg, session_tokens=10**9, episode_elapsed_s=1e9
        )
        is None
    )


def test_each_regler_arms_individually():
    for name in (
        "kv_session_offload_budget_total_tokens",
        "kv_session_offload_budget_session_tokens",
        "kv_session_offload_budget_prefill_tokens",
        "kv_session_offload_budget_decode_tokens",
        "kv_session_offload_budget_max_sessions",
        "kv_session_offload_spill_progress_lock_tokens",
        "kv_session_offload_spill_hysteresis_steps",
    ):
        cfg = SpillBudgetConfig.from_server_args(_sa(**{name: 7}))
        assert cfg.armed, name
    for name in (
        "kv_session_offload_budget_rate_tokens_per_s",
        "kv_session_offload_budget_episode_seconds",
        "kv_session_offload_spill_cooldown_seconds",
    ):
        cfg = SpillBudgetConfig.from_server_args(_sa(**{name: 1.5}))
        assert cfg.armed and cfg.needs_clock, name
    # grace alone does not arm (it only qualifies a demotion)
    cfg = SpillBudgetConfig.from_server_args(
        _sa(kv_session_offload_budget_demote_grace_iters=9)
    )
    assert not cfg.armed


# ---------------------------------------------------------------------------
# Admission: first violated regler wins, in the fixed documented order
# ---------------------------------------------------------------------------


def _all_binding_cfg():
    return SpillBudgetConfig(
        total_tokens=10,
        session_tokens=10,
        prefill_tokens=10,
        decode_tokens=10,
        rate_tokens_per_s=1.0,
        max_sessions=1,
    )


def test_admission_first_binding_regler_wins_in_fixed_order():
    cfg = _all_binding_cfg()
    kw = dict(
        n_open_slots=5,
        spill_tokens=100,
        phase="decode",
        session_tokens_after=100,
        prefill_tokens_after=100,
        decode_tokens_after=100,
        total_tokens_after=100,
        rate_ready=False,
    )
    # everything violated -> the order names the event
    assert budget_admission_violation(cfg, **kw) == "max-sessions"
    kw["n_open_slots"] = 0
    assert budget_admission_violation(cfg, **kw) == "session-tokens"
    kw["session_tokens_after"] = 5
    assert budget_admission_violation(cfg, **kw) == "decode-tokens"
    kw["decode_tokens_after"] = 5
    assert budget_admission_violation(cfg, **kw) == "total-tokens"
    kw["total_tokens_after"] = 5
    assert budget_admission_violation(cfg, **kw) == "rate"
    kw["rate_ready"] = True
    assert budget_admission_violation(cfg, **kw) is None
    assert BUDGET_ADMISSION_ORDER.index("max-sessions") == 0
    assert BUDGET_ADMISSION_ORDER.index("rate") == len(BUDGET_ADMISSION_ORDER) - 1


def test_admission_phase_budgets_are_separate():
    cfg = SpillBudgetConfig(prefill_tokens=50, decode_tokens=200)
    kw = dict(
        n_open_slots=0,
        spill_tokens=100,
        session_tokens_after=100,
        prefill_tokens_after=100,  # over the prefill budget
        decode_tokens_after=100,  # under the decode budget
        total_tokens_after=100,
        rate_ready=True,
    )
    # a DECODE admission is untouched by the exhausted prefill budget
    assert budget_admission_violation(cfg, phase="decode", **kw) is None
    # a PREFILL admission binds on it
    assert budget_admission_violation(cfg, phase="prefill", **kw) == (
        "prefill-tokens"
    )


def test_episode_violation_session_volume_and_window():
    cfg = SpillBudgetConfig(session_tokens=100, episode_seconds=10.0)
    ok = dict(session_tokens=100, episode_elapsed_s=10.0)  # at the cap: fine
    assert budget_episode_violation(cfg, **ok) is None
    assert (
        budget_episode_violation(cfg, session_tokens=101, episode_elapsed_s=0.0)
        == "session-tokens"
    )
    assert (
        budget_episode_violation(cfg, session_tokens=1, episode_elapsed_s=10.1)
        == "episode-window"
    )
    # session volume is checked before the window (fixed order)
    assert (
        budget_episode_violation(
            cfg, session_tokens=101, episode_elapsed_s=10.1
        )
        == "session-tokens"
    )


# ---------------------------------------------------------------------------
# Rate bucket: debt model (throttle, never starve), deterministic
# ---------------------------------------------------------------------------


def test_rate_bucket_debt_model_never_starves():
    b = SpillRateBucket(1000.0, burst_seconds=1.0)  # cap 1000
    b.advance(0.0)
    assert b.ready()
    # one consumption LARGER than the burst: allowed (debt), then throttled
    b.consume(5000)
    assert not b.ready()
    # refill pays the debt off -- ~4 seconds at 1000 tok/s
    b.advance(3.9)
    assert not b.ready()
    b.advance(4.1)
    assert b.ready()  # debt cleared: the big consumer was throttled, not starved


def test_rate_bucket_is_deterministic_for_identical_inputs():
    # rank-uniformity proxy: two ranks feeding the identical (uniform clock,
    # replicated token) sequence hold bit-identical levels
    ops = [("a", 0.0), ("c", 1234), ("a", 0.5), ("c", 10), ("a", 2.25)]
    levels = []
    for _ in range(2):
        b = SpillRateBucket(777.0)
        for op, v in ops:
            b.advance(v) if op == "a" else b.consume(int(v))
        levels.append(b.level)
    assert levels[0] == levels[1]


def test_rate_bucket_refill_caps_at_burst():
    b = SpillRateBucket(100.0, burst_seconds=1.0)
    b.advance(0.0)
    b.advance(1000.0)  # a long idle must not bank unbounded burst
    assert b.level == b.cap == 100.0


# ---------------------------------------------------------------------------
# Cooldown: progress lock (a) primary, time cap (c) on top, lazy expiry
# ---------------------------------------------------------------------------


def test_progress_lock_blocks_until_n_tokens_produced():
    reg = SpillCooldownRegistry(progress_lock_tokens=50, cooldown_seconds=0.0)
    reg.note_restore("r1", output_len=100, now=0.0)
    assert reg.blocked("r1", output_len_now=100, now=99.0)  # no progress
    assert reg.blocked("r1", output_len_now=149, now=99.0)  # 49 < 50
    assert not reg.blocked("r1", output_len_now=150, now=99.0)  # progressed
    # entry expired -> free forever after
    assert not reg.blocked("r1", output_len_now=150, now=99.0)


def test_time_cap_is_a_coarse_bound_on_top():
    reg = SpillCooldownRegistry(progress_lock_tokens=10, cooldown_seconds=30.0)
    reg.note_restore("r1", output_len=0, now=100.0)
    # progress made, but the time cap still holds
    assert reg.blocked("r1", output_len_now=50, now=120.0)
    # both caps passed -> unblocked and expired
    assert not reg.blocked("r1", output_len_now=50, now=131.0)
    assert "r1" not in reg._entries


def test_unknown_session_is_never_blocked():
    reg = SpillCooldownRegistry(50, 30.0)
    assert not reg.blocked("never-restored", output_len_now=0, now=0.0)


def test_in_window_probe_is_non_mutating():
    reg = SpillCooldownRegistry(progress_lock_tokens=50, cooldown_seconds=0.0)
    reg.note_restore("r1", output_len=0, now=0.0)
    assert reg.in_window("r1", output_len_now=10, now=0.0)
    # expired by progress: the probe reports False but keeps the entry --
    # only blocked() performs the lazy expiry
    assert not reg.in_window("r1", output_len_now=50, now=0.0)
    assert "r1" in reg._entries
    assert not reg.blocked("r1", output_len_now=50, now=0.0)
    assert "r1" not in reg._entries


def test_pendulum_counters_are_split_actual_vs_blocked():
    c = SpillBudgetCounters()
    d = c.as_dict()
    # the guarantee counter (actual rounds inside the lock) and the
    # visibility counter (prevented attempts) are distinct fields
    assert d["pendulum_events"] == 0
    assert d["pendulum_blocked"] == 0


# ---------------------------------------------------------------------------
# Victim selection under the cooldown: protections are untouchable
# ---------------------------------------------------------------------------


class _FakeReq:
    def __init__(self, seq, fast=False):
        self.kv_arrival_seq = seq
        self.is_fast_lane = fast


def test_blocked_none_is_byte_identical():
    reqs = [_FakeReq(s) for s in (0, 1, 2, 3)]
    assert select_spill_victim(reqs) == select_spill_victim(reqs, blocked=None)
    assert select_spill_victim(reqs, blocked=set()) == select_spill_victim(reqs)


def test_blocked_youngest_shifts_to_next_youngest():
    reqs = [_FakeReq(s) for s in (0, 1, 2, 3)]
    assert select_spill_victim(reqs) == 3
    assert select_spill_victim(reqs, blocked={3}) == 2
    assert select_spill_victim(reqs, blocked={3, 2}) == 1
    # everything but the tabu oldest blocked -> no victim, never the oldest
    assert select_spill_victim(reqs, blocked={1, 2, 3}) is None


def test_blocked_oldest_does_not_shift_the_tabu():
    # HARD LIMIT: the oldest-tabu is resolved BEFORE the cooldown exclusion.
    # Blocking the oldest must not promote the second-oldest into the tabu
    # slot (which would wrongly protect it and skew selection).
    reqs = [_FakeReq(s) for s in (0, 1, 2)]
    # oldest (idx 0) blocked: selection among {1, 2} unchanged -> youngest 2
    assert select_spill_victim(reqs, blocked={0}) == 2


def test_sole_visible_generating_session_is_never_a_budget_victim():
    # HARD LIMIT: a sole running session never self-spills -- with or without
    # a cooldown set, under any blocked content. (The budget can only REMOVE
    # candidates, never add one.)
    solo = [_FakeReq(7)]
    assert select_spill_victim(solo) is None
    assert select_spill_victim(solo, blocked=set()) is None
    assert select_spill_victim(solo, blocked={0}) is None


# ---------------------------------------------------------------------------
# GDN accounting: counted, never quantized
# ---------------------------------------------------------------------------


def test_gdn_token_equivalent_math():
    # ~75 MB bf16 state on the 27B at ~12 KiB/token KV -> ~6400 tokens
    eq = gdn_token_equivalent(75 * (1 << 20), 12 * 1024)
    assert eq == -(-75 * (1 << 20) // (12 * 1024))
    assert gdn_token_equivalent(0, 100) == 0
    assert gdn_token_equivalent(100, 0) == 0
    # ceil, never floor-to-zero for a real state
    assert gdn_token_equivalent(1, 10**9) == 1


def test_gdn_state_quantization_is_rejected():
    # HARD LIMIT: GDN states are NEVER quantized. The budget layer refuses to
    # account a sub-bf16 state instead of normalizing it away.
    assert GDN_STATE_MIN_ITEMSIZE == 2
    with pytest.raises(ValueError, match="never quantized"):
        gdn_token_equivalent(1000, 100, state_itemsize=1)
    # bf16 and wider pass
    assert gdn_token_equivalent(1000, 100, state_itemsize=2) == 10
    assert gdn_token_equivalent(1000, 100, state_itemsize=4) == 10


# ---------------------------------------------------------------------------
# HARD LIMIT #217: the budget must not re-introduce free-list-only gating.
# Restore-readiness (which counts the radix-evictable memory) stays the
# unchanged _maybe_restore_flow / _restore_memory_ok; no budget code path may
# consult the allocator or the tree cache at all.
# ---------------------------------------------------------------------------


def test_budget_layer_never_touches_allocator_or_tree_state():
    import inspect

    from sglang.srt.managers import kv_session_offload as kvso

    sources = [
        inspect.getsource(kvso.SpillBudgetConfig),
        inspect.getsource(kvso.budget_admission_violation),
        inspect.getsource(kvso.budget_episode_violation),
        inspect.getsource(kvso.SpillRateBucket),
        inspect.getsource(kvso.SpillCooldownRegistry),
    ]
    for name in (
        "_budget_evaluate_episodes",
        "_budget_demote",
        "_budget_resident_volumes",
        "_budget_admission_check",
        "_budget_blocked_victims",
        "_budget_begin_iteration",
    ):
        # getattr without default: a renamed method breaks the guard loudly
        sources.append(inspect.getsource(getattr(KVSessionOffloadManager, name)))
    forbidden = ("available_size", "free_pages", "evict_from_tree_cache",
                 "_tree_evictable_size", "_restore_memory_ok")
    for src in sources:
        for tok in forbidden:
            assert tok not in src, (
                f"budget code references {tok}: budget policy must never gate "
                "on rank-local memory state (the #217 free-list-only bug must "
                "not return through the budget)"
            )
    # ... and the restore gate itself still counts the evictable radix memory
    # and is budget-free (demoted sessions drain through the SAME gate).
    restore_src = inspect.getsource(KVSessionOffloadManager._maybe_restore_flow)
    assert "_tree_evictable_size" in restore_src
    assert "_budget" not in restore_src


# ---------------------------------------------------------------------------
# Manager-level: demotion flow, tick exclusion, grace, admission decline
# ---------------------------------------------------------------------------


def _budget_manager(cfg: SpillBudgetConfig):
    mgr = KVSessionOffloadManager.__new__(KVSessionOffloadManager)
    mgr.spills = {}
    mgr._free_regions = [0, 1, 2]
    mgr.scheduler = types.SimpleNamespace(
        spec_algorithm=None, running_batch=None, waiting_queue=()
    )
    mgr._iter_ct = 10
    mgr._budget = cfg
    mgr._budget_armed = cfg.armed
    mgr._budget_counters = SpillBudgetCounters()
    mgr._budget_bucket = (
        SpillRateBucket(cfg.rate_tokens_per_s) if cfg.rate_tokens_per_s > 0 else None
    )
    mgr._budget_cooldown = (
        SpillCooldownRegistry(cfg.progress_lock_tokens, cfg.cooldown_seconds)
        if (cfg.progress_lock_tokens > 0 or cfg.cooldown_seconds > 0)
        else None
    )
    mgr._budget_now = 0.0
    mgr._budget_gdn_eq = 0
    mgr._fast_lane_enabled = False
    mgr.tick_controller = None
    mgr._log = lambda *a, **k: None
    return mgr


def _spilled_req(rpi, n_in=100, n_out=21, boundary=20, seq=1):
    return types.SimpleNamespace(
        req_pool_idx=rpi,
        rid=f"r{rpi}",
        origin_input_ids=list(range(n_in)),
        output_ids=list(range(n_out)),
        kv_spill_boundary=boundary,
        kv_arrival_seq=seq,
        sampling_params=types.SimpleNamespace(max_new_tokens=4096),
        finished=lambda: False,
    )


def _slot_for(mgr, req, region, phase="decode", initial_tail=None, start=0.0):
    slot = SpillSlot(
        req=req,
        region=region,
        spill_iter=0,
        wave=WaveBackController(8, 1),
        hysteresis=RestoreHysteresis(1),
    )
    slot.budget_phase = phase
    L = len(req.origin_input_ids) + len(req.output_ids) - 1
    tail = L - req.kv_spill_boundary
    slot.budget_initial_tail = tail if initial_tail is None else initial_tail
    slot.budget_episode_start = start
    mgr.spills[req.req_pool_idx] = slot
    return slot


def test_resident_volumes_split_phases_and_charge_gdn():
    mgr = _budget_manager(SpillBudgetConfig(total_tokens=10**6))
    mgr._budget_gdn_eq = 10
    # decode episode: L = 100 + 21 - 1 = 120, boundary 20 -> tail 100
    _slot_for(mgr, _spilled_req(1), region=0, phase="decode")
    # born-spilled: initial (prompt) tail 80, grown to 100 -> 20 decode growth
    _slot_for(mgr, _spilled_req(2, seq=2), region=1, phase="prefill",
              initial_tail=80)
    total, pre, dec, per_slot = mgr._budget_resident_volumes()
    assert per_slot[1] == 110 and per_slot[2] == 110  # tail + gdn each
    assert pre == 80 + 10  # prompt share + this session's GDN charge
    assert dec == 110 + 20  # decode session (incl. gdn) + born growth
    assert total == pre + dec == 220


def test_episode_window_exhaustion_demotes_with_cap():
    cfg = SpillBudgetConfig(episode_seconds=10.0)
    mgr = _budget_manager(cfg)
    req = _spilled_req(1)
    slot = _slot_for(mgr, req, region=0, start=0.0)
    mgr._budget_now = 9.0
    mgr._budget_evaluate_episodes()
    assert not slot.budget_demoted  # inside the window
    mgr._budget_now = 10.5
    mgr._budget_evaluate_episodes()
    assert slot.budget_demoted
    assert not slot.budget_tick_release  # drain grace running (no spec guard)
    # Herabstufung: generation capped at the CURRENT output -> the stock
    # finish delivers what exists and donates the prefix after the drain.
    assert req.sampling_params.max_new_tokens == len(req.output_ids)
    assert mgr._budget_counters.episodes_demoted == 1
    assert mgr._budget_counters.exhaustions == {"episode-window": 1}
    # idempotent: a second pass does not double-demote
    mgr._budget_evaluate_episodes()
    assert mgr._budget_counters.episodes_demoted == 1


def test_total_volume_exhaustion_demotes_the_youngest_live():
    cfg = SpillBudgetConfig(total_tokens=150)
    mgr = _budget_manager(cfg)
    old = _slot_for(mgr, _spilled_req(1, seq=1), region=0)  # tail 100
    young = _slot_for(mgr, _spilled_req(2, seq=9), region=1)  # tail 100
    mgr._budget_evaluate_episodes()
    assert young.budget_demoted and not old.budget_demoted
    assert mgr._budget_counters.exhaustions == {"total-tokens": 1}
    # one demotion per class per iteration; the remainder re-evaluates later
    assert mgr._budget_counters.episodes_demoted == 1


def test_session_volume_exhaustion_demotes_that_session():
    cfg = SpillBudgetConfig(session_tokens=90)
    mgr = _budget_manager(cfg)
    small = _slot_for(mgr, _spilled_req(1, n_out=5, boundary=60, seq=1), region=0)
    big = _slot_for(mgr, _spilled_req(2, seq=2), region=1)  # tail 100 > 90
    mgr._budget_evaluate_episodes()
    assert big.budget_demoted and not small.budget_demoted


def test_demoted_session_is_excluded_from_ticks_until_grace_expiry():
    cfg = SpillBudgetConfig(episode_seconds=1.0, demote_grace_iters=100)
    mgr = _budget_manager(cfg)
    req = _spilled_req(1)
    slot = _slot_for(mgr, req, region=0, start=0.0)
    mgr._budget_now = 2.0
    mgr._budget_evaluate_episodes()
    assert slot.budget_demoted
    # in grace: not tickable (liveness over, waiting for the drain handover)
    assert mgr._pick_tick_slot(None) is None
    # grace expiry: the finishing host tick is released
    mgr._iter_ct = slot.budget_demote_iter + cfg.demote_grace_iters + 1
    mgr._budget_evaluate_episodes()
    assert slot.budget_tick_release
    assert mgr._pick_tick_slot(None) is slot
    assert mgr._budget_counters.episodes_demoted == 1


def test_spec_host_finish_guard_skips_the_drain_grace():
    # spec active without KVSO_RESUME: no restore path exists for the drain,
    # so the demotion releases the finishing tick immediately.
    cfg = SpillBudgetConfig(episode_seconds=1.0)
    mgr = _budget_manager(cfg)
    mgr.scheduler.spec_algorithm = types.SimpleNamespace(is_none=lambda: False)
    slot = _slot_for(mgr, _spilled_req(1), region=0, start=0.0)
    mgr._budget_now = 2.0
    mgr._budget_evaluate_episodes()
    assert slot.budget_demoted and slot.budget_tick_release


def test_try_spill_declines_at_the_session_count_budget():
    cfg = SpillBudgetConfig(max_sessions=1)
    mgr = _budget_manager(cfg)
    _slot_for(mgr, _spilled_req(1), region=0)
    batch = types.SimpleNamespace(reqs=[_spilled_req(9)])
    assert mgr.try_spill(batch, fast_pressure=False) is False
    assert mgr._budget_counters.admission_declines == 1
    assert mgr._budget_counters.exhaustions == {"max-sessions": 1}


def test_admission_check_binds_volume_and_charges_nothing_on_decline():
    cfg = SpillBudgetConfig(total_tokens=150)
    mgr = _budget_manager(cfg)
    _slot_for(mgr, _spilled_req(1), region=0)  # 100 resident
    assert mgr._budget_admission_check(100) == "total-tokens"  # 200 > 150
    assert mgr._budget_admission_check(40) is None  # 140 <= 150
    assert mgr._budget_counters.admission_declines == 1


def test_note_spill_opens_episode_and_charges_rate_and_counters():
    cfg = SpillBudgetConfig(rate_tokens_per_s=1000.0)
    mgr = _budget_manager(cfg)
    mgr._budget_bucket.advance(0.0)
    req = _spilled_req(1)
    slot = _slot_for(mgr, req, region=0)
    mgr._budget_now = 5.0
    mgr._budget_note_spill(slot, "decode", 400)
    assert slot.budget_phase == "decode"
    assert slot.budget_initial_tail == 400
    assert slot.budget_episode_start == 5.0
    assert mgr._budget_counters.episodes_started == 1
    assert mgr._budget_counters.spilled_tokens_decode == 400
    assert mgr._budget_bucket.level == 1000.0 - 400
    mgr._budget_note_spill(slot, "prefill", 100)
    assert mgr._budget_counters.spilled_tokens_prefill == 100


def test_prefill_gate_closes_on_budget_and_is_inert_unarmed():
    cfg = SpillBudgetConfig(prefill_tokens=90)
    mgr = _budget_manager(cfg)
    mgr.prefill_spill = True
    assert mgr.prefill_spill_free_regions() == 3  # nothing resident yet
    _slot_for(mgr, _spilled_req(1), region=0, phase="prefill",
              initial_tail=100)
    assert mgr.prefill_spill_free_regions() == 0  # 100 >= 90 -> gate closed
    assert mgr._budget_counters.prefill_gate_closures == 1
    # unarmed budget: the gate never engages
    mgr2 = _budget_manager(SpillBudgetConfig())
    mgr2.prefill_spill = True
    _slot_for(mgr2, _spilled_req(1), region=0, phase="prefill",
              initial_tail=10**9)
    assert mgr2.prefill_spill_free_regions() == 3


def test_blocked_victims_reads_the_cooldown_registry():
    cfg = SpillBudgetConfig(progress_lock_tokens=50)
    mgr = _budget_manager(cfg)
    r0, r1 = _spilled_req(1, seq=1), _spilled_req(2, seq=2)
    mgr._budget_cooldown.note_restore(r1.rid, len(r1.output_ids), now=0.0)
    blocked = mgr._budget_blocked_victims([r0, r1])
    assert blocked == {1}
    # progress unblocks
    r1.output_ids = list(range(21 + 50))
    assert mgr._budget_blocked_victims([r0, r1]) is None


def test_budget_stats_reports_counters_and_the_four_states():
    cfg = SpillBudgetConfig(total_tokens=10**6)
    mgr = _budget_manager(cfg)
    live = _slot_for(mgr, _spilled_req(1, seq=1), region=0)
    demoted = _slot_for(mgr, _spilled_req(2, seq=2), region=1)
    demoted.budget_demoted = True
    mgr.scheduler.running_batch = types.SimpleNamespace(reqs=[object()] * 3)
    stats = mgr.budget_stats()
    assert stats["state_device_resident"] == 3
    assert stats["state_spilled_live"] == 1
    assert stats["state_demoted_pending"] == 1
    assert stats["state_retracted_by_budget_decline"] == 0
    for key in (
        "spilled_tokens_prefill",
        "spilled_tokens_decode",
        "episodes_started",
        "episodes_restored",
        "episodes_finished_on_host",
        "episodes_demoted",
        "demotions_drained",
        "demotions_host_finished",
        "pendulum_events",
        "rate_throttled_ticks",
        "exhaustions",
    ):
        assert key in stats
    assert live is not None


def test_counters_as_dict_roundtrip():
    c = SpillBudgetCounters()
    c.note_exhaustion("rate")
    c.note_exhaustion("rate")
    c.note_exhaustion("total-tokens")
    d = c.as_dict()
    assert d["exhaustions"] == {"rate": 2, "total-tokens": 1}
    # as_dict is a snapshot, not a live view
    c.note_exhaustion("rate")
    assert d["exhaustions"]["rate"] == 2


# ---------------------------------------------------------------------------
# (b) pressure hysteresis: the spill-side mirror of the restore hysteresis
# ---------------------------------------------------------------------------


def test_pressure_hysteresis_mirrors_restore_hysteresis_semantics():
    gate = RestoreHysteresis(3)
    # shortfall must HOLD for N consecutive evaluations
    assert not gate.update(True)
    assert not gate.update(True)
    assert gate.update(True)  # fires on the 3rd consecutive
    # pressure clearing resets the streak (flutter damped)
    assert not gate.update(False)
    assert not gate.update(True)


# ---------------------------------------------------------------------------
# server_args validation of the new flags
# ---------------------------------------------------------------------------


def _fake_server_args(**over):
    ns = types.SimpleNamespace(
        enable_kv_session_offload=True,
        kv_session_offload_prefill=False,
        kv_session_offload_host_ram_gib=0.0,
        kv_session_offload_block_size=8192,
        kv_session_offload_tick_interval=1,
        kv_session_offload_tick_floor=8,
        kv_session_offload_restore_hysteresis_steps=4,
        kv_session_offload_max_spills=1,
        kv_session_offload_restore_margin_tokens=4096,
        kv_session_offload_wave_back_min_free_tokens=0,
        kv_session_offload_mtp_resident_slices=0,
        kv_session_offload_spec_in_tick=False,
        kv_session_offload_budget_total_tokens=0,
        kv_session_offload_budget_session_tokens=0,
        kv_session_offload_budget_prefill_tokens=0,
        kv_session_offload_budget_decode_tokens=0,
        kv_session_offload_budget_rate_tokens_per_s=0.0,
        kv_session_offload_budget_episode_seconds=0.0,
        kv_session_offload_budget_max_sessions=0,
        kv_session_offload_spill_progress_lock_tokens=0,
        kv_session_offload_spill_hysteresis_steps=0,
        kv_session_offload_spill_cooldown_seconds=0.0,
        kv_session_offload_budget_demote_grace_iters=256,
        speculative_algorithm=None,
        attention_backend="flashinfer",
        page_size=1,
        disaggregation_mode="null",
        weightless_kv_fastlane=False,
        enable_hierarchical_cache=False,
        enable_unified_memory=False,
        enable_hisparse=False,
        pp_size=1,
        dp_size=1,
        enable_mixed_chunk=False,
    )
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


def _validate(ns):
    from sglang.srt.server_args import ServerArgs

    ServerArgs._handle_kv_session_offload(ns)


def test_server_args_budget_defaults_validate():
    _validate(_fake_server_args())  # all reglers off -> must not raise


def test_server_args_reject_budget_without_the_feature():
    for name, val in (
        ("kv_session_offload_budget_total_tokens", 1000),
        ("kv_session_offload_budget_rate_tokens_per_s", 10.0),
        ("kv_session_offload_spill_progress_lock_tokens", 5),
        ("kv_session_offload_spill_cooldown_seconds", 1.0),
    ):
        with pytest.raises(ValueError, match="requires"):
            _validate(
                _fake_server_args(
                    enable_kv_session_offload=False, **{name: val}
                )
            )


def test_server_args_reject_negative_budget_values():
    for name, val in (
        ("kv_session_offload_budget_session_tokens", -1),
        ("kv_session_offload_budget_episode_seconds", -0.5),
        ("kv_session_offload_budget_demote_grace_iters", -1),
    ):
        with pytest.raises(ValueError, match=">= 0"):
            _validate(_fake_server_args(**{name: val}))


def test_server_args_accept_an_armed_budget():
    _validate(
        _fake_server_args(
            kv_session_offload_budget_total_tokens=100000,
            kv_session_offload_budget_episode_seconds=10.0,
            kv_session_offload_spill_progress_lock_tokens=64,
        )
    )
