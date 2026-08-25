# SPDX-License-Identifier: Apache-2.0
"""#861j: METAL-FAITHFUL REPRO -- why no decode batch ever forms in TP.

THE SPECIMEN (boot_w37f2_0825_1223.log, pin 35b9914e50, and identically
w37d4/w37f): three consecutive boots behind GREEN desk gates produced ZERO
real decode batches (`Decode batch phase=` == 0 -- the only honest counter),
21-24 flips per ~6 min. The falsifiers so far exercised policy TERMS; this
file drives the REAL batch-formation path those terms protect: real `Req`
through the real `reset_for_retract`, the seam's own stamp, the real
`readmit_seam_residents` queue authority, the real `maybe_arm_phase_policy`,
the real `prefill_blocked_here` gate, and the real `get_next_batch_to_run`.

THE TWO CLOSED DOORS, from the specimen (12:27:43):

    PHASE-FLIP cutover complete: active stack tp, ps tp=3 pp=1
    PHASE-POLICY arming tp_to_pp: pending prefill 18288 tok > 0 (purity: ...)

DOOR 1 -- the demand term arms away before TP gets one scheduling round.
  * `_pending_prefill_tokens` reads ~0 (the retract credit covers the whole
    prompt) and the W32 boundary subtraction (scheduler.py:11159) removes the
    seam-transport remainder. So `decide()`'s drain-exit block, gated
    `pending > pp_exit_tokens` (phase_policy.py:2788), is SKIPPED -- and the
    W32 seam-transport hold at :2869, the ONLY reader of
    `seam_transport_tokens`, sits INSIDE that block. The subtraction zeroes
    the gate guarding its own only consumer: a self-disabling exclusion.
  * Control falls to the #861d-2 demand branch: `demand_prefill_tokens()` =
    max(pending=0, admissible=full prompts of the 7 stamped requests) > 0
    under strict purity -> TP_TO_PP armed, in the recv hook, BEFORE any batch
    build. The `starved` term at the min-dwell floor reads the same
    unclassified quantity, so even the dwell does not hold it: the arm is
    INSTANT. An armed flip parks all new work, so the seam-transport
    exemption is never consulted (measured: SEAM TRANSPORT ADMITTED=0,
    REFUSED=0, `Prefill batch phase=tp`=0 in all three logs).

DOOR 2 -- even held, the premise refuses: `seam_transport_premise_holds`
  (phase_purity.py:846) requires a stamped request with
  `cache_protected_len > 0`, and `reset_for_retract`
  (schedule_batch.py:1589) zeroes exactly that field three lines after
  stamping the credit. The premise reads state the transition it guards just
  manufactured (the W37-D class), so it is structurally False for the entire
  population the exemption exists for. The only TP-residency path stays
  closed even with Door 1 fixed.

Every test below asserts the HEALTHY property and therefore dies RED on
35b9914e50. The cold-readmission pin is the can-fail counterweight: the
dangerous direction (a request retracted before ANY fill re-prefilling in
TP) must STAY refused by any fix.
"""

import dataclasses
import os
import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.managers.phase_policy import PhasePolicyState
from sglang.srt.managers.phase_policy import config_from_env as policy_config_from_env
from sglang.srt.managers.phase_purity import (
    SEAM_READMIT_ATTR,
    parse_purity,
    prefill_blocked_here,
)
from sglang.srt.managers.schedule_batch import NextBatchPlan, Req
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.chunk_cache import ChunkCache
from sglang.srt.sampling.sampling_params import SamplingParams

# The metal policy environment, from the boot's own armed line:
#   "PHASE-POLICY armed: N=7004 tok (break-even 3.2s / (1/1681 - 1/7245.5)),
#    min dwell 3s, idle dwell 3s, pp window 15s, tp decode floor 10s ..."
# plus --phase-policy-drain-mode-strict from boot_w37f2.sh.
_METAL_ENV = {
    "SGLANG_PHASE_POLICY_TP_TOK_S": "1681",
    "SGLANG_PHASE_POLICY_PP_TOK_S": "7245.5",
    "SGLANG_PHASE_POLICY_FLIP_COST_S": "3.2",
    "SGLANG_PHASE_POLICY_DRAIN_MODE_STRICT": "1",
    "SGLANG_PHASE_POLICY_PP_WINDOW_S": "15",
    "SGLANG_PHASE_POLICY_TP_DECODE_FLOOR_S": "10",
    "SGLANG_PHASE_POLICY_MIN_DWELL_S": "3",
    "SGLANG_PHASE_POLICY_IDLE_DWELL_S": "3",
}
# Env that must NOT leak in from the invoking shell.
_MUST_BE_UNSET = ("SGLANG_PHASE_POLICY_FLIP_TOKENS", "SGLANG_PHASE_POLICY_DRAIN_MODE")


def _metal_policy_cfg():
    """The REAL config builder under the metal env, then the REAL purity
    binding `Scheduler.__init__` applies under `--phase-flip-purity strict`
    (scheduler.py:668-681): prefill_runs_in_tp=False, pp_exit_tokens=chunk."""
    saved = {}
    for k, v in _METAL_ENV.items():
        saved[k] = os.environ.get(k)
        os.environ[k] = v
    for k in _MUST_BE_UNSET:
        saved[k] = os.environ.get(k)
        os.environ.pop(k, None)
    try:
        cfg = policy_config_from_env(enabled=True, chunk_tokens=4096)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return dataclasses.replace(cfg, prefill_runs_in_tp=False, pp_exit_tokens=4096)


def _mk_prefilled_req(i: int, prompt_len: int = 3047) -> Req:
    """A request as the PP window leaves it: prefill complete, ONE output
    token (the OUTTRACE `n=1` of the specimen)."""
    sp = SamplingParams(max_new_tokens=64, temperature=0)
    sp.normalize(None)
    req = Req(
        rid=f"w37f2-{i}",
        origin_input_text="",
        origin_input_ids=list(range(prompt_len)),
        sampling_params=sp,
    )
    req.output_ids = [271]
    return req


def _seam_retract(reqs, epoch: int = 3):
    """What the #856 cutover does to each resident: the REAL
    `Req.reset_for_retract`, then the seam's own stamp -- byte-for-byte the
    per-request effect of `build_cutover_release._retract`
    (phase_flip_runtime.py:1444-1454)."""
    for r in reqs:
        r.reset_for_retract()
        setattr(r, SEAM_READMIT_ATTR, epoch)


def _tp_scheduler_after_cutover(cfg, *, epoch: int = 3):
    """A Scheduler whose METHODS are the real ones; only construction is
    hand-rolled (the class cannot be __init__'ed hermetically). State is the
    instant after a pp_to_tp cutover: phase tp, nothing resident, empty
    queue (the re-admission is driven by the caller through the REAL
    `readmit_seam_residents`)."""
    s = Scheduler.__new__(Scheduler)
    s.server_args = SimpleNamespace(
        enable_phase_flip=True,
        phase_flip_purity="strict",
        chunked_prefill_size=4096,
    )
    s._phase_purity = parse_purity("strict")
    s.phase_flip_active_stack = "tp"
    s.phase_policy_cfg = cfg
    s.phase_policy_state = PhasePolicyState()
    s.phase_flip_runtime = SimpleNamespace(
        phase="tp", pending=None, epoch=epoch, armed_idle_locked=False
    )
    s.waiting_queue = []
    s.running_batch = None
    s.chunked_req = None
    s.last_batch = None
    s.running_mbs = []
    s.last_mbs = []
    # A freshly re-dispatched loop: no round has failed to build yet.
    s._round_built_nothing = False
    s._decode_steps_this_phase = 0
    # Queue-authority requirements of the REAL readmit path.
    s.enable_hicache_storage = False
    s.enable_priority_scheduling = False
    s.abort_on_priority_when_disabled = False
    s.max_queued_requests = None
    s._kv_arrival_ct = 0
    s.disaggregation_mode = DisaggregationMode.NULL
    s.parked_decode_set = None
    s._pp_live_mb_id = -1
    return s


class TestDoor1PolicyArmsAwayFromItsOwnReadmission(CustomTestCase):
    """The first TP policy round after the seam re-admission."""

    def _armed(self, dwell_ago_s: float):
        cfg = _metal_policy_cfg()
        reqs = [_mk_prefilled_req(i) for i in range(7)]
        _seam_retract(reqs)
        s = _tp_scheduler_after_cutover(cfg)
        n = s.readmit_seam_residents(reqs)
        assert n == 7, f"the REAL queue authority re-admitted {n} of 7"
        s.phase_policy_state.last_flip_at = time.perf_counter() - dwell_ago_s
        ret = s.maybe_arm_phase_policy()
        return s, ret

    def test_the_first_tp_round_must_not_arm_away_from_the_readmission(self):
        """RED ON 35b9914e50. The 7 requests the cutover just retracted and
        re-admitted are the flip's own justification; the seam-transport
        exemption exists to serve them IN THIS LAYOUT. A tp_to_pp arm on the
        first round destroys that -- the arm's execution voids the arm's own
        justification (W30's rule), and the layout oscillates for ever with
        zero decode batches, which is the W37-F metal specimen."""
        s, ret = self._armed(dwell_ago_s=8.0)
        self.assertIsNone(
            ret,
            "the policy armed away from the seam re-admission on the first "
            f"TP round: {s.phase_policy_state.last_reason!r}. These tokens "
            "are flip transport, served in THIS layout by read-through -- "
            "counting them as PP demand makes the tp-ward flip undo itself "
            "(21+21 flips, 0 decode batches on metal)",
        )

    def test_not_even_the_dwell_floor_holds_the_instant_rearm(self):
        """RED ON 35b9914e50, same root, sharper clock: 0.1 s after the
        cutover the `starved` bypass at the min-dwell floor reads the same
        unclassified existence quantity, so the arm fires with the dwell
        floor live -- the specimen's same-second re-arm."""
        s, ret = self._armed(dwell_ago_s=0.1)
        self.assertIsNone(
            ret,
            "armed tp_to_pp 0.1s after the cutover (min dwell 3s bypassed by "
            f"the starved term): {s.phase_policy_state.last_reason!r}",
        )


class TestDoor2PremiseRefusesThePopulationItExistsFor(CustomTestCase):
    def test_the_purity_gate_must_admit_the_seam_stamped_population(self):
        """RED ON 35b9914e50. `seam_transport_premise_holds` demands a
        stamped request with `cache_protected_len > 0` -- but
        `reset_for_retract` zeroes that field on every request the seam
        retracts, three lines after stamping the credit
        (`cached_prompt_tokens_at_retract` = the full prompt: these requests
        WERE fully prefilled and their KV fenced to the canonical store).
        The premise reads state the transition it guards manufactures, so
        the exemption can never admit the population it exists for."""
        cfg = _metal_policy_cfg()
        reqs = [_mk_prefilled_req(i) for i in range(7)]
        _seam_retract(reqs)
        s = _tp_scheduler_after_cutover(cfg)
        s.readmit_seam_residents(reqs)
        blocked = prefill_blocked_here(s, running_bs=0)
        self.assertFalse(
            blocked,
            "TP refused the read-through prefill of the 7 requests the seam "
            "itself stamped: every one carries the full-prompt retract "
            "credit (KV computed and fenced), yet the premise checks "
            "cache_protected_len, which the retraction zeroed. The only "
            "path to TP residency is closed",
        )

    def test_a_cold_readmission_is_still_refused(self):
        """CAN-FAIL COUNTERWEIGHT, green before AND after any fix. A request
        retracted before ANY fill (no output, no extend_range) has no KV in
        the canonical store; re-admitting it in TP would be a cold prefill
        of real work -- the W37-D violation (258 batches at #cached-token 0).
        Whatever evidence the premise reads, this population must stay
        blocked."""
        cfg = _metal_policy_cfg()
        req = _mk_prefilled_req(0)
        req.output_ids = []  # never produced a token
        req.extend_range = None  # never filled a chunk
        _seam_retract([req])
        assert req.cached_prompt_tokens_at_retract == 0
        s = _tp_scheduler_after_cutover(cfg)
        s.readmit_seam_residents([req])
        self.assertTrue(
            prefill_blocked_here(s, running_bs=0),
            "a cold re-admission (zero retract credit) was admitted in TP -- "
            "that is real work in the wrong layout, the exact W37-D breach",
        )


def _batch_formation_scheduler(cfg):
    """The `get_next_batch_to_run` drive, following the proven construction
    of test_scheduler_chunked_req_gate._scheduler_for_get_next_batch (its
    stub-drift notes apply here verbatim). The PATH UNDER TEST is the
    TP-phase purity gate inside the real `get_next_batch_to_run`; the
    timeout/abort prologue and the downstream decode plumbing are pinned
    exactly as that suite pins them."""
    s = _tp_scheduler_after_cutover(cfg)
    s._abort_on_waiting_timeout = MagicMock()
    s._abort_on_running_timeout = MagicMock()
    s.dllm_config = None
    s.dllm_manager = None
    s.enable_hisparse = False
    s.enable_fpm = False
    s.require_mlp_sync = False
    s.spec_algorithm = MagicMock()
    # The gate consults the REAL phase machinery, so enable_phase_flip stays
    # True; the runtime round hook is deferred exactly as the PP loop defers
    # it, so the consensus machinery (which needs a live group) never runs.
    s._defer_flip_round_to_pp_loop = True
    sa = MagicMock(
        speculative_skip_dp_mlp_sync=True,
        kv_reshard_vectors=None,
        kv_pressure_ladder=None,
        regime_controller="off",
        gdn_state_set_ladder=None,
        enable_vram_dial=False,
        enable_phase_flip=True,
        phase_flip_purity="strict",
        chunked_prefill_size=4096,
    )
    s.server_args = sa
    running = MagicMock()
    running.is_empty.return_value = True
    running.is_prefill_only = False
    running.batch_is_full = False
    running.reqs = []
    s.running_batch = running
    # THE RECORDER. On 35b9914e50 the purity gate blocks and this is never
    # consulted; the healthy property is that the builder is reached for the
    # stamped queue. It builds nothing (that half has its own suites) -- the
    # assertion is on the DOOR, not the batch.
    s.get_new_batch_prefill = MagicMock(
        return_value=NextBatchPlan(batch_to_run=None, running_batch=running)
    )
    s.dp_attn_adapter = MagicMock()
    s.dp_attn_adapter.maybe_prepare_mlp_sync_batch = MagicMock(
        side_effect=lambda batch, **_: batch
    )
    s._maybe_prepare_ngram_embedding = MagicMock(side_effect=lambda batch: batch)
    s.update_running_batch = MagicMock(side_effect=lambda batch: batch)
    pool = SimpleNamespace(
        req_to_token=torch.zeros((8, 64), dtype=torch.int32), mamba_allocator=None
    )
    s.tree_cache = ChunkCache(
        SimpleNamespace(
            req_to_token_pool=pool, token_to_kv_pool_allocator=None, page_size=1
        )
    )
    s.req_to_token_pool = pool
    s._pending_chunked_abort_req = None
    s.enable_hierarchical_cache = False
    s.kv_session_offload = None
    s.tp_cpu_group = None
    s.token_to_kv_pool_allocator = MagicMock(available_size=lambda: 1 << 30)
    s.kv_reshard_runtime = None
    s.kv_pressure_runtime = None
    s.kv_capacity_runtime = None
    s.regime_observer = None
    s.regime_stage_table = None
    s._regime_observer_mode = None
    s.gdn_slot_executor = None
    s.congruent_prefill_lane = None
    s.ps = ParallelState(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        dp_rank=None,
        dp_size=1,
        attn_tp_rank=0,
        attn_tp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        attn_dp_rank=0,
        attn_dp_size=1,
        moe_ep_rank=0,
        moe_ep_size=1,
        moe_dp_rank=None,
        moe_dp_size=1,
        gpu_id=0,
    )
    return s


class TestDoor2TheBatchFormationPathNeverOpens(CustomTestCase):
    def test_tp_must_consult_the_builder_for_the_stamped_queue(self):
        """RED ON 35b9914e50 -- the metal arm auditor's own words: "the
        cutover COMMITTED into the target layout, and it still built no
        batch in 8 rounds". Eight REAL `get_next_batch_to_run` rounds over
        the re-admitted stamped queue: the purity gate refuses every one
        (premise reads the zeroed field), so the prefill builder is never
        consulted, no request can become resident, and TP can never form a
        decode batch -- zero `Decode batch phase=` on three boots."""
        cfg = _metal_policy_cfg()
        reqs = [_mk_prefilled_req(i) for i in range(7)]
        _seam_retract(reqs)
        s = _batch_formation_scheduler(cfg)
        s.readmit_seam_residents(reqs)
        for _ in range(8):
            Scheduler.get_next_batch_to_run(
                s, running_batch=s.running_batch, last_batch=None
            )
        self.assertGreater(
            s.get_new_batch_prefill.call_count,
            0,
            "8 real scheduler rounds and the prefill builder was never "
            "consulted for the 7 seam-stamped requests: the purity gate is "
            "closed (premise refuses the retraction-zeroed field), so the "
            "re-admitted population can never become resident in TP and "
            "decode formation is structurally impossible -- the W37-F "
            "specimen at the desk",
        )


class TestTheHoldHasABoundedExit(CustomTestCase):
    """W37-E is the standing proof that a hold without an exit is a deadlock
    with better manners. Every test here is a CAN-FAIL for the #861j hold."""

    def _cutover_state(self):
        cfg = _metal_policy_cfg()
        reqs = [_mk_prefilled_req(i) for i in range(7)]
        _seam_retract(reqs)
        s = _tp_scheduler_after_cutover(cfg)
        s.readmit_seam_residents(reqs)
        s.phase_policy_state.last_flip_at = time.perf_counter() - 8.0
        return s

    def test_a_lapsed_transport_debt_lets_the_demand_fire(self):
        """The transport-debt clock is the hold's exit: past the drain-stall
        deadline (10 s at the metal seam cost) the serviceable credit lapses
        and the demand MUST arm tp_to_pp -- the work goes to PP exactly as it
        did before #861j, one bounded delay later, never a wedge."""
        s = self._cutover_state()
        s._seam_transport_debt_since = time.perf_counter() - 11.0
        ret = s.maybe_arm_phase_policy()
        self.assertIsNotNone(
            ret,
            "the hold outlived its own deadline: 7 stamped requests sat "
            "unadmitted past the drain-stall bound and nothing armed -- that "
            "is the W37-E wedge shape reintroduced by the fix meant to close "
            "W37-F",
        )
        self.assertEqual(ret.direction, "tp_to_pp")

    def test_fresh_unstamped_work_still_demands_the_pp_layout(self):
        """Stamped transport plus ONE fresh queued request: the fresh prompt
        is not serviceable in TP (the exemption's builder filter excludes
        it), so the demand must fire undiminished -- the subtraction may
        remove exactly the transport, never a token more."""
        s = self._cutover_state()
        fresh = _mk_prefilled_req(99, prompt_len=3047)
        fresh.output_ids = []
        s.waiting_queue.append(fresh)
        ret = s.maybe_arm_phase_policy()
        self.assertIsNotNone(
            ret,
            "a fresh request's prefill demand was swallowed by the seam "
            "subtraction -- the transport credit is claiming work that is "
            "not transport",
        )
        self.assertEqual(ret.direction, "tp_to_pp")

    def test_the_hold_names_its_state_in_the_log(self):
        """The W37-F specimen's whole cost was an invisible closed door; the
        hold this fix introduces must be readable from the boot log."""
        from sglang.srt.managers import scheduler as sched_mod

        s = self._cutover_state()
        with self.assertLogs(sched_mod.logger, level="INFO") as cm:
            ret = s.maybe_arm_phase_policy()
        self.assertIsNone(ret)
        self.assertIn(
            "seam transport",
            "\n".join(cm.output),
            "the hold fired but did not name the seam-transport state",
        )


if __name__ == "__main__":
    unittest.main()
