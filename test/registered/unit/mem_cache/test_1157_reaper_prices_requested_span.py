"""#1157 -- the prefetch reaper priced itself on a quantity that did not exist yet.

THE SPECIMEN (boot weg1b3 @ 6980c75eac, log lines 69215 -> 72543 -> 72553/72556,
23:56:17-18). The pp_to_tp cutover re-admitted six retracted residents; the
LONG one (rid 679e4568, 84,027 tokens, 84,026 pages at page_size=1) queued
LAST. `_prefetch_timeout_check_linear_func` charged
``base + len(operation.hash_value) x per_page`` and ``hash_value`` is empty
until the serial prefetch thread's single whole-list store probe has run --
so the drain, which runs every scheduler pass over the whole waiting queue,
reaped the unprobed operation at the BARE base (~1.2 s) with ``completed=0``.
The store HAD the pages (84,200 files written 23:54:21-23:55:02); nobody asked
it in time. Then a P=0 admission and six recomputed 4096-token TP chunks.

The effective base was 1.0 s, not the 2.0 s the pin declared (#968/#1065):
those locals in `UnifiedRadixCache.init_hicache` were dead, overwritten by the
tuple unpack from `HybridCacheController.parse_storage_backend_extra_config`
whose own defaults were 1 / 0.25; `PrefetchTimeoutConfig` (2.0 / 1.0) was read
only by hiradix / hi_mamba, neither the serving tree.

WHAT THIS FILE PINS (one matched check per fix item, error class named):

F1a  ARITHMETIC ON THE OPERATION OBJECT. A real `PrefetchOperation` with
     84,026 token ids and an empty ``hash_value`` on a tree-shaped stub whose
     base / per_page come from the parse: the linear check is False at
     t = 1.5 s and the budget equals ``base + 84,026 x per_page``; once the
     probe fills ``hash_value`` the hit-priced form stands. Same form on the
     reachable sibling `HiRadixCache`.
F1b  ONE SOURCE OF TRUTH. The parse pops its defaults FROM
     `PrefetchTimeoutConfig`; the attached tree's effective base equals the
     config's with no extra config and the override with one; no other
     numeric default for ``prefetch_timeout_base`` exists under
     python/sglang/srt; the effective pair is printed at attach.
F1c  THE REAP IS A LINE. Driving the real `check_prefetch_progress` over an
     unprobed, timed-out operation (the #937 harness: a real CPU
     `UnifiedRadixCache`, a real host pool, a real `PrefetchOperation`)
     emits ``#1157 PREFETCH REAPED ... probed=False ... completed=0`` and
     records a `PrefetchOutcome` with ``probed=False`` on the one record the
     admission loop pops.
N1   THE REAP ANNOTATION IS RANK-UNIFORM (review fix). `probed` and
     `hit_tokens` ride two extra slots of the EXISTING packed MIN all_reduce
     in `check_prefetch_progress` (the one that reduces min_completed_tokens);
     the record is derived from the reduced vector only. Two fake ranks --
     one whose prefetch thread has stamped the probe, one whose reap landed
     before its stamp -- both derive probed=False. Mutant: read the local
     stamp instead of the reduced slot -> red.

Hermetic: CUDA_VISIBLE_DEVICES="" and the worktree's python on PYTHONPATH. Red
on the parent (e63fd081dc): the budget helper does not exist, the parse
answers (1.0, 0.25), and no REAPED line is emitted.
"""

import json
import os
import queue
import re
import time
import types
import unittest

import torch

from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.managers.cache_controller import (
    PrefetchOperation as BasePrefetchOperation,
)
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.hicache_phase_binding import binding_state
from sglang.srt.mem_cache.hicache_storage import PrefetchOutcome, PrefetchTimeoutConfig
from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
    PrefetchOperation,
)
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache_components.tree_component import (
    ComponentType,
)
from sglang.srt.mem_cache.unified_radix_cache import (
    _POOL_SLOT_COUNT,
    _REAP_PACKED_LEN,
    _REAP_SLOT_HIT_TOKENS,
    _REAP_SLOT_PROBED,
    UnifiedRadixCache,
    _OngoingPrefetch,
)
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

REQUESTED_TOKENS = 84_026  # the LONG re-admission of boot weg1b3, page_size=1
PROBED_PAGES = 6009  # c4e85437's probed hit on the same cutover (log 100171)
PAGE_SIZE = 1
SRT_ROOT = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "python", "sglang", "srt"
)


def _parsed_defaults():
    _, _, base, per_ki, _ = HybridCacheController.parse_storage_backend_extra_config(
        None
    )
    return float(base), float(per_ki)


def _tree_stub(base: float, per_ki: float, page_size: int = PAGE_SIZE):
    """A UnifiedRadixCache-shaped stub carrying exactly the three fields the
    reaper reads (`_deferred_prefetch_bound_s` reads the same three)."""
    stub = types.SimpleNamespace(
        page_size=page_size,
        prefetch_timeout_base=base,
        prefetch_timeout_per_page=page_size / 1024 * per_ki,
        prefetch_stop_policy="timeout",
    )
    for name in (
        "_prefetch_priced_pages",
        "_prefetch_timeout_budget_s",
        "_prefetch_timeout_check_linear_func",
        "_reap_annotation_local",
    ):
        setattr(stub, name, getattr(UnifiedRadixCache, name).__get__(stub))
    return stub


def _unprobed_operation(tokens: int, *, age_s: float) -> BasePrefetchOperation:
    op = BasePrefetchOperation(
        request_id="679e4568",
        host_indices=torch.zeros(0, dtype=torch.int64),
        token_ids=list(range(tokens)),
    )
    assert op.hash_value == []  # the probe has not run
    op.start_time = time.monotonic() - age_s
    return op


class F1aTheReaperPricesTheRequestedSpanUntilProbed(CustomTestCase):
    def test_an_unprobed_84k_operation_is_not_reaped_at_1_5s(self):
        """THE MATCHED CHECK. Red on the parent: base + 0 x per_page = 1.0 s,
        and 1.5 s is past it."""
        base, per_ki = _parsed_defaults()
        tree = _tree_stub(base, per_ki)
        op = _unprobed_operation(REQUESTED_TOKENS, age_s=1.5)
        self.assertFalse(tree._prefetch_timeout_check_linear_func(op))

    def test_the_unprobed_budget_is_base_plus_requested_pages_x_per_page(self):
        base, per_ki = _parsed_defaults()
        tree = _tree_stub(base, per_ki)
        op = _unprobed_operation(REQUESTED_TOKENS, age_s=0.0)
        expected = base + (REQUESTED_TOKENS // PAGE_SIZE) * (PAGE_SIZE / 1024 * per_ki)
        self.assertAlmostEqual(tree._prefetch_timeout_budget_s(op), expected, places=6)
        # With the pin's own numbers (2.0 s + 1.0 s/KiToken) that is ~84 s,
        # which covers the measured ~2,800 pages/s store transfer.
        self.assertGreater(tree._prefetch_timeout_budget_s(op), 80.0)

    def test_after_the_probe_the_hit_priced_form_stands(self):
        base, per_ki = _parsed_defaults()
        tree = _tree_stub(base, per_ki)
        op = _unprobed_operation(REQUESTED_TOKENS, age_s=0.0)
        op.hash_value = [f"h{i}" for i in range(PROBED_PAGES)]
        expected = base + PROBED_PAGES * (PAGE_SIZE / 1024 * per_ki)
        self.assertAlmostEqual(tree._prefetch_timeout_budget_s(op), expected, places=6)

    def test_the_reaper_can_still_fire_past_the_priced_budget(self):
        """CAN-FAIL COUNTERWEIGHT: the budget is a bound, not a disable."""
        base, per_ki = _parsed_defaults()
        tree = _tree_stub(base, per_ki)
        op = _unprobed_operation(16, age_s=0.0)
        op.start_time = time.monotonic() - (tree._prefetch_timeout_budget_s(op) + 1.0)
        self.assertTrue(tree._prefetch_timeout_check_linear_func(op))

    def test_page_size_divides_the_requested_span(self):
        base, per_ki = _parsed_defaults()
        tree = _tree_stub(base, per_ki, page_size=64)
        op = _unprobed_operation(REQUESTED_TOKENS, age_s=0.0)
        self.assertEqual(tree._prefetch_priced_pages(op), REQUESTED_TOKENS // 64)

    def test_the_hiradix_sibling_prices_the_requested_span(self):
        """registry.py:113-115 constructs HiRadixCache for non-hybrid models
        under hierarchical cache, so the sibling is reachable and carries the
        same form (capped by its own `PrefetchTimeoutConfig.max`).
        HiMambaRadixCache is NOT touched: registry.py:106-107 -- no
        construction site anywhere on this tree."""
        cfg = PrefetchTimeoutConfig()
        stub = types.SimpleNamespace(page_size=PAGE_SIZE, prefetch_timeout_config=cfg)
        check = HiRadixCache._prefetch_timeout_check_linear_func.__get__(stub)
        op = _unprobed_operation(REQUESTED_TOKENS, age_s=1.5)
        self.assertFalse(check(op))
        op.start_time = time.monotonic() - (cfg.max + 1.0)
        self.assertTrue(check(op))


class F1bOneSourceOfTruthForTheTimeout(CustomTestCase):
    def test_the_parse_pops_its_defaults_from_prefetch_timeout_config(self):
        """Red on the parent: the parse answered (1.0, 0.25)."""
        cfg = PrefetchTimeoutConfig()
        self.assertEqual(_parsed_defaults(), (cfg.base, cfg.per_ki_token))

    def test_an_extra_config_override_still_wins(self):
        extra = json.dumps(
            {"prefetch_timeout_base": 3, "prefetch_timeout_per_ki_token": 0.5}
        )
        _, _, base, per_ki, _ = HybridCacheController.parse_storage_backend_extra_config(
            extra
        )
        self.assertEqual((base, per_ki), (3.0, 0.5))

    def test_no_second_numeric_default_exists_under_srt(self):
        """The grep the operator named: exactly one default definition. Every
        line under python/sglang/srt naming `prefetch_timeout_base` beside a
        numeric literal is a second default; there must be none (the config
        field is `base`)."""
        offenders = []
        pat = re.compile(r"prefetch_timeout_base\b[^#\n]*?(=|,)\s*[0-9]")
        for dirpath, _, files in os.walk(SRT_ROOT):
            for fn in files:
                if not fn.endswith(".py"):
                    continue
                p = os.path.join(dirpath, fn)
                with open(p, encoding="utf-8", errors="replace") as fh:
                    for lineno, line in enumerate(fh, 1):
                        if pat.search(line) and not line.lstrip().startswith("#"):
                            offenders.append(f"{os.path.relpath(p, SRT_ROOT)}:{lineno}")
        self.assertEqual(offenders, [], f"second timeout defaults: {offenders}")
        self.assertTrue(hasattr(PrefetchTimeoutConfig(), "base"))
        self.assertTrue(hasattr(PrefetchTimeoutConfig(), "per_ki_token"))

    def _attach(self, base: float, per_ki: float):
        stub = types.SimpleNamespace(
            page_size=PAGE_SIZE,
            prefetch_stop_policy="timeout",
            storage_metrics_collector=None,
            extra_metric_labels=None,
        )
        apply = UnifiedRadixCache._apply_storage_runtime_config.__get__(stub)
        with self.assertLogs("sglang.srt.mem_cache.unified_radix_cache", "INFO") as cm:
            apply(
                storage_backend="file",
                prefetch_threshold=256,
                prefetch_timeout_base=base,
                prefetch_timeout_per_ki_token=per_ki,
                hicache_storage_pass_prefix_keys=False,
                enable_storage=True,
                enable_storage_metrics=False,
                extra_metric_labels=None,
            )
        return stub, "\n".join(cm.output)

    def test_the_attached_tree_runs_the_configs_base_without_extra_config(self):
        cfg = PrefetchTimeoutConfig()
        base, per_ki = _parsed_defaults()
        stub, out = self._attach(base, per_ki)
        self.assertEqual(stub.prefetch_timeout_base, cfg.base)
        self.assertEqual(stub.prefetch_timeout_per_ki_token, cfg.per_ki_token)
        self.assertAlmostEqual(
            stub.prefetch_timeout_per_page, PAGE_SIZE / 1024 * cfg.per_ki_token
        )
        self.assertIn(
            f"#1157 PREFETCH TIMEOUT base={cfg.base:.2f}s "
            f"per_ki_token={cfg.per_ki_token:.2f}s source=PrefetchTimeoutConfig",
            out,
        )

    def test_the_attached_tree_runs_the_extra_config_value_when_given(self):
        stub, out = self._attach(3.0, 0.5)
        self.assertEqual(stub.prefetch_timeout_base, 3.0)
        self.assertIn(
            "#1157 PREFETCH TIMEOUT base=3.00s per_ki_token=0.50s source=extra_config",
            out,
        )

    def test_the_tree_carries_the_config_before_any_attach(self):
        """The __init__ pair (formerly 1.0 / 0.25 literals) is the config's
        too, so `_deferred_prefetch_bound_s` and the reaper never see a
        third default between construction and attach."""
        cache = _build_cache()
        cfg = PrefetchTimeoutConfig()
        self.assertEqual(cache.prefetch_timeout_base, cfg.base)
        self.assertEqual(cache.prefetch_timeout_per_ki_token, cfg.per_ki_token)


# ---- the #937 harness: a real CPU tree, a real host pool, a real operation ----


def _build_cache() -> UnifiedRadixCache:
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    return UnifiedRadixCache(
        params=CacheInitParams(
            req_to_token_pool=ReqToTokenPool(
                size=8, max_context_len=64, device="cpu", enable_memory_saver=False
            ),
            token_to_kv_pool_allocator=TokenToKVPoolAllocator(
                size=64,
                dtype=torch.float16,
                device="cpu",
                kvcache=MHATokenToKVPool(
                    size=64,
                    page_size=PAGE_SIZE,
                    dtype=torch.float16,
                    head_num=2,
                    head_dim=4,
                    layer_num=2,
                    device="cpu",
                    enable_memory_saver=False,
                ),
                need_sort=False,
            ),
            page_size=PAGE_SIZE,
            disable=False,
            tree_components=(ComponentType.FULL,),
        )
    )


REAP_REQ = "req-1157"
REAP_TOKENS = 16


def _reap_scenario(*, probed: bool):
    """An operation registered on the tree under policy 'timeout' whose
    clock is past its priced budget. ``probed=False`` leaves ``hash_value``
    empty (the weg1b3 shape); ``probed=True`` fills it with the whole span
    and lets the transfer complete, i.e. the healthy completion."""
    binding_state().reset()
    cache = _build_cache()
    cache.prefetch_stop_policy = "timeout"
    device_pool = cache.token_to_kv_pool_allocator.get_kvcache()
    pool = MHATokenToKVPoolHost(
        device_pool=device_pool,
        host_to_device_ratio=0.5,
        host_size=0,
        page_size=PAGE_SIZE,
        layout="layer_first",
        pin_memory=False,
        device="cpu",
        allocator_type="default",
        budget_label="gen1",
    )
    binding_state().advance("pp", host_pool=pool)
    host_indices = pool.alloc(REAP_TOKENS)
    operation = PrefetchOperation(
        request_id=REAP_REQ,
        host_indices=host_indices,
        token_ids=list(range(1, REAP_TOKENS + 1)),
    )
    if probed:
        operation.hash_value = [f"h{i}" for i in range(REAP_TOKENS // PAGE_SIZE)]
        operation.probed_hit_tokens = REAP_TOKENS
        operation.increment(REAP_TOKENS)
    fake_cc = types.SimpleNamespace(
        mem_pool_host=pool,
        host_mem_release_queue=queue.Queue(),
        prefetch_revoke_queue=queue.Queue(),
        ack_backup_queue=queue.Queue(),
        prefetch_tokens_occupied=REAP_TOKENS,
        write_policy="write_through_selective",
    )
    fake_cc.terminate_prefetch = HiCacheController.terminate_prefetch.__get__(fake_cc)
    fake_cc.append_host_mem_release = HiCacheController.append_host_mem_release.__get__(
        fake_cc
    )
    cache.cache_controller = fake_cc
    cache.ongoing_prefetch[REAP_REQ] = _OngoingPrefetch(
        cache.root_node,
        RadixKey(list(range(1, REAP_TOKENS + 1))),
        host_indices,
        operation,
        None,
        {},
    )
    # Past the PRICED budget (base + 16 pages x per_page), so the reap is
    # legitimate under the fix and the line must still appear.
    operation.start_time = time.monotonic() - (
        cache._prefetch_timeout_budget_s(operation) + 1.0
    )
    return cache, operation


class F1cTheReapIsALine(CustomTestCase):
    def setUp(self):
        self.addCleanup(binding_state().reset)

    def test_a_timed_out_unprobed_operation_prints_the_reap(self):
        """THE MATCHED CHECK: the line with probed=False and the numbers."""
        cache, _ = _reap_scenario(probed=False)
        with self.assertLogs("sglang.srt.mem_cache.unified_radix_cache", "WARNING") as cm:
            self.assertTrue(cache.check_prefetch_progress(REAP_REQ))
        out = "\n".join(cm.output)
        self.assertIn(
            f"#1157 PREFETCH REAPED req={REAP_REQ} probed=False "
            f"requested_pages={REAP_TOKENS // PAGE_SIZE} hit_pages=0 completed=0 "
            "elapsed=",
            out,
        )
        m = re.search(r"elapsed=([0-9.]+)s budget=([0-9.]+)s", out)
        self.assertIsNotNone(m, out)
        elapsed, budget = float(m.group(1)), float(m.group(2))
        self.assertGreater(elapsed, budget)
        # The line prints %.2f; compare within its own rounding.
        self.assertAlmostEqual(
            budget,
            PrefetchTimeoutConfig().base
            + REAP_TOKENS * (PAGE_SIZE / 1024 * PrefetchTimeoutConfig().per_ki_token),
            delta=0.006,
        )

    def test_the_reap_records_an_unprobed_outcome_on_the_one_record(self):
        cache, _ = _reap_scenario(probed=False)
        cache.check_prefetch_progress(REAP_REQ)
        outcome = cache.prefetch_loaded_tokens_by_reqid[REAP_REQ]
        self.assertIsInstance(outcome, PrefetchOutcome)
        self.assertFalse(outcome.probed)
        self.assertEqual(outcome.hit_tokens, 0)
        self.assertEqual(int(outcome), 0)
        # The existing reader is unchanged: an int, popped to 0-or-loaded.
        self.assertEqual(cache.pop_prefetch_loaded_tokens(REAP_REQ), 0)
        self.assertNotIn(REAP_REQ, cache.ongoing_prefetch)

    def test_a_completed_probed_operation_is_not_a_reap(self):
        """CAN-FAIL COUNTERWEIGHT: the line is per reap, not per completion."""
        cache, _ = _reap_scenario(probed=True)
        with self.assertLogs("sglang.srt.mem_cache.unified_radix_cache", "INFO") as cm:
            self.assertTrue(cache.check_prefetch_progress(REAP_REQ))
        out = "\n".join(cm.output)
        self.assertNotIn("#1157 PREFETCH REAPED", out)
        outcome = cache.prefetch_loaded_tokens_by_reqid[REAP_REQ]
        self.assertTrue(outcome.probed)
        self.assertEqual(outcome.hit_tokens, REAP_TOKENS)

    def test_a_revoke_records_the_probed_miss(self):
        """The revoke arm (probe answered below the prefetch threshold) used
        to delete the record without a trace; it now records the probed
        answer on the same record, which is what the seam witness reads."""
        cache, operation = _reap_scenario(probed=False)
        operation.probed_hit_tokens = 0
        # The revoke arm releases through the HYBRID controller's signature
        # (`extra_pools=`); the release itself is not the subject here.
        cache.cache_controller.append_host_mem_release = lambda *a, **k: None
        cache.cache_controller.prefetch_revoke_queue.put(REAP_REQ)
        cache._drain_storage_control_queues_local()
        self.assertNotIn(REAP_REQ, cache.ongoing_prefetch)
        outcome = cache.prefetch_loaded_tokens_by_reqid[REAP_REQ]
        self.assertTrue(outcome.probed)
        self.assertEqual(outcome.hit_tokens, 0)
        self.assertEqual(int(outcome), 0)


class N1TheReapAnnotationIsRankUniform(CustomTestCase):
    """The rank-divergent admission death class of boot weg1b3, on this path: a reap landing on rank B between
    B's gloo all_reduce and B's stamp reads probed=False locally while rank A
    reads True; a witness premise derived from the local stamp would then
    diverge across ranks. The annotation must come from the reduced vector.
    """

    def setUp(self):
        self.addCleanup(binding_state().reset)

    @staticmethod
    def _packed_for(tree, operation, hash_value):
        probed, hit = tree._reap_annotation_local(operation, hash_value)
        packed = [0] * _REAP_PACKED_LEN
        packed[_REAP_SLOT_PROBED] = probed
        packed[_REAP_SLOT_HIT_TOKENS] = hit
        return packed

    def test_two_ranks_one_stamped_one_not_both_derive_unprobed(self):
        """THE MATCHED CHECK: the MIN over the two ranks' proposed slots,
        read back through `_reap_annotation_from_packed` on BOTH ranks."""
        tree = _tree_stub(*_parsed_defaults())
        stamped = _unprobed_operation(REAP_TOKENS, age_s=0.0)
        stamped.probed_hit_tokens = REAP_TOKENS  # rank A: thread stamped
        unstamped = _unprobed_operation(REAP_TOKENS, age_s=0.0)  # rank B: not yet
        a = self._packed_for(tree, stamped, [])
        b = self._packed_for(tree, unstamped, [])
        self.assertEqual((a[_REAP_SLOT_PROBED], a[_REAP_SLOT_HIT_TOKENS]), (1, REAP_TOKENS))
        self.assertEqual((b[_REAP_SLOT_PROBED], b[_REAP_SLOT_HIT_TOKENS]), (0, 0))
        reduced = torch.tensor([min(x, y) for x, y in zip(a, b)], dtype=torch.int)
        for rank_name in ("A", "B"):
            with self.subTest(rank=rank_name):
                probed, hit = UnifiedRadixCache._reap_annotation_from_packed(reduced)
                self.assertFalse(probed)
                self.assertEqual(hit, 0)

    def test_the_slots_sit_after_the_pool_slots(self):
        self.assertEqual(_REAP_SLOT_PROBED, 1 + _POOL_SLOT_COUNT)
        self.assertEqual(_REAP_SLOT_HIT_TOKENS, 2 + _POOL_SLOT_COUNT)
        self.assertEqual(_REAP_PACKED_LEN, 3 + _POOL_SLOT_COUNT)

    def test_the_real_reap_records_the_groups_reading_not_the_local_stamp(self):
        """Drive the real `check_prefetch_progress` with a stamped LOCAL
        operation under a two-rank stub whose peer proposes probed=0 (the
        all_reduce stub zeroes the two slots as a MIN with that peer would):
        the recorded outcome must say probed=False / hit_tokens=0, and the
        REAPED line must print the group's reading."""
        cache, operation = _reap_scenario(probed=False)
        operation.probed_hit_tokens = REAP_TOKENS  # this rank's thread stamped
        cache.tp_world_size = 2
        seen = {}

        def _peer_min(packed, op, label=""):
            # Only the check_prefetch_progress vector carries the two slots;
            # the terminate clock's MAX and the #580 vote pair pass through
            # untouched (a MIN with a single agreeing peer is the identity).
            if label != "check_prefetch_progress":
                return
            seen["len"] = int(packed.numel())
            seen["proposed"] = (
                int(packed[_REAP_SLOT_PROBED].item()),
                int(packed[_REAP_SLOT_HIT_TOKENS].item()),
            )
            packed[_REAP_SLOT_PROBED] = 0
            packed[_REAP_SLOT_HIT_TOKENS] = 0

        cache._all_reduce_attn_groups = _peer_min
        with self.assertLogs("sglang.srt.mem_cache.unified_radix_cache", "WARNING") as cm:
            self.assertTrue(cache.check_prefetch_progress(REAP_REQ))
        self.assertEqual(seen["len"], _REAP_PACKED_LEN)
        self.assertEqual(seen["proposed"], (1, REAP_TOKENS))
        outcome = cache.prefetch_loaded_tokens_by_reqid[REAP_REQ]
        self.assertFalse(outcome.probed)
        self.assertEqual(outcome.hit_tokens, 0)
        self.assertIn("#1157 PREFETCH REAPED req=req-1157 probed=False", "\n".join(cm.output))

    def test_single_rank_keeps_its_own_reading(self):
        """tp_world_size == 1: the local proposal IS the reduced vector."""
        cache, operation = _reap_scenario(probed=False)
        operation.probed_hit_tokens = 0  # probed, answered nothing
        cache.check_prefetch_progress(REAP_REQ)
        outcome = cache.prefetch_loaded_tokens_by_reqid[REAP_REQ]
        self.assertTrue(outcome.probed)
        self.assertEqual(outcome.hit_tokens, 0)


if __name__ == "__main__":
    unittest.main()
