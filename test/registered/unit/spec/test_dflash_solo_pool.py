"""Unit tests for T156 task D: the small DFLASH solo draft-KV pool.

CPU-only tests of the pure pieces: the global->draft slot mapper
(allocation, read/write translation, hole semantics, free-listener drain,
clear reset, LRU reclaim, readable exhaustion error) and the ctx-cap
resolution over the cross-algo shapes stash. GPU behavior (pool shrink,
corridor, DFLASH function on the small pool) is covered by the live
validation protocol, not here.
"""

import os
import types
import unittest

import torch

from sglang.srt.speculative.dflash_solo_pool import (
    SOLO_POOL_CAP_ENV,
    DraftKVSlotMapper,
    resolve_dflash_solo_pool_cap,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


def _mapper(num_global=1000, num_draft=16, cap=64):
    return DraftKVSlotMapper(
        num_global_slots=num_global,
        num_draft_slots=num_draft,
        ctx_cap=cap,
        device="cpu",
    )


def _t(vals):
    return torch.tensor(vals, dtype=torch.int64)


class TestMapperBasics(CustomTestCase):
    def test_write_allocates_and_read_translates(self):
        m = _mapper()
        d = m.translate_write(_t([100, 200, 300]))
        self.assertEqual(d.shape, (3,))
        # Distinct fresh globals -> distinct non-hole draft slots.
        self.assertEqual(len(set(d.tolist())), 3)
        self.assertNotIn(0, d.tolist())
        # A later read of the same globals sees the same slots.
        r = m.translate_read(_t([300, 100, 200]))
        self.assertEqual(r.tolist(), [d[2], d[0], d[1]])
        # Re-write hits the existing mapping (no growth).
        d2 = m.translate_write(_t([100, 200]))
        self.assertEqual(d2.tolist(), [d[0], d[1]])
        self.assertEqual(m.stats()["mapped"], 3)

    def test_global_slot_zero_is_hole(self):
        m = _mapper()
        self.assertEqual(m.translate_read(_t([0])).tolist(), [0])
        self.assertEqual(m.translate_write(_t([0])).tolist(), [0])

    def test_unmapped_read_returns_hole_and_counts(self):
        m = _mapper()
        m.translate_write(_t([5]))
        r = m.translate_read(_t([5, 77, 88]))
        self.assertEqual(r[1].item(), 0)
        self.assertEqual(r[2].item(), 0)
        self.assertNotEqual(r[0].item(), 0)
        self.assertEqual(m.holes_read_total, 2)

    def test_valid_mask_blocks_allocation(self):
        m = _mapper()
        locs = torch.tensor([[10, 11, 12], [20, 21, 22]], dtype=torch.int64)
        valid = torch.tensor([[True, True, False], [False, False, False]])
        d = m.translate_write(locs, valid=valid)
        self.assertEqual(d.shape, (2, 3))
        self.assertNotEqual(d[0, 0].item(), 0)
        self.assertNotEqual(d[0, 1].item(), 0)
        # Invalid rows -> hole, nothing allocated for them.
        self.assertEqual(d[0, 2].item(), 0)
        self.assertEqual(d[1].tolist(), [0, 0, 0])
        self.assertEqual(m.stats()["mapped"], 2)

    def test_free_listener_recycles_via_drain(self):
        m = _mapper(num_draft=4)  # slots 1..3 usable
        d = m.translate_write(_t([10, 11, 12]))
        self.assertEqual(m.stats()["free"], 0)
        # Scheduler thread frees two globals; applied at the next translate.
        m.on_global_free(_t([10, 12]))
        r = m.translate_write(_t([50, 51]))  # needs the recycled slots
        self.assertNotIn(0, r.tolist())
        self.assertEqual(m.stats()["mapped"], 3)
        # The freed globals are unmapped now.
        self.assertEqual(m.translate_read(_t([10])).tolist(), [0])
        # Slot of the surviving global unchanged.
        self.assertEqual(m.translate_read(_t([11])).tolist(), [d[1].item()])

    def test_free_ignores_unmapped_and_out_of_range(self):
        m = _mapper()
        m.translate_write(_t([10]))
        m.on_global_free(_t([999999, -3, 55]))  # none mapped / out of range
        m.translate_read(_t([10]))  # drains without error
        self.assertEqual(m.stats()["mapped"], 1)

    def test_clear_resets_everything(self):
        m = _mapper(num_draft=4)
        m.translate_write(_t([10, 11, 12]))
        m.on_global_clear()
        r = m.translate_read(_t([10]))  # drains the reset
        self.assertEqual(r.tolist(), [0])
        s = m.stats()
        self.assertEqual(s["mapped"], 0)
        self.assertEqual(s["free"], 3)

    def test_lru_reclaim_prefers_oldest(self):
        m = _mapper(num_draft=5)  # 4 usable slots
        m.begin_round()
        m.translate_write(_t([1, 2]))  # round 1: old
        m.begin_round()
        m.translate_write(_t([3, 4]))  # round 2: newer
        m.begin_round()
        # Pool full; two more allocations force a reclaim of the oldest.
        d = m.translate_write(_t([5, 6]))
        self.assertNotIn(0, d.tolist())
        self.assertGreaterEqual(m.reclaim_events, 1)
        # The oldest entries (globals 1, 2) were dropped.
        self.assertEqual(m.translate_read(_t([1, 2])).tolist(), [0, 0])
        # The newer entries survive unless the sweep needed them.
        self.assertEqual(m.stats()["mapped"], 4)

    def test_exhaustion_raises_readable_error(self):
        m = _mapper(num_draft=4)
        m.begin_round()
        m.translate_write(_t([1, 2, 3]))
        # Everything was touched THIS round -> nothing reclaimable.
        with self.assertRaises(RuntimeError) as cm:
            m.translate_write(_t([4, 5]))
        self.assertIn("exhausted", str(cm.exception))

    def test_epoch_protects_current_round_from_reclaim(self):
        m = _mapper(num_draft=4)
        m.begin_round()
        m.translate_write(_t([1]))
        m.begin_round()
        m.translate_write(_t([2, 3]))  # full now
        # Same round: global 1 (older epoch) is the only candidate.
        d = m.translate_write(_t([9]))
        self.assertNotIn(0, d.tolist())
        self.assertEqual(m.translate_read(_t([1])).tolist(), [0])
        r = m.translate_read(_t([2, 3]))
        self.assertNotIn(0, r.tolist())


class _Args:
    def __init__(self, shapes=None):
        if shapes is not None:
            self.speculative_cross_shapes = shapes


class TestConsumptionGate(CustomTestCase):
    """The measured-KV consumption gate's pure verdict, incl. the T156-D
    component-shift bypass (a structural pool release must not freeze the
    correction as 'unconsumed fantasy growth')."""

    @staticmethod
    def _frozen(**kw):
        from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
            ModelRunnerKVCacheMixin,
        )

        base = dict(
            delta_b=1 << 30,
            prev_b=291 << 20,
            prev_leftover_b=4 << 30,
            free_b=8 << 30,
            component_shift=False,
        )
        base.update(kw)
        return ModelRunnerKVCacheMixin.correction_growth_frozen(**base)

    def test_unconsumed_growth_frozen(self):
        # Leftover did not shrink -> freeze (the stock rule).
        self.assertTrue(self._frozen())

    def test_consumed_growth_passes(self):
        # Leftover shrank by > 128 MiB -> previous growth consumed.
        self.assertFalse(self._frozen(free_b=3 << 30, prev_leftover_b=4 << 30))

    def test_component_shift_bypasses_freeze(self):
        # The T156-D case: pool released GiBs, leftover grew structurally.
        self.assertFalse(self._frozen(component_shift=True))

    def test_negative_delta_always_applies(self):
        self.assertFalse(self._frozen(delta_b=-(1 << 30)))

    def test_no_previous_correction_passes(self):
        self.assertFalse(self._frozen(prev_b=0))

    def test_no_previous_leftover_passes(self):
        self.assertFalse(self._frozen(prev_leftover_b=None))


class TestCapResolution(CustomTestCase):
    def test_env_off_and_explicit(self):
        os.environ[SOLO_POOL_CAP_ENV] = "off"
        try:
            cap, _ = resolve_dflash_solo_pool_cap(_Args())
            self.assertIsNone(cap)
        finally:
            del os.environ[SOLO_POOL_CAP_ENV]
        os.environ[SOLO_POOL_CAP_ENV] = "12345"
        try:
            cap, src = resolve_dflash_solo_pool_cap(_Args())
            self.assertEqual(cap, 12345)
            self.assertIn("explicit", src)
        finally:
            del os.environ[SOLO_POOL_CAP_ENV]

    def test_not_cross_algo_disabled(self):
        cap, src = resolve_dflash_solo_pool_cap(_Args())
        self.assertIsNone(cap)
        self.assertIn("full-context", src)

    def test_auto_uses_gate_threshold(self):
        args = _Args({"force": "auto", "ctx_gate": {"threshold": 8192}})
        self.assertEqual(resolve_dflash_solo_pool_cap(args)[0], 8192)
        args = _Args({"force": "auto", "ctx_gate": {"threshold": None}})
        self.assertIsNone(resolve_dflash_solo_pool_cap(args)[0])

    def test_policy_stage_bound(self):
        table = [(0, ("dflash", 16)), (4096, ("nextn", 3))]
        args = _Args(
            {
                "force": "policy",
                "policy_table": table,
                "ctx_gate": {"threshold": 8192},
            }
        )
        self.assertEqual(resolve_dflash_solo_pool_cap(args)[0], 4096)

    def test_policy_gate_caps_stage_bound(self):
        # Stage runs to 8192 but the gate fences at 6000 -> cap 6000.
        table = [(0, ("dflash", 16)), (8192, ("nextn", 3))]
        args = _Args(
            {
                "force": "policy",
                "policy_table": table,
                "ctx_gate": {"threshold": 6000},
            }
        )
        self.assertEqual(resolve_dflash_solo_pool_cap(args)[0], 6000)

    def test_policy_unbounded_stage_disables(self):
        # DFLASH is the LAST stage and the gate is off -> no cap.
        table = [(0, ("nextn", 3)), (4096, ("dflash", 16))]
        args = _Args(
            {
                "force": "policy",
                "policy_table": table,
                "ctx_gate": {"threshold": None},
            }
        )
        self.assertIsNone(resolve_dflash_solo_pool_cap(args)[0])
        # With a gate, the gate bounds the trailing stage.
        args = _Args(
            {
                "force": "policy",
                "policy_table": table,
                "ctx_gate": {"threshold": 8192},
            }
        )
        self.assertEqual(resolve_dflash_solo_pool_cap(args)[0], 8192)

    def test_policy_without_dflash_stage_minimal(self):
        table = [(0, ("nextn", 3))]
        args = _Args(
            {
                "force": "policy",
                "policy_table": table,
                "ctx_gate": {"threshold": 8192},
            }
        )
        self.assertEqual(resolve_dflash_solo_pool_cap(args)[0], 0)

    def test_static_and_schedule_disabled(self):
        for force in ("dflash", "nextn", "schedule"):
            args = _Args({"force": force, "ctx_gate": {"threshold": 8192}})
            self.assertIsNone(resolve_dflash_solo_pool_cap(args)[0])


# ---------------------------------------------------------------------------
# Sync-free mode (SGLANG_DFLASH_WINDOW_POOL_SYNC_FREE, window pool, default off)
# ---------------------------------------------------------------------------


def _free_count(m):
    return int(m._counters[0]) if m.sync_free else m._free_count


def _state(m, with_holes=True):
    fc = _free_count(m)
    st = [
        m.map.tolist(),
        m._slot_global.tolist(),
        m._slot_epoch.tolist(),
        fc,
        m._free[:fc].tolist(),
        m.reclaim_events,
        m.reclaimed_slots_total,
    ]
    if with_holes:
        st.append(m.holes_read_total)
    return st


class TestSyncFreeMapperEquivalence(CustomTestCase):
    """The sync-free mapper keeps the LEGACY mapper's state bit for bit.

    Random op sequences (writes with and without a valid mask, 1-D and 2-D,
    reads with holes, frees with duplicates and out-of-range ids, round
    bumps, clears, pools small enough to reclaim and to exhaust) run on a
    legacy and a sync-free mapper side by side; after EVERY op the returned
    slots, the forward map, the reverse map, the epochs, the free count and
    the live free stack must be equal. Mode "off" additionally starves the
    host of snapshots, so the free-count bound only ever shrinks and the
    exact fallback carries every write it cannot prove safe.
    """

    def _run(self, seed, num_global, num_draft, steps, starve):
        g = torch.Generator().manual_seed(seed)
        legacy = DraftKVSlotMapper(num_global, num_draft, 64, device="cpu")
        fast = DraftKVSlotMapper(num_global, num_draft, 64, device="cpu", sync_free=True)
        self.assertTrue(fast.sync_free)
        self.assertFalse(legacy.sync_free)
        if starve:
            fast._stream_mode = "off"

        def rnd(n, lo, hi):
            return torch.randint(lo, hi, (n,), generator=g)

        for step in range(steps):
            op = int(rnd(1, 0, 100))
            if op < 40:
                k = int(rnd(1, 1, 13))
                locs = torch.randperm(num_global + 1, generator=g)[:k].to(torch.int64)
                valid = None
                if int(rnd(1, 0, 2)):
                    valid = rnd(k, 0, 2).to(torch.bool)
                if k % 2 == 0 and int(rnd(1, 0, 2)):
                    locs = locs.view(2, k // 2)
                    valid = None if valid is None else valid.view(2, k // 2)
                outs = []
                for m in (legacy, fast):
                    try:
                        outs.append(m.translate_write(locs.clone(), valid=valid))
                    except RuntimeError as e:
                        outs.append(("raise", "exhausted" in str(e)))
                if isinstance(outs[0], tuple) or isinstance(outs[1], tuple):
                    self.assertEqual(outs[0], outs[1], f"seed {seed} step {step}")
                else:
                    self.assertEqual(outs[0].tolist(), outs[1].tolist())
            elif op < 70:
                k = int(rnd(1, 1, 17))
                locs = rnd(k, 0, num_global + 1)
                a = legacy.translate_read(locs.clone())
                b = fast.translate_read(locs.clone())
                self.assertEqual(a.tolist(), b.tolist())
            elif op < 85:
                k = int(rnd(1, 1, 9))
                ids = rnd(k, -3, num_global + 6)
                if k > 2:
                    ids[1] = ids[0]  # a duplicate
                legacy.on_global_free(ids.clone())
                fast.on_global_free(ids.clone())
            elif op < 97:
                legacy.begin_round()
                fast.begin_round()
            else:
                legacy.on_global_clear()
                fast.on_global_clear()
            self.assertEqual(
                _state(legacy, with_holes=False),
                _state(fast, with_holes=False),
                f"seed {seed} step {step} op {op}",
            )
        ls, fs = legacy.stats(), fast.stats()
        self.assertEqual(ls, fs)
        self.assertEqual(legacy.holes_read_total, fast.holes_read_total)

    def test_roomy_pool(self):
        for seed in range(6):
            self._run(seed, num_global=300, num_draft=200, steps=250, starve=False)

    def test_tight_pool_reclaims_and_exhausts_alike(self):
        for seed in range(6):
            self._run(seed, num_global=120, num_draft=12, steps=250, starve=False)

    def test_starved_bound_falls_back_exactly(self):
        for seed in range(6):
            self._run(seed, num_global=200, num_draft=40, steps=250, starve=True)

    def test_clear_rebuilds_from_host_constants(self):
        # A phase release can hand the mapper's device bytes back arbitrary;
        # the reset must rebuild from host constants, as the legacy one does.
        fresh = DraftKVSlotMapper(500, 40, 64, device="cpu")
        m = DraftKVSlotMapper(500, 40, 64, device="cpu", sync_free=True)
        m.translate_write(_t([3, 4, 5]))
        m._free.fill_(-123456)
        m._counters.fill_(-7)
        m.map.fill_(999)
        m._slot_global.fill_(-9)
        m._slot_epoch.fill_(77)
        m.on_global_clear()
        fresh.on_global_clear()
        a = fresh.translate_write(_t([10, 11, 12]))
        b = m.translate_write(_t([10, 11, 12]))
        self.assertEqual(a.tolist(), b.tolist())
        self.assertEqual(_state(fresh, with_holes=False), _state(m, with_holes=False))
        self.assertEqual(int(m._counters[1]), 0)

    def test_holes_are_counted_like_legacy(self):
        legacy = DraftKVSlotMapper(1000, 16, 64, device="cpu")
        fast = DraftKVSlotMapper(1000, 16, 64, device="cpu", sync_free=True)
        for m in (legacy, fast):
            m.translate_write(_t([5]))
            m.translate_read(_t([5, 77, 88]))
        self.assertEqual(fast.holes_read_total, 2)
        self.assertEqual(legacy.holes_read_total, fast.holes_read_total)

    def test_default_is_the_legacy_mapper(self):
        m = _mapper()
        self.assertFalse(m.sync_free)
        self.assertFalse(hasattr(m, "_counters"))
        self.assertEqual(m._free.numel(), m.num_draft_slots - 1)

    def test_env_switch_defaults_off(self):
        from sglang.srt.environ import envs

        old = os.environ.pop("SGLANG_DFLASH_WINDOW_POOL_SYNC_FREE", None)
        try:
            self.assertFalse(envs.SGLANG_DFLASH_WINDOW_POOL_SYNC_FREE.get())
        finally:
            if old is not None:
                os.environ["SGLANG_DFLASH_WINDOW_POOL_SYNC_FREE"] = old


class TestSyncFreeNeverReadsTheDevice(CustomTestCase):
    """On the META device a tensor has a shape and no data: any host read
    (``.item()``, ``bool()``, boolean-mask indexing, ``unique``) raises. The
    sync-free paths must run there end to end; the legacy mapper must not
    (negative control -- proves the probe can see a host read)."""

    @staticmethod
    def _meta(*shape, dtype=torch.int64):
        return torch.empty(*shape, dtype=dtype, device="meta")

    def test_sync_free_paths_run_on_meta(self):
        m = DraftKVSlotMapper(1000, 256, 64, device="meta", sync_free=True)
        m.on_global_free(self._meta(7))
        m.begin_round()
        d = m.translate_write(self._meta(3, 8), valid=self._meta(3, 8, dtype=torch.bool))
        self.assertEqual(tuple(d.shape), (3, 8))
        self.assertEqual(tuple(m.translate_write(self._meta(24)).shape), (24,))
        self.assertEqual(tuple(m.translate_read(self._meta(40)).shape), (40,))
        rows = m.translate_read_rows(self._meta(3, 40), self._meta(3, 40, dtype=torch.bool))
        self.assertEqual(tuple(rows.shape), (3, 40))
        m.on_global_free(self._meta(5))
        m.translate_read(self._meta(4))  # drains the free on meta, too

    def test_window_rows_rebuild_runs_on_meta(self):
        from sglang.srt.speculative.dflash_solo_pool import (
            rebuild_window_rows_sync_free,
        )

        m = DraftKVSlotMapper(1000, 256, 64, device="meta", sync_free=True)
        rebuild_window_rows_sync_free(
            mapper=m,
            target_req_to_token=self._meta(8, 4096, dtype=torch.int32),
            draft_req_to_token=self._meta(8, 4096, dtype=torch.int32),
            req_pool_indices=self._meta(3),
            start=self._meta(3),
            lengths=self._meta(3, dtype=torch.int32),
            max_len=2048,
        )

    def test_legacy_mapper_reads_the_device(self):
        m = DraftKVSlotMapper(1000, 256, 64, device="meta")
        with self.assertRaises(Exception):
            m.translate_write(self._meta(24))
        with self.assertRaises(Exception):
            m.translate_read(self._meta(24))


class TestWindowRowsSyncFree(CustomTestCase):
    """rebuild_window_rows_sync_free == the legacy gather -> translate_read ->
    assign chain, per row, and leaves every other column alone."""

    def test_matches_the_legacy_chain(self):
        from sglang.srt.speculative.dflash_solo_pool import (
            rebuild_window_rows_sync_free,
        )

        g = torch.Generator().manual_seed(7)
        num_global, width = 5000, 700
        for trial in range(20):
            legacy = DraftKVSlotMapper(num_global, 3000, 64, device="cpu")
            fast = DraftKVSlotMapper(num_global, 3000, 64, device="cpu", sync_free=True)
            mapped = torch.randperm(num_global, generator=g)[:1500] + 1
            for m in (legacy, fast):
                m.translate_write(mapped.clone())
                m.begin_round()
            target = torch.randint(1, num_global + 1, (16, width), generator=g).to(torch.int32)
            bs = int(torch.randint(1, 7, (1,), generator=g))
            rpi = torch.randperm(16, generator=g)[:bs]
            lengths = torch.randint(0, 300, (bs,), generator=g)
            start = torch.stack(
                [torch.randint(0, width - int(n) + 1, (1,), generator=g)[0] for n in lengths]
            )
            max_len = int(lengths.max()) + int(torch.randint(0, 40, (1,), generator=g))
            sentinel = -7
            draft_ref = torch.full((16, width), sentinel, dtype=torch.int32)
            draft_new = draft_ref.clone()
            # Legacy chain, emulated row by row (the Triton assign writes row
            # b's segment [0, len_b) from the flat translated list, in order).
            flat = torch.cat(
                [target[int(r), int(s) : int(s) + int(n)] for r, s, n in zip(rpi, start, lengths)]
            ).to(torch.int64)
            translated = legacy.translate_read(flat)
            off = 0
            for r, n in zip(rpi, lengths):
                draft_ref[int(r), : int(n)] = translated[off : off + int(n)].to(torch.int32)
                off += int(n)
            rebuild_window_rows_sync_free(
                mapper=fast,
                target_req_to_token=target,
                draft_req_to_token=draft_new,
                req_pool_indices=rpi,
                start=start,
                lengths=lengths.to(torch.int32),
                max_len=max_len,
            )
            self.assertEqual(draft_ref.tolist(), draft_new.tolist(), f"trial {trial}")
            self.assertEqual(legacy.holes_read_total, fast.holes_read_total)
            self.assertEqual(legacy._slot_epoch.tolist(), fast._slot_epoch.tolist())

    def test_rows_path_refuses_the_legacy_mapper(self):
        m = _mapper()
        with self.assertRaises(RuntimeError):
            m.translate_read_rows(
                torch.zeros(1, 4, dtype=torch.int64), torch.ones(1, 4, dtype=torch.bool)
            )

    def test_bound_wider_than_the_table_is_refused(self):
        from sglang.srt.speculative.dflash_solo_pool import (
            rebuild_window_rows_sync_free,
        )

        m = DraftKVSlotMapper(100, 16, 64, device="cpu", sync_free=True)
        with self.assertRaises(RuntimeError):
            rebuild_window_rows_sync_free(
                mapper=m,
                target_req_to_token=torch.zeros(2, 64, dtype=torch.int32),
                draft_req_to_token=torch.zeros(2, 32, dtype=torch.int32),
                req_pool_indices=_t([0]),
                start=_t([0]),
                lengths=_t([8]),
                max_len=40,
            )


class TestWorkerArmsSyncFreeOnlyForTheWindowPool(CustomTestCase):
    """_maybe_init_solo_small_pool builds a sync-free mapper exactly when the
    window pool is on AND SGLANG_DFLASH_WINDOW_POOL_SYNC_FREE=1."""

    def _init(self, env):
        from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
        from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

        alloc = object.__new__(TokenToKVPoolAllocator)
        alloc.size = 70000
        alloc.page_size = 1
        alloc.register_free_listener = lambda on_free, on_clear: None
        fake = types.SimpleNamespace(
            use_compact_draft_cache=True,
            draft_window_size=2048,
            block_size=8,
            device="cpu",
            _spec_solo_active=True,
            _spec_solo_is_host=True,
            server_args=types.SimpleNamespace(max_running_requests=6),
        )
        cfg = types.SimpleNamespace(max_running_requests=6, max_total_num_tokens=70000)
        keys = ("SGLANG_DFLASH_WINDOW_POOL", "SGLANG_DFLASH_WINDOW_POOL_SYNC_FREE")
        saved = {k: os.environ.get(k) for k in keys}
        try:
            for k in keys:
                os.environ.pop(k, None)
            os.environ.update(env)
            return DFlashWorkerV2._maybe_init_solo_small_pool(fake, cfg, alloc)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_window_pool_default_stays_legacy(self):
        m = self._init({"SGLANG_DFLASH_WINDOW_POOL": "1"})
        self.assertIsNotNone(m)
        self.assertFalse(m.sync_free)
        self.assertEqual(m.num_draft_slots, 1 + (2048 + 8) * 6 * 2)

    def test_window_pool_with_switch_is_sync_free(self):
        m = self._init(
            {"SGLANG_DFLASH_WINDOW_POOL": "1", "SGLANG_DFLASH_WINDOW_POOL_SYNC_FREE": "1"}
        )
        self.assertIsNotNone(m)
        self.assertTrue(m.sync_free)

    def test_switch_without_window_pool_changes_nothing(self):
        m = self._init({"SGLANG_DFLASH_WINDOW_POOL_SYNC_FREE": "1"})
        # compact draft cache without the window pool -> no small pool at all
        self.assertIsNone(m)



# ---------------------------------------------------------------------------
# Radix-dedup draft-row carry (SGLANG_DFLASH_WINDOW_POOL_DEDUP_CARRY, default
# off): the tree keeps its own slots, frees the request's fresh duplicates;
# the fresh draft rows move to the kept slots instead of being dropped.
# ---------------------------------------------------------------------------


class TestDedupCarry(CustomTestCase):
    def _both(self):
        return (
            DraftKVSlotMapper(1000, 16, 64, device="cpu"),
            DraftKVSlotMapper(1000, 16, 64, device="cpu", sync_free=True),
        )

    def test_carry_moves_row_to_kept_slot(self):
        for m in self._both():
            d = m.translate_write(_t([500, 501, 502]))  # fresh D prefill rows
            free0 = _free_count(m)
            # Insert dedup: tree keeps 100..102 (never written), frees 500..502.
            m.on_global_alias(_t([500, 501, 502]), _t([100, 101, 102]))
            m.on_global_free(_t([500, 501, 502]))
            r = m.translate_read(_t([100, 101, 102]))
            self.assertEqual(r.tolist(), d.tolist())  # same draft rows, no hole
            self.assertEqual(m.translate_read(_t([500])).tolist(), [0])
            self.assertEqual(_free_count(m), free0)  # owner change, no free
            self.assertEqual(m._slot_global[d.to(torch.int64)].tolist(), [100, 101, 102])
            self.assertEqual(m.stats()["alias_carried_total"], 3)
            self.assertEqual(m.holes_read_total, 1)  # only the [500] probe

    def test_kept_slot_with_row_keeps_it_and_fresh_row_recycles(self):
        for m in self._both():
            kept = m.translate_write(_t([100]))
            m.translate_write(_t([500]))
            free0 = _free_count(m)
            m.on_global_alias(_t([500]), _t([100]))
            m.on_global_free(_t([500]))
            self.assertEqual(m.translate_read(_t([100])).tolist(), kept.tolist())
            self.assertEqual(_free_count(m), free0 + 1)
            self.assertEqual(m.stats()["alias_carried_total"], 0)

    def test_unmapped_fresh_slot_carries_nothing(self):
        for m in self._both():
            m.on_global_alias(_t([500]), _t([100]))
            m.on_global_free(_t([500]))
            self.assertEqual(m.translate_read(_t([100])).tolist(), [0])
            self.assertEqual(m.stats()["mapped"], 0)

    def test_free_before_alias_is_the_old_loss(self):
        # Queue order is the contract: a free queued BEFORE the alias drops
        # the row exactly as without the carry.
        for m in self._both():
            m.translate_write(_t([500]))
            m.on_global_free(_t([500]))
            m.on_global_alias(_t([500]), _t([100]))
            self.assertEqual(m.translate_read(_t([100])).tolist(), [0])

    def test_mismatched_lengths_ignored_and_clear_drops_queue(self):
        for m in self._both():
            d = m.translate_write(_t([500, 501]))
            m.on_global_alias(_t([500, 501]), _t([100]))  # not element-wise
            self.assertEqual(m.translate_read(_t([500, 501])).tolist(), d.tolist())
            m.on_global_alias(_t([500]), _t([100]))
            m.on_global_clear()
            self.assertEqual(m.translate_read(_t([100])).tolist(), [0])
            self.assertEqual(m.stats()["mapped"], 0)

    def test_legacy_and_sync_free_agree(self):
        a, b = self._both()
        for m in (a, b):
            m.translate_write(_t([500, 501, 502, 7]))
            m.translate_write(_t([100]))
            m.on_global_alias(_t([500, 501, 502]), _t([100, 101, 102]))
            m.on_global_free(_t([500, 501, 502]))
            m.translate_read(_t([100, 101, 102, 7]))
        self.assertEqual(_state(a), _state(b))

    def test_allocator_alias_listener(self):
        from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator

        got = []
        alloc = types.SimpleNamespace()
        for name in ("register_alias_listener", "has_alias_listeners", "notify_alias"):
            setattr(alloc, name, types.MethodType(getattr(BaseTokenToKVPoolAllocator, name), alloc))
        self.assertFalse(alloc.has_alias_listeners())
        alloc.notify_alias(_t([1]), _t([2]))  # nobody subscribed: no call
        alloc.register_alias_listener(lambda s, d: got.append((s.tolist(), d.tolist())))
        self.assertTrue(alloc.has_alias_listeners())
        alloc.notify_alias(_t([1]), _t([2]))
        self.assertEqual(got, [([1], [2])])

    def test_env_switch_defaults_off(self):
        from sglang.srt.environ import envs

        os.environ.pop("SGLANG_DFLASH_WINDOW_POOL_DEDUP_CARRY", None)
        self.assertFalse(envs.SGLANG_DFLASH_WINDOW_POOL_DEDUP_CARRY.get())


if __name__ == "__main__":
    unittest.main()
