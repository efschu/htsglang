# SPDX-License-Identifier: Apache-2.0
"""27B D rounds (24.09.): HiCache work between two D decode rounds, behind switches.

User order: HiCache work "darf nie den laufenden prefill oder decode
beeintraechtigen". Two things sat in the path between two D rounds:

1. `check_hicache_events` -> `drain_storage_control_queues` MIN-reduces the
   storage queue sizes over the attention group on EVERY scheduler round -- on
   group D (TP 3) one gloo all_reduce per decode round, parked on its own waiter
   thread, lock-stepping the three schedulers, for queues that are empty through
   a decode phase. SGLANG_HICACHE_DRAIN_AGREE_EVERY=N runs it on a rank-uniform
   cadence instead (every round while it drains, every N-th while it finds
   nothing).
2. The load-back's index copies block the scheduler thread, and since upstream
   #36738 the load stream waits for the forward in flight: a request admitted
   with a store hit between two D rounds holds the thread until the running
   decode forward is done. SGLANG_HICACHE_LOAD_ASYNC_INDEX=1 moves those index
   tensors through pinned memory / selects them on the device.

Both default OFF; what must hold: off is the unchanged path, on is rank-uniform
and moves the same bytes into the same rows.
"""

import os
import types
import unittest
from queue import Queue
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch

from sglang.srt.mem_cache import unified_radix_cache as u
from sglang.srt.mem_cache.pool_host import arena_mamba_pool as amp
from sglang.srt.mem_cache.pool_host import arena_pool as ap
from sglang.test.test_utils import CustomTestCase

_ENVS = ("SGLANG_HICACHE_DRAIN_AGREE_EVERY", "SGLANG_HICACHE_ROUND_TIMING",
         "SGLANG_HICACHE_LOAD_ASYNC_INDEX")


class _EnvCase(CustomTestCase):
    def setUp(self):
        super().setUp()
        self._saved = {k: os.environ.pop(k, None) for k in _ENVS}

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _cache(tp_world_size=3):
    """A UnifiedRadixCache with only what check_hicache_events reads; the ack
    polls and the PP reap are stubbed (they are not what this changes)."""
    c = u.UnifiedRadixCache.__new__(u.UnifiedRadixCache)
    c.attn_cp_group = None
    c.attn_tp_group = None
    c.tp_world_size = tp_world_size
    c.enable_storage = True
    c.enable_storage_metrics = False
    c.storage_metrics_collector = None
    c._pin_trace_every = 0
    c._drain_async_work = lambda: None
    c.writing_check = lambda *a, **k: None
    c.loading_check = lambda: None
    return c


class TheDefaultIsTheUnchangedPath(_EnvCase):
    def test_unset_drains_on_every_round(self):
        c = _cache()
        c.drain_storage_control_queues = mock.Mock(return_value=False)
        c._gated_drain_storage_control_queues = mock.Mock()
        for _ in range(7):
            c.check_hicache_events()
        self.assertEqual(c.drain_storage_control_queues.call_count, 7)
        c._gated_drain_storage_control_queues.assert_not_called()
        self.assertFalse(hasattr(c, "_drain_gate_round"))
        self.assertFalse(hasattr(c, "_hc_round_acc"))

    def test_every_1_is_the_default(self):
        os.environ["SGLANG_HICACHE_DRAIN_AGREE_EVERY"] = "1"
        c = _cache()
        c.drain_storage_control_queues = mock.Mock(return_value=False)
        for _ in range(5):
            c.check_hicache_events()
        self.assertEqual(c.drain_storage_control_queues.call_count, 5)

    def test_garbage_and_nonpositive_values_are_every_round(self):
        for v in ("x", "0", "-3", ""):
            os.environ["SGLANG_HICACHE_DRAIN_AGREE_EVERY"] = v
            self.assertEqual(u._hicache_drain_agree_every(), 1, v)
        os.environ["SGLANG_HICACHE_ROUND_TIMING"] = "x"
        self.assertEqual(u._hicache_round_timing_every(), 0)

    def test_group_p_takes_no_collective_so_the_gate_never_engages(self):
        """TP 1 / PP 3: the drain reduces over nobody -- a gate there would only
        delay P's local drain, so P keeps draining every round even when the
        variable reaches its environment."""
        os.environ["SGLANG_HICACHE_DRAIN_AGREE_EVERY"] = "8"
        c = _cache(tp_world_size=1)
        self.assertFalse(c._drain_agreement_is_collective())
        c.drain_storage_control_queues = mock.Mock(return_value=False)
        for _ in range(9):
            c.check_hicache_events()
        self.assertEqual(c.drain_storage_control_queues.call_count, 9)


class TheDrainReportsWhatTheGroupAgreed(_EnvCase):
    def _cc(self):
        return types.SimpleNamespace(prefetch_revoke_queue=Queue(), ack_backup_queue=Queue(),
                                     host_mem_release_queue=Queue(),
                                     extra_host_mem_release_queues={})

    def test_empty_queues_agree_on_nothing_and_drain_zero(self):
        c = _cache()
        c.cache_controller = self._cc()
        c._all_reduce_attn_groups = lambda t, op, label="": None  # one rank: MIN is itself
        c._drain_storage_control_queues_impl = mock.Mock()
        self.assertFalse(c.drain_storage_control_queues())
        kw = c._drain_storage_control_queues_impl.call_args.kwargs
        # fnFL2 H74 (x172/x174): the backup acks drain rank-locally (None = every
        # ready ack of THIS rank); `hot` and the revoke/release counts stay the MIN
        self.assertEqual((kw["n_revoke"], kw["n_backup"], kw["n_release"]), (0, None, 0))

    def test_a_pending_ack_is_agreed_and_drained_with_the_same_count(self):
        c = _cache()
        c.cache_controller = self._cc()
        c.cache_controller.ack_backup_queue.put("op1")
        c.cache_controller.ack_backup_queue.put("op2")
        c._all_reduce_attn_groups = lambda t, op, label="": None
        c._drain_storage_control_queues_impl = mock.Mock()
        self.assertTrue(c.drain_storage_control_queues())
        kw = c._drain_storage_control_queues_impl.call_args.kwargs
        self.assertEqual((kw["n_revoke"], kw["n_backup"], kw["n_release"]), (0, None, 0))

    def test_switched_off_the_backup_acks_drain_the_agreed_count(self):
        from sglang.srt.environ import envs

        c = _cache()
        c.cache_controller = self._cc()
        c.cache_controller.ack_backup_queue.put("op1")
        c.cache_controller.ack_backup_queue.put("op2")
        c._all_reduce_attn_groups = lambda t, op, label="": None
        c._drain_storage_control_queues_impl = mock.Mock()
        with envs.SGLANG_WEG2_ENABLE_LOCAL_BACKUP_ACK_DRAIN.override(False):
            self.assertTrue(c.drain_storage_control_queues())
        kw = c._drain_storage_control_queues_impl.call_args.kwargs
        self.assertEqual((kw["n_revoke"], kw["n_backup"], kw["n_release"]), (0, 2, 0))


class TheCadenceIsRankUniform(_EnvCase):
    """Three D ranks whose queues fill at DIFFERENT rounds. Every rank must
    enter the agreement on exactly the same rounds -- one rank alone in a gloo
    all_reduce is the wedge -- and drain the same MIN."""

    def _run(self, every, arrivals, rounds=120):
        os.environ["SGLANG_HICACHE_DRAIN_AGREE_EVERY"] = str(every)
        ranks = [_cache() for _ in range(3)]
        pending = [0, 0, 0]
        entered = [[] for _ in range(3)]
        drained = [[] for _ in range(3)]
        cur = {"round": 0, "mins": None}

        def make_drain(i):
            def drain():
                if cur["mins"] is None:  # the first rank into this round's collective
                    cur["mins"] = min(pending)
                agreed = cur["mins"]
                entered[i].append(cur["round"])
                drained[i].append(agreed)
                pending[i] -= agreed
                return agreed > 0
            return drain

        for i, r in enumerate(ranks):
            r.drain_storage_control_queues = make_drain(i)
        for rnd in range(1, rounds + 1):
            cur["round"], cur["mins"] = rnd, None
            for i in range(3):
                pending[i] += arrivals(i, rnd)
            for r in ranks:
                r.check_hicache_events()
        return ranks, entered, drained, pending

    def test_every_rank_enters_on_the_same_rounds_with_the_same_min(self):
        def arrivals(i, rnd):  # rank i sees each ack i rounds late; bursts at 10, 50
            return 1 if rnd in (10 + i, 11 + i, 50 + 2 * i) else 0

        ranks, entered, drained, pending = self._run(8, arrivals)
        self.assertEqual(entered[0], entered[1])
        self.assertEqual(entered[1], entered[2])
        self.assertEqual(drained[0], drained[1])
        self.assertEqual(drained[1], drained[2])
        self.assertEqual(pending, [0, 0, 0], "an agreed ack was never drained")
        self.assertLess(len(entered[0]), 120 // 4, "the cadence did not thin the collective")

    def test_idle_group_agrees_every_nth_round(self):
        ranks, entered, _d, _p = self._run(8, lambda i, rnd: 0, rounds=64)
        self.assertEqual(entered[0], [1, 9, 17, 25, 33, 41, 49, 57])

    def test_a_draining_group_stays_hot_every_round(self):
        # all ranks hold work at every round -> the MIN is > 0 -> agree every round
        _r, entered, _d, _p = self._run(8, lambda i, rnd: 1, rounds=30)
        self.assertEqual(entered[0], list(range(1, 31)))


class TheRoundTimerOnlyMeasures(_EnvCase):
    def test_summary_every_n_rounds_and_no_state_when_off(self):
        os.environ["SGLANG_HICACHE_ROUND_TIMING"] = "4"
        c = _cache()
        c.drain_storage_control_queues = mock.Mock(return_value=False)
        with self.assertLogs(u.logger, level="INFO") as cm:
            for _ in range(8):
                c.check_hicache_events()
        lines = [m for m in cm.output if "HICACHE-ROUND-TIMING" in m]
        self.assertEqual(len(lines), 2)
        self.assertIn("rounds=4", lines[0])
        self.assertEqual(c.drain_storage_control_queues.call_count, 8)


class TheAsyncIndexMovesTheSameRows(_EnvCase):
    def test_index_helpers_equal_the_blocking_forms(self):
        t = torch.tensor([7, 3, 9, 1, 4], dtype=torch.int32)
        mask = torch.tensor([True, False, True, True, False])
        self.assertTrue(torch.equal(ap._select_rows_async(t, mask), t[mask]))
        got = ap.index_to_device_async(t, "cpu")
        self.assertEqual(got.dtype, torch.int64)
        self.assertTrue(torch.equal(got, t.to(torch.int64)))

    def test_the_arena_transfer_hands_the_kernel_the_same_indices(self):
        """`_transfer` (KV and draft rows alike): the switch changes how the
        index crosses to the card, never which rows the kernel reads/writes."""
        seen = {}
        dst_k = torch.zeros(8, 2)
        dp = types.SimpleNamespace(k_buffer=[dst_k], v_buffer=[torch.zeros(8, 2)])
        for flag in ("0", "1"):
            os.environ["SGLANG_HICACHE_LOAD_ASYNC_INDEX"] = flag
            p = ap.ArenaMHAHostPool.__new__(ap.ArenaMHAHostPool)
            p.can_use_jit = True
            p.element_dim = 2
            kern = mock.Mock()
            with mock.patch("sglang.jit_kernel.hicache.transfer_hicache_one_layer", kern):
                p._transfer(dp, object(), object(), torch.tensor([4, 0, 2], dtype=torch.int32),
                            torch.tensor([1, 6, 3]), 0)
            kw = kern.call_args.kwargs
            seen[flag] = (kw["indices_src"].tolist(), kw["indices_dst"].tolist(),
                          kw["indices_src"].dtype, kw["indices_dst"].dtype)
        self.assertEqual(seen["0"], seen["1"])
        self.assertEqual(seen["1"][2], torch.int64)

    def test_the_state_loader_lands_the_same_bytes(self):
        """The xsn335 synthetic blob, loaded with the switch off and on."""
        L, A = 2, 8
        t_shape, width, conv_shape, e = (2, 4), 3, (6, 3), 2
        t_row, c_row = 8 * e, 6 * width * e
        slot_bytes = L * (t_row + c_row)
        blob = torch.randint(0, 255, (A, slot_bytes), dtype=torch.uint8)
        t_ext, c_ext, off = [], [], 0
        for _l in range(L):
            t_ext.append((off, t_row)); off += t_row
            segs = []
            for n_j in (2, 3, 1):
                segs.append((off, n_j * width * e)); off += n_j * width * e
            c_ext.append(segs)
        results = {}
        for flag in ("0", "1"):
            os.environ["SGLANG_HICACHE_LOAD_ASYNC_INDEX"] = flag
            pool = amp.ArenaMambaPoolHost.__new__(amp.ArenaMambaPoolHost)
            pool._slot_view = blob
            pool._page_bytes = slot_bytes
            pool._state_stage = None
            pool.temporal_dtype = torch.bfloat16
            pool.conv_dtype = torch.bfloat16
            pool._layout = {"L": L, "t_shape": t_shape, "conv_shape": conv_shape,
                            "width": width, "t_ext": t_ext, "c_ext": c_ext}
            temporal = [torch.zeros((16,) + t_shape, dtype=torch.bfloat16) for _ in range(L)]
            conv = [torch.zeros((16,) + conv_shape, dtype=torch.bfloat16) for _ in range(L)]
            dp = types.SimpleNamespace(mamba_cache=types.SimpleNamespace(temporal=temporal, conv=[conv]))
            pool._load_states_all_layers(dp, torch.tensor([5, 1, 7]), torch.tensor([3, 9, 0]))
            results[flag] = [t.clone() for t in temporal] + [c.clone() for c in conv]
        # bitwise: random bytes read as bf16 contain NaN patterns, and
        # torch.equal on NaN is False even for identical bits
        for a, b in zip(results["0"], results["1"]):
            self.assertTrue(torch.equal(a.view(torch.int16), b.view(torch.int16)))
        self.assertTrue(any(bool(t.view(torch.int16).any()) for t in results["1"]))


if __name__ == "__main__":
    unittest.main()
