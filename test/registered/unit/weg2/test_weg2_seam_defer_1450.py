# SPDX-License-Identifier: Apache-2.0
"""#1450: the seam-digest grade leaves the wake RPC.

py-spy on TP0 (boot weg2xsn206, flip P->D, 75 s at 50 Hz): 47 % of the
samples sat in the fold (_seam_fold.fold_piece) while the wake was on the
clock.  Now _weg2_seam_digest_after hands join + assemble + compare + verdict
to a finisher thread and returns; a refusal is parked on the object and
raised at this rank's next leg (release/resume head) or idle tick -- later
and named, never lost.  SGLANG_WEG2_SEAM_DIGEST_DEFER=0 restores the sync
form.  The fold block grows 4 -> 16 MiB (SGLANG_WEG2_SEAM_CHUNK_BYTES).

Hermetic: fake updater with the grade monkeypatched; AST pins for the three
raise sites.
"""
import inspect
import os
import threading
import time
import unittest
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers.scheduler_components import weight_updater as wu
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

M = wu.SchedulerWeightUpdaterManager


def _fake(threads, finish):
    f = SimpleNamespace(
        weg2_seam_before="BEFORE", weg2_seam_pending_refusal=None, weg2_seam_ref=None,
        _weg2_seam_after_threads=threads, _weg2_seam_after_parts={},
        _weg2_group_name=lambda: "D", _weg2_rank=lambda: 0, _weg2_device_index=lambda: 1,
        _weg2_seam_inventory=lambda: ([("w", None)], [], 1, ""),
        _weg2_seam_digest_finish=finish,
    )
    f._weg2_seam_finisher = None
    f._weg2_raise_pending_seam_refusal = lambda **kw: M._weg2_raise_pending_seam_refusal(f, **kw)
    return f


class Deferred(CustomTestCase):
    def setUp(self):
        self._armed = wu.seam_digest.seam_digest_armed
        wu.seam_digest.seam_digest_armed = lambda *a, **k: True
        os.environ.pop("SGLANG_WEG2_SEAM_DIGEST_DEFER", None)

    def tearDown(self):
        wu.seam_digest.seam_digest_armed = self._armed
        os.environ.pop("SGLANG_WEG2_SEAM_DIGEST_DEFER", None)

    def _run(self, refusal):
        seen = {}
        done = threading.Event()

        def finish(**kw):
            seen.update(kw)
            done.set()
            return refusal

        th = threading.Thread(target=lambda: None)
        th.start()
        f = _fake([th], finish)
        t0 = time.perf_counter()
        M._weg2_seam_digest_after(f, SimpleNamespace(epoch=7), ["weights"])
        rpc_ms = (time.perf_counter() - t0) * 1000
        self.assertTrue(done.wait(5.0))
        for _ in range(50):
            if f.weg2_seam_pending_refusal is refusal:
                break
            time.sleep(0.02)
        return f, seen, rpc_ms

    def test_refusal_is_parked_and_raised_at_the_next_leg(self):
        boom = RuntimeError("W-seam MISMATCH")
        f, seen, _ = self._run(boom)
        self.assertIs(f.weg2_seam_pending_refusal, boom)
        self.assertEqual(seen["before"], "BEFORE")
        self.assertEqual(seen["epoch"], 7)
        self.assertEqual(f._weg2_seam_after_threads, [])       # handed to the finisher
        with self.assertRaises(RuntimeError):
            f._weg2_raise_pending_seam_refusal()
        self.assertIsNone(f.weg2_seam_pending_refusal)          # raised once, then clear
        f._weg2_raise_pending_seam_refusal()                    # idempotent afterwards

    def test_next_leg_joins_the_finisher_before_reading_the_verdict_1450b(self):
        """boot weg2xsn208: the grade was still folding when the next release
        unmapped the pages.  The leg head now waits for the finisher."""
        gate = threading.Event()
        boom = RuntimeError("late MISMATCH")

        def finish(**kw):
            gate.wait(5.0)
            return boom

        th = threading.Thread(target=lambda: None)
        th.start()
        f = _fake([th], finish)
        M._weg2_seam_digest_after(f, SimpleNamespace(epoch=3), ["weights"])
        self.assertIsNotNone(f._weg2_seam_finisher)
        self.assertTrue(f._weg2_seam_finisher.is_alive())
        # idle tick: never blocks, grade not ready -> nothing raised
        f._weg2_raise_pending_seam_refusal(join=False)
        self.assertIsNone(f.weg2_seam_pending_refusal)
        # leg head: joins, then raises the refusal the grade produced
        gate.set()
        with self.assertRaises(RuntimeError):
            f._weg2_raise_pending_seam_refusal()
        self.assertIsNone(f._weg2_seam_finisher)

    def test_match_parks_nothing(self):
        f, _, _ = self._run(None)
        time.sleep(0.05)
        self.assertIsNone(f.weg2_seam_pending_refusal)

    def test_sync_form_still_raises_on_the_rpc(self):
        os.environ["SGLANG_WEG2_SEAM_DIGEST_DEFER"] = "0"
        boom = RuntimeError("W-seam MISMATCH")
        th = threading.Thread(target=lambda: None)
        th.start()
        f = _fake([th], lambda **kw: boom)
        with self.assertRaises(RuntimeError):
            M._weg2_seam_digest_after(f, SimpleNamespace(epoch=1), ["weights"])


class Wiring(CustomTestCase):
    def test_raise_sites(self):
        src = inspect.getsource(wu)
        self.assertEqual(src.count("self._weg2_raise_pending_seam_refusal()  # #1450"), 2)
        rel = inspect.getsource(M.release_memory_occupation)
        res = inspect.getsource(M.resume_memory_occupation)
        for body in (rel, res):
            self.assertLess(body.index("_weg2_raise_pending_seam_refusal()"), body.index("_weg2_leg_replay("))
        from sglang.srt.managers.scheduler import Scheduler
        idle = inspect.getsource(Scheduler.on_idle)
        self.assertIn("_wu_chk(join=False)", idle)
        # slots=True dataclass: the field is declared (the #1437b lesson)
        self.assertIn("weg2_seam_pending_refusal", M.__dataclass_fields__)
        self.assertIn("_weg2_seam_finisher", M.__dataclass_fields__)

    def test_index_cache_released_after_a_reading_1454(self):
        from sglang.srt.weg2 import _seam_fold, seam_digest
        _seam_fold._IDX_CACHE[("cpu", 7)] = (__import__("torch").arange(7), __import__("torch").arange(7))
        self.assertGreater(_seam_fold.release_index_cache(), 0)
        self.assertEqual(_seam_fold._IDX_CACHE, {})
        src = inspect.getsource(seam_digest)
        self.assertEqual(src.count("_fold.release_index_cache()"), 2)

    def test_fold_block_default(self):
        from sglang.srt.weg2 import _seam_fold
        if not os.environ.get("SGLANG_WEG2_SEAM_CHUNK_BYTES"):
            self.assertEqual(_seam_fold.DEFAULT_CHUNK_BYTES, 16 << 20)


if __name__ == "__main__":
    unittest.main()
