# SPDX-License-Identifier: Apache-2.0
"""Warmup flashinfer autotune is GROUP-UNIFORM on a heterogeneous TP group.

rc9b D (RC9 e914e89fde, 25.09. 19:53Z): TP0 = 5090 (flashinfer_cutlass,
should_run_flashinfer_autotune -> True), TP1/TP2 = 3080 (w4a8_int8, sm_86 ->
False). TP0 entered the autotune dummy forward alone; its
enter_capture_group_barrier had no partner and barlink raised
"group barrier (tp:0) made no progress" after 120 s.

Two real CPU processes on a gloo group play TP0 (tunes) and TP1 (does not).
The dummy forward is stood in for by what makes it a TP forward -- a group
collective -- and every collective each rank issues is an all_gather of its
own label, so a mis-pairing is DETECTED (labels differ) instead of hanging.
Both ranks must issue the identical collective sequence, the tuning rank
under the autotuner, the other one untuned.
"""

from __future__ import annotations

import os
import socket
import types
import unittest
from datetime import timedelta

import torch.multiprocessing as mp

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _worker(rank, port, tunes, out):
    import torch.distributed as dist

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=15),
    )
    from unittest import mock

    from sglang.srt.model_executor.runner import base_runner as BR

    group = dist.new_group([0, 1], backend="gloo", timeout=timedelta(seconds=15))
    seq = []
    tuned = []

    def collective(label):
        got = [None, None]
        dist.all_gather_object(got, label, group=group)
        seq.append(label)
        if got[0] != got[1]:
            raise RuntimeError(f"MISPAIRED collective on rank {rank}: {got}")

    mr = types.SimpleNamespace(
        device="cuda",
        canary_manager=None,
        is_draft_kv_only_producer=False,
        tp_group=types.SimpleNamespace(cpu_group=group, world_size=2),
    )
    runner = types.SimpleNamespace(
        model_runner=mr,
        _pre_initialize_flashinfer_allreduce_workspace=lambda: None,
        _autotune_buffers=lambda: (object(), 4),
        _dummy_run=lambda **kw: (collective("tp-barrier"), collective("tp-allreduce")),
    )
    runner._flashinfer_autotune = types.MethodType(BR.BaseRunner._flashinfer_autotune, runner)

    def fake_run(model_runner, fn, **kw):
        tuned.append(kw.get("tune", True))
        fn()

    err = None
    try:
        with mock.patch.object(BR, "should_run_flashinfer_autotune", lambda m: tunes[rank]), mock.patch.object(
            BR, "run_flashinfer_autotune_forward", side_effect=fake_run
        ):
            BR.BaseRunner.warmup(runner)
        collective("after-warmup: decode graph capture")
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {str(e)[:200]}"
    out.put((rank, seq, tuned, err))
    try:
        dist.destroy_process_group()
    except Exception:  # noqa: BLE001
        pass


def _run(tunes):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = _free_port()
    ps = [ctx.Process(target=_worker, args=(r, port, tunes, q)) for r in (0, 1)]
    for p in ps:
        p.start()
    res = {}
    for _ in ps:
        r, seq, tuned, err = q.get(timeout=90)
        res[r] = (seq, tuned, err)
    for p in ps:
        p.join(timeout=30)
        if p.is_alive():
            p.kill()
    return res


class TestGroupUniformAutotune(unittest.TestCase):
    def test_heterogeneous_group_runs_one_paired_forward(self):
        """TP0 tunes, TP1 does not: same collective sequence, TP1 untuned."""
        res = _run({0: True, 1: False})
        (s0, t0, e0), (s1, t1, e1) = res[0], res[1]
        self.assertIsNone(e0, e0)
        self.assertIsNone(e1, e1)
        self.assertEqual(s0, s1)
        self.assertEqual(s0, ["tp-barrier", "tp-allreduce", "after-warmup: decode graph capture"])
        self.assertEqual(t0, [True])
        self.assertEqual(t1, [False])

    def test_nobody_tunes_means_no_dummy_forward(self):
        res = _run({0: False, 1: False})
        for r in (0, 1):
            seq, tuned, err = res[r]
            self.assertIsNone(err, err)
            self.assertEqual(seq, ["after-warmup: decode graph capture"])
            self.assertEqual(tuned, [])

    def test_everybody_tunes_unchanged(self):
        res = _run({0: True, 1: True})
        for r in (0, 1):
            seq, tuned, err = res[r]
            self.assertIsNone(err, err)
            self.assertEqual(tuned, [True])
        self.assertEqual(res[0][0], res[1][0])


class TestLocalShortcuts(unittest.TestCase):
    def test_single_rank_and_rank_local_runners_issue_no_collective(self):
        from sglang.srt.model_executor.runner.flashinfer_autotune import (
            agree_flashinfer_autotune_across_group as agree,
        )

        boom = types.SimpleNamespace(cpu_group=object(), world_size=2)
        solo = types.SimpleNamespace(spec_solo_rank_local_graphs=True, tp_group=boom)
        self.assertTrue(agree(solo, True))
        self.assertFalse(agree(solo, False))
        one = types.SimpleNamespace(tp_group=types.SimpleNamespace(cpu_group=object(), world_size=1))
        self.assertTrue(agree(one, True))
        self.assertFalse(agree(one, False))
        self.assertFalse(agree(types.SimpleNamespace(tp_group=None), False))


if __name__ == "__main__":
    unittest.main()
