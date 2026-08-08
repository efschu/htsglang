# SPDX-License-Identifier: Apache-2.0
"""#631 dual group-set init: the creation-order/manifest pin (operator pin 1).

Real 3-process gloo world, CPU-only (CUDA masked in the workers -- the
cards belong to whoever holds the GPU window, and this test needs none).
Gates:

* green path -- a world that already carries the primary PP topology
  (tp=1, pp=3) builds the secondary flip set (tp=3, dcp=3, pp=1) on every
  rank; both sets are usable side by side (smoke all_reduce over the
  secondary tp group next to the primary pp group);
* manifest falsifier -- ONE rank whose intended manifest diverges (the
  test-only salt) makes EVERY rank raise the loud manifest error BEFORE
  any secondary group is created (the #431/#616B/#645 rank-divergent
  collective family dies at the plan exchange, not inside a half-built
  communicator).
"""

import os
import tempfile
import unittest

import torch.multiprocessing as mp

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=60, suite="base-a-test-cpu")

WORLD = 3


def _worker(rank, store, q, salt_for_rank):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    fails = []
    raised = None
    try:
        import torch.distributed  # noqa: F401

        # CPU-world scaffolding: this test targets GROUP CREATION ORDER,
        # not the device-comm stack. On a CUDA-platform build the pynccl/
        # custom-AR gates are env-uniform (deliberately device-blind), so
        # a device-masked world would still construct them and die on the
        # cpu device. Mask the constructors, not the gate semantics.
        import sglang.srt.distributed.parallel_state as _ps

        _ps.should_build_pynccl = lambda *a, **k: False
        _ps.should_build_custom_allreduce = lambda *a, **k: False

        from sglang.srt.distributed.parallel_state import (
            get_pp_group,
            get_phase_flip_group,
            get_tp_group,
            init_distributed_environment,
            initialize_model_parallel,
            initialize_phase_flip_secondary_groups,
            phase_flip_groups_initialized,
        )

        init_distributed_environment(
            world_size=WORLD,
            rank=rank,
            distributed_init_method=f"file://{store}",
            local_rank=rank,
            backend="gloo",
        )
        # Primary topology: the PP prefill phase (tp=1, pp=3).
        initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=WORLD,
            backend="gloo",
        )
        if get_pp_group().world_size != WORLD:
            fails.append(f"primary pp world {get_pp_group().world_size}")
        if get_tp_group().world_size != 1:
            fails.append(f"primary tp world {get_tp_group().world_size}")

        try:
            initialize_phase_flip_secondary_groups(
                tp_size=WORLD,
                pp_size=1,
                dcp_size=WORLD,
                backend="gloo",
                _manifest_salt=salt_for_rank(rank),
            )
        except RuntimeError as e:
            raised = str(e)

        if raised is None:
            if not phase_flip_groups_initialized():
                fails.append("flip groups not initialized after green init")
            ftp = get_phase_flip_group("tp")
            fdcp = get_phase_flip_group("dcp")
            fpp = get_phase_flip_group("pp")
            if ftp.world_size != WORLD or ftp.rank_in_group != rank:
                fails.append(
                    f"flip tp world {ftp.world_size} rank {ftp.rank_in_group}"
                )
            if fdcp.world_size != WORLD:
                fails.append(f"flip dcp world {fdcp.world_size}")
            if fpp.world_size != 1:
                fails.append(f"flip pp world {fpp.world_size}")
            # Smoke: both sets usable side by side.
            import torch
            import torch.distributed as dist

            t = torch.tensor([float(rank + 1)])
            dist.all_reduce(t, group=ftp.device_group)
            if float(t.item()) != 6.0:
                fails.append(f"flip tp all_reduce got {t.item()}")
            t2 = torch.tensor([float(rank + 1)])
            dist.all_reduce(t2, group=get_pp_group().device_group)
            if float(t2.item()) != 6.0:
                fails.append(f"primary pp all_reduce got {t2.item()}")
        else:
            if phase_flip_groups_initialized():
                fails.append("flip groups exist despite manifest error")
    except Exception as e:  # noqa: BLE001
        import traceback

        fails.append(f"exception: {e}\n{traceback.format_exc()}")
    q.put((rank, fails, raised))


def _salt_none(rank):
    return 0


def _salt_rank1(rank):
    return 1 if rank == 1 else 0


class TestPhaseFlipGroupInit(CustomTestCase):
    def _run(self, salt_fn):
        ctx = mp.get_context("spawn")
        with tempfile.TemporaryDirectory() as d:
            store = os.path.join(d, "store")
            q = ctx.Queue()
            procs = [
                ctx.Process(target=_worker, args=(r, store, q, salt_fn))
                for r in range(WORLD)
            ]
            for p in procs:
                p.start()
            try:
                results = [q.get(timeout=300) for _ in range(WORLD)]
            finally:
                for p in procs:
                    p.join(timeout=60)
                    if p.is_alive():
                        p.terminate()
            return sorted(results)

    def test_dual_group_sets_green_path(self):
        for rank, fails, raised in self._run(_salt_none):
            self.assertIsNone(raised, f"rank {rank}: {raised}")
            self.assertEqual(fails, [], f"rank {rank}: {fails}")

    def test_manifest_divergence_all_ranks_loud_nothing_created(self):
        results = self._run(_salt_rank1)
        for rank, fails, raised in results:
            self.assertEqual(fails, [], f"rank {rank}: {fails}")
            self.assertIsNotNone(
                raised, f"rank {rank} did not raise on manifest divergence"
            )
            self.assertIn("manifest DIVERGES", raised)


if __name__ == "__main__":
    unittest.main()
