# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Two real ranks must RENDEZVOUS, not merely time out on schedule (#630).

THE GAP THIS FILLS. test_hicache_bounded_waits_630.py asserts that the bounded
calls raise on schedule and post the same operation ``recv`` would -- against
mocked ``Work`` objects and fake groups. It proves the wait is BOUNDED. It never
proves two real ranks EXCHANGE ANYTHING, and that difference is not academic: a
guard was removed on the strength of that suite, and the configuration it had
been protecting wedged on metal for eleven minutes on 2026-08-17 --

    PP0  pp_sync/isend[0]->pp1   waited 649.1 s
    PP1  pp_sync/recv<-pp0       waited 649.2 s
    PP2  pp_sync/recv<-pp1       waited 649.1 s

-- with every rank inside the collective and every op correctly posted.

THE DEFECT the harness below found: ``bounded_wait`` polled
``work.is_completed()`` and only called ``work.wait()`` once the poll had
already succeeded. For gloo that never happens -- ``is_completed()`` REPORTS
state, ``wait()`` DRIVES the transfer -- so with both peers polling, neither
side advanced the exchange. The bound written to stop a hang produced a
livelock.

These tests use THREE REAL PROCESSES over a REAL gloo group (the shape
test_lockstep_sentinel_622.py established) driving the REAL
``UnifiedRadixCache._pp_sync`` / ``_drain_async_work``. No mocks: a mock cannot
fail the way this failed.
"""

import json
import os
import tempfile
import types
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=40)

WORLD = 3
ROUNDS = 2
TIMEOUT_S = 5.0


def _holder(rank, group, timeout_s):
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    h = types.SimpleNamespace(
        pp_rank=rank,
        pp_size=WORLD,
        pp_group=group,
        work_list=[],
        collective_timeout_s=timeout_s,
        tp_world_size=1,
    )
    for name in ("_pp_sync", "_drain_async_work", "_wait_bounded"):
        setattr(h, name, types.MethodType(getattr(UnifiedRadixCache, name), h))
    return h


def _worker(rank, init_file, out_dir, dead_rank):
    res = {"rank": rank, "ok": False, "error": None, "values": []}
    try:
        dist.init_process_group(
            "gloo", init_method=f"file://{init_file}", rank=rank, world_size=WORLD
        )
        if rank == dead_rank:
            # Never enters the collective: the survivor must still be bounded.
            import time

            time.sleep(TIMEOUT_S * 2.5)
        else:
            h = _holder(rank, dist.group.WORLD, TIMEOUT_S)
            for r in range(ROUNDS):
                h._drain_async_work()
                for which in (0, 1):  # writing_check, then loading_check
                    data = torch.tensor(r * 10 + which, dtype=torch.int)
                    h._pp_sync(data)
                    res["values"].append(int(data.item()))
            h._drain_async_work()
        res["ok"] = True
    except BaseException as exc:  # noqa: BLE001 - the failure IS the result
        res["error"] = f"{type(exc).__name__}: {exc}"[:200]
    finally:
        with open(os.path.join(out_dir, f"r{rank}.json"), "w") as f:
            json.dump(res, f)
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def _run(dead_rank=-1):
    with tempfile.TemporaryDirectory() as tmp:
        init_file = os.path.join(tmp, "pg_init")
        try:
            mp.spawn(_worker, args=(init_file, tmp, dead_rank), nprocs=WORLD, join=True)
        except Exception:
            pass  # a worker raising is a verdict, not a harness error
        out = []
        for r in range(WORLD):
            p = os.path.join(tmp, f"r{r}.json")
            out.append(
                json.load(open(p))
                if os.path.exists(p)
                else {"rank": r, "ok": False, "error": "no verdict", "values": []}
            )
        return out


class ThreeRealRanksExchangeValues(unittest.TestCase):
    def test_the_ring_rendezvouses_with_the_bound_ACTIVE(self):
        """The regression test proper: bounded AND completing.

        Before the fix this wedged with the exact three labels seen on metal.
        The bound is deliberately left ON -- running it green only with the
        timeout disabled is what hid the defect in the first place.
        """
        results = _run()
        for res in results:
            self.assertTrue(res["ok"], f"rank {res['rank']} failed: {res['error']}")

    def test_downstream_ranks_receive_rank0s_values(self):
        """Completion is not enough -- the BYTES must arrive.

        A wait that returns without transferring would satisfy the test above.
        """
        results = _run()
        expected = [r * 10 + w for r in range(ROUNDS) for w in (0, 1)]
        self.assertEqual(expected, results[0]["values"])
        for res in results[1:]:
            self.assertEqual(
                expected,
                res["values"],
                f"rank {res['rank']} did not receive rank 0's values",
            )


class TheBoundStillFires(unittest.TestCase):
    """CAN-FAIL: the #630 property must survive the fix.

    Driving progress with a blocking wait is only correct if a DEAD peer still
    cannot park a survivor for the group's two-hour timeout. If this regresses,
    the fix has traded a livelock for the original hang.
    """

    def test_a_dead_peer_still_raises_a_named_bounded_error(self):
        from sglang.srt.mem_cache.hicache_collective import (
            HiCacheCollectiveTimeoutError,
        )

        results = _run(dead_rank=1)
        rank0 = results[0]
        self.assertFalse(rank0["ok"], "rank 0 must not claim success")
        self.assertIn(HiCacheCollectiveTimeoutError.__name__, rank0["error"])
        self.assertIn("pp_sync/isend[0]->pp1", rank0["error"])


if __name__ == "__main__":
    unittest.main()
