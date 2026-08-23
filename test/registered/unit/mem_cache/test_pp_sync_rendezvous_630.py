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
import sys
import tempfile
import time
import types
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# AT MODULE SCOPE ON PURPOSE (#825) -- do not push this back into ``_holder``.
# The note under ``_pay_import_cost`` explains what that costs.
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60)

WORLD = 3
ROUNDS = 2
TIMEOUT_S = 5.0

_IMPORT_COST_ENV = "SGLANG_TEST_630_IMPORT_DELAY_S"


def _pay_import_cost():
    """Stand-in for the cost of importing the subject, paid where the real
    import is paid -- so that moving one moves the other.

    #825, MEASURED 2026-08-23 (hermetic, CVD=""): importing
    ``sglang.srt.mem_cache.unified_radix_cache`` costs **7.38 s cold** and
    5.3-6.0 s warm, and pulls 6081 modules. Two arbitrary branch states of this
    tree differ from each other by ~0.5 s warm, because the import graph is
    exactly what a branch changes.

    THE MARGIN THAT COST WAS BEING CHARGED AGAINST WAS 7.5 s, and it was a
    wall clock. ``_worker`` starts the mute peer's ``sleep(timeout_s * 2.5)``
    immediately after ``init_process_group``, while the SURVIVORS still had to
    run this import before reaching the collective. The peer's clock therefore
    ran during an import the peer itself never paid:

        margin  = timeout_s * 2.5 - timeout_s = 7.5 s   (at TIMEOUT_S = 5.0)
        import  = 7.38 s cold
        headroom = 0.12 s                                (1.6 %)

    When the import crossed the margin the peer's process EXITED while the
    survivor was still inside its bounded wait. That closes the gloo pair, the
    wait ends early with a transport error, and #734's discriminator correctly
    answered ``HiCacheCollectiveError`` -- which this file does not expect. The
    test went red for a real reason that had nothing to do with the property it
    asserts. Strand 17c measured one such run at "FAILED after 4.3s" against a
    5 s bound, which fixes the survivor's arrival at 12.5 - 4.3 = 8.2 s, i.e.
    0.7 s past the margin.

    THE FIX IS THE IMPORT'S POSITION, not a bigger margin. Hoisted to module
    scope it is paid by EVERY rank, including the mute peer, BEFORE
    ``init_process_group`` -- and that call is itself the rendezvous, so it
    absorbs whatever skew the imports introduced. The peer's clock then starts
    with the survivors already at the collective, and the margin no longer has
    an import inside it. A bigger margin would only have bought time until the
    next module was added to the graph.

    The knob exists so the property is TESTABLE rather than argued: it injects
    additional import cost at the same site the real import occupies. With the
    import at module scope no injected cost can move the verdict; with the
    import back inside ``_holder`` a cost above the margin flips it. That is
    the red-first arm below.
    """
    delay = float(os.environ.get(_IMPORT_COST_ENV, "") or 0.0)
    if delay > 0:
        time.sleep(delay)


# Module scope: every rank pays this before it reaches ``init_process_group``.
_pay_import_cost()


def _holder(rank, group, timeout_s):
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


def _worker(rank, init_file, out_dir, dead_rank, timeout_s=TIMEOUT_S):
    res = {"rank": rank, "ok": False, "error": None, "values": []}
    try:
        dist.init_process_group(
            "gloo", init_method=f"file://{init_file}", rank=rank, world_size=WORLD
        )
        if rank == dead_rank:
            # Never enters the collective: the survivor must still be bounded.
            # This clock starts at the rendezvous, so nothing a survivor still
            # has to do before the collective may be charged against it (#825).
            time.sleep(timeout_s * 2.5)
        else:
            h = _holder(rank, dist.group.WORLD, timeout_s)
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


def _run(dead_rank=-1, timeout_s=TIMEOUT_S, import_delay_s=0.0):
    prev = os.environ.get(_IMPORT_COST_ENV)
    if import_delay_s:
        # Set BEFORE spawn so the children inherit it and pay it during their
        # own module import, which is the site under test.
        os.environ[_IMPORT_COST_ENV] = str(import_delay_s)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            init_file = os.path.join(tmp, "pg_init")
            try:
                mp.spawn(
                    _worker,
                    args=(init_file, tmp, dead_rank, timeout_s),
                    nprocs=WORLD,
                    join=True,
                )
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
    finally:
        if prev is None:
            os.environ.pop(_IMPORT_COST_ENV, None)
        else:
            os.environ[_IMPORT_COST_ENV] = prev


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


class TheMarginMustNotDependOnImportCost(unittest.TestCase):
    """#825: the verdict above must not turn on how long an import takes.

    Both arms are RED with the subject imported inside ``_holder`` (i.e. after
    ``init_process_group``) and GREEN with it at module scope. Measured on this
    tree 2026-08-23: cold import 7.38 s against a 7.5 s margin -- 0.12 s of
    headroom, which is what the #622-family branch spent.
    """

    def test_the_subject_is_resident_before_any_process_group(self):
        """Structural arm: the import must already be paid at module scope.

        This is the whole fix in one assertion. It is deterministic and costs
        nothing, and it fails the moment someone pushes the import back into
        ``_holder`` -- where it is paid AFTER the rendezvous, by the survivors
        only, inside the mute peer's clock.
        """
        # assertTrue, not assertIn: assertIn renders the container on failure,
        # and that container is sys.modules -- 264 KB of dict into the CI log.
        self.assertTrue(
            "sglang.srt.mem_cache.unified_radix_cache" in sys.modules,
            "the subject must be imported at module scope, before any process "
            "group exists -- see _pay_import_cost (#825)",
        )

    def test_an_import_cost_over_the_margin_still_yields_the_bounded_timeout(self):
        """Behavioural arm: inject more import cost than the margin can absorb.

        The bound is scaled down so the arm is cheap: at ``bound`` the mute
        peer lives ``bound * 2.5`` and the margin is ``bound * 1.5``. The
        injected cost exceeds that margin outright, so if it were paid after
        the rendezvous -- as the shipped import was -- the peer would exit
        mid-wait and the verdict would flip to HiCacheCollectiveError. Paid at
        module scope it cannot, because ``init_process_group`` is itself the
        rendezvous and absorbs it.
        """
        from sglang.srt.mem_cache.hicache_collective import (
            HiCacheCollectiveTimeoutError,
        )

        bound = 1.0
        margin = bound * 2.5 - bound  # 1.5 s
        results = _run(dead_rank=1, timeout_s=bound, import_delay_s=margin + 0.5)
        rank0 = results[0]
        self.assertFalse(rank0["ok"], "rank 0 must not claim success")
        self.assertIn(
            HiCacheCollectiveTimeoutError.__name__,
            rank0["error"],
            "an alive-but-mute peer must still read as a TIMEOUT; a transport "
            "error here means the peer exited during the wait, i.e. the margin "
            "was eaten before the collective (#825)",
        )
        self.assertIn("pp_sync/isend[0]->pp1", rank0["error"])


if __name__ == "__main__":
    unittest.main()
