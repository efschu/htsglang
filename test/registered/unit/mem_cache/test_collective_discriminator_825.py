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
"""A dead peer must NEVER read as a timeout, whatever the clock says (#825).

#734 established the property: transport failure and deadline expiry are
different events and the log must not confuse them. It enforced that with
``waited < timeout_s * 0.95``, which was correct for the shape it was written
against -- ``Work.wait(timeout=...)`` raised for BOTH causes, so only the
elapsed time separated them.

#829 changed the shape. The deadline moved onto ``ParkedWait.join``, and the
two causes stopped meeting:

    expiry            -> join() returns False   -> the `not completed` raise
    transport failure -> join() re-raises       -> the `except RuntimeError`

The comparison then had nothing left to decide and one thing left to break: a
peer death landing in the last 5 % of the bound was relabelled a timeout. At
the 600 s default that band is 30 seconds wide.

MEASURED 2026-08-23, hermetic 2-process gloo, CVD="", with both raise sites
instrumented to name which branch fired:

    real expiry, 4/4              -> `not completed` path, NEVER the except
    real SIGKILL peer, 14 trials  -> the except, detection latency 15-21 ms
    death at 0.98 / 0.97 / 0.96 of the bound -> reached the except and was
        relabelled HiCacheCollectiveTimeoutError. Three specimens.

These arms generate BOTH causes for real. ``TheBandIsGone`` is the red-first
pair; ``TheGuard`` covers the one configuration in which control flow alone
would not be sound.
"""

import os
import sys
import tempfile
import time
import unittest

import torch

from sglang.srt.mem_cache.hicache_collective import (
    DEFAULT_PG_TIMEOUT_S,
    HiCacheCollectiveError,
    HiCacheCollectiveTimeoutError,
    assert_bound_is_discriminable,
    bounded_wait,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=40)

BOUND = 4.0


class _RaisingWork:
    """A ``Work`` whose ``wait()`` fails like a closed gloo pair.

    ParkedWait only ever calls ``wait()`` on it, on its own thread, so this is
    the whole surface. The delay is the point: it puts the transport failure at
    a chosen fraction of the bound WITHOUT a race, which is what makes the
    red-first arm deterministic where a real kill would be timing-dependent.
    """

    def __init__(self, fail_after_s: float, message: str):
        self._fail_after_s = fail_after_s
        self._message = message

    def wait(self, timeout=None):
        time.sleep(self._fail_after_s)
        raise RuntimeError(self._message)


class _SilentWork:
    """A ``Work`` that never completes and never fails: a peer that went mute.

    ``timeout`` is HONOURED even though ``ParkedWait`` never passes one, and
    that is not decoration. The #829 mutant harness has a mutant (M1) that
    hands the deadline back to the ``Work`` -- ``work.wait(timeout=...)``, the
    shipped #630 line. A fake that ignored the argument would sleep through
    that mutant instead of failing under it, the suite would hit the harness
    timeout with no verdict, and M1 would be scored SURVIVED for a reason that
    has nothing to do with the law it tests. Measured exactly that on the first
    run of this file.

    Expiring by RAISING is also what the real timed gloo wait does, so the
    mutant sees the behaviour it is meant to see.
    """

    def wait(self, timeout=None):
        if timeout is not None:
            time.sleep(getattr(timeout, "total_seconds", lambda: float(timeout))())
            raise RuntimeError("Application timeout caused pair closure")
        time.sleep(3600)


class TheBandIsGone(unittest.TestCase):
    """RED-FIRST: both arms fail with the 0.95 comparison in place."""

    def test_a_transport_failure_inside_the_last_five_percent_is_not_a_timeout(self):
        """The band, hit deterministically at 0.98 of the bound.

        With the comparison in place this raises HiCacheCollectiveTimeoutError
        -- the wrong-instrument reading #734 exists to prevent, produced by the
        machinery #734 installed to prevent it.
        """
        work = _RaisingWork(
            BOUND * 0.98,
            "[../gloo/transport/tcp/pair.cc:547] Connection closed by peer",
        )
        with self.assertRaises(HiCacheCollectiveError) as caught:
            bounded_wait(work, "pp_sync/isend[0]->pp1", BOUND, "probe-rank0")
        self.assertNotIsInstance(
            caught.exception,
            HiCacheCollectiveTimeoutError,
            "a closed pair is a dead peer, not a slow one -- at any elapsed "
            "time, including inside the last 5 % of the bound (#825)",
        )
        self.assertIn("Connection closed by peer", str(caught.exception))

    def test_the_verdict_does_not_move_with_the_clock(self):
        """The same cause at four points of the bound must give one answer.

        This is the property in one assertion: elapsed time is evidence, never
        the decision. With the comparison in place the last of these flips.
        """
        seen = []
        for fraction in (0.10, 0.50, 0.90, 0.99):
            work = _RaisingWork(BOUND * fraction, "Connection closed by peer")
            try:
                bounded_wait(work, f"probe@{fraction}", BOUND, "probe-rank0")
                seen.append("NO_RAISE")
            except BaseException as exc:  # noqa: BLE001 - the type IS the result
                seen.append(type(exc).__name__)
        self.assertEqual(
            [HiCacheCollectiveError.__name__] * 4,
            seen,
            f"one cause must yield one verdict; got {seen} across the bound",
        )


class ExpiryStillReadsAsExpiry(unittest.TestCase):
    """Non-regression: removing the comparison must not relabel real expiries."""

    def test_a_mute_peer_still_raises_the_bounded_timeout(self):
        work = _SilentWork()
        started = time.monotonic()
        with self.assertRaises(HiCacheCollectiveTimeoutError):
            bounded_wait(work, "pp_sync/recv<-pp0", 1.0, "probe-rank1")
        waited = time.monotonic() - started
        self.assertGreaterEqual(waited, 0.95, "the bound must actually be waited")


class RealRanksRealCauses(unittest.TestCase):
    """The same two causes over a REAL gloo pair, in real processes.

    The arms above are deterministic but synthetic. These are neither: a real
    peer process really dies, and a real peer really goes mute. A synthetic
    Work cannot fail the way a closed TCP pair fails.
    """

    @staticmethod
    def _run(scenario, bound, kill_at):
        import torch.multiprocessing as mp

        ctx = mp.get_context("spawn")
        mgr = ctx.Manager()
        out = mgr.dict()
        with tempfile.TemporaryDirectory() as tmp:
            init_file = os.path.join(tmp, "pg_init")
            procs = [
                ctx.Process(
                    target=_gloo_worker,
                    args=(r, init_file, scenario, bound, kill_at, out),
                )
                for r in range(2)
            ]
            for p in procs:
                p.start()
            deadline = time.time() + bound * 3 + 60
            for p in procs:
                p.join(timeout=max(1, deadline - time.time()))
            for p in procs:
                if p.is_alive():
                    p.kill()
                    p.join(timeout=5)
        return dict(out).get(1, {"err_type": "no verdict"})

    def test_a_real_dead_peer_is_never_reported_as_a_timeout(self):
        """Real SIGKILL, placed inside the band the comparison was blind in."""
        res = self._run("peerdeath", BOUND, BOUND * 0.97)
        self.assertEqual(
            HiCacheCollectiveError.__name__,
            res["err_type"],
            f"a killed peer must read as a dead peer; got {res}",
        )

    def test_a_real_mute_peer_is_still_reported_as_a_timeout(self):
        """Real alive-but-silent peer: the other direction, over real gloo."""
        res = self._run("expiry", 1.0, 0.0)
        self.assertEqual(
            HiCacheCollectiveTimeoutError.__name__,
            res["err_type"],
            f"an alive mute peer must read as a timeout; got {res}",
        )


def _gloo_worker(rank, init_file, scenario, bound, kill_at, out):
    """Rank 0 is the peer under test; rank 1 runs the real ``bounded_wait``."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import datetime as dt

    import torch.distributed as dist

    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
        timeout=dt.timedelta(seconds=600),
    )
    dist.barrier()

    if rank == 0:
        if scenario == "peerdeath":
            time.sleep(kill_at)
            os.kill(os.getpid(), 9)  # no close, no teardown: a real corpse
        else:
            time.sleep(bound * 3.0)  # alive, healthy, mute
        try:
            dist.destroy_process_group()
        except Exception:
            pass
        return

    tensor = torch.zeros(1, dtype=torch.long)
    work = dist.irecv(tensor, src=0)
    started = time.monotonic()
    err_type = "NO_RAISE"
    try:
        bounded_wait(work, "pp_sync/isend[0]->pp1", bound, "probe-rank1")
    except BaseException as exc:  # noqa: BLE001 - the type IS the result
        err_type = type(exc).__name__
    out[1] = {
        "err_type": err_type,
        "waited": round(time.monotonic() - started, 3),
        "ratio": round((time.monotonic() - started) / bound, 4),
    }
    try:
        dist.destroy_process_group()
    except Exception:
        pass


class TheGuard(unittest.TestCase):
    """The one configuration where control flow alone would not be sound.

    The parked wait is unbounded, so the causes stay separated only while the
    bound sits under the group's own timeout. That is a configuration property,
    so it is checked by name, once -- not re-derived from a stopwatch at every
    raise, which is what the retired comparison did.
    """

    def test_a_bound_under_the_group_timeout_is_accepted(self):
        assert_bound_is_discriminable(600.0, 7200.0)  # the shipped 12x margin

    def test_a_bound_at_or_over_the_group_timeout_is_refused_by_name(self):
        for bad in (7200.0, 7201.0):
            with self.assertRaises(HiCacheCollectiveError) as caught:
                assert_bound_is_discriminable(bad, 7200.0)
            msg = str(caught.exception)
            self.assertIn("7200", msg)
            self.assertIn(f"{bad:g}", msg)
            self.assertIn("SGLANG_HICACHE_COLLECTIVE_TIMEOUT_S", msg)

    def test_the_documented_fallback_is_the_group_default(self):
        """An unreadable group must not silently disable the guard."""
        self.assertEqual(120.0 * 60.0, DEFAULT_PG_TIMEOUT_S)
        with self.assertRaises(HiCacheCollectiveError):
            assert_bound_is_discriminable(DEFAULT_PG_TIMEOUT_S + 1.0, None)


if __name__ == "__main__":
    sys.exit(unittest.main())
