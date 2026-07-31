"""One rank builds the BAR1 JIT extension; the rest wait on a bounded barrier.

WHY THIS FILE EXISTS
--------------------
``load_collective_ext`` used to let every rank call ``load_inline`` on the
same build directory at the same moment. torch serialises that with a
``FileBaton``: one rank creates ``lock`` and compiles, the others poll
``os.path.exists(lock)``. A builder killed before ``release()`` leaves the
lock behind, and the survivors keep polling.

``baton_health`` bounds that wait, but under co-location -- several ranks of
one group in one build directory -- neither of its fast rules can fire:
rule 2 needs sources registered outside the build directory, which
``load_inline`` does not have, and rule 3 needs the directory to stop
changing, which it cannot while every waiting rank rewrites ``main.cpp``
into it. Only the 30-minute backstop is left. Task #366 killed such a run
after 3.5 minutes and reported it as a hang; it was not wrong to.

So the tests below pin the property that removes the race rather than the
one that survives it: the build happens on exactly ONE rank, and a build
that FAILS ends every rank with a named error instead of leaving anyone in
a wait.

CPU only, and deliberately NOT over gloo. What is under test is the
branching protocol -- who builds, who waits, who raises -- not a transport.
Driving it through a real process group would buy no coverage of that logic
and would import the file-store flakiness that already accounts for a dozen
known failures in this suite. So the ranks are threads and the two group
primitives the function uses are substituted with in-process equivalents
that have the same semantics: a barrier every rank must reach, and an
all-gather every rank must contribute to. A rank that skips either one
deadlocks the fake exactly as it would deadlock gloo -- which is the failure
these tests exist to catch, so the substitution keeps the sharp edge.
"""

import os
import tempfile
import threading
import unittest
from unittest import mock

from sglang.jit_kernel import baton_health
from sglang.srt.distributed.device_communicators.barlink_bar1_ext import (
    ENV_GROUPED_BUILD,
    build_once_per_group,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

WORLD = 3
#: Every wait in the fake is bounded. A regression here is a deadlock, and a
#: test that hangs forever is a worse report than one that fails.
WAIT_S = 20


class _FakeGroup:
    """One rank's handle. Carries its own rank; the collectives are shared."""

    def __init__(self, rank, world, barrier, gathered, gate):
        self.rank = rank
        self.world = world
        self.barrier = barrier
        self.gathered = gathered
        self.gate = gate


def _run_group(body, world=WORLD):
    """Run ``body(rank, group)`` on ``world`` threads with fake collectives."""
    results = [None] * world
    errors = [None] * world
    barrier = threading.Barrier(world, timeout=WAIT_S)
    gathered = [None] * world
    gate = threading.Barrier(world, timeout=WAIT_S)
    groups = [_FakeGroup(r, world, barrier, gathered, gate) for r in range(world)]

    def fake_barrier(group, label="barrier", **kw):
        group.barrier.wait()

    def fake_all_gather_object(out, obj, group=None):
        group.gathered[group.rank] = obj
        group.gate.wait()
        out[:] = list(group.gathered)

    def one(rank):
        try:
            results[rank] = body(rank, groups[rank])
        except BaseException as exc:  # noqa: BLE001 -- reported per rank
            errors[rank] = exc

    patches = [
        mock.patch(
            "sglang.srt.distributed.device_communicators.barlink_liveness"
            ".bounded_barrier",
            fake_barrier,
        ),
        mock.patch("torch.distributed.get_rank", lambda g: g.rank),
        mock.patch("torch.distributed.get_world_size", lambda g: g.world),
        mock.patch("torch.distributed.all_gather_object", fake_all_gather_object),
    ]
    for p in patches:
        p.start()
    try:
        threads = [threading.Thread(target=one, args=(r,)) for r in range(world)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=WAIT_S + 10)
        alive = [i for i, t in enumerate(threads) if t.is_alive()]
    finally:
        for p in patches:
            p.stop()
    return results, errors, alive


class TestExactlyOneRankBuilds(CustomTestCase):
    """The property that removes the baton race."""

    def test_only_one_rank_runs_the_build_body(self):
        calls = []
        lock = threading.Lock()

        def body(rank, group):
            def build():
                with lock:
                    calls.append(rank)
                return f"artifact-for-rank-{rank}"

            return build_once_per_group(group, "probe_ext", build)

        results, errors, alive = _run_group(body)
        self.assertEqual(alive, [], "ranks still running -- the wait is not bounded")
        self.assertEqual([e for e in errors if e], [])
        # The builder builds; the others build too, but only AFTER it finished,
        # so they meet a warm cache and never contend for the baton. What must
        # never happen is a build before the builder is done.
        self.assertEqual(
            calls[0], 0, f"rank 0 must build first, order was {calls}"
        )
        self.assertEqual(len(results), WORLD)
        for r in results:
            self.assertIsNotNone(r)

    def test_a_single_rank_group_does_not_go_through_the_handshake(self):
        """world==1 must stay byte-for-byte the old path: build and return."""
        seen = []
        out = build_once_per_group(None, "probe_ext", lambda: seen.append(1) or "x")
        self.assertEqual(out, "x")
        self.assertEqual(seen, [1])

    def test_the_escape_hatch_restores_the_concurrent_path(self):
        prev = os.environ.get(ENV_GROUPED_BUILD)
        os.environ[ENV_GROUPED_BUILD] = "0"
        try:
            calls = []
            lock = threading.Lock()

            def body(rank, group):
                def build():
                    with lock:
                        calls.append(rank)
                    return rank

                return build_once_per_group(group, "probe_ext", build)

            _, errors, alive = _run_group(body)
            self.assertEqual(alive, [])
            self.assertEqual([e for e in errors if e], [])
            self.assertEqual(sorted(calls), list(range(WORLD)))
        finally:
            if prev is None:
                os.environ.pop(ENV_GROUPED_BUILD, None)
            else:
                os.environ[ENV_GROUPED_BUILD] = prev


class TestAFailedBuildEndsEveryRank(CustomTestCase):
    """A builder that raises must not leave the others in a wait.

    This is the #94/#303 shape: the rank that takes the exceptional branch
    must still reach the collective its peers are entering.
    """

    def test_builder_failure_raises_on_every_rank_with_the_reason(self):
        def body(rank, group):
            def build():
                if rank == 0:
                    raise RuntimeError("ninja: build stopped")
                return "should-not-be-reached"

            return build_once_per_group(group, "probe_ext", build)

        results, errors, alive = _run_group(body)
        self.assertEqual(alive, [], "a failed build left a rank waiting")
        for rank in range(WORLD):
            self.assertIsNotNone(
                errors[rank], f"rank {rank} did not fail with the group"
            )
            self.assertIn("probe_ext", str(errors[rank]))
            self.assertIn("ninja: build stopped", str(errors[rank]))
        self.assertEqual(results, [None] * WORLD)

    def test_a_non_builder_never_loads_when_the_build_failed(self):
        """The artifact is never touched by a rank whose group did not build it."""
        loaded = []

        def body(rank, group):
            def build():
                if rank == 0:
                    raise RuntimeError("nvcc died")
                loaded.append(rank)
                return "artifact"

            return build_once_per_group(group, "probe_ext", build)

        _, errors, alive = _run_group(body)
        self.assertEqual(alive, [])
        self.assertEqual(len([e for e in errors if e]), WORLD)
        self.assertEqual(loaded, [], f"ranks {loaded} loaded a missing artifact")


class TestTheBatonRuleThisReplaces(CustomTestCase):
    """Why the bounded baton alone was not enough under co-location.

    Pins the reading of ``baton_health`` that motivates the change above, so
    that if someone later makes rule 2 or rule 3 fire for ``load_inline``,
    this test tells them the grouped build may be reconsidered.
    """

    def test_an_unregistered_build_dir_only_gets_the_backstop(self):
        with tempfile.TemporaryDirectory() as d:
            lock = os.path.join(d, "lock")
            open(lock, "w").close()
            open(os.path.join(d, "probe.so"), "w").close()
            # No registered sources (rule 2 unavailable) and the directory
            # just changed (rule 3 unavailable): the verdict must be "wait",
            # bounded only by the 30-minute rule 4.
            verdict = baton_health.baton_verdict(
                lock, quiet_seconds=120.0, max_wait_seconds=1800.0
            )
            self.assertEqual(verdict.action, "wait")

    def test_rule_four_is_half_an_hour_by_default(self):
        """The number this change exists to avoid waiting out."""
        self.assertEqual(baton_health.DEFAULT_MAX_WAIT_SECONDS, 1800.0)


if __name__ == "__main__":
    unittest.main()
