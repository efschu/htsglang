"""#547 -> #550: the kvso x HiCache pair, from blanket refusal to budgeted
opt-in.

WHAT #547 FOUND
---------------
The refusal read, in full::

    "--enable-kv-session-offload is its own host tier; it cannot be combined
     with --enable-hierarchical-cache or --enable-unified-memory."

That is a DESCRIPTION of the two features, not a reason for the exclusion, and
it leaves a reader unable to tell a physical impossibility from an unbuilt
piece. Reading the tree answered it: it was the latter.

* the host buffers are two INDEPENDENT ``MHATokenToKVPoolHost`` objects with
  separate lifecycles (``hiradix_cache.HiRadixCache.__init__`` vs
  ``model_runner_kv_cache_mixin``);
* the key spaces are DISJOINT -- kvso addresses host rows by sentinel slot ids
  strictly above the device allocator's range (``sentinel_base``), HiCache by
  its own page indices and hash keys;
* neither module reads the other's state.

#547 named two genuinely missing pieces: (1) a JOINT pinned-host-RAM budget --
each feature validated its own alone, and over-committing pinned memory invokes
the OOM killer rather than swapping -- and (2) a MEASUREMENT of the contention
between kvso's in-critical-path spill copies and HiCache's backup/prefetch
transfers over the same link.

WHAT #550 CHANGES, AND WHAT IT DELIBERATELY DOES NOT
----------------------------------------------------
Gap (1) is CLOSED in code: ``mem_cache/pinned_host_budget`` is one owner for
"may this pinned host buffer be allocated", every pinned host pool routes its
check through it, and ``_handle_kv_session_offload`` sums both posts once in
the launcher over configured numbers only. Gap (2) is NOT closed by writing a
guard -- it is a number that only a hardware run produces. So the pair became
an OPT-IN (``KVSO_ALLOW_HICACHE=1``) rather than a default, and the message
says which of the two gaps is still open. That distinction is the point of
this file: an opt-in whose message still claimed the RAM gap would be lying,
and a default-on pair whose latency interaction was never measured would be
shipping the thing #547 refused.

This file therefore pins THREE things:
  * the pair is reachable at all (it was not, before #550);
  * the refusal that remains names the measurement gap and NOT the RAM gap;
  * the joint budget guard actually refuses an over-commit.

CAN-FAIL PROOF: delete the ``KVSO_ALLOW_HICACHE`` gate and
``test_the_pair_is_opt_in_not_refused`` goes red. Make the sum guard a per-post
check again (drop the other post from the demand) and
``TestTheJointBudgetGuardRefusesOverCommit`` goes red -- both of its
over-commit cases are individually plausible and only jointly impossible, which
is exactly the shape the old code could not see. Restore the #547 blanket
refusal and most of this file goes red.

Run:
  CUDA_VISIBLE_DEVICES=99 PYTHONPATH=python python -m pytest \
      test/registered/unit/server_args/test_kvso_hicache_exclusion_547.py -q
"""

import os
import pathlib
import re
import unittest
from unittest import mock

from sglang.srt.mem_cache.pinned_host_budget import (
    PinnedHostPost,
    hicache_configured_host_bytes,
    joint_pinned_host_error,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

_SRT = pathlib.Path(__file__).resolve().parents[4] / "python" / "sglang" / "srt"

_GIB = 1024**3


def _handle(env=None, **over):
    """Run the kvso validation and return the refusal message, or None."""
    kw = dict(
        model_path="dummy",
        enable_kv_session_offload=True,
        enable_hierarchical_cache=True,
    )
    kw.update(over)
    args = ServerArgs(**kw)
    with mock.patch.dict(os.environ, env or {}, clear=False):
        try:
            args._handle_kv_session_offload()
        except ValueError as e:
            return str(e)
    return None


class TestThePairIsNoLongerUnbuildable(unittest.TestCase):
    def test_the_pair_is_opt_in_not_refused(self):
        """With the opt-in set, the pair passes validation. Before #550 no
        environment could reach this."""
        self.assertIsNone(_handle(env={"KVSO_ALLOW_HICACHE": "1"}))

    def test_unified_memory_takes_the_same_opt_in(self):
        self.assertIsNone(
            _handle(
                env={"KVSO_ALLOW_HICACHE": "1"},
                enable_hierarchical_cache=False,
                enable_unified_memory=True,
            )
        )

    def test_without_the_opt_in_it_still_stops(self):
        self.assertIsNotNone(_handle(env={"KVSO_ALLOW_HICACHE": "0"}))


class TestTheRemainingRefusalNamesTheRightGap(unittest.TestCase):
    """The message must have moved WITH the code. A refusal that still cites
    the pinned-RAM gap after that gap was closed sends the operator to fix
    something that is already fixed."""

    def setUp(self):
        self.msg = _handle(env={"KVSO_ALLOW_HICACHE": "0"})

    def test_it_names_the_task_that_owns_the_lift(self):
        self.assertIn("#550", self.msg)

    def test_it_names_the_opt_in_that_reaches_the_pair(self):
        self.assertIn("KVSO_ALLOW_HICACHE=1", self.msg)

    def test_it_does_not_claim_a_physical_impossibility(self):
        self.assertIn("disjoint", self.msg)
        self.assertIn("independent", self.msg)

    def test_it_no_longer_claims_the_pinned_ram_gap_is_open(self):
        """The half #550 closed. The old message said nothing sums the pinned
        RAM; saying that now would be false."""
        self.assertNotIn("never been measured", self.msg.split("measurement")[0])
        self.assertIn("summed", self.msg)

    def test_it_names_the_measurement_as_the_open_half(self):
        self.assertIn("contention", self.msg)
        self.assertIn("measurement, not a mechanism", self.msg)

    def test_it_no_longer_reads_as_a_bare_restatement(self):
        self.assertNotIn("is its own host tier", self.msg)


class TestTheJointBudgetGuardRefusesOverCommit(unittest.TestCase):
    """The gap #547 named, closed and pinned.

    Both posts below are individually plausible against the machine and only
    JOINTLY impossible -- the exact shape two independent per-pool checks
    cannot see, and the reason the pair was refused rather than budgeted.
    """

    def _jointly_impossible(self):
        return [
            PinnedHostPost("HiCache host tier", "--hicache-size", 60 * _GIB),
            PinnedHostPost(
                "kv-session-offload spill pool",
                "--kv-session-offload-host-ram-gib",
                60 * _GIB,
            ),
        ]

    def test_two_individually_plausible_posts_are_jointly_refused(self):
        posts = self._jointly_impossible()
        # Each post alone fits in 100 GiB available minus a 10 GiB reserve.
        for p in posts:
            self.assertIsNone(
                joint_pinned_host_error([p], 108 * _GIB, 100 * _GIB, 10 * _GIB)
            )
        # Together they do not.
        self.assertIsNotNone(
            joint_pinned_host_error(posts, 108 * _GIB, 100 * _GIB, 10 * _GIB)
        )

    def test_the_refusal_prices_every_post_by_name_and_flag(self):
        msg = joint_pinned_host_error(
            self._jointly_impossible(), 108 * _GIB, 100 * _GIB, 10 * _GIB
        )
        # An operator who is told only the total cannot tell which flag to
        # lower, so both posts and both flags must appear.
        self.assertIn("HiCache host tier", msg)
        self.assertIn("kv-session-offload spill pool", msg)
        self.assertIn("--hicache-size", msg)
        self.assertIn("--kv-session-offload-host-ram-gib", msg)

    def test_it_exceeds_total_and_says_so_differently(self):
        """Over TOTAL is a configuration no machine state could satisfy; over
        AVAILABLE is one this machine cannot satisfy now. Different fixes."""
        posts = [
            PinnedHostPost("HiCache host tier", "--hicache-size", 200 * _GIB),
        ]
        msg = joint_pinned_host_error(posts, 108 * _GIB, 100 * _GIB, 10 * _GIB)
        self.assertIn("TOTAL host RAM", msg)

    def test_a_fitting_pair_is_admitted(self):
        posts = [
            PinnedHostPost("HiCache host tier", "--hicache-size", 20 * _GIB),
            PinnedHostPost(
                "kv-session-offload spill pool",
                "--kv-session-offload-host-ram-gib",
                20 * _GIB,
            ),
        ]
        self.assertIsNone(
            joint_pinned_host_error(posts, 108 * _GIB, 100 * _GIB, 10 * _GIB)
        )

    def test_no_honest_number_means_no_guard(self):
        """Refusing to guess beats refusing a boot on a fabricated figure --
        the lxcfs lesson (#549/#551)."""
        posts = [PinnedHostPost("x", "--x", 500 * _GIB)]
        self.assertIsNone(joint_pinned_host_error(posts, None, None, 10 * _GIB))

    def test_zero_sized_posts_are_not_a_refusal(self):
        self.assertIsNone(
            joint_pinned_host_error(
                [PinnedHostPost("x", "--x", 0)], 108 * _GIB, 100 * _GIB, 10 * _GIB
            )
        )


class TestParseTimePricing(unittest.TestCase):
    """--hicache-size is absolute and can be summed at parse time.
    --hicache-ratio is a multiple of a device pool that does not exist yet, so
    there is no honest parse-time number and the code must not invent one."""

    def test_absolute_hicache_size_is_priced(self):
        self.assertEqual(hicache_configured_host_bytes(20, 0), int(20 * 1e9))

    def test_ratio_only_has_no_parse_time_number(self):
        self.assertIsNone(hicache_configured_host_bytes(0, 4))

    def test_the_launcher_refuses_a_jointly_impossible_pair(self):
        """End-to-end through ServerArgs, on a FIXED machine size so the result
        does not depend on whatever RAM the test box happens to have free.

        60 GiB of kvso alone passes its own pre-existing check (100 GiB
        available minus the 10 GiB reserve), and 60 GB of HiCache alone fits
        too. Only the SUM is impossible -- so this case is red against any
        version of the code that checks the two posts separately, which is
        precisely what #547 refused the pair for.
        """
        fixed = (120 * _GIB, 100 * _GIB)
        with mock.patch(
            "sglang.srt.memtier.profile.host_memory_bytes_for_pinning",
            return_value=fixed,
        ):
            msg = _handle(
                env={"KVSO_ALLOW_HICACHE": "1"},
                hicache_size=60,  # GB, absolute
                kv_session_offload_host_ram_gib=60,  # GiB, node-wide
            )
        self.assertIsNotNone(msg)
        self.assertIn("Pinned host RAM over-committed", msg)
        self.assertIn("--hicache-size", msg)
        self.assertIn("--kv-session-offload-host-ram-gib", msg)


class TestTheForcedWriteThroughSeamIsNowReachable(unittest.TestCase):
    """``force_host_write_through`` (#242) exists so a kvso budget DEMOTION
    hands its prefix over losslessly under HiCache: the donating insert is
    exempted from HiCache's hit-count write-through heuristic, which would
    otherwise drop exactly the leaves the session just produced.

    #547 recorded it as dead BY CONSTRUCTION -- its one producer
    (``kv_session_offload._budget_demote``) sat on the far side of a refusal
    from its only readers (the hierarchical caches). #550 removes that
    refusal, so the honest status changes from "unreachable" to "reachable,
    opt-in". The structural facts that made the reasoning true are still worth
    pinning: one producer, readers confined to the prefix-cache layer.
    """

    def _sources(self):
        return {p: p.read_text() for p in _SRT.rglob("*.py")}

    def test_the_forced_write_through_seam_has_exactly_one_producer(self):
        setters = []
        for path, text in self._sources().items():
            for line in text.splitlines():
                if re.search(r"\.force_host_write_through\s*=", line):
                    setters.append(str(path.relative_to(_SRT)))
        self.assertEqual(
            setters,
            ["managers/kv_session_offload.py"],
            "force_host_write_through gained a producer outside "
            "kv-session-offload; the interop reasoning in this file must be "
            "re-derived.",
        )

    def test_it_is_consumed_only_inside_the_prefix_cache_layer(self):
        """Every reader lives under ``mem_cache/`` -- i.e. in a prefix cache,
        whose host tier only exists when hierarchical caching is initialised
        (``mem_cache/registry.py``). A reader anywhere else would be a path
        that could fire without HiCache."""
        elsewhere = set()
        for path, text in self._sources().items():
            rel = str(path.relative_to(_SRT))
            if "force_host_write_through" not in text:
                continue
            if rel.startswith("mem_cache/"):
                continue
            if rel in (
                "managers/kv_session_offload.py",  # the sole producer
                "server_args.py",  # the #550 reasoning
            ):
                continue
            elsewhere.add(rel)
        self.assertEqual(
            elsewhere,
            set(),
            "the forced-write-through seam is read outside the prefix cache "
            "layer; the interop reasoning must be re-derived.",
        )

    def test_the_hierarchical_caches_are_among_its_readers(self):
        readers = {
            str(p.relative_to(_SRT))
            for p, t in self._sources().items()
            if "force_host_write_through" in t
        }
        self.assertIn("mem_cache/hiradix_cache.py", readers)
        self.assertIn("mem_cache/hi_mamba_radix_cache.py", readers)

    def test_the_producer_can_now_reach_its_readers(self):
        """The half that #550 inverts: the pair the producer needs is bootable.
        This is the test that was ``assertRaises`` before."""
        self.assertIsNone(_handle(env={"KVSO_ALLOW_HICACHE": "1"}))


if __name__ == "__main__":
    unittest.main()
