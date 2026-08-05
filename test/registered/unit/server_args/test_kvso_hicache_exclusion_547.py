"""#547: the kvso x HiCache refusal must name its reasons, and the interop
code written for that pair must be known to be unreachable.

WHAT THE AUDIT FOUND
--------------------
The refusal read, in full::

    "--enable-kv-session-offload is its own host tier; it cannot be combined
     with --enable-hierarchical-cache or --enable-unified-memory."

That is a DESCRIPTION of the two features, not a reason for the exclusion, and
it leaves a reader unable to tell a physical impossibility from an unbuilt
piece. Reading the tree answers it: it is the latter.

* the host buffers are two INDEPENDENT ``MHATokenToKVPoolHost`` objects with
  separate lifecycles (``hiradix_cache.HiRadixCache.__init__`` vs
  ``model_runner_kv_cache_mixin``);
* the key spaces are DISJOINT -- kvso addresses host rows by sentinel slot ids
  strictly above the device allocator's range (``sentinel_base``), HiCache by
  its own page indices and hash keys;
* neither module reads the other's state.

What is genuinely missing is (1) a JOINT pinned-host-RAM budget -- each feature
validates its own alone, and over-committing pinned memory invokes the OOM
killer rather than swapping -- and (2) a MEASUREMENT of the contention between
kvso's in-critical-path spill copies and HiCache's backup/prefetch transfers
over the same link.

THE SHARPER HALF: DEAD INTEROP CODE
-----------------------------------
``force_host_write_through`` (#242) exists so that a kvso budget DEMOTION hands
its prefix over losslessly under HiCache: the donating insert is exempted from
HiCache's hit-count write-through heuristic, which would otherwise drop exactly
the leaves the session just produced. It has ONE producer
(``kv_session_offload._budget_demote``) and its consumers are ONLY the
hierarchical caches (``hiradix_cache``, ``hi_mamba_radix_cache``). The boot
refuses that pair -- so the mechanism cannot run, and the feature table's claim
that "the hand-over is lossless under HiCache" describes behaviour no boot can
reach.

This file pins both halves so neither can rot further: the refusal must keep
naming its reasons, and the dead-interop fact must stay TRUE until #547
deliberately changes it. When the pair becomes bootable, this file goes red and
that is the point -- it is the ratchet, not an obstacle.

CAN-FAIL PROOF: restore the old one-sentence refusal and every test in
``TestTheRefusalNamesItsReasons`` goes red. Remove the exclusivity check
entirely and ``test_the_pair_is_still_refused`` goes red. Give
``force_host_write_through`` a second producer outside kvso and
``test_the_forced_write_through_seam_has_exactly_one_producer`` goes red.

Run:
  CUDA_VISIBLE_DEVICES=99 PYTHONPATH=python python -m pytest \
      test/registered/unit/server_args/test_kvso_hicache_exclusion_547.py -q
"""

import pathlib
import re
import unittest

from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

_SRT = pathlib.Path(__file__).resolve().parents[4] / "python" / "sglang" / "srt"


def _refusal(**over):
    kw = dict(
        model_path="dummy",
        enable_kv_session_offload=True,
        enable_hierarchical_cache=True,
    )
    kw.update(over)
    args = ServerArgs(**kw)
    try:
        args._handle_kv_session_offload()
    except ValueError as e:
        return str(e)
    raise AssertionError("the pair was NOT refused")


class TestThePairIsStillRefused(unittest.TestCase):
    def test_the_pair_is_still_refused(self):
        self.assertTrue(_refusal())

    def test_unified_memory_is_refused_the_same_way(self):
        msg = _refusal(enable_hierarchical_cache=False, enable_unified_memory=True)
        self.assertIn("#547", msg)


class TestTheRefusalNamesItsReasons(unittest.TestCase):
    """A refusal that only restates what the flags are teaches nobody what to
    build. Each of these asserts one thing the message must carry."""

    def test_it_names_the_task_that_owns_the_lift(self):
        self.assertIn("#547", _refusal())

    def test_it_does_not_claim_a_physical_impossibility(self):
        msg = _refusal()
        self.assertIn("Not a physical impossibility", msg)
        # ...and it says WHY that is so, rather than merely asserting it.
        self.assertIn("key spaces are disjoint", msg)

    def test_it_names_the_pinned_ram_gap(self):
        msg = _refusal()
        self.assertIn("PINNED host RAM", msg)
        self.assertIn("OOM killer", msg)

    def test_it_names_the_unmeasured_transfer_contention(self):
        msg = _refusal()
        self.assertIn("never been measured", msg)

    def test_it_no_longer_reads_as_a_bare_restatement(self):
        self.assertNotIn("is its own host tier", _refusal())


class TestTheForcedWriteThroughSeamIsKnownDead(unittest.TestCase):
    """``force_host_write_through`` is written for a pair the boot refuses.

    Structural, not behavioural: with the pair unbootable there is no runtime
    in which to observe the seam, so the honest test is over the code that
    would have to change for it to become live.
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
            "kv-session-offload; the dead-interop reasoning in this file and "
            "in the #547 refusal must be re-derived.",
        )

    def test_it_is_consumed_only_inside_the_prefix_cache_layer(self):
        """Every reader lives under ``mem_cache/`` -- i.e. in a prefix cache,
        whose host tier only exists when hierarchical caching is initialised
        (``mem_cache/registry.py``: ``if ctx.enable_hierarchical_cache:
        cache.init_hicache(...)``). A reader anywhere else would be a path that
        could fire without HiCache, and the reasoning here would not hold."""
        elsewhere = set()
        for path, text in self._sources().items():
            rel = str(path.relative_to(_SRT))
            if "force_host_write_through" not in text:
                continue
            if rel.startswith("mem_cache/"):
                continue
            if rel in (
                "managers/kv_session_offload.py",  # the sole producer
                "server_args.py",  # the #547 refusal's own reasoning
            ):
                continue
            elsewhere.add(rel)
        self.assertEqual(
            elsewhere,
            set(),
            "the forced-write-through seam is read outside the prefix cache "
            "layer; the #547 dead-interop reasoning must be re-derived.",
        )

    def test_the_hierarchical_caches_are_among_its_readers(self):
        """The other half of the claim: the seam really is aimed at HiCache."""
        readers = {
            str(p.relative_to(_SRT))
            for p, t in self._sources().items()
            if "force_host_write_through" in t
        }
        self.assertIn("mem_cache/hiradix_cache.py", readers)
        self.assertIn("mem_cache/hi_mamba_radix_cache.py", readers)

    def test_the_producer_needs_the_refused_pair_to_ever_fire(self):
        """Both ends together: the only writer lives in the module the refusal
        makes mutually exclusive with the only readers."""
        with self.assertRaises(ValueError):
            ServerArgs(
                model_path="dummy",
                enable_kv_session_offload=True,
                enable_hierarchical_cache=True,
            )._handle_kv_session_offload()


if __name__ == "__main__":
    unittest.main()
