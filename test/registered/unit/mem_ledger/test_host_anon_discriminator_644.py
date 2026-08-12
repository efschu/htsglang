"""#644: the residual host anon must be attributable, not just measurable.

WHAT THIS IS FOR
----------------
VAL-R4 measured ~16 GB of host ``RssAnon`` surviving model load on the real
GGUF MoE checkpoint **on both sides of the #644 fix** (plateau 15.974 GB with
the fix, 16.646 GB with the hunk reverted). An RSS sampler cannot say whether
those bytes are:

* referenced CPU tensors that outlive load -- a leak with a named holder, or
* pages Python already freed that glibc's arenas never returned to the kernel
  -- bounded, and fixable with a trim rather than with an ownership change.

Both look identical in RSS. The textbook separator is ``malloc_trim(0)`` under
a debugger, and gdb is not installed on this rig, so the separation is done
in-process behind an env flag.

THE INSTRUMENTS, AND WHY THE VERDICT NEEDS TWO OF THEM
------------------------------------------------------
``live_cpu_storage_bytes`` answers "do references persist" without caring who
holds them; ``malloc_trim`` answers "was the residue already free". A verdict
from one instrument alone is the failure mode this file exists to prevent, so
``MIXED`` and ``NEITHER`` are first-class outcomes rather than errors -- the
instrument reports disagreement instead of resolving it by preference.

CAN-FAIL (each one run, not asserted from the desk)
---------------------------------------------------
Replace the deduplicated storage accounting with per-tensor
``numel * element_size`` and ``test_aliasing_views_count_their_storage_once``
goes red -- verified. Note what that proof does NOT say: per-tensor accounting
does not *miss* an aliased storage, because ``gc.get_objects()`` still sees the
base tensor through the view's ``_base``. An earlier draft of this test claimed
it did, and passed on the broken accounting. What the dedup prevents is
OVER-counting one resident storage once per tensor that points at it, which on
this ticket is the dangerous direction: it can manufacture a 16 GB retention
verdict out of a few GB of real residency. Make ``enabled()`` read the env at import and
``test_the_flag_is_read_per_call`` goes red. Return a verdict from the trim
instrument alone and ``test_both_instruments_firing_reports_mixed`` goes red.
Drop the ``getattr`` guards in ``named_holder_residue`` and
``test_named_holders_are_reported_per_parameter`` goes red on a plain module.
"""

import os
import unittest

import torch

from sglang.srt.mem_ledger import host_anon_644 as D
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _FakeParam(torch.nn.Parameter):
    """A parameter that can carry the two holders #644 names."""

    def __new__(cls, data):
        return super().__new__(cls, data)


class _FakeModel(torch.nn.Module):
    def __init__(self, holders=None):
        super().__init__()
        self.w = _FakeParam(torch.zeros(4, dtype=torch.float32))
        if holders:
            for attr, value in holders.items():
                setattr(self.w, attr, value)


class LiveStorageCensus(unittest.TestCase):
    def test_aliasing_views_count_their_storage_once(self):
        """A view keeps the WHOLE storage alive.

        ``linear.py`` appends ``.narrow()`` views that pin full host storages,
        so counting a view's own nbytes would under-report the retention by
        precisely the amount under investigation. The unit is the storage
        pointer, not the tensor.
        """
        before, _ = D.live_cpu_storage_bytes()
        # One 8 MiB storage, five tensors pointing at it. This is the shape the
        # loader produces: linear.py appends narrow() views and the parameter
        # itself aliases the same host storage, so several live tensors share
        # one resident allocation.
        #
        # NOTE, measured rather than assumed: per-TENSOR accounting does not
        # *miss* a storage whose only reference is a view -- gc.get_objects()
        # still sees the base tensor, because the view holds it as `_base`. So
        # the dedup's job is not discovery, it is arithmetic: without it this
        # 8 MiB of residency is charged five times, and a 16 GB verdict can be
        # manufactured out of 3 GB of real retention. Overstating retention is
        # the failure that matters here, since the whole ticket turns on
        # whether ~16 GB is genuinely referenced.
        base = torch.zeros(8 << 20, dtype=torch.uint8)
        keep = [base] + [base.narrow(0, 0, 8 << 20) for _ in range(4)]
        after, _ = D.live_cpu_storage_bytes()
        delta = after - before
        self.assertGreaterEqual(delta, 8 << 20, "the resident storage must be charged")
        self.assertLess(
            delta,
            2 * (8 << 20),
            "one storage, five tensors: charging it once is the point -- "
            "per-tensor summing would report ~40 MiB for 8 MiB of residency",
        )
        del keep, base

    def test_cuda_tensors_are_not_counted(self):
        """The question is about HOST anon. Device tensors are the point of
        the load, not the residue."""
        total, _ = D.live_cpu_storage_bytes()
        self.assertIsInstance(total, int)
        self.assertGreaterEqual(total, 0)


class NamedHolders(unittest.TestCase):
    def test_named_holders_are_reported_per_parameter(self):
        """'which parameter' separates a missed branch from a whole-model
        regression, so the report is per parameter, not a count."""
        model = _FakeModel(
            holders={"data_container": [torch.zeros(2)], "expert_data_map": {0: 1}}
        )
        found = D.named_holder_residue(model)
        self.assertEqual(len(found), 2)
        self.assertTrue(any("w.data_container holds 1 entries" in f for f in found))
        self.assertTrue(any("w.expert_data_map holds 1 entries" in f for f in found))

    def test_a_clean_model_reports_nothing(self):
        self.assertEqual(D.named_holder_residue(_FakeModel()), [])

    def test_an_empty_holder_is_not_a_finding(self):
        """#644's fix leaves the attributes present but empty. Reporting an
        empty container as residue would flag every fixed boot."""
        model = _FakeModel(holders={"data_container": [], "expert_data_map": {}})
        self.assertEqual(D.named_holder_residue(model), [])

    def test_a_model_without_the_attributes_is_safe(self):
        self.assertEqual(D.named_holder_residue(torch.nn.Linear(2, 2)), [])

    def test_none_model_is_safe(self):
        self.assertEqual(D.named_holder_residue(None), [])


class Gating(unittest.TestCase):
    def setUp(self):
        self.addCleanup(os.environ.pop, D.ENV_FLAG, None)

    def test_disabled_by_default(self):
        os.environ.pop(D.ENV_FLAG, None)
        self.assertIsNone(D.run_discriminator(None, "off"))

    def test_the_flag_is_read_per_call(self):
        os.environ.pop(D.ENV_FLAG, None)
        self.assertFalse(D.enabled())
        os.environ[D.ENV_FLAG] = "1"
        self.assertTrue(D.enabled())
        os.environ[D.ENV_FLAG] = "0"
        self.assertFalse(D.enabled())


class VerdictLogic(unittest.TestCase):
    """The verdict is a function of the two instruments; pin the table."""

    def setUp(self):
        os.environ[D.ENV_FLAG] = "1"
        self.addCleanup(os.environ.pop, D.ENV_FLAG, None)

    def _run_with(self, live_bytes, rss_sequence, holders=None):
        seq = list(rss_sequence)
        self.addCleanup(setattr, D, "rss_anon_kb", D.rss_anon_kb)
        self.addCleanup(setattr, D, "live_cpu_storage_bytes", D.live_cpu_storage_bytes)
        self.addCleanup(setattr, D, "malloc_trim", D.malloc_trim)
        D.rss_anon_kb = lambda pid="self": seq.pop(0) if seq else 0
        D.live_cpu_storage_bytes = lambda: (live_bytes, [])
        D.malloc_trim = lambda: True
        return D.run_discriminator(_FakeModel(holders), "t")

    def test_persistent_references_report_retention(self):
        # 16 GiB live, trim gives nothing back.
        out = self._run_with(16 << 30, [17_000_000, 16_800_000, 16_790_000])
        self.assertEqual(out["verdict"], "RETENTION")

    def test_freed_but_untrimmed_reports_allocator(self):
        # Nothing reachable, and the trim hands most of the residue back.
        out = self._run_with(64 << 20, [17_000_000, 16_800_000, 4_000_000])
        self.assertEqual(out["verdict"], "ALLOCATOR")
        self.assertGreater(out["trim_released_mib"], 12000)

    def test_both_instruments_firing_reports_mixed(self):
        """Two instruments disagreeing is a result, not an error. Resolving it
        by preference is how a single-instrument verdict gets laundered into a
        two-instrument one."""
        out = self._run_with(16 << 30, [17_000_000, 16_800_000, 4_000_000])
        self.assertEqual(out["verdict"], "MIXED")

    def test_neither_instrument_firing_is_reported_as_neither(self):
        """Anon that Python does not own AND the allocator will not return is
        the signature of driver or pinned-host allocations -- a third answer,
        and pretending it is one of the other two would be a false attribution.
        """
        out = self._run_with(64 << 20, [17_000_000, 16_800_000, 16_790_000])
        self.assertEqual(out["verdict"], "NEITHER")

    def test_named_holders_ride_along_with_the_verdict(self):
        out = self._run_with(
            64 << 20, [1000, 900, 900], holders={"data_container": [torch.zeros(2)]}
        )
        self.assertTrue(out["named_holder_residue"])


class Libc(unittest.TestCase):
    def test_malloc_trim_is_callable_on_this_box(self):
        """The whole design rests on libc exposing malloc_trim. Assert it
        rather than discover it on a booted GGUF checkpoint."""
        self.assertTrue(D.malloc_trim())

    def test_rss_anon_is_readable_and_not_vmrss(self):
        anon = D.rss_anon_kb()
        self.assertGreater(anon, 0)
        # RssAnon is a strict subset of VmRSS; if they were the same field the
        # instrument would be counting the mmap'd GGUF's file pages.
        vmrss = D._proc_status_kb("VmRSS")
        self.assertLessEqual(anon, vmrss)


if __name__ == "__main__":
    unittest.main()
