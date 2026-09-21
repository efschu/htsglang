"""fnFL2 v43 (21.09.): W84 refused a boot on 273 findings that name no page.

Two populations, both live under an exchanged weights tag and in no plan:

* 265 Marlin PLACEHOLDERS (``weight_g_idx``, ``g_idx_sort_indices``,
  ``weight_zero_point``, every one ``shape=[0]``).  Zero bytes hold no page,
  so the refusal's own reason -- "the destination would serve whatever its
  arena held" -- has nothing to refer to.
* 8 Marlin WORKSPACES (``...mlp.experts.workspace``, 272 int32 = 1088 B),
  built by ``marlin_make_workspace`` with ``torch.zeros``.  Sourceless by
  construction AND load-bearing: Marlin needs the semaphores at zero, and the
  resume maps the peer's released pages under them.  So this one is not
  waved through -- it is zeroed on the waking side.
"""

import unittest

import torch

from sglang.srt.weg2 import weight_exchange as WX


class LocalScratchPredicate(unittest.TestCase):
    def test_the_leaf_name_decides_not_a_substring(self):
        self.assertTrue(WX.is_local_scratch("model.layers.40.mlp.experts.workspace"))
        self.assertTrue(WX.is_local_scratch("workspace"))
        self.assertFalse(WX.is_local_scratch("model.workspace.weight"))
        self.assertFalse(WX.is_local_scratch("model.layers.0.mlp.gate.weight"))


class ZeroLocalScratch(unittest.TestCase):
    def _model(self):
        m = torch.nn.Module()
        inner = torch.nn.Module()
        inner.workspace = torch.nn.Parameter(
            torch.full((272,), 7, dtype=torch.int32), requires_grad=False
        )
        inner.weight = torch.nn.Parameter(
            torch.full((4,), 3.0), requires_grad=False
        )
        inner.g_idx_sort_indices = torch.nn.Parameter(
            torch.zeros((0,), dtype=torch.int32), requires_grad=False
        )
        m.experts = inner
        return m

    def test_it_zeroes_the_workspace_and_nothing_else(self):
        m = self._model()
        done = WX.zero_local_scratch(m)
        self.assertEqual(done, ["experts.workspace"])
        self.assertEqual(int(m.experts.workspace.data.abs().sum()), 0)
        # the real weight is untouched
        self.assertEqual(float(m.experts.weight.data.sum()), 12.0)

    def test_an_empty_scratch_is_not_reported_as_zeroed(self):
        m = torch.nn.Module()
        m.workspace = torch.nn.Parameter(
            torch.zeros((0,), dtype=torch.int32), requires_grad=False
        )
        self.assertEqual(WX.zero_local_scratch(m), [])


class TheResumeCallsIt(unittest.TestCase):
    def test_the_waking_side_is_wired(self):
        import inspect

        from sglang.srt.managers.scheduler_components import weight_updater

        src = inspect.getsource(weight_updater)
        self.assertIn("from sglang.srt.weg2.weight_exchange import zero_local_scratch", src)
        self.assertIn("WEG2-RESUME local-scratch zeroed=%d", src)
        # and it must sit AFTER the weights legs, not before them
        self.assertLess(
            src.index('_weg2_ph("leg_collects")'),
            src.index("zero_local_scratch"),
        )


class CoverLineNamesBothPopulations(unittest.TestCase):
    def test_counts_are_appended_and_the_ground_is_printed(self):
        row = WX.TagCoverage(
            rank=0, tag="weights_0", mode="exchange",
            planned_bytes=10, buffers_bytes=0, tms_bytes=10,
            uncovered=(), short=(), missing=(),
            n_parameters=3, n_buffers=0, n_attributes=0,
            empty=("a.weight_g_idx",), local_scratch=("b.workspace",),
        )
        line = row.cover_line()
        self.assertIn("empty=1", line)
        self.assertIn("local_scratch=1", line)
        self.assertIn("scratch_reason=", line)
        # neither is a refusal
        self.assertTrue(row.ok)

    def test_a_real_uncovered_tensor_still_refuses(self):
        t = WX.LiveTensor(
            name="x.weight", kind=WX.PARAMETER, module_path="x",
            dtype="torch.float16", shape=(8, 8), nbytes=128,
            storage_key=("s", 0), tag="weights_0",
        )
        row = WX.TagCoverage(
            rank=0, tag="weights_0", mode="exchange",
            planned_bytes=0, buffers_bytes=0, tms_bytes=0,
            uncovered=(t,), short=(), missing=(),
            n_parameters=1, n_buffers=0, n_attributes=0,
        )
        self.assertFalse(row.ok)


if __name__ == "__main__":
    unittest.main()


class TheKindIsNotStable(unittest.TestCase):
    """fnFL2 v44: Marlin rebinds ``layer.workspace`` to a fresh Parameter over
    the same storage, so the coverage walk sees it as an ATTRIBUTE.  The first
    fix classified only Parameters and the boot still died, on 8 of the
    original 273.  Classification must not depend on the kind, and neither may
    the memset."""

    def test_a_plain_attribute_workspace_is_zeroed_too(self):
        m = torch.nn.Module()
        inner = torch.nn.Module()
        # NOT registered as a parameter: a plain tensor attribute
        object.__setattr__(inner, "workspace", torch.full((272,), 5, dtype=torch.int32))
        m.experts = inner
        done = WX.zero_local_scratch(m)
        self.assertEqual(done, ["experts.workspace"])
        self.assertEqual(int(inner.workspace.abs().sum()), 0)

    def test_one_tensor_reached_by_both_walks_is_reported_once(self):
        m = torch.nn.Module()
        inner = torch.nn.Module()
        inner.workspace = torch.nn.Parameter(
            torch.full((272,), 5, dtype=torch.int32), requires_grad=False
        )
        m.experts = inner
        self.assertEqual(WX.zero_local_scratch(m), ["experts.workspace"])

    def test_the_attribute_branch_classifies_the_same_two_populations(self):
        import inspect

        src = inspect.getsource(WX.build_coverage)
        after = src.split("if t.kind != ATTRIBUTE:")[1]
        self.assertIn("empty.append(t.name)", after)
        self.assertIn("is_local_scratch(t.name)", after)
