# SPDX-License-Identifier: Apache-2.0
"""Auftrag 1301: scripts/weg2/sm89/derive_sm89_kernel_list.py -- the sm_89 twin
lines of a delta_prebuild kernel list (P1b of PLAN_HWGEN_N_KARTEN_1003).

Pins: every rule, that only twins are written (the rig lists stay the rig's, so a
build with the existing lists is unchanged), that no 12.0 / py line is copied, and
-- where the build lane is present on this host -- that the real lists derive into
lines ``delta_prebuild.py`` itself can parse.
"""

from __future__ import annotations

import importlib.util
import os
import unittest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_HERE = os.path.dirname(os.path.abspath(__file__))
_TOOL = os.path.normpath(
    os.path.join(_HERE, "..", "..", "..", "..", "scripts", "weg2", "sm89", "derive_sm89_kernel_list.py")
)
_spec = importlib.util.spec_from_file_location("derive_sm89_kernel_list", _TOOL)
derive = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(derive)

LANE = "/spinning/gpu-arb/docker"


class TestRules(unittest.TestCase):
    def test_tvmffi_86_gets_an_89_twin_with_kwargs_and_env_kept(self):
        line = ('tvmffi 8.6 sglang.jit_kernel.hicache:_jit_hicache_module '
                '{"element_size":512,"unroll":4} PYTHONPATH=/opt/htsglang/src-nf/python')
        self.assertEqual(derive.derive_line(line), [line.replace("tvmffi 8.6", "tvmffi 8.9", 1)])

    def test_tvmffi_120_and_other_archs_have_no_twin(self):
        for line in ("tvmffi 12.0 sglang.jit_kernel.hicache:_jit_hicache_module {}",
                     "tvmffi 8.9 sglang.jit_kernel.hicache:_jit_hicache_module {}",
                     "py sglang.srt.x:fn", "# tvmffi 8.6 x:y", "", "   "):
            self.assertEqual(derive.derive_line(line), [], line)

    def test_fi_86_becomes_fi_89_120f_does_not(self):
        self.assertEqual(derive.derive_line("fi 86 norm"), ["fi 89 norm"])
        self.assertEqual(derive.derive_line("fi 86 batch_prefill_x server"), ["fi 89 batch_prefill_x server"])
        self.assertEqual(derive.derive_line("fi 120f norm"), [])

    def test_w4a8_prebuild_module_line_targets_8_9_with_its_own_report(self):
        line = "module sglang.jit_kernel.prebuild_nvfp4_w4a8 --arch 8.6 --report /opt/htsglang/JIT_PREBUILD-27b-w4a8.json"
        self.assertEqual(
            derive.derive_line(line),
            ["module sglang.jit_kernel.prebuild_nvfp4_w4a8 --arch 8.9 --report /opt/htsglang/JIT_PREBUILD-27b-w4a8-sm89.json"],
        )
        self.assertEqual(derive.derive_line("module sglang.jit_kernel.other --arch 8.6"), [])

    def test_barlink_gets_the_mixed_union_and_the_pure_ada_name(self):
        line = "barlink 8.6,12.0 sglang.srt.distributed.device_communicators.barlink_bar1_pipe_ext:load_pipe_ext"
        rest = "sglang.srt.distributed.device_communicators.barlink_bar1_pipe_ext:load_pipe_ext"
        self.assertEqual(derive.derive_line(line), [f"barlink 8.6,8.9,12.0 {rest}", f"barlink 8.9 {rest}"])
        self.assertEqual(derive.derive_line(f"barlink 8.9 {rest}"), [])
        # ordering is numeric, not lexical
        self.assertEqual(derive._barlink_union("12.0,8.6"), "8.6,8.9,12.0")

    def test_whole_list_is_ordered_deduped_and_idempotent(self):
        text = ["# c", "tvmffi 8.6 a:b", "tvmffi 12.0 a:b", "tvmffi 8.6 a:b", "fi 86 u", "barlink 8.6,12.0 m:f"]
        out = derive.derive(text)
        self.assertEqual(out, ["tvmffi 8.9 a:b", "fi 89 u", "barlink 8.6,8.9,12.0 m:f", "barlink 8.9 m:f"])
        self.assertEqual(derive.derive(out), [], "twins have no further twins")
        for line in out:
            self.assertNotIn(" 8.6 ", " " + line + " ")
            self.assertNotIn(" 12.0 ", " " + line + " ")


@unittest.skipUnless(os.path.isfile(f"{LANE}/delta_prebuild.py")
                     and os.path.isfile(f"{LANE}/delta_kernels_rc10u.txt"),
                     "build lane not on this host")
class TestRealBuildLane(unittest.TestCase):
    LISTS = ["delta_kernels_rc10u.txt", "delta_kernels_rc9dwin.txt", "delta_kernels_rc9f.txt",
             "delta_kernels_rc9c.txt", "delta_kernels_rc9b.txt", "delta_kernels_rc9.txt"]

    def test_every_derived_line_parses_in_delta_prebuild(self):
        spec = importlib.util.spec_from_file_location("dp_lane", f"{LANE}/delta_prebuild.py")
        dp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dp)
        lines = []
        for name in self.LISTS:
            path = f"{LANE}/{name}"
            if os.path.isfile(path):
                with open(path) as fh:
                    lines.extend(fh.read().splitlines())
        twins = derive.derive(lines)
        self.assertGreater(len(twins), 10)
        for t in twins:
            argv, _env = dp._cmd(t)
            self.assertTrue(argv, t)
        # every tvmffi 8.6 line of the lists has exactly one 8.9 twin
        n86 = len({ln.strip() for ln in lines if ln.strip().startswith("tvmffi 8.6 ")})
        n89 = sum(1 for t in twins if t.startswith("tvmffi 8.9 "))
        self.assertEqual(n86, n89)


if __name__ == "__main__":
    unittest.main()
