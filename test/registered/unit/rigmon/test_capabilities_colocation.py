"""CPU unit tests for the co-location rows of the capability table.

Several ranks on ONE physical GPU are gated by two host properties that the
table previously did not show: the NCCL RUNTIME version (>= 2.30, decided by
the library ctypes loads, not by build metadata) and a reachable CUDA MPS
control daemon (decided by the control directory only — no process scan).
"""

import unittest

from sglang.srt.rigmon.capabilities import (
    ACTIVE,
    AVAILABLE,
    UNAVAILABLE,
    ProbeEnv,
    _format_nccl_version,
    probe_all,
    probe_mps,
    probe_nccl_colocation,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _env(**kw):
    base = dict(
        exists=lambda p: False,
        listdir=lambda p: [],
        read_text=lambda p: "",
        run=lambda cmd: (127, "not found"),
        import_module=lambda n: (_ for _ in ()).throw(ImportError("absent")),
        env={},
        nccl_version=lambda environ: (23102, "libnccl.so.2", None),
    )
    base.update(kw)
    return ProbeEnv(**base)


COLOCATED_ENGINE = {"rank_gpu_id": "0,1,1,2"}
SPREAD_ENGINE = {"rank_gpu_id": "0,1,2"}


class TestNcclVersionFormat(CustomTestCase):
    def test_modern_encoding(self):
        self.assertEqual(_format_nccl_version(22809), "2.28.9")
        self.assertEqual(_format_nccl_version(23102), "2.31.2")

    def test_legacy_encoding(self):
        self.assertEqual(_format_nccl_version(2804), "2.8.4")


class TestNcclColocationProbe(CustomTestCase):
    def test_runtime_below_minimum_names_version_and_requirement(self):
        cap = probe_nccl_colocation(
            _env(nccl_version=lambda environ: (22809, "libnccl.so.2", None))
        )
        self.assertEqual(cap.state, UNAVAILABLE)
        # The reason must name the measured runtime version, the library it
        # came from, the hard threshold, and what it would take.
        self.assertIn("2.28.9", cap.reason)
        self.assertIn("libnccl.so.2", cap.reason)
        self.assertIn("2.30", cap.reason)
        self.assertIn("SGLANG_NCCL_SO_PATH", cap.reason)

    def test_no_loadable_library_carries_the_loader_error(self):
        cap = probe_nccl_colocation(
            _env(
                nccl_version=lambda environ: (
                    None,
                    None,
                    "libnccl.so.2: cannot open shared object file",
                )
            )
        )
        self.assertEqual(cap.state, UNAVAILABLE)
        self.assertIn("cannot open shared object file", cap.reason)
        self.assertIn("2.30", cap.reason)

    def test_table_shows_the_runtime_version_when_capable(self):
        cap = probe_nccl_colocation(_env())
        self.assertEqual(cap.state, AVAILABLE)
        # The table must show the RUNTIME version, in every state.
        self.assertIn("2.31.2", cap.reason)
        self.assertEqual(cap.evidence["version"], "2.31.2")
        self.assertEqual(cap.evidence["library"], "libnccl.so.2")

    def test_active_when_the_run_co_locates_ranks(self):
        cap = probe_nccl_colocation(_env(engine_info=COLOCATED_ENGINE))
        self.assertEqual(cap.state, ACTIVE)
        self.assertIn("2.31.2", cap.reason)

    def test_spread_ranks_are_available_not_active(self):
        cap = probe_nccl_colocation(_env(engine_info=SPREAD_ENGINE))
        self.assertEqual(cap.state, AVAILABLE)

    def test_rank_gpu_id_list_form(self):
        cap = probe_nccl_colocation(_env(engine_info={"rank_gpu_id": [0, 1, 1, 2]}))
        self.assertEqual(cap.state, ACTIVE)


class TestMpsProbe(CustomTestCase):
    def test_missing_control_directory_names_dir_and_remedy(self):
        cap = probe_mps(_env())
        self.assertEqual(cap.state, UNAVAILABLE)
        self.assertIn("/tmp/nvidia-mps", cap.reason)
        self.assertIn("nvidia-cuda-mps-control -d", cap.reason)

    def test_present_control_directory_is_available(self):
        cap = probe_mps(
            _env(
                exists=lambda p: p == "/tmp/nvidia-mps",
                listdir=lambda p: ["control", "log"],
            )
        )
        self.assertEqual(cap.state, AVAILABLE)
        self.assertEqual(cap.evidence["entries"], ["control", "log"])

    def test_active_when_the_run_co_locates_ranks(self):
        cap = probe_mps(
            _env(
                exists=lambda p: p == "/tmp/nvidia-mps",
                engine_info=COLOCATED_ENGINE,
            )
        )
        self.assertEqual(cap.state, ACTIVE)

    def test_pipe_directory_override_is_respected(self):
        seen = []

        def exists(p):
            seen.append(p)
            return p == "/run/mps"

        cap = probe_mps(
            _env(exists=exists, env={"CUDA_MPS_PIPE_DIRECTORY": "/run/mps"})
        )
        self.assertEqual(cap.state, AVAILABLE)
        self.assertIn("/run/mps", seen)
        self.assertNotIn("/tmp/nvidia-mps", seen)

    def test_decided_by_directory_not_by_process_scan(self):
        # The probe must never shell out (no pgrep/ps pattern matching);
        # a run() call would show up here.
        calls = []

        def run(cmd):
            calls.append(cmd)
            return (0, "")

        probe_mps(_env(run=run))
        self.assertEqual(calls, [])


class TestFullTableRows(CustomTestCase):
    def test_both_rows_present_with_reasons(self):
        report = probe_all(_env())
        keys = {c.key for c in report.capabilities}
        self.assertIn("nccl_colocation", keys)
        self.assertIn("mps", keys)
        for key in ("nccl_colocation", "mps"):
            cap = report.by_key(key)
            if cap.state == UNAVAILABLE:
                self.assertTrue(cap.reason)


if __name__ == "__main__":
    unittest.main()
