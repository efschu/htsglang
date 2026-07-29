"""CPU tests for the P2P readiness re-probe package (scripts/p2p_readiness/).

Only the pure parsing / planning / diff logic: NVML topo parsing, BAR
parsing and classification (nominal vs effective separation), the NCCL
transport grep, the size ladder crossing the 256-MiB window, the
effective-aperture search plan, and the JSON schema. No CUDA, no torch.cuda,
no subprocesses onto the cards -- the package is built to be dry-testable
while the driver update runs.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts", "p2p_readiness")
sys.path.insert(0, SCRIPTS_DIR)

import p2p_common as pc  # noqa: E402


class TestBarParsing(CustomTestCase):
    SMI_TEXT = """\
==============NVSMI LOG==============

Attached GPUs                             : 2
GPU 00000000:01:00.0
    FB Memory Usage
        Total                             : 20480 MiB
    BAR1 Memory Usage
        Total                             : 256 MiB
        Used                              : 5 MiB
        Free                              : 251 MiB

GPU 00000000:C1:00.0
    FB Memory Usage
        Total                             : 32768 MiB
    BAR1 Memory Usage
        Total                             : 32768 MiB
        Used                              : 32 MiB
        Free                              : 32736 MiB
"""

    def test_parse_smi_bar1(self):
        out = pc.parse_smi_bar1(self.SMI_TEXT)
        self.assertEqual(out["0000:01:00.0"], 256 * pc.MIB)
        self.assertEqual(out["0000:c1:00.0"], 32768 * pc.MIB)

    def test_parse_lspci_regions(self):
        text = """\
01:00.0 VGA compatible controller: NVIDIA Corporation GA102
	Region 0: Memory at a0000000 (32-bit, non-prefetchable) [size=16M]
	Region 1: Memory at 4000000000 (64-bit, prefetchable) [size=256M]
	Region 3: Memory at 4010000000 (64-bit, prefetchable) [size=32M]
"""
        self.assertEqual(
            pc.parse_lspci_regions(text),
            [256 * pc.MIB, 32 * pc.MIB, 16 * pc.MIB],
        )

    def test_classify_bar(self):
        # 3080 shape: 256-MiB window on 20-GiB VRAM
        self.assertEqual(pc.classify_bar(256 * pc.MIB, 20 * pc.GIB), "windowed")
        # 5090 shape: full-VRAM BAR
        self.assertEqual(pc.classify_bar(32 * pc.GIB, 32 * pc.GIB), "full")
        self.assertEqual(pc.classify_bar(None, 20 * pc.GIB), "unknown")

    def test_normalize_pci(self):
        for raw in ("00000000:01:00.0", "0000:01:00.0", "01:00.0"):
            self.assertEqual(pc.normalize_pci(raw), "0000:01:00.0")


class TestTopoParsing(CustomTestCase):
    TOPO = """\
	GPU0	GPU1	GPU2	CPU Affinity	NUMA Affinity
GPU0	 X 	PHB	PHB	0-31	0
GPU1	PHB	 X 	PHB	0-31	0
GPU2	PHB	PHB	 X 	0-31	0

Legend:
  X    = Self
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge
"""

    def test_parse_topo_matrix(self):
        out = pc.parse_topo_matrix(self.TOPO)
        self.assertEqual(out[("GPU0", "GPU1")], "PHB")
        self.assertEqual(out[("GPU2", "GPU0")], "PHB")
        self.assertEqual(out[("GPU1", "GPU1")], "X")
        self.assertEqual(len(out), 9)


class TestNcclTransportGrep(CustomTestCase):
    LOG = """\
host:123:145 [0] NCCL INFO Channel 00/0 : 0[1000] -> 1[c1000] via P2P/CUMEM
host:123:145 [0] NCCL INFO Channel 01/0 : 0[1000] -> 1[c1000] via P2P/CUMEM
host:124:146 [1] NCCL INFO Channel 00 : 1[c1000] -> 0[1000] via SHM/direct/direct
host:124:146 [1] NCCL INFO Channel 01/0 : 1[c1000] -> 0[1000] via NET/Socket/0
irrelevant line
"""

    def test_parse_rows(self):
        rows = pc.parse_nccl_transports(self.LOG)
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]["transport_class"], "P2P")
        self.assertEqual(rows[2]["transport_class"], "SHM")
        self.assertEqual(rows[3]["transport_class"], "NET")
        self.assertEqual(rows[0]["src_rank"], 0)
        self.assertEqual(rows[0]["dst_busid"], "c1000")

    def test_summary_reports_mixture_as_finding(self):
        rows = pc.parse_nccl_transports(self.LOG)
        summary = pc.summarize_transport_classes(rows)
        self.assertEqual(summary["0->1"], "P2P")
        # mixed transports on one pair are reported, not collapsed
        self.assertEqual(summary["1->0"], "NET+SHM")


class TestProbePlanning(CustomTestCase):
    def test_ladder_crosses_the_window_boundary(self):
        ladder = pc.size_ladder(64 * pc.KIB, pc.GIB)
        window = pc.SMALL_BAR_WINDOW_BYTES
        for must_have in (window - pc.MIB, window, window + pc.MIB, pc.GIB):
            self.assertIn(must_have, ladder)
        self.assertEqual(ladder, sorted(ladder))
        self.assertEqual(len(ladder), len(set(ladder)))
        # doubling ladder actually reaches beyond the boundary
        self.assertGreater(max(ladder), window)

    def test_aperture_search_plan_grows_then_bisects(self):
        # growing phase
        self.assertEqual(pc.aperture_search_plan(0, None), pc.MIB)
        self.assertEqual(pc.aperture_search_plan(pc.MIB, None), 2 * pc.MIB)
        # first failure at 256 MiB with 128 MiB known-good: bisect
        nxt = pc.aperture_search_plan(128 * pc.MIB, 256 * pc.MIB)
        self.assertEqual(nxt, 192 * pc.MIB)
        # terminates at 1-MiB resolution
        self.assertIsNone(pc.aperture_search_plan(255 * pc.MIB, 256 * pc.MIB))

    def test_search_terminates(self):
        """The plan converges for any monotone failure boundary."""
        boundary = 251 * pc.MIB  # an "effectively less than nominal" aperture
        largest_ok, first_fail, steps = 0, None, 0
        while True:
            size = pc.aperture_search_plan(largest_ok, first_fail)
            if size is None or (first_fail is None and size > pc.GIB):
                break
            if size < boundary:
                largest_ok = size
            else:
                first_fail = size
            steps += 1
            self.assertLess(steps, 64, "search must terminate")
        self.assertLessEqual(largest_ok, boundary)
        self.assertGreaterEqual(largest_ok, boundary - pc.MIB)


class TestSchemaAndDiff(CustomTestCase):
    def test_directed_pair_nominal_vs_effective_are_separate(self):
        """The register consumes ONLY effective values; both must exist as
        distinct fields and default to unmeasured."""
        p = pc.DirectedPairResult(src_pci="a", dst_pci="b")
        d = pc.asdict_fallback(p)
        self.assertIn("dst_bar1_nominal_bytes", d)
        self.assertIn("effective_max_single_copy_bytes", d)
        self.assertIn("effective_max_region_chunked_bytes", d)
        self.assertIsNone(d["effective_max_single_copy_bytes"])
        self.assertEqual(d["probe_errors"], [])

    def test_write_json_roundtrip_with_dataclasses(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "x", "out.json")
            payload = pc.result_envelope("capability_matrix")
            payload["devices"] = [
                pc.DeviceInfo(
                    pci_bus_id="0000:01:00.0",
                    name="RTX 3080",
                    uuid="GPU-x",
                    nvml_index=0,
                    cuda_index=2,
                    vram_total_bytes=20 * pc.GIB,
                    bar1_total_bytes=256 * pc.MIB,
                    bar1_classification="windowed",
                )
            ]
            pc.write_json(path, payload)
            with open(path) as f:
                loaded = json.load(f)
            self.assertEqual(loaded["schema_version"], pc.SCHEMA_VERSION)
            self.assertEqual(loaded["kind"], "capability_matrix")
            dev = loaded["devices"][0]
            self.assertEqual(dev["bar1_classification"], "windowed")
            # the device-order trap: NVML and cuda indices are BOTH present
            self.assertEqual(dev["nvml_index"], 0)
            self.assertEqual(dev["cuda_index"], 2)


class TestDryRuns(CustomTestCase):
    """The scripts' argument/plan paths run without CUDA."""

    def _run(self, script, *extra):
        r = subprocess.run(
            [sys.executable, os.path.join(SCRIPTS_DIR, script), "--dry-run", *extra],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=SCRIPTS_DIR,
        )
        self.assertEqual(r.returncode, 0, f"{script}: {r.stdout}\n{r.stderr}")
        return r.stdout

    def test_capability_matrix_dry(self):
        out = self._run("capability_matrix.py")
        self.assertIn("NOMINAL", out)
        self.assertIn("effective-aperture", out)

    def test_d2d_bench_dry(self):
        out = self._run("d2d_bench.py")
        self.assertIn("255MiB", out)
        self.assertIn("256MiB", out)
        self.assertIn("257MiB", out)
        self.assertIn("dual-window", out)

    def test_nccl_transport_dry(self):
        out = self._run("nccl_transport_check.py")
        self.assertIn("P2P/SHM/NET", out)
        self.assertIn("baseline", out)

    def test_run_all_dry(self):
        r = subprocess.run(
            [
                "bash",
                os.path.join(SCRIPTS_DIR, "run_all.sh"),
                "--dry-run",
                "--results-dir",
                tempfile.mkdtemp(prefix="p2p_dry_"),
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        self.assertEqual(r.returncode, 0, f"{r.stdout}\n{r.stderr}")
        self.assertIn("skipping lock acquisition", r.stdout)

    def test_run_all_lock_paths_are_directories_with_info(self):
        """The arbitration contract: mkdir-atomic lock DIRECTORY + info file.
        Verified against the script text, since the dry run (correctly)
        never takes locks."""
        with open(os.path.join(SCRIPTS_DIR, "run_all.sh")) as f:
            text = f.read()
        self.assertIn("/tmp/gpu-card-$i.lock", text)
        self.assertIn('mkdir "$lock"', text)
        self.assertIn("holder=", text)
        self.assertIn("heartbeat=", text)
        self.assertNotIn("rm -rf /tmp/gpu-card-*", text, "never steal locks")


if __name__ == "__main__":
    unittest.main()
