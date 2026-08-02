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
"""§13.10 window harness: the gates as arithmetic, and the smoke as a test.

The chunking window gets one boot; everything that can be wrong on the desk
must fail here instead. Two layers:

* the PURE gates (``structure_gate`` / ``coherence_gate``) over hand-built
  rows -- including every red and the VOID state, so the window can trust a
  green;
* the whole driver through ``run()`` against the fake lane server, once
  clean and once per dirty arm -- the desk-written-never-executed rule, plus
  the proof that each red branch fires for its own reason (a gate that fires
  everywhere localizes nothing).
"""

import importlib.util
import os
import sys
import threading
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_CHUNKING = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "..",
    "..",
    "..",
    "scripts",
    "dual_group",
    "chunking",
)


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_CHUNKING, f"{name}.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


pa = _load("probe_arms")
fls = _load("fake_lane_server")


def _row(n=1600, chunk=0, ms=100.0, ids=None):
    row = {"prefill_ms": ms, "output_ids": ids or []}
    if chunk > 0:
        import math

        k = math.ceil(n / chunk)
        row["prefill_chunks"] = k
        row["prefill_chunk_ms"] = [round(ms / k, 3)] * k
    return row


class TestStructureGate(CustomTestCase):
    def test_green_chunked_and_green_single(self):
        self.assertEqual(pa.structure_gate(_row(chunk=512), 1600, 512).state, "green")
        self.assertEqual(pa.structure_gate(_row(), 1600, 0).state, "green")

    def test_chunk_fields_on_a_single_forward_row_are_red(self):
        v = pa.structure_gate(_row(chunk=512), 1600, 0)
        self.assertEqual(v.state, "red")

    def test_missing_chunk_fields_are_red(self):
        v = pa.structure_gate(_row(), 1600, 512)
        self.assertEqual(v.state, "red")
        self.assertIn("override", v.reason)

    def test_wrong_chunk_count_is_red(self):
        row = _row(chunk=512)
        row["prefill_chunk_ms"] = row["prefill_chunk_ms"][:-1]
        row["prefill_chunks"] -= 1
        self.assertEqual(pa.structure_gate(row, 1600, 512).state, "red")

    def test_sum_mismatch_is_red_but_rounding_is_not(self):
        row = _row(chunk=512)
        row["prefill_chunk_ms"] = [x + 0.0001 for x in row["prefill_chunk_ms"]]
        self.assertEqual(pa.structure_gate(row, 1600, 512).state, "green")
        row["prefill_chunk_ms"] = [x / 2 for x in row["prefill_chunk_ms"]]
        self.assertEqual(pa.structure_gate(row, 1600, 512).state, "red")


class TestCoherenceGate(CustomTestCase):
    REF = list(range(64))

    def test_in_set_is_green(self):
        v = pa.coherence_gate(list(self.REF), [list(self.REF), list(self.REF)])
        self.assertEqual(v.state, "green")

    def test_divergence_from_exact_set_is_red(self):
        arm = list(self.REF)
        arm[18] = 7777
        v = pa.coherence_gate(arm, [list(self.REF), list(self.REF)])
        self.assertEqual(v.state, "red")
        self.assertEqual(v.detail["divergence_index"], 18)

    def test_divergence_inside_the_band_is_green(self):
        ref_b = list(self.REF)
        ref_b[10] = 5555  # the instrument's own noise floor: 10
        arm = list(self.REF)
        arm[12] = 7777
        v = pa.coherence_gate(arm, [list(self.REF), ref_b])
        self.assertEqual(v.state, "green")
        self.assertEqual(v.detail["floor"], 10)

    def test_divergence_before_the_band_is_red(self):
        ref_b = list(self.REF)
        ref_b[10] = 5555
        arm = list(self.REF)
        arm[3] = 7777
        v = pa.coherence_gate(arm, [list(self.REF), ref_b])
        self.assertEqual(v.state, "red")

    def test_floorless_reference_set_is_void_not_a_verdict(self):
        ref_b = [9999] + list(self.REF)[1:]
        v = pa.coherence_gate(list(self.REF), [list(self.REF), ref_b])
        self.assertEqual(v.state, "void")

    def test_length_difference_is_a_divergence(self):
        v = pa.coherence_gate(list(self.REF)[:32], [list(self.REF)])
        self.assertEqual(v.state, "red")
        self.assertEqual(v.detail["divergence_index"], 32)


class TestDriverSmoke(CustomTestCase):
    """The whole driver against the fake server, one process, no card."""

    def _run(self, dirty):
        fls.set_dirty(dirty)
        fls.STATE["total"] = 0
        fls.STATE["results"] = []
        srv = fls.serve(0)
        port = srv.server_address[1]
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            out = f"/tmp/chunking_harness_{dirty}_{port}.json"
            rc = pa.run(
                [
                    "--base",
                    f"http://127.0.0.1:{port}",
                    "--chunks",
                    "512,1024,2048",
                    "--prompt-tokens",
                    "1600",
                    "--ref-draws",
                    "3",
                    "--spec",
                    "off",
                    "--strict",
                    "--out",
                    out,
                ]
            )
            import json

            with open(out) as f:
                report = json.load(f)
            os.unlink(out)
            return rc, report
        finally:
            srv.shutdown()
            fls.set_dirty("none")

    def _states(self, report, gate):
        return {
            name: v["state"]
            for name, v in report["arms"]["nospec"]["verdicts"].items()
            if name.endswith(gate)
        }

    def test_clean_run_is_green_everywhere(self):
        rc, report = self._run("none")
        self.assertEqual(rc, 0)
        states = self._states(report, "structure")
        states.update(self._states(report, "coherence"))
        self.assertTrue(all(s == "green" for s in states.values()), states)

    def test_dirty_tail_fails_only_coherence(self):
        rc, report = self._run("tail")
        self.assertEqual(rc, 1)
        self.assertTrue(
            all(s == "green" for s in self._states(report, "structure").values())
        )
        coherence = self._states(report, "coherence")
        self.assertTrue(all(s == "red" for s in coherence.values()), coherence)

    def test_dirty_chunks_fails_only_structure(self):
        rc, report = self._run("chunks")
        self.assertEqual(rc, 1)
        chunked = {
            k: s
            for k, s in self._states(report, "structure").items()
            if k.startswith("chunk")
        }
        # The plant removes one chunk, so it cannot act on the degenerate
        # single-chunk arm (2048 over a 1600-token prompt) -- that arm STAYS
        # green, which is itself worth pinning: a gate that went red there
        # would be firing on something other than the plant.
        self.assertEqual(
            chunked,
            {
                "chunk512/structure": "red",
                "chunk1024/structure": "red",
                "chunk2048/structure": "green",
            },
        )
        self.assertTrue(
            all(s == "green" for s in self._states(report, "coherence").values())
        )

    def test_dirty_sum_fails_only_structure(self):
        rc, report = self._run("sum")
        self.assertEqual(rc, 1)
        chunked = {
            k: s
            for k, s in self._states(report, "structure").items()
            if k.startswith("chunk")
        }
        self.assertTrue(all(s == "red" for s in chunked.values()), chunked)

    def test_dirty_band_is_void_and_does_not_fail_the_run(self):
        rc, report = self._run("band")
        self.assertEqual(rc, 0)
        coherence = self._states(report, "coherence")
        self.assertTrue(all(s == "void" for s in coherence.values()), coherence)


if __name__ == "__main__":
    unittest.main()
