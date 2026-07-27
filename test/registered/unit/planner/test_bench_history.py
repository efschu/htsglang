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
"""Unit tests for the per-model benchmark run history.

Pure filesystem; no server, no network. The history root is redirected to a
temp dir through the env var the module reads.
"""

import json
import os
import tempfile
import unittest
from unittest import mock

from sglang.srt.planner import bench_history


class _Rooted(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="bhist_")
        self._env = mock.patch.dict(
            os.environ, {"SGLANG_PLANNER_BENCH_HISTORY": self.tmp})
        self._env.start()

    def tearDown(self):
        self._env.stop()
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, model, **kw):
        rec = {
            "model": model,
            "endpoint": "http://127.0.0.1:30000",
            "results": [{"test_id": 1, "status": "pass", "label": "Basic"}],
            "transcript": [{"test_id": 1, "request": {"messages": []},
                            "answer": "Paris"}],
        }
        rec.update(kw)
        return bench_history.save_run(rec)


class TestSlug(unittest.TestCase):
    def test_path_tail_is_kept_readable(self):
        self.assertEqual(
            bench_history.model_slug("/models/cache/Qwen3.6-27B-GGUF"),
            "cache__Qwen3.6-27B-GGUF")

    def test_hf_id_keeps_both_halves(self):
        self.assertEqual(bench_history.model_slug("Qwen/Qwen3.6-27B"),
                         "Qwen__Qwen3.6-27B")

    def test_unnamed_target_is_still_stored(self):
        # A run without a model name is still a run someone made.
        self.assertEqual(bench_history.model_slug(""), "_unknown")
        self.assertEqual(bench_history.model_slug(None), "_unknown")

    def test_separators_cannot_escape_the_directory(self):
        slug = bench_history.model_slug("../../etc/passwd")
        self.assertNotIn("/", slug)
        self.assertFalse(slug.startswith("."))


class TestSaveAndList(_Rooted):
    def test_round_trip_keeps_the_transcript(self):
        rid = self._run("/m/Model-A")
        rec = bench_history.load_run(rid)
        self.assertIsNotNone(rec)
        self.assertEqual(rec["model"], "/m/Model-A")
        self.assertEqual(rec["transcript"][0]["answer"], "Paris")

    def test_runs_are_grouped_per_model(self):
        self._run("/m/Model-A")
        self._run("/m/Model-B")
        a = bench_history.list_runs(model="/m/Model-A")
        self.assertEqual(len(a), 1)
        self.assertEqual(a[0]["model"], "/m/Model-A")
        self.assertEqual(len(bench_history.list_runs()), 2)

    def test_summary_counts_statuses(self):
        self._run("/m/Model-A", results=[
            {"test_id": 1, "status": "pass"},
            {"test_id": 2, "status": "fail"},
            {"test_id": 3, "status": "pass"},
        ])
        s = bench_history.list_runs(model="/m/Model-A")[0]
        self.assertEqual(s["counts"], {"pass": 2, "fail": 1})
        self.assertEqual(s["n_tests"], 3)

    def test_newest_first(self):
        self._run("/m/Model-A", started_at=1000.0, run_id="old")
        self._run("/m/Model-A", started_at=2000.0, run_id="new")
        ids = [r["run_id"] for r in bench_history.list_runs(model="/m/Model-A")]
        self.assertEqual(ids, ["new", "old"])

    def test_corrupt_file_is_skipped_not_fatal(self):
        self._run("/m/Model-A")
        d = os.path.join(self.tmp, bench_history.model_slug("/m/Model-A"))
        with open(os.path.join(d, "broken.json"), "w") as f:
            f.write("{ not json")
        self.assertEqual(len(bench_history.list_runs(model="/m/Model-A")), 1)

    def test_missing_root_lists_nothing(self):
        self.assertEqual(bench_history.list_runs(model="/never/created"), [])
        self.assertEqual(bench_history.list_runs(), [])

    def test_load_rejects_path_traversal(self):
        self._run("/m/Model-A")
        for bad in ("../secret", "a/b", ".hidden", ""):
            self.assertIsNone(bench_history.load_run(bad))

    def test_delete(self):
        rid = self._run("/m/Model-A")
        self.assertTrue(bench_history.delete_run(rid))
        self.assertIsNone(bench_history.load_run(rid))
        self.assertFalse(bench_history.delete_run(rid))

    def test_write_is_atomic_no_tmp_left_behind(self):
        self._run("/m/Model-A")
        d = os.path.join(self.tmp, bench_history.model_slug("/m/Model-A"))
        self.assertFalse([n for n in os.listdir(d) if n.endswith(".tmp")])

    def test_stored_file_is_plain_json(self):
        rid = self._run("/m/Model-A")
        d = os.path.join(self.tmp, bench_history.model_slug("/m/Model-A"))
        with open(os.path.join(d, rid + ".json"), encoding="utf-8") as f:
            self.assertEqual(json.load(f)["run_id"], rid)


if __name__ == "__main__":
    unittest.main()
