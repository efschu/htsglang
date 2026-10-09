"""F0-K (RC1 acceptance 09.10.2026, M1 + M2 + the generic hold): what the rename must NOT have changed in a persisted format.

The acceptance found two spellings that go into FILES and HASHES and were renamed anyway: the header magic of the L3 index snapshot
(``store_journal.MAGIC``, M1) and the seed of the salted KV page keys (``mem_cache.utils._NAMESPACE_SEED_TAG``, M2).  The first costs a full
L3 walk on the way in and on the way back, the second turns every salted L2/L3 page into a silent miss.  The rename kit cannot see this class
(its AST comparison hides strings, the rest inventory counts what is left over), so this test holds the renamed tree against the FREEZE tree:

* ``TestNamedPins``        the two RC1 findings by name, plus the L3 store identity (directory name digest, key suffix hash, generation) against
                           values computed with the FREEZE code (the model of the Flash-Next release, ``dc6a8c2062``, cannot be recomputed on a box
                           without the weights; the functions that make it are pinned instead);
* ``TestBytesVsFreeze``    every ``bytes`` constant of every Python file of the freeze tree is still there, byte for byte, in the renamed file -- except
                           the four wire/NVRTC names that are consistent inside one file (listed below with the reason);
* ``TestFormatIdsVsFreeze`` every old-family format id (``<name>/<n>``) is either unchanged or on the RENAMED table, and the table says how the
                           old spelling is still read (dual reader) or why it need not be.

The fixture is made by ``tools/release/persisted_consts_1009.py --ref <freeze>`` (27B: 86ff356d0d; Flash-Next: c651892375 = freeze + 22 fix commits).
"""

import hashlib
import importlib
import importlib.util
import json
import pathlib
import subprocess
import sys
import types
import unittest

from flliper.test.ci.ci_register import register_cpu_ci
from flliper.test.test_utils import CustomTestCase

register_cpu_ci(est_time=60, suite="base-a-test-cpu")

ROOT = pathlib.Path(__file__).resolve().parents[4]
FIX = pathlib.Path(__file__).resolve().parent / "fixtures" / "f0k_persisted_consts_1009.json"
SG = "sg" "lang"
W2 = "WE" "G2"
w2 = "we" "g2"

#: byte constants that DID change, with the reason they may.  Key: (renamed path, freeze repr).  Value: (current repr, why)
ALLOWED_BYTES = {
    ("python/flliper/srt/pdflip/front.py", "b'%s_resumable_depth'" % w2): ("b'pdflip_resumable_depth'",
        "JSON field name of the P/D handoff (RC1 acceptance m2): producer and reader are the same tree; an EXTERNAL client of the field has no alias yet"),
    ("python/flliper/srt/pdflip/front.py", "b'%s_seq_hash'" % w2): ("b'pdflip_seq_hash'", "same as above (SEQ-HASH mark of the leg-2 stream)"),
    ("python/flliper/srt/pdflip/lane_sm_copy.py", "b'%s_lane_copy'" % w2): ("b'pdflip_lane_copy'",
        "NVRTC kernel name: the CUDA source string of the same file declares the kernel under the new name (checked below)"),
    ("python/flliper/srt/pdflip/lane_sm_copy.py", "b'%s_lane_copy.cu'" % w2): ("b'pdflip_lane_copy.cu'", "NVRTC program label, no reader"),
}

#: format ids that changed spelling; each names how the old one stays readable or why no reader is needed.
RENAMED = {
    "%s.state/1" % w2: ("pdflip.state/1", "dual", "dashboard names.schema_ok reads both; the engine reads only the state.json of its own boot"),
    "%s.event/1" % w2: ("pdflip.event/1", "dual", "dashboard names.schema_ok reads both; events.jsonl of the own boot"),
    "%s.rankstats/1" % w2: ("pdflip.rankstats/1", "dual", "dashboard names.schema_ok reads both; rank files are timer-written per boot"),
    "%s.rank_vram/1" % w2: ("pdflip.rank_vram/1", "boot-local", "block written by a rank and read by the front of the SAME boot (state dir is per boot id); no other reader"),
    "%s.vram_plan/1" % w2: ("pdflip.vram_plan/1", "boot-local", "plan written by the launcher of a boot and read by its ranks; each boot writes its own"),
    "%s-node-0" % SG: ("flliper-node-0", "ci-name", "pod name of the Ascend NPU CI cluster test; not a rig artefact"),
}
#: format ids that are PERSISTED and must stay as they were (the kit protects them: tools/release/rename_to_flliper.py DENY_SPAN / W_DENY_SPAN)
FROZEN = ("%s.expert_stats/1" % SG, "%s.forward_peak/1" % SG, "%s-footprint/1" % w2, "%s-lane-coverage-1" % w2, "%s-pp-calib/1" % w2,
          "%s-x-curves/1" % w2, "%s.form_measures/2" % w2, "%s.form_measures/3" % w2, "htsglang-rig-artifact/v1")


def _fixture():
    return json.loads(FIX.read_text(encoding="utf-8"))


def _current():
    p = str(ROOT / "tools" / "release")
    if p not in sys.path:
        sys.path.insert(0, p)
    return importlib.import_module("persisted_consts_1009").from_root(str(ROOT))


class TestNamedPins(CustomTestCase):
    def test_l3_snapshot_magic_is_the_freeze_spelling(self):
        from flliper.srt.mem_cache.storage.file import store_journal as sj
        self.assertEqual(sj.MAGIC, ("%s-L3-INDEX v3" % W2).encode())

    def test_l3_snapshot_written_now_is_readable_by_the_freeze_header_check(self):
        """The header line of a snapshot this tree writes must start with the freeze magic + a space (what the freeze ``load_snapshot`` tests)."""
        from flliper.srt.mem_cache.storage.file import store_journal as sj
        self.assertTrue((sj.MAGIC + b" n=7\n").startswith(("%s-L3-INDEX v3 " % W2).encode()))

    def test_namespace_seed_is_the_freeze_spelling_and_the_page_key_digest_is_pinned(self):
        from flliper.srt.mem_cache import utils as mu
        self.assertEqual(mu._NAMESPACE_SEED_TAG, ("%s-kv-namespace-v1" % SG).encode() + b"\0")
        # computed with the freeze tree's namespace_root_hash (86ff356d0d and c651892375 give the same value)
        self.assertEqual(mu.namespace_root_hash("tenant-a"), "f745ea8fe246ceb0f3cd3eea690938e1897be800f245ba0200ad0eb7d01cb77e")
        self.assertIsNone(mu.namespace_root_hash(None))   # the unsalted chain keeps its key exactly

    def test_l3_store_identity_functions_are_the_freeze_ones(self):
        from flliper.srt.pdflip import launcher as L
        ident = {"model_path": "/models/synthetic-27b", "model_config_sha1": "0123456789abcdef0123456789abcdef01234567", "weights_fp": "w-fp-0001",
                 "profile": "27b-nvfp4", "form_kv": "fp8", "kv_cache_dtype": "fp8_e4m3", "override_p": "", "override_d": "", "vision": "",
                 "generation": "706"}
        # computed with the freeze launcher's l3_persist_dir_name (86ff356d0d and c651892375 agree): the digest of the NAMED identity dict
        self.assertEqual(L.l3_persist_dir_name(ident), "l3-27b-nvfp4-synthetic-27b-8f9c80316e")
        self.assertEqual((L.L3_PERSIST_PREFIX, L.L3_PERSIST_GENERATION, L.L3_ROPE_APPLY), ("l3-", "706", "merge-v1"))

    def test_storage_key_suffix_hash_is_the_freeze_one(self):
        from flliper.srt.mem_cache.hicache_storage import compute_model_identity_hash
        a = types.SimpleNamespace(model_path="/models/synthetic-27b", revision=None, dtype="bfloat16", quantization="compressed-tensors",
                                  kv_cache_dtype="fp8_e4m3", rank_tp_ratio=None, rank_kv_ratio=None)
        self.assertEqual(compute_model_identity_hash(a), "421234b6ef4b57d3")
        a.rank_tp_ratio = "13,6,6"
        self.assertEqual(compute_model_identity_hash(a), "a993cb59e3131585")

    def test_the_pinned_digests_are_what_the_seed_says(self):
        self.assertEqual(hashlib.sha256(("%s-kv-namespace-v1" % SG).encode() + b"\0" + b"tenant-a").hexdigest(),
                         "f745ea8fe246ceb0f3cd3eea690938e1897be800f245ba0200ad0eb7d01cb77e")


class TestBytesVsFreeze(CustomTestCase):
    def test_every_bytes_constant_of_the_freeze_tree_is_still_there(self):
        fx, cur = _fixture()["bytes"], _current()["bytes"]
        self.assertGreater(len(fx), 50, "fixture looks empty")
        bad = []
        for path, old in sorted(fx.items()):
            if path not in cur:
                bad.append((path, "no bytes constants left in the file (file gone or constants removed)", old[:3]))
                continue
            have = list(cur[path])
            for r in old:
                if r in have:
                    have.remove(r)
                    continue
                alt = ALLOWED_BYTES.get((path, r))
                if alt and alt[0] in have:
                    have.remove(alt[0])
                    continue
                bad.append((path, "missing", r))
        self.assertEqual(bad, [], "a byte constant of the freeze tree is gone or changed: a persisted format / seed / wire name renamed?")

    def test_the_allowed_nvrtc_name_is_consistent_inside_its_file(self):
        t = (ROOT / "python/flliper/srt/pdflip/lane_sm_copy.py").read_text(encoding="utf-8")
        self.assertIn("pdflip_lane_copy(unsigned char*", t)       # the CUDA source declares the kernel under the name KERNEL_NAME carries
        self.assertIn('KERNEL_NAME = b"pdflip_lane_copy"', t)

    def test_allowed_table_has_no_dead_entries(self):
        fx = _fixture()["bytes"]
        for (path, r) in ALLOWED_BYTES:
            self.assertIn(r, fx.get(path, []), (path, r))


class TestFormatIdsVsFreeze(CustomTestCase):
    def test_every_old_family_format_id_is_frozen_or_on_the_renamed_table(self):
        fx, cur = _fixture()["format_ids"], _current()["format_ids"]
        bad = []
        for path, ids in sorted(fx.items()):
            have = list(cur.get(path, []))
            # the renamed spelling no longer carries an OLD word, so it is not in ``format_ids`` of the current side: read it from the source
            text = (ROOT / path).read_text(encoding="utf-8") if (ROOT / path).is_file() else ""
            for i in ids:
                if i in have:
                    continue                                            # unchanged
                new = RENAMED.get(i)
                if new and ('"%s"' % new[0]) in text:
                    continue
                bad.append((path, i))
        self.assertEqual(bad, [], "a format id changed its spelling and is not on the RENAMED table (or its new spelling is missing)")

    def test_the_frozen_ids_are_all_still_in_the_tree(self):
        fx, cur = _fixture()["format_ids"], _current()["format_ids"]
        seen_old = {i for ids in fx.values() for i in ids}
        for i in FROZEN:
            self.assertIn(i, seen_old, "fixture lacks %s" % i)
            self.assertTrue(any(i in ids for ids in cur.values()), "%s was renamed or removed" % i)

    def test_renamed_table_has_no_dead_entries(self):
        seen_old = {i for ids in _fixture()["format_ids"].values() for i in ids}
        for i in RENAMED:
            self.assertIn(i, seen_old, i)

    def test_dual_ids_are_read_in_both_spellings_by_the_dashboard_reader(self):
        spec = importlib.util.spec_from_file_location("rigdash_names_f0k", ROOT / "tools/rig_dashboard/rigdash/names.py")
        names = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = names
        spec.loader.exec_module(names)
        for old, (new, kind, _why) in RENAMED.items():
            if kind != "dual":
                continue
            self.assertTrue(names.schema_ok(old, new), old)
            self.assertTrue(names.schema_ok(new, new), new)
            self.assertFalse(names.schema_ok(new + "x", new))

    def test_no_consumer_of_the_renamed_planner_contracts_still_asks_for_the_old_spelling(self):
        """flliper.balken/1 -> flliper.bar/1 and flliper.verdikt/1 -> flliper.verdict/1 (German -> English, F0-D): producer AND consumers moved together (string literals; the comments that still quote the old id are prose)."""
        out = subprocess.run(["git", "-C", str(ROOT), "grep", "-nE", r"[\"']flliper\.(balken|verdikt)/1[\"']", "--", "python", "tools", "test"], capture_output=True, text=True)
        self.assertEqual(out.stdout.strip(), "")


if __name__ == "__main__":
    unittest.main()
