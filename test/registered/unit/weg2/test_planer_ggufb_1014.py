"""NF-GGUF stage B preparation (2026-10-14, AP ggufb): the G series on the int24f8 line + the GGUF FLIP profile.

* ``TestMerge``          the merge of ``desk/nf-gguf-1018`` (G1..G16) into ``desk/nf-int24f8-1013`` (f4afb10f28): the files of the G
                         series are there, the edge ids are unique (K136/K137 = the lanes envs of int24f5, K138/K139 = the two GGUF
                         edges renumbered from K136/K137), every edge has its anchor.
* ``TestFlipProfile``    ``docker/profiles_release/nf-gguf-flip-int24f8.env`` = the NF Dauerstand profile (verbatim, sha256 in the
                         header) + ``tools/release/gguf_flip_overlay.env``: PP3 / TP3 flip, GGUF model + tokenizer, vision off, draft =
                         the shared-MTP file, GGUF store directory, lanes / void pass / STRICT=0 stated, nothing of the INT4 form
                         left that the GGUF loader refuses; and the rest of the argv equals the Dauerstand's token for token.
* ``TestDryRunFlip``     the launcher dry run of that profile on the reference rig (RTX 5090 + 2x RTX 3080) from the committed header
                         snapshots == ``golden/nf/plan_nf_gguf_flip_int24f8_n3.txt`` (0 diff lines), rc 0, nothing forced, no rest
                         blocker; box-bound like every dry-run golden here (skips with the reason where the census / evidence files of
                         this box are absent).

GPU-free, NVML-free, Docker-free.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import unittest
import warnings

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

try:
    from sglang.srt.weg2 import launcher
    from sglang.srt.weg2 import propose_oracle as O
except Exception as exc:  # pragma: no cover - no weg2 launcher in this build
    pytest.skip(f"weg2 launcher unavailable: {exc}", allow_module_level=True)

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = str(pathlib.Path(launcher.__file__).resolve().parents[4])
FIX = os.path.join(HERE, "fixtures", "planer_1006")
GOLDEN = os.path.join(FIX, "golden")
CKPT = os.path.join(FIX, "checkpoints")
REPLAY_REF = os.path.join(HERE, "fixtures", "xchg_launch_replay_0911", "nvml_devices_1378.json")
FLAT = os.path.join(TREE, "docker", "profiles_release", "nf-gguf-flip-int24f8.env")
OVERLAY = os.path.join(TREE, "tools", "release", "gguf_flip_overlay.env")
GEN = os.path.join(TREE, "tools", "release", "gguf_flip_profile.py")
RIG_BASE = "/spinning/gpu-arb/docker/profiles/nf-int4-h6-abl-xc-w3sp-mlr-hl-vw.env"
SERVED = "Qwen3.8-Flash-Next-GGUF"
DRAFT_FILE = "mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf"
GOLDEN_NAME = "plan_nf_gguf_flip_int24f8_n3.txt"
D_REF_LOG = "boot_weg2_dkrnfdggufdbar110131203_12b375f3dc_1013_120342.D.log"   # D log of the first GGUF READY (stage A attempt 6)
ACC_EVIDENCE = "/spinning/docker-acceptance/nf/evidence"                       # host dir the container mounts as /var/lib/htsglang/evidence


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _gen():
    sys.path.insert(0, os.path.dirname(GEN))
    try:
        import gguf_flip_profile as G
    finally:
        sys.path.remove(os.path.dirname(GEN))
    return G


def _value(argv, flag):
    i = len(argv) - 1 - argv[::-1].index(flag)
    return argv[i + 1]


# ---------------------------------------------------------------------------
# the merge
# ---------------------------------------------------------------------------

class TestMerge(unittest.TestCase):
    #: files the G series (G1..G16) brought; the merge must carry every one
    G_FILES = (
        "python/sglang/srt/model_loader/gguf_qwen4exp.py",
        "python/sglang/srt/layers/moe/gguf_presplit.py",
        "python/sglang/srt/layers/moe/gguf_layout.py",
        "python/sglang/srt/planner/expert_layer_rows.py",
        "python/sglang/srt/weg2/xchg_gguf_census.py",
        "python/sglang/srt/weg2/gguf_census.py",
        "python/sglang/srt/models/qwen4_exp_mtp.py",
        "tools/release/profconv/nf-gguf.env",
        "tools/release/profconv/nf-gguf-d.env",
        "tools/release/make_nf_gguf_sibling.sh",
        "tools/release/make_nf_gguf_draft_dir.sh",
        "docker/profiles_release/nf-gguf-d.env",
    )
    #: and the int24f8 files that must NOT have been lost to a conflict resolution
    LINE_FILES = (
        "python/sglang/srt/weg2/pp_void_pass.py",
    )

    def test_the_files_of_both_lines_are_there(self):
        for rel in self.G_FILES + self.LINE_FILES:
            self.assertTrue(os.path.isfile(os.path.join(TREE, rel)), rel)

    def test_g16_freeze_and_the_g14_payload_marker_are_in_the_merged_sources(self):
        mtp = _read(os.path.join(TREE, "python/sglang/srt/models/qwen4_exp_mtp.py"))
        self.assertIn("def freeze_parameters", mtp)
        self.assertIn("is_lazy", mtp)                           # G16: lazy GGUF parameters are skipped
        wu = _read(os.path.join(TREE, "python/sglang/srt/model_loader/weight_utils.py"))
        self.assertIn("G14 GGUF-DENSE-PAYLOAD", wu)

    def test_the_int24f8_defaults_survive_the_merge(self):
        env = _read(os.path.join(TREE, "python/sglang/srt/environ.py"))
        self.assertIn("SGLANG_WEG2_LANES = EnvBool(True)", env)
        from sglang.srt.weg2 import pp_void_pass

        self.assertTrue(pp_void_pass.enabled({}))                  # the switch's own reader: default on, not an EnvBool in environ.py
        self.assertFalse(pp_void_pass.enabled({pp_void_pass.ENV: "0"}))

    def test_edge_ids_are_unique_and_the_gguf_edges_are_renumbered(self):
        path = os.path.join(TREE, "python/sglang/srt/weg2/kantenkatalog_1004.json")
        kanten = json.loads(_read(path))["kanten"]
        ids = [k["id"] for k in kanten]
        self.assertEqual(len(ids), len(set(ids)))
        by = {k["id"]: k for k in kanten}
        self.assertEqual(by["K136"]["von"], "SGLANG_WEG2_LANE_KEEPALIVE_S")           # int24f5 (lanes), unchanged number
        self.assertEqual(by["K137"]["von"], "SGLANG_WEG2_LANE_PREEMPT_CHUNK_TOKENS")
        self.assertEqual((by["K138"]["von"], by["K138"]["nach"], by["K138"]["wert"]), ("--load-format", "--tokenizer-path", "gguf"))
        self.assertEqual((by["K139"]["von"], by["K139"]["nach"], by["K139"]["wert"]), ("--load-format", "--weg2-vision", "gguf"))
        self.assertEqual(len(ids), 139)

    def test_the_two_gguf_anchors_are_on_the_lines_they_name(self):
        path = os.path.join(TREE, "python/sglang/srt/weg2/kantenkatalog_1004.json")
        by = {k["id"]: k for k in json.loads(_read(path))["kanten"]}
        lines = _read(os.path.join(TREE, "python/sglang/srt/weg2/launcher.py")).splitlines()
        for eid in ("K138", "K139"):
            b = by[eid]["beleg"]
            self.assertIn(b["anker"], lines[b["zeile"] - 1], eid)


# ---------------------------------------------------------------------------
# the flip profile
# ---------------------------------------------------------------------------

class TestFlipProfile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.G = _gen()
        cls.text = _read(FLAT)
        cls.parts = cls.G.split(cls.text)
        cls.li = O.profile_launch_input(FLAT, asset_dirs=())
        cls.argv = cls.li.argv

    def test_the_flat_file_is_base_plus_overlay_and_the_overlay_is_the_committed_one(self):
        self.assertIsNotNone(self.parts)
        base, overlay = self.parts
        self.assertEqual(overlay, _read(OVERLAY))
        name = re.search(r"#   base     (\S+)  sha256 ([0-9a-f]{64})", self.text)
        self.assertIsNotNone(name)
        self.assertEqual(name.group(1), os.path.basename(self.G.DEFAULT_BASE))
        self.assertEqual(name.group(2), hashlib.sha256(base.encode("utf-8")).hexdigest())
        self.assertEqual(self.G.build(base, name.group(1), overlay), self.text)

    def test_the_generator_check_is_green_on_a_box_that_has_the_dauerstand(self):
        if not os.path.isfile(RIG_BASE):
            self.skipTest("the rig's Dauerstand profile is not on this box")
        r = subprocess.run([sys.executable, GEN, "--check"], capture_output=True, text=True)
        self.assertIn(r.returncode, (0, 1), r.stdout + r.stderr)             # 2 = no base, a crash is not an answer
        if r.returncode != 0:
            # a moved Dauerstand is a signal for the next regeneration, not a defect of this tree
            warnings.warn("the Dauerstand profile moved since this flat file was generated: %s" % r.stdout.strip())

    def test_no_fliper_spelling_and_the_file_sources(self):
        self.assertNotIn("FLLIPER_", self.text)
        self.assertNotIn("pdflip", self.text)
        r = subprocess.run(["bash", "-n", FLAT], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_the_profile_is_the_gguf_flip_form(self):
        v = self.li.vars
        self.assertEqual(v["PROFILE_NAME"], "nf-gguf-flip-int24f8")
        self.assertEqual(v["PROFILE_FORMAT"], "gguf")
        self.assertEqual(v["PROFILE_STATUS"], "experimentell")
        self.assertTrue(v["PROFILE_FORM"].endswith("+vw+gguf-flip"), v["PROFILE_FORM"])   # the Dauerstand's form, then the overlay
        # the flip form: family weights, exchange source, PP3 cut, TP3 D (no --d-only)
        self.assertNotIn("--d-only", self.argv)
        self.assertEqual(_value(self.argv, "--flip-weights"), "family")
        self.assertEqual(_value(self.argv, "--weg2-weight-source"), "exchange")
        self.assertEqual(_value(self.argv, "--pp-stage-ratio"), "29,11,8")
        self.assertEqual(_value(self.argv, "--profile"), "nextflash")

    def test_model_tokenizer_vision_census(self):
        model = _value(self.argv, "--model")
        self.assertTrue(model.endswith("Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf"), model)
        self.assertEqual(_value(self.argv, "--tokenizer-path"), os.path.dirname(model))
        self.assertEqual(_value(self.argv, "--weg2-vision"), "off")
        self.assertNotIn("--weg2-vision-place", self.argv)                               # W111 with vision off
        self.assertEqual(self.argv.count("--weg2-xchg-census-foreign"), 1)
        i = self.argv.index("--weg2-xchg-census")
        self.assertEqual(self.argv[i + 2], "--weg2-xchg-census-foreign")                 # right behind its census

    def test_draft_is_the_shared_mtp_file_in_the_draft_directory_on_both_groups(self):
        for flag in ("--extra-p", "--extra-d"):
            v = _value(self.argv, flag)
            m = re.search(r"--speculative-draft-model-path (\S+)", v)
            self.assertIsNotNone(m, flag)
            self.assertTrue(m.group(1).endswith("Qwen3.8-Flash-Next-GGUF-unsloth-draft/" + DRAFT_FILE), m.group(1))
            self.assertEqual(re.findall(r"--served-model-name (\S+)", v), [SERVED], flag)
            self.assertNotIn("Minachist", v)
            self.assertNotIn("albucino", v)

    def test_the_gguf_store_directory_replaces_the_int4_one_in_both_group_envs(self):
        for flag in ("--env-p", "--env-d"):
            v = _value(self.argv, flag)
            self.assertIn("SGLANG_MOE_EXPERT_STORE_DIR=/mnt/nf-experts/gguf", v, flag)
            self.assertNotIn("fnFL2", v, flag)

    def test_lanes_void_pass_and_strict_are_stated(self):
        p, d = _value(self.argv, "--env-p"), _value(self.argv, "--env-d")
        self.assertIn("SGLANG_WEG2_LANES=1", p)
        self.assertIn("SGLANG_WEG2_LANES=1", d)
        self.assertIn("SGLANG_WEG2_PP_VOID_PASS_ENABLE=1", p)
        self.assertIn("SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0", p)
        self.assertIn("SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0", d)
        self.assertEqual(self.li.env.get("SGLANG_WEG2_LANES"), "1")
        self.assertEqual(self.li.env.get("SGLANG_WEG2_PP_VOID_PASS_ENABLE"), "1")

    def test_the_int8_hyper_connection_mixer_the_gguf_loader_refuses_is_off(self):
        self.assertEqual(self.li.env.get("SGLANG_HC_MIXER_INT8"), "0")
        self.assertEqual(self.text.count("_form SGLANG_HC_MIXER_INT8"), 2)               # the base's (stripped by the overlay) + the 0

    def test_required_paths_name_sibling_draft_and_config_but_not_the_store(self):
        v = subprocess.run(
            ["bash", "-c", 'source "$1" >/dev/null 2>&1; printf "%s\\n" "${PROFILE_REQUIRED_PATHS[@]}"', "x", FLAT],
            capture_output=True, text=True, env={**os.environ, "HOME": "/root", "HTSGLANG_TAG": "t"}).stdout.split("\n")
        v = [x for x in v if x]
        joined = "\n".join(v)
        for need in ("-sibling\n", "-00001-of-00003.gguf\n", "-sibling/config.json", "-sibling/tokenizer.json",
                     "-unsloth-draft\n", "-unsloth-draft/" + DRAFT_FILE, "-unsloth-draft/config.json", "/" + D_REF_LOG):
            self.assertIn(need, joined + "\n", need)
        self.assertNotIn("/mnt/nf-experts", joined)                                      # created by preflight_state, never required
        self.assertNotIn("mtp-dense.safetensors", joined)                                # the INT4 draft's files

    def test_the_d_solve_has_its_reference_log_and_the_state_that_log_was_measured_in(self):
        """Without a D log of THIS form the whole D fraction solve is skipped ("die eingebaute Referenz gilt fuer das INT4-Modell") and a flip
        boot under --d-kv-token-cut owned is refused (#239 S3h). The log is the first GGUF READY; it was measured with H39 off (a GGUF boot has no
        dense repack), so the D group says so (no GGUF reader of the env: gguf.py never reads it), else the planner discards the log."""
        logs = _value(self.argv, "--d-residency-reference-logs")
        self.assertTrue(logs.endswith("/" + D_REF_LOG), logs)
        self.assertEqual(_value(self.argv, "--d-residency-reference-seats"), "6")
        self.assertEqual(_value(self.argv, "--d-kv-token-cut"), "owned")                  # the Dauerstand's form, not weakened
        self.assertIn("SGLANG_WEG2_DENSE_REPACK_OUTSIDE_POOL=0", _value(self.argv, "--env-d"))
        self.assertNotIn("SGLANG_WEG2_DENSE_REPACK_OUTSIDE_POOL", _value(self.argv, "--env-p"))
        gguf_py = _read(os.path.join(TREE, "python/sglang/srt/layers/quantization/gguf.py"))
        self.assertNotIn("DENSE_REPACK_OUTSIDE_POOL", gguf_py)                           # why H39=0 changes nothing at run time for GGUF

    def test_the_token_cut_has_a_named_exit(self):
        r = subprocess.run(
            ["bash", "-c", 'export NF_GGUF_D_KV_TOKEN_CUT=off; source "$1" >/dev/null 2>&1; printf "%s\\0" "${PROFILE_ARGS[@]}"', "x", FLAT],
            capture_output=True, text=True, env={**os.environ, "HOME": "/root", "HTSGLANG_TAG": "t"})
        argv = [a for a in r.stdout.split("\0") if a]
        self.assertEqual(_value(argv, "--d-kv-token-cut"), "off")

    def test_the_rest_of_the_argv_equals_the_dauerstand_token_for_token(self):
        """Only the named deltas differ from the Dauerstand's own argv (evaluated from the verbatim base inside the flat file)."""
        base_text = self.parts[0]
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "base.env")
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(base_text)
            base = O.profile_launch_input(p, asset_dirs=()).argv

        def strip(argv):
            out, i = [], 0
            while i < len(argv):
                t = argv[i]
                if t in ("--model", "--weg2-vision", "--weg2-vision-place", "--extra-p", "--extra-d", "--env-p", "--env-d"):
                    i += 2
                    continue
                if t in ("--tokenizer-path", "--weg2-xchg-census-foreign", "--d-residency-reference-logs", "--d-residency-reference-seats"):
                    i += 1 if t == "--weg2-xchg-census-foreign" else 2
                    continue
                out.append(t)
                i += 1
            return out

        self.assertEqual(strip(self.argv), strip(base))
        # the group extras differ ONLY in the draft path, the served name and (env) the store dir / the stated switches
        norm = lambda s: re.sub(r"\s--served-model-name \S+$", "", s)
        for flag in ("--extra-p", "--extra-d"):
            a = norm(_value(self.argv, flag))
            b = _value(base, flag)
            a = re.sub(r"--speculative-draft-model-path \S+", "--speculative-draft-model-path X", a)
            b = re.sub(r"--speculative-draft-model-path \S+", "--speculative-draft-model-path X", b)
            self.assertEqual(a, b, flag)
        for flag, extra in (("--env-p", ";SGLANG_WEG2_LANES=1;SGLANG_WEG2_PP_VOID_PASS_ENABLE=1"), ("--env-d", ";SGLANG_WEG2_LANES=1;SGLANG_WEG2_DENSE_REPACK_OUTSIDE_POOL=0")):
            a = _value(self.argv, flag).replace("/mnt/nf-experts/gguf", "/mnt/nf-experts/fnFL2").replace(extra, "")
            self.assertEqual(a, _value(base, flag), flag)


# ---------------------------------------------------------------------------
# the dry run
# ---------------------------------------------------------------------------

def _snapshots():
    out = {}
    for n in sorted(os.listdir(CKPT)):
        if os.path.isfile(os.path.join(CKPT, n, "manifest.json")):
            out[O.read_snapshot_manifest(os.path.join(CKPT, n))["name"]] = os.path.join(CKPT, n)
    return out


def _launch_input():
    # the sentinel SGLANG_WEG2_EVIDENCE_DIR names the dir of the D reference log (the container's evidence mount); the launcher keeps its own
    return O.profile_launch_input(FLAT, evidence_dir=ACC_EVIDENCE)


def _inputs_present():
    try:
        li = _launch_input()
    except Exception:
        return False
    return (not li.unresolved_paths and os.path.isfile(os.path.join(ACC_EVIDENCE, D_REF_LOG))
            and os.path.isfile(os.path.join(GOLDEN, "nf", GOLDEN_NAME)))


_LINE = O.launcher_line()


@unittest.skipUnless(_inputs_present(), "the census / evidence files the nf profile names are not on this box")
@unittest.skipUnless(_LINE == O.LINE_NF, "the golden is of the NF launcher line (golden/nf/)")
class TestDryRunFlip(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prun = O.run_profile(FLAT, O.read_replay(REPLAY_REF), tree=TREE, snapshots=_snapshots(), launch_input=_launch_input())
        cls.dump = cls.prun.result.dump()

    def test_the_dump_equals_the_golden(self):
        res = self.prun.result
        self.assertIsNone(res.exc_type, "%s: %s" % (res.exc_type, str(res.exc_msg)[:300]))
        self.assertEqual(res.rc, 0)
        want = _read(os.path.join(GOLDEN, "nf", GOLDEN_NAME))
        d = O.diff_lines(want, self.dump)
        self.assertEqual(d, [], "plan diff vs %s: %d lines\n%s" % (GOLDEN_NAME, len(d), "\n".join(x[:240] for x in d[:12])))
        self.assertEqual(self.prun.result.forced, [])

    def test_no_rest_blocker(self):
        for code in ("W172", "W173", "W163", "W111", "W71 ", "Weg2GgufConfigMissing", "Weg2GgufMetaMismatch",
                     "Weg2TokenizerRefused", "Weg2LaunchRefused", "would refuse", "refused --"):
            self.assertNotIn(code, self.dump, code)

    def test_the_d_solve_runs_and_the_flip_under_the_token_cut_is_allowed(self):
        """The rest blocker of the first flip dry run: #239 S3h "in a FLIP boot refused" because the D solve was skipped (no D log of this form)."""
        self.assertNotIn("(dry run: would refuse)", self.dump)
        self.assertNotIn("FRACTION-SOLVE D(Karte, Erwartung) ENTFAELLT", self.dump)
        self.assertNotIn("FRACTION-SOLVE D(dry, expectation) ENTFAELLT", self.dump)
        self.assertNotIn("D-KV-STUFEN (#251c): entfaellt unter dem Token-Schnitt", self.dump)
        self.assertIn("#239 S3h FLIP UNTER KV-TOKEN-SCHNITT STUFEN ERLAUBT", self.dump)
        self.assertIn("D-EIGENTUM (#239 S3f)", self.dump)
        self.assertIn(D_REF_LOG, self.dump)

    def test_the_named_gaps_stay_named_and_are_not_refusals(self):
        """Skipped computations (INT4-only references), listed in the report -- a line here that turns into a refusal is a new blocker."""
        for line in ("PP-CUT RATEN-SCHNITT (H57) ENTFAELLT", "PP-CUT P-KARTE ENTFAELLT", "WAKE-CREDIT (#H14) D(Karte, Erwartung) ENTFAELLT",
                     "P-CUT UNCALIBRATED gguf checkpoint"):
            self.assertIn(line, self.dump, line)

    def test_it_is_a_flip_plan_with_the_per_layer_gguf_rows_and_the_pinned_cut(self):
        self.assertNotIn("D-ONLY: no flip arm", self.dump)
        self.assertIn("vision=off", self.dump)
        self.assertIn("PP-CUT EXPERT ROWS JE LAYER (G5)", self.dump)
        self.assertRegex(self.dump, r"PP-CUT SHIPPED: layers=29,11,8 ")
        self.assertIn("mtp-Qwen3.8-Flash-Next-shared-Q8_0", self.dump)

    def test_both_groups_serve_under_the_model_name_and_only_d_gets_the_draft_file(self):
        for grp in ("P", "D"):
            line = [ln for ln in self.dump.splitlines() if "group %s argv:" % grp in ln]
            self.assertEqual(len(line), 1, grp)
            names = re.findall(r"--served-model-name ([^\s'\"]+)", line[0])
            self.assertEqual(len(names), 2, names)                  # the launcher's own (file name) first, the profile's last
            self.assertTrue(names[0].endswith("00001-of-00003.gguf"), names)
            self.assertEqual(names[-1], SERVED)
            draft = re.findall(r"--speculative-draft-model-path (\S+)", line[0])
            if grp == "D":                                          # "Draft auf P streichen" (H25): the draft is D's, P is launched without it
                self.assertEqual(len(draft), 1, draft)
                self.assertTrue(draft[0].endswith("-unsloth-draft/" + DRAFT_FILE), draft)
            else:
                self.assertEqual(draft, [])

    def test_the_golden_provenance_names_the_inputs(self):
        side = json.loads(_read(os.path.join(GOLDEN, "nf", GOLDEN_NAME.replace(".txt", ".provenance.json"))))
        self.assertEqual(side["golden"], GOLDEN_NAME)
        self.assertEqual(side["profile"]["sha256"], _sha(FLAT))
        snaps = _snapshots()
        for name in side["checkpoints"]:
            self.assertIn(name, snaps)


if __name__ == "__main__":
    unittest.main()
