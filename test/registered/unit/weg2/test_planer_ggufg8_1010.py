"""NF-GGUF AP G8 (2026-10-10): stage A of the metal order -- the D-only profile ``nf-gguf-d``, the draft-directory contract and the
two helper scripts with their ``--check`` mode.

* ``TestProfile``       ``tools/release/profconv/nf-gguf-d.env``: D-only (``--d-only`` once, PROFILE_D_ONLY / port 30032), the served
                        model name instead of the file name, the draft as the symlink FILE inside the draft directory (never the
                        bare directory), the required paths (sibling, draft dir + file + config.json, store dir), vision off; and the
                        rest of the argv equals nf-gguf.env's, token for token (the form cannot drift from the GGUF flip profile).
* ``TestHelpers``       ``make_nf_gguf_sibling.sh`` / ``make_nf_gguf_draft_dir.sh``: build, idempotence, ``--check`` (read only; each
                        failure named: dead symlink, no GGUF magic, missing / foreign config.json, incomplete split set).
* ``TestDryRunGolden``  the launcher dry run of nf-gguf-d on the reference rig (RTX 5090 + 2x RTX 3080) from the committed header
                        snapshots == ``golden/nf/plan_nf_gguf_d_n3.txt`` (0 diff lines), rc 0, nothing forced, and NO W172 / W173 /
                        W163 prediction; box-bound like every dry-run golden here (skips with the reason where the census /
                        evidence files of this box are absent).

GPU-free, NVML-free, Docker-free.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import shlex
import stat
import subprocess
import tempfile
import unittest

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
PROFILES = os.path.join(TREE, "tools", "release", "profconv")
PROFILE = os.path.join(PROFILES, "nf-gguf-d.env")
BASE = os.path.join(PROFILES, "nf-gguf.env")
SIB_SH = os.path.join(TREE, "tools", "release", "make_nf_gguf_sibling.sh")
DRAFT_SH = os.path.join(TREE, "tools", "release", "make_nf_gguf_draft_dir.sh")
SERVED = "Qwen3.8-Flash-Next-GGUF"
DRAFT_FILE = "mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf"


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _argv_value(argv, flag):
    i = len(argv) - 1 - argv[::-1].index(flag)
    return argv[i + 1]


class TestProfile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.li = O.profile_launch_input(PROFILE, asset_dirs=())
        cls.base = O.profile_launch_input(BASE, asset_dirs=())

    def test_the_profile_is_d_only_on_the_gguf_form(self):
        v = self.li.vars
        self.assertEqual(v["PROFILE_NAME"], "nf-gguf-d")
        self.assertEqual(v["PROFILE_D_ONLY"], "1")
        self.assertEqual(v["PROFILE_SERVE_PORT"], "30032")
        self.assertEqual(v["PROFILE_FORMAT"], "gguf")
        self.assertEqual(v["PROFILE_STATUS"], "experimentell")
        self.assertEqual(self.li.argv.count("--d-only"), 1)
        self.assertEqual(self.li.argv[0], "--d-only")

    def test_the_rest_of_the_argv_is_nf_gguf_env_with_exactly_three_deltas(self):
        """--d-only is new; the draft path moved into the draft directory; --served-model-name was added to both extras. Nothing else."""
        want = [t for t in self.base.argv]
        got = [t for t in self.li.argv if t != "--d-only"]
        old, new = self.base.draft, self.li.draft
        self.assertNotEqual(old, new)
        self.assertEqual(len(got), len(want))
        changed = []
        for a, b in zip(want, got):
            if a == b:
                continue
            self.assertEqual(b, a.replace(old, new) + " --served-model-name " + SERVED if old in a else None, (a[:80], b[:80]))
            changed.append(b)
        self.assertEqual(len(changed), 2)  # --extra-p and --extra-d
        for t in changed:
            self.assertTrue(t.endswith(" --served-model-name " + SERVED))

    def test_the_served_name_is_not_the_file_name(self):
        extra_d = _argv_value(self.li.argv, "--extra-d")
        words = shlex.split(extra_d)
        self.assertEqual(words[-2:], ["--served-model-name", SERVED])
        self.assertEqual(words.count("--served-model-name"), 1)
        self.assertNotIn("00001-of-00003", SERVED)

    def test_the_draft_is_the_symlink_file_inside_the_draft_dir_never_the_directory(self):
        draft = self.li.draft
        self.assertTrue(draft.endswith(DRAFT_FILE))
        self.assertEqual(os.path.basename(os.path.dirname(draft)), "Qwen3.8-Flash-Next-GGUF-unsloth-draft")
        self.assertEqual(self.li.vars["PROFILE_DRAFT_DIR"], os.path.dirname(draft))
        for flag in ("--extra-p", "--extra-d"):
            words = shlex.split(_argv_value(self.li.argv, flag))
            self.assertEqual(words[words.index("--speculative-draft-model-path") + 1], draft)
        # the model/tokenizer pair of the base is unchanged
        self.assertEqual(_argv_value(self.li.argv, "--model"), _argv_value(self.base.argv, "--model"))
        self.assertEqual(_argv_value(self.li.argv, "--tokenizer-path"), _argv_value(self.base.argv, "--tokenizer-path"))

    def test_the_required_paths_name_sibling_draft_dir_draft_file_config_and_store(self):
        raw = _read(PROFILE)
        block = raw.split("PROFILE_REQUIRED_PATHS=(")[1].split("\n)\n")[0]
        for need in (
            '"$PROFILE_SIBLING"',
            '"$PROFILE_MODEL"',
            '"$PROFILE_TOKENIZER/config.json"',
            '"$PROFILE_TOKENIZER/tokenizer.json"',
            '"$PROFILE_DRAFT_DIR"',
            '"$PROFILE_DRAFT"',
            '"$PROFILE_DRAFT_DIR/config.json"',
            '"$NF_STORE_DIR"',
        ):
            self.assertIn(need, block)
        self.assertNotIn("safetensors", block)
        for flag in ("--env-p", "--env-d"):  # NF_STORE_DIR is not a PROFILE_* variable: read it where the launcher gets it
            self.assertIn("SGLANG_MOE_EXPERT_STORE_DIR=/mnt/nf-experts/gguf", _argv_value(self.li.argv, flag).replace("FLLIPER_", "SGLANG_"))

    def test_vision_is_off_and_ple_checkpoint_backend_is_kept(self):
        self.assertEqual(_argv_value(self.li.argv, "--pdflip-vision"), "off")
        for flag in ("--extra-p", "--extra-d"):
            e = _argv_value(self.li.argv, flag)
            self.assertIn("--ple-offload-embedding", e)
            self.assertIn("--ple-offload-backend checkpoint", e)

    def test_the_committed_launch_json_is_this_profile(self):
        want = json.loads(_read(os.path.join(GOLDEN, "nf", "launch_nf-gguf-d.json")))
        got = O.launch_input_doc(O.profile_launch_input(PROFILE, asset_dirs=()))
        self.assertEqual(got, want, "regenerate: python -m sglang.srt.weg2.propose_oracle launch --profile %s --out ..." % PROFILE)


def _gguf_stub(path, payload=b"GGUF\x03\x00\x00\x00"):
    with open(path, "wb") as fh:
        fh.write(payload)


def _sh(script, *args, env=None):
    return subprocess.run(["bash", script, *args], capture_output=True, text=True, env={**os.environ, **(env or {})})


class TestHelpers(unittest.TestCase):
    def test_both_scripts_are_executable_and_parse(self):
        for s in (SIB_SH, DRAFT_SH):
            self.assertTrue(os.access(s, os.X_OK), s)
            self.assertEqual(subprocess.run(["bash", "-n", s], capture_output=True).returncode, 0)

    # -- sibling --------------------------------------------------------------------------------------------------------

    def _sibling_sources(self, td):
        parts, orig = os.path.join(td, "parts"), os.path.join(td, "orig")
        os.makedirs(parts)
        os.makedirs(orig)
        for i in (1, 2, 3):
            _gguf_stub(os.path.join(parts, "m-0000%d-of-00003.gguf" % i))
        for n in ("config.json", "tokenizer.json", "tokenizer_config.json"):
            with open(os.path.join(orig, n), "w") as fh:
                json.dump({"model_type": "qwen4_exp", "hidden_size": 8} if n == "config.json" else {"x": 1}, fh)
        return {"NF_GGUF_PARTS_DIR": parts, "NF_GGUF_CONFIG_SRC": orig}

    def test_sibling_check_passes_on_what_the_builder_made_and_names_every_defect(self):
        with tempfile.TemporaryDirectory(prefix="g8-sib-") as td:
            env = self._sibling_sources(td)
            dest = os.path.join(td, "out")
            self.assertEqual(_sh(SIB_SH, dest, env=env).returncode, 0)
            snap = sorted(os.listdir(dest))
            ok = _sh(SIB_SH, "--check", dest, env=env)
            self.assertEqual(ok.returncode, 0, ok.stderr)
            self.assertIn("OK (3 gguf part(s)", ok.stdout)
            self.assertEqual(sorted(os.listdir(dest)), snap, "--check wrote something")
            # a dead symlink
            os.symlink("/nonexistent/x-00004-of-00003.gguf", os.path.join(dest, "z-00003-of-00003.gguf"))
            bad = _sh(SIB_SH, "--check", dest, env=env)
            self.assertEqual(bad.returncode, 1)
            self.assertIn("does not resolve", bad.stderr)
            os.unlink(os.path.join(dest, "z-00003-of-00003.gguf"))
            # a part without the GGUF magic
            real = os.path.realpath(os.path.join(dest, "m-00002-of-00003.gguf"))
            with open(real, "wb") as fh:
                fh.write(b"NOPE")
            bad = _sh(SIB_SH, "--check", dest, env=env)
            self.assertEqual(bad.returncode, 1)
            self.assertIn("no GGUF magic", bad.stderr)
            _gguf_stub(real)
            # a missing part of the split set
            os.unlink(os.path.join(dest, "m-00003-of-00003.gguf"))
            bad = _sh(SIB_SH, "--check", dest, env=env)
            self.assertEqual(bad.returncode, 1)
            self.assertIn("split set says 3 parts", bad.stderr)
            os.symlink(os.path.join(env["NF_GGUF_PARTS_DIR"], "m-00003-of-00003.gguf"), os.path.join(dest, "m-00003-of-00003.gguf"))
            # config.json that is not JSON, tokenizer missing
            with open(os.path.join(dest, "config.json"), "w") as fh:
                fh.write("not json")
            os.unlink(os.path.join(dest, "tokenizer.json"))
            bad = _sh(SIB_SH, "--check", dest, env=env)
            self.assertEqual(bad.returncode, 1)
            self.assertIn("config.json is not valid JSON", bad.stderr)
            self.assertIn("tokenizer.json missing", bad.stderr)
            # no directory at all
            self.assertEqual(_sh(SIB_SH, "--check", os.path.join(td, "nope"), env=env).returncode, 1)

    # -- draft dir ------------------------------------------------------------------------------------------------------

    def _draft_sources(self, td):
        mtp = os.path.join(td, "MTP", DRAFT_FILE)
        os.makedirs(os.path.dirname(mtp))
        _gguf_stub(mtp)
        orig = os.path.join(td, "orig")
        os.makedirs(orig)
        cfg = {"model_type": "qwen4_exp", "text_config": {"hidden_size": 2560, "hc_count": 4, "num_hidden_layers": 48, "vocab_size": 248320}}
        with open(os.path.join(orig, "config.json"), "w") as fh:
            json.dump(cfg, fh)
        sib = os.path.join(td, "sib")
        os.makedirs(sib)
        with open(os.path.join(sib, "config.json"), "w") as fh:
            json.dump(cfg, fh)
        return {"NF_GGUF_MTP_FILE": mtp, "NF_GGUF_CONFIG_SRC": orig, "NF_GGUF_SIBLING": sib}, cfg

    def test_draft_dir_is_a_symlink_plus_the_config_and_check_reads_nothing_big(self):
        with tempfile.TemporaryDirectory(prefix="g8-draft-") as td:
            env, cfg = self._draft_sources(td)
            dest = os.path.join(td, "draft")
            r = _sh(DRAFT_SH, dest, env=env)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(sorted(os.listdir(dest)), ["config.json", DRAFT_FILE])
            self.assertTrue(os.path.islink(os.path.join(dest, DRAFT_FILE)))
            self.assertEqual(json.load(open(os.path.join(dest, "config.json"))), cfg)
            self.assertEqual(_sh(DRAFT_SH, dest, env=env).returncode, 0)  # idempotent
            snap = {n: os.lstat(os.path.join(dest, n)).st_mtime_ns for n in os.listdir(dest)}
            ok = _sh(DRAFT_SH, "--check", dest, env=env)
            self.assertEqual(ok.returncode, 0, ok.stderr)
            self.assertIn("not the directory", ok.stdout)  # the contract sentence: the server gets the FILE
            self.assertEqual({n: os.lstat(os.path.join(dest, n)).st_mtime_ns for n in os.listdir(dest)}, snap, "--check wrote something")

    def test_draft_dir_config_falls_back_to_the_target_sibling_when_the_original_is_absent(self):
        with tempfile.TemporaryDirectory(prefix="g8-draft-fb-") as td:
            env, cfg = self._draft_sources(td)
            env["NF_GGUF_CONFIG_SRC"] = os.path.join(td, "no-original")
            dest = os.path.join(td, "draft")
            self.assertEqual(_sh(DRAFT_SH, dest, env=env).returncode, 0)
            self.assertEqual(json.load(open(os.path.join(dest, "config.json"))), cfg)
            env["NF_GGUF_SIBLING"] = os.path.join(td, "no-sibling")
            self.assertEqual(_sh(DRAFT_SH, os.path.join(td, "d2"), env=env).returncode, 1)  # no config anywhere: refused by name

    def test_draft_dir_check_names_every_defect(self):
        with tempfile.TemporaryDirectory(prefix="g8-draft-bad-") as td:
            env, cfg = self._draft_sources(td)
            dest = os.path.join(td, "draft")
            self.assertEqual(_sh(DRAFT_SH, dest, env=env).returncode, 0)
            link = os.path.join(dest, DRAFT_FILE)
            # the geometry of ANOTHER model (what W173 would refuse at boot)
            foreign = json.loads(json.dumps(cfg))
            foreign["text_config"]["hidden_size"] = 4096
            with open(os.path.join(dest, "config.json"), "w") as fh:
                json.dump(foreign, fh)
            bad = _sh(DRAFT_SH, "--check", dest, env=env)
            self.assertEqual(bad.returncode, 1)
            self.assertIn("foreign to the target sibling", bad.stderr)
            with open(os.path.join(dest, "config.json"), "w") as fh:
                json.dump(cfg, fh)
            # no config.json
            os.unlink(os.path.join(dest, "config.json"))
            bad = _sh(DRAFT_SH, "--check", dest, env=env)
            self.assertIn("config.json missing or empty (W172 at boot)", bad.stderr)
            with open(os.path.join(dest, "config.json"), "w") as fh:
                json.dump(cfg, fh)
            # a regular copy instead of the symlink, and a file without the magic
            real = os.path.realpath(link)
            os.unlink(link)
            with open(link, "wb") as fh:
                fh.write(b"NOPE")
            bad = _sh(DRAFT_SH, "--check", dest, env=env)
            self.assertEqual(bad.returncode, 1)
            self.assertIn("is not a symlink", bad.stderr)
            self.assertIn("no GGUF magic", bad.stderr)
            os.unlink(link)
            # a dead symlink
            os.symlink(os.path.join(td, "gone.gguf"), link)
            bad = _sh(DRAFT_SH, "--check", dest, env=env)
            self.assertIn("does not resolve", bad.stderr)
            os.unlink(link)
            os.symlink(real, link)
            self.assertEqual(_sh(DRAFT_SH, "--check", dest, env=env).returncode, 0)
            # two .gguf files: ambiguous
            _gguf_stub(os.path.join(dest, "other.gguf"))
            self.assertIn("exactly one .gguf", _sh(DRAFT_SH, "--check", dest, env=env).stderr)
            self.assertEqual(_sh(DRAFT_SH, "--check", os.path.join(td, "nope"), env=env).returncode, 1)


# ---------------------------------------------------------------------------
# the dry run
# ---------------------------------------------------------------------------

def _snapshots():
    out = {}
    for n in sorted(os.listdir(CKPT)):
        if os.path.isfile(os.path.join(CKPT, n, "manifest.json")):
            out[O.read_snapshot_manifest(os.path.join(CKPT, n))["name"]] = os.path.join(CKPT, n)
    return out


def _inputs_present():
    try:
        li = O.profile_launch_input(PROFILE)
    except Exception:
        return False
    return not li.unresolved_paths and os.path.isfile(os.path.join(GOLDEN, "nf", "plan_nf_gguf_d_n3.txt"))


_LINE = O.launcher_line()


@unittest.skipUnless(_inputs_present(), "the census / evidence files the nf profile names are not on this box")
@unittest.skipUnless(_LINE == O.LINE_NF, "the golden is of the NF launcher line (golden/nf/)")
class TestDryRunGolden(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prun = O.run_profile(PROFILE, O.read_replay(REPLAY_REF), tree=TREE, snapshots=_snapshots())
        cls.dump = cls.prun.result.dump()

    def test_the_dump_equals_the_golden(self):
        res = self.prun.result
        self.assertIsNone(res.exc_type, "%s: %s" % (res.exc_type, res.exc_msg[:300]))
        self.assertEqual(res.rc, 0)
        want = _read(os.path.join(GOLDEN, "nf", "plan_nf_gguf_d_n3.txt"))
        d = O.diff_lines(want, self.dump)
        self.assertEqual(d, [], "plan diff vs plan_nf_gguf_d_n3.txt: %d lines\n%s" % (len(d), "\n".join(x[:240] for x in d[:12])))
        self.assertEqual(self.prun.result.forced, [])

    def test_no_config_refusal_is_predicted_for_the_target_or_the_draft(self):
        """The metal order's expectation without the draft directory was a W172 for the draft; with it the dry run is clean."""
        for code in ("W172", "W173", "W163", "Weg2GgufConfigMissing", "Weg2GgufMetaMismatch", "Weg2TokenizerRefused"):
            self.assertNotIn(code, self.dump, code)

    def test_it_is_a_d_only_plan_with_the_draft_priced_from_the_gguf_header(self):
        self.assertIn("D-ONLY: no flip arm", self.dump)
        self.assertIn("budget D(d-only, expectation)", self.dump)
        self.assertIn("vision=off", self.dump)
        self.assertIn("draft=mtp", self.dump)
        self.assertIn("d_draft_host=2711 MiB", self.dump)  # the shared-Q8_0 file: checkpoint minus the target-shared tables
        self.assertIn("mtp-Qwen3.8-Flash-Next-shared-Q8_0", self.dump)

    def test_group_d_serves_under_the_model_name_and_gets_the_draft_file(self):
        line = [ln for ln in self.dump.splitlines() if "group D argv:" in ln]
        self.assertEqual(len(line), 1)
        import re

        names = re.findall(r"--served-model-name ([^\s'\"]+)", line[0])
        # the launcher's own (derived: the FILE name) comes first, the profile's after it; argparse takes the last
        self.assertEqual(len(names), 2, names)
        self.assertTrue(names[0].endswith("00001-of-00003.gguf"), names)
        self.assertEqual(names[-1], SERVED)
        self.assertIn("-unsloth-draft/mtp-Qwen3", line[0].replace("<TREE>", "."))

    def test_the_golden_provenance_names_the_inputs(self):
        side = json.loads(_read(os.path.join(GOLDEN, "nf", "plan_nf_gguf_d_n3.provenance.json")))
        self.assertEqual(side["golden"], "plan_nf_gguf_d_n3.txt")
        self.assertEqual(side["profile"]["sha256"], _sha(PROFILE))
        self.assertEqual(side["profile_base"]["sha256"], _sha(BASE))
        snaps = _snapshots()
        for name in side["checkpoints"]:
            self.assertIn(name, snaps)
