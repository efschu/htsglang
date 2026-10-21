"""NCARDS 1021: the N x RTX 3090 release profiles are consistent, the NF ones are PLANNER OUTPUT (not hand copies), the reference rig is untouched.

* every ``*-<n>x3090.env`` profile (27B INT8 / NVFP4 / NVFP4 dual, NF) names N cards, an N-entry inventory and N-entry per-card vectors;
* the generated block of ``nf-<n>x3090.env`` equals what ``tools/ncards_profile_gen.py`` (= ``propose()``) prints now: a drift is red;
* proposing again with the generated profile as the BASIS changes nothing (``same_as_profile``: the profile is a fixed point of the planner);
* the reference rig: the proposal of each rig release profile on the rig's own NVML recording is the profile byte for byte.

Profile files are read through ``propose_oracle.profile_launch_input`` (bash evaluates them exactly as the entrypoint does).
GPU-free.  The generator inputs are checkpoint HEADERS (``planer_1006/checkpoints``), no weights.
"""

import importlib.util
import os
import pathlib
import re
import subprocess
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest

try:
    from flliper.srt.pdflip import propose as P
    from flliper.srt.pdflip import propose_oracle as O
except Exception as exc:  # pragma: no cover - no pdflip planner in this build
    pytest.skip(f"pdflip planner unavailable: {exc}", allow_module_level=True)

HERE = pathlib.Path(__file__).resolve().parent
TREE = HERE.parents[3]
PROFILES = TREE / "docker" / "flliper" / "profiles_release"
REPLAY_REF = HERE / "fixtures" / "xchg_launch_replay_0911" / "nvml_devices_1378.json"
GEN = TREE / "tools" / "ncards_profile_gen.py"

FAMILIES = ("27b", "27b-nvfp4", "27b-nvfp4-dual", "nf")
NS = (2, 3, 4)
#: positional per-card vectors a profile may carry (flag / extra-group word): their entry count is N on an N-card profile
VECTOR_FLAGS = (("flag", "-", "--user-reserve-mib"), ("flag", "-", "--d-foreign-context-mib"), ("flag", "-", "--d-nontorch-mib"),
                ("flag", "-", "--pp-cut-expert-lru-rows"), ("flag", "-", "--pp-cut-expert-device-fraction"),
                ("extra", "p", "--rank-moe-resident-fraction"), ("extra", "d", "--rank-moe-resident-fraction"),
                ("extra", "d", "--rank-role"), ("extra", "d", "--rank-tp-ratio"), ("extra", "d", "--rank-moe-ratio"),
                ("extra", "p", "--rank-user-reserve-mib"), ("extra", "d", "--rank-user-reserve-mib"))


def _gen():
    spec = importlib.util.spec_from_file_location("ncards_profile_gen", str(GEN))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _li(name):
    return O.profile_launch_input(str(PROFILES / (name + ".env")))


def _csv_len(v):
    return len([x for x in str(v).split(",") if x != ""])


class TestProfileFiles(unittest.TestCase):
    def test_every_profile_names_n_cards_and_n_entry_vectors(self):
        for fam in FAMILIES:
            for n in NS:
                name = "%s-%dx3090" % (fam, n)
                path = PROFILES / (name + ".env")
                if not path.is_file():
                    self.fail("profile missing: %s" % name)
                with self.subTest(profile=name):
                    li = _li(name)
                    self.assertEqual(li.vars.get("PROFILE_CARD_COUNT"), str(n))
                    self.assertEqual(li.vars.get("PROFILE_INVENTORY"), ",".join(["RTX3090"] * n))
                    self.assertEqual(li.vars.get("PROFILE_GPUS"), "NVIDIA GeForce RTX 3090=%d" % n)
                    # NF on 2 and 4 identical 3090s: the launcher's D-EIGENTUM (#239 S3f) solve finds no bearable form, forced past only as the named D-BUDGET
                    # (review 1021 A1; the W122/W130 residency balance passes there); N = 3 is bootable
                    self.assertEqual(li.vars.get("PROFILE_STATUS"), "blockiert-d-budget" if name in ("nf-2x3090", "nf-4x3090") else "hochrechnung")
                    la = P.LaunchArgv(li.argv, li.env)
                    for kind, grp, flag in VECTOR_FLAGS:
                        val = la.get_flag(flag) if kind == "flag" else la.extra_get(grp, flag)
                        if val is None or "," not in str(val):
                            continue
                        self.assertEqual(_csv_len(val), n, "%s %s %s = %r" % (name, kind, flag, val))

    def test_shell_syntax(self):
        for p in sorted(PROFILES.glob("*x3090.env")) + [PROFILES / "_3090_common.sh"]:
            with self.subTest(p=p.name):
                r = subprocess.run(["bash", "-n", str(p)], capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, r.stderr)

    def test_setters_edit_in_place(self):
        script = (
            'source "%s/_3090_common.sh"; '
            'PROFILE_ARGS=(--x 1 --extra-d "--a 1 --rank-tp-ratio 1,0,0 --b 2" --env-d "K1=a;K2=b;K3=c" "--extra-p=--max-running-requests=1 --rank-gpu-memory-mib 6,7"); '
            '_p3090_extra d --rank-tp-ratio 5,5; _p3090_extra d --new 9; _p3090_extra_drop d --a; _p3090_env d K2 zz; _p3090_env d K9 q; '
            '_p3090_env_drop d K1; _p3090_extra p --rank-gpu-memory-mib 8,9; printf "%%s\\n" "${PROFILE_ARGS[@]}"' % PROFILES)
        out = subprocess.run(["bash", "-c", script], capture_output=True, text=True).stdout.splitlines()
        self.assertEqual(out, ["--x", "1", "--extra-d", "--rank-tp-ratio 5,5 --b 2 --new 9", "--env-d", "K2=zz;K3=c;K9=q",
                               "--extra-p=--max-running-requests=1 --rank-gpu-memory-mib 8,9"])


class TestNfIsPlannerOutput(unittest.TestCase):
    def test_generated_block_has_not_drifted(self):
        gen = _gen()
        for n in NS:
            with self.subTest(n=n):
                path = PROFILES / ("nf-%dx3090.env" % n)
                self.assertTrue(path.is_file(), path)
                have = gen.extract(path.read_text(encoding="utf-8"))
                self.assertTrue(have, "no generated block in %s" % path.name)
                self.assertEqual(have, gen.block("nf", n))

    def test_the_proposal_for_the_generated_profile_is_a_fixed_point(self):
        gen = _gen()
        for n in NS:
            with self.subTest(n=n):
                _, v = gen.proposal("nf", n)          # the proposal against the BASIS (release nf)
                li = _li("nf-%dx3090" % n)
                # the profile IS that proposal: no flag, group word or env item differs (order of appended flags is not a difference)
                self.assertEqual(gen.diff_lines(li.argv, li.env, v["argv"], v["env"]), [])


class TestReferenceRigUnchanged(unittest.TestCase):
    def test_rig_proposal_is_the_profile(self):
        gen = _gen()
        rows = O.read_replay(str(REPLAY_REF))
        for basis in ("nf",):
            with self.subTest(basis=basis):
                li = _li(basis)
                model = _model(gen, basis)
                rig = P.propose(rows, model, "flip", {}, basis=li, draft=_draft(gen, basis),
                                rates={"RTX 5090": 203.42, "RTX 3080 20GB": 50.97})
                self.assertTrue(rig["inventory"]["same_as_profile"])
                self.assertEqual(rig["argv"], list(li.argv))   # the rig is proposed as it is, byte for byte
                self.assertEqual(rig["env"], dict(li.env))


def _model(gen, basis):
    import tempfile
    from flliper.srt.pdflip import model_profile as MP
    env_file, model_dir, draft_dir = gen.BASES[basis]
    tmp = tempfile.mkdtemp(prefix="ncards-test-")
    os.environ["FLLIPER_CARD_LIBRARY"] = os.path.join(tmp, "none.json")
    for name in (model_dir, draft_dir):
        O.materialize_checkpoint(os.path.join(gen.CKPT, name), tmp)
    _model.cache = (tmp, model_dir, draft_dir)
    mp = MP.estimate_or_state(os.path.join(tmp, model_dir))
    assert mp["ok"], mp
    return mp["profile"]


def _draft(gen, basis):
    from flliper.srt.pdflip import model_profile as MP
    tmp, _m, draft_dir = _model.cache
    return MP.estimate_draft(os.path.join(tmp, draft_dir))


if __name__ == "__main__":
    unittest.main()
