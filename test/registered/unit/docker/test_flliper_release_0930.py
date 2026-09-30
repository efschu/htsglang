"""fLLiper flat release image (30.09.): the build files under docker/flliper/.

What is pinned here, without docker, GPU or network:
  * Dockerfile.flliper FLAT-2 (two frozen heads): the baked FLLIPER_ONE_TREE decides whether REV_27B must equal
    REV_NF (1) or both must be 40-hex revisions of their own (0); an unbaked file is refused. The guard is
    EXECUTED (extracted from the RUN body), not grepped. Every RUN body parses in dash (the build's /bin/sh).
  * make_flat_ctx.sh: bash syntax; bake points 5/5; --plan with --rev-nf on the real repo reports TWO trees, resolves
    each kernel line in its own slot's tree and names the ctx -nf<sha10> (skipped where the rig inputs are absent).
  * flliper_postcheck.sh: every verdict path against a docker STUB (canned inspect/run answers): a good image is
    GREEN; a revision mismatch, a dirty tree, a failed selfcheck, a JIT report not OK, a shadowing package, an
    unparsable or unaccepted profile and a registry row without the 27B Leistungsschalter each turn it RED.
"""

import json
import os
import pathlib
import re
import shutil
import stat
import subprocess
import tempfile
import textwrap
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[4]
D = ROOT / "docker" / "flliper"
DOCKERFILE = D / "Dockerfile.flliper"
MAKE = D / "make_flat_ctx.sh"
POST = D / "flliper_postcheck.sh"

R27 = "a" * 40
RNF = "b" * 40


def _run_bodies(text):
    """The shell body of every RUN instruction (continuations joined, --mount options dropped)."""
    lines, bodies, cur = text.splitlines(), [], None
    for ln in lines:
        if cur is None:
            if ln.startswith("RUN "):
                cur = ln[4:]
                if not cur.rstrip().endswith("\\"):
                    bodies.append(cur)
                    cur = None
        else:
            if ln.lstrip().startswith("#"):
                continue  # Docker drops comment lines inside a continuation
            cur += "\n" + ln
            if not ln.rstrip().endswith("\\"):
                bodies.append(cur)
                cur = None
    out = []
    for b in bodies:
        b = b.replace("\\\n", "")  # Docker joins continuations (backslash + newline removed)
        out.append(re.sub(r"^\s*(--mount=\S+\s+)+", "", b))
    return out


def _flat2_guard():
    body = next(b for b in _run_bodies(DOCKERFILE.read_text()) if "FLLIPER_ONE_TREE" in b)
    m = re.search(r'case "\$\{FLLIPER_ONE_TREE\}" in.*?esac;', body, re.S)
    assert m, "FLAT-2 guard not found in the step-4 RUN body"
    return m.group(0).rstrip(";")


class TestDockerfileFlat2(unittest.TestCase):
    def _guard(self, one, r27, rnf):
        env = dict(os.environ, FLLIPER_ONE_TREE=one, REV_27B=r27, REV_NF=rnf)
        return subprocess.run(["dash", "-euc", _flat2_guard()], env=env, capture_output=True, text=True)

    def test_one_tree_requires_equal_revisions(self):
        self.assertEqual(self._guard("1", R27, R27).returncode, 0)
        r = self._guard("1", R27, RNF)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("FLAT-2", r.stdout + r.stderr)

    def test_two_trees_need_two_40hex_revisions(self):
        self.assertEqual(self._guard("0", R27, RNF).returncode, 0)
        self.assertNotEqual(self._guard("0", R27, "abc123").returncode, 0)
        self.assertNotEqual(self._guard("0", "", RNF).returncode, 0)

    def test_unbaked_point_is_refused(self):
        r = self._guard("__FLLIPER_ONE_TREE__", R27, R27)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("FLAT-5", r.stdout + r.stderr)

    def test_bake_points_and_label(self):
        t = DOCKERFILE.read_text()
        for ph in ("VERSION", "SOURCE", "PLACEHOLDERS", "LOCK_SHA256", "ONE_TREE"):
            self.assertRegex(t, rf"(?m)^ARG FLLIPER_{ph}=__FLLIPER_{ph}__$")
        self.assertIn('io.github.efschu.flliper.one_tree="${FLLIPER_ONE_TREE}"', t)
        # declared once (a second bare ARG in the same stage would read as a new, empty declaration to a reviewer)
        self.assertEqual(len(re.findall(r"(?m)^ARG FLLIPER_ONE_TREE", t)), 1)

    @unittest.skipUnless(shutil.which("dash"), "dash missing")
    def test_every_run_body_parses_in_dash(self):
        bad = []
        for i, b in enumerate(_run_bodies(DOCKERFILE.read_text())):
            r = subprocess.run(["dash", "-n"], input=b, capture_output=True, text=True)
            if r.returncode:
                bad.append((i, r.stderr.strip()[:200]))
        self.assertEqual(bad, [])
        self.assertGreaterEqual(len(_run_bodies(DOCKERFILE.read_text())), 20)


RIG = pathlib.Path("/spinning/htsglang")
RIG_OK = (RIG / ".git").exists() and pathlib.Path("/spinning/htsglang-gpu/.venv/bin/python").exists() \
    and pathlib.Path("/spinning/gpu-arb/docker/profiles").is_dir()


def _rev(ref):
    r = subprocess.run(["git", "-C", str(RIG), "rev-parse", "--verify", "-q", ref + "^{commit}"],
                       capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ""


class TestMakeFlatCtx(unittest.TestCase):
    def test_syntax_and_help(self):
        self.assertEqual(subprocess.run(["bash", "-n", str(MAKE)]).returncode, 0)
        h = subprocess.run(["bash", str(MAKE), "--help"], capture_output=True, text=True)
        self.assertEqual(h.returncode, 0)
        self.assertIn("--rev-nf", h.stdout)
        self.assertIn("CTX_ROOT", h.stdout)

    def test_needs_exactly_one_mode(self):
        r = subprocess.run(["bash", str(MAKE)], capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)

    @unittest.skipUnless(RIG_OK, "rig repo / reference venv / profiles not present")
    def test_plan_two_trees(self):
        r27 = _rev("refs/heads/desk/27b-z30y3-integ-0929")
        rnf = _rev("refs/remotes/origin/desk/nf-wake-tail-0930") or _rev("refs/heads/desk/nf-wake-tail-0930")
        if not r27 or not rnf or r27 == rnf:
            self.skipTest("the two line heads are not both present")
        env = dict(os.environ, CTX_ROOT=tempfile.mkdtemp())
        p = subprocess.run(["bash", str(MAKE), "--plan", "--no-hash", "--rev", r27, "--branch", "desk/27b-z30y3-integ-0929",
                            "--rev-nf", rnf, "--branch-nf", "desk/nf-wake-tail-0930", "--release", "0.0.0-test"],
                           capture_output=True, text=True, env=env, timeout=300)
        out = p.stdout
        self.assertIn("TWO trees", out)
        self.assertIn("-- slot nf", out)
        self.assertIn(f"NF lines in {rnf[:10]}", out)
        self.assertIn(f"flat-0.0.0-test-{r27[:10]}-nf{rnf[:10]}-cu130", out)
        self.assertIn(f"revision.nf={rnf[:10]}", out)
        self.assertIn("bake points 5/5", out)
        self.assertNotIn("module missing", out)
        self.assertNotIn("lacks the bake point", out)

    @unittest.skipUnless(RIG_OK, "rig repo / reference venv / profiles not present")
    def test_plan_one_tree_unchanged(self):
        r27 = _rev("refs/heads/desk/27b-z30y3-integ-0929")
        if not r27:
            self.skipTest("27B line head not present")
        p = subprocess.run(["bash", str(MAKE), "--plan", "--no-hash", "--rev", r27, "--branch", "desk/27b-z30y3-integ-0929",
                            "--release", "0.0.0-test"], capture_output=True, text=True, timeout=300,
                           env=dict(os.environ, CTX_ROOT=tempfile.mkdtemp()))
        self.assertIn(f"ONE tree: both slots carry {r27[:10]}", p.stdout)
        self.assertIn(f"flat-0.0.0-test-{r27[:10]}-cu130", p.stdout)
        self.assertNotIn("-- slot nf", p.stdout)


    @unittest.skipUnless(RIG_OK, "rig repo / reference venv / profiles not present")
    def test_merge_point_is_enforced(self):
        """--require-in-27b: the dual layout is planned as a MERGE POINT of the 27B head -- a ref that is not an
        ancestor of the 27B revision is a ctx blocker (plan rc 3); an ancestor passes."""
        r27 = _rev("refs/heads/desk/27b-z30y3-integ-0929")
        if not r27:
            self.skipTest("27B line head not present")
        base = ["bash", str(MAKE), "--plan", "--no-hash", "--rev", r27, "--branch", "desk/27b-z30y3-integ-0929",
                "--release", "0.0.0-test"]
        env = dict(os.environ, CTX_ROOT=tempfile.mkdtemp())
        anc = subprocess.run(["git", "-C", str(RIG), "rev-parse", r27 + "~3"], capture_output=True, text=True).stdout.strip()
        p = subprocess.run(base + ["--require-in-27b", anc], capture_output=True, text=True, env=env, timeout=300)
        self.assertIn(f"merge point {anc} ({anc[:10]}) is in the 27B tree", p.stdout)
        # a commit that is NOT in the 27B tree: the 27B head's parent's sibling never is -- use the NF head if present
        other = _rev("refs/remotes/origin/desk/nf-wake-tail-0930") or _rev("refs/heads/desk/nf-wake-tail-0930")
        if not other:
            self.skipTest("no foreign head to test the refusal")
        p = subprocess.run(base + ["--require-in-27b", other], capture_output=True, text=True, env=env, timeout=300)
        self.assertEqual(p.returncode, 3)
        self.assertIn("is NOT in the 27B tree", p.stdout)


# ---------------------------------------------------------------------------------------------------------------------
# flliper_postcheck.sh against a docker stub
# ---------------------------------------------------------------------------------------------------------------------

STUB = r'''#!/usr/bin/env python3
import json, os, sys
a = sys.argv[1:]
S = json.load(open(os.environ["STUB_STATE"]))
def out(key, rc=0):
    v = S.get(key, "")
    sys.stdout.write(v if v.endswith("\n") or not v else v + "\n"); sys.exit(S.get(key + "_rc", rc))
if a[:2] == ["image", "inspect"]:
    out("inspect")
if a[:1] == ["run"]:
    s = " ".join(a)
    if "rev-parse HEAD" in s: out("trees")
    if "selfcheck" in a: out("selfcheck_" + next(x.split("=", 1)[1] for x in a if x.startswith("HTSGLANG_PROFILE=")))
    if "JIT_PREBUILD" in s: out("jit")
    if "find_spec" in s: out("shadow")
    if "SLOT_ROOT" in s: out("import_" + ("nf" if "src-nf" in s else "27b"))
    if "profile_form_env" in s: out("profiles")
    if "PROFILE_SWITCH_DEFAULTS" in s: out("defaults")
    if "pip" in a and "check" in a: out("pip")
    if "accepted_first_docker" in s: out("accepted")
    if "dryrun" in a: out("dryrun")
    if "boot_deadman" in s: out("deadman")
sys.stderr.write("stub: unexpected %r\n" % a); sys.exit(97)
'''


def _good_state():
    labels = {
        "org.opencontainers.image.title": "fLLiper", "org.opencontainers.image.version": "0.1.0",
        "org.opencontainers.image.revision": R27,
        "io.github.efschu.flliper.revision.27b": R27, "io.github.efschu.flliper.revision.nf": RNF,
        "io.github.efschu.flliper.build": "flat", "io.github.efschu.flliper.one_tree": "0",
        "io.github.efschu.flliper.placeholders": "none",
    }
    inspect = [{"Config": {"Labels": labels, "Healthcheck": {"Test": ["CMD", "/opt/htsglang/healthcheck.sh"]},
                           "StopSignal": "SIGTERM", "Entrypoint": ["/opt/htsglang/entrypoint.sh"]},
                "RootFS": {"Layers": ["sha256:x"] * 52}}]
    return {
        "inspect": json.dumps(inspect),
        "trees": f"27b {R27} 0\nnf {RNF} 0",
        "selfcheck_27b": "D1 ok", "selfcheck_nf": "D1 ok",
        "jit": "N 3\nR JIT_PREBUILD-27b.json OK 0\nR JIT_PREBUILD-27b-heavy.json OK 0\nR JIT_PREBUILD-nf.json OK 0",
        "shadow": "SHADOW -",
        "import_27b": "PKG flliper fi 0.7.0 torch 2.11.0+cu130\nBAD -",
        "import_nf": "PKG flliper fi 0.7.0 torch 2.11.0+cu130\nBAD -",
        "profiles": "ACCEPTED 27b nf\nP 27b OK abgenommen 40 model\nP nf OK abgenommen 55 model\n"
                    "P 27b-nvfp4 OK experimentell 44 model\nP nf-gguf OK geplant 3",
        "defaults": "MISS -\nOFF -\nON 31 of 44",
        "pip": "No broken requirements found.",
        "accepted": "27b nf", "dryrun": "LAUNCH-DRYRUN ok",
        "deadman": "DM md5=b436391f63b7c64f27a10bf3d9d926a7 progress=2 want=b436391f63b7c64f27a10bf3d9d926a7\n"
                   "S 27b sf=/opt/htsglang/src-27b/python/flliper/srt/pdflip/state_file.py progress=2 rank=1 death=1\n"
                   "S nf sf=/opt/htsglang/src-nf/python/flliper/srt/pdflip/state_file.py progress=2 rank=1 death=1",
    }


class TestPostcheck(unittest.TestCase):
    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.stub = self.tmp / "docker"
        self.stub.write_text(STUB)
        self.stub.chmod(self.stub.stat().st_mode | stat.S_IEXEC)
        self.state = self.tmp / "state.json"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, state, *args):
        self.state.write_text(json.dumps(state))
        env = dict(os.environ, DOCKER=str(self.stub), STUB_STATE=str(self.state))
        return subprocess.run(["bash", str(POST), "flliper:0.1.0-cu130", *args], capture_output=True, text=True, env=env)

    def _verdicts(self, r):
        return {(m.group(1), m.group(2)) for m in re.finditer(r"^CHECK (C\d+) (\w+)", r.stdout, re.M)}

    def test_syntax_help_plan(self):
        self.assertEqual(subprocess.run(["bash", "-n", str(POST)]).returncode, 0)
        r = subprocess.run(["bash", str(POST), "img", "--plan"], capture_output=True, text=True,
                           env=dict(os.environ, DOCKER="/nonexistent/docker"))
        self.assertEqual(r.returncode, 0)  # --plan never calls docker
        self.assertIn("selfcheck", r.stdout)
        self.assertEqual(subprocess.run(["bash", str(POST)], capture_output=True).returncode, 2)

    def test_good_image_is_green(self):
        r = self._run(_good_state())
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("-> GREEN", r.stdout)
        v = self._verdicts(r)
        for c in ("C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C11"):
            self.assertIn((c, "PASS"), v, r.stdout)
        self.assertIn(("C10", "SKIP"), v)
        self.assertNotRegex(r.stdout, r"(?m)^CHECK C\d+ FAIL")

    def test_ctx_revisions_must_match_labels(self):
        ctx = self.tmp / "ctx"
        ctx.mkdir()
        (ctx / "BUILD_INFO.json").write_text(json.dumps({"lines": {"27b": {"revision": R27}, "nf": {"revision": RNF}}}))
        self.assertEqual(self._run(_good_state(), "--ctx", str(ctx)).returncode, 0)
        (ctx / "BUILD_INFO.json").write_text(json.dumps({"lines": {"27b": {"revision": R27}, "nf": {"revision": "c" * 40}}}))
        r = self._run(_good_state(), "--ctx", str(ctx))
        self.assertEqual(r.returncode, 1)
        self.assertIn(("C1", "FAIL"), self._verdicts(r))

    def _red(self, mutate, check):
        s = _good_state()
        mutate(s)
        r = self._run(s)
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn((check, "FAIL"), self._verdicts(r), r.stdout)
        self.assertIn("-> RED", r.stdout)

    def test_one_tree_label_disagreeing_with_revisions(self):
        def m(s):
            d = json.loads(s["inspect"]); d[0]["Config"]["Labels"]["io.github.efschu.flliper.one_tree"] = "1"
            s["inspect"] = json.dumps(d)
        self._red(m, "C1")

    def test_missing_healthcheck(self):
        def m(s):
            d = json.loads(s["inspect"]); d[0]["Config"]["Healthcheck"] = None; s["inspect"] = json.dumps(d)
        self._red(m, "C2")

    def test_dirty_or_wrong_tree(self):
        self._red(lambda s: s.update(trees=f"27b {R27} 3\nnf {RNF} 0"), "C3")
        self._red(lambda s: s.update(trees=f"27b {R27} 0\nnf {R27} 0"), "C3")

    def test_selfcheck_fails(self):
        self._red(lambda s: s.update(selfcheck_nf="FATAL: torch_extensions missing", selfcheck_nf_rc=1), "C4")

    def test_jit_report_not_ok(self):
        self._red(lambda s: s.update(jit="N 1\nR JIT_PREBUILD-nf.json PROBLEMS 2"), "C5")
        self._red(lambda s: s.update(jit="N 0"), "C5")

    def test_import_and_shadow(self):
        self._red(lambda s: s.update(import_nf="PKG sglang fi 0.6.14 torch 2.11\nBAD flashinfer 0.6.14"), "C6")
        self._red(lambda s: s.update(shadow="SHADOW sglang"), "C6")

    def test_profiles(self):
        self._red(lambda s: s.update(profiles="ACCEPTED 27b nf\nP 27b OK abgenommen 40 model\nP nf OK experimentell 55 model"),
                  "C7")
        self._red(lambda s: s.update(profiles="ACCEPTED 27b\nP 27b profiles/27b.env: line 3: syntax error"), "C7")

    def test_release_defaults_missing(self):
        self._red(lambda s: s.update(defaults="MISS d_hostgap_base d_release_fixes\nOFF -\nON 25 of 40"), "C8")
        self._red(lambda s: s.update(defaults="MISS -\nOFF p_row_authority\nON 30 of 44"), "C8")

    def test_dryrun_with_models(self):
        r = self._run(_good_state(), "--models", "/models")
        self.assertIn(("C10", "PASS"), self._verdicts(r))
        s = _good_state(); s["dryrun_rc"] = 3
        r = self._run(s, "--models", "/models")
        self.assertIn(("C10", "FAIL"), self._verdicts(r))

    def test_deadman_in_the_image(self):
        """C11 (30.09., DEADMAN-IM-IMAGE): the image's deadman fits BOTH slots' state_file.py and BUILD_INFO's md5."""
        good = _good_state()
        self.assertIn(("C11", "PASS"), self._verdicts(self._run(good, "--only", "C11")))
        s = dict(good)   # the host copy baked in (no progress-check) while the trees have it
        s["deadman"] = good["deadman"].replace("progress=2 want", "progress=0 want")
        self.assertIn(("C11", "FAIL"), self._verdicts(self._run(s, "--only", "C11")))
        s = dict(good)   # the NF tree lacks progress-check the deadman calls
        s["deadman"] = good["deadman"].replace("S nf sf=/opt/htsglang/src-nf/python/flliper/srt/pdflip/state_file.py progress=2",
                                               "S nf sf=/opt/htsglang/src-nf/python/flliper/srt/pdflip/state_file.py progress=0")
        self.assertIn(("C11", "FAIL"), self._verdicts(self._run(s, "--only", "C11")))
        s = dict(good)   # note_rank_death without the rank writer
        s["deadman"] = good["deadman"].replace("rank=1 death=1\nS nf", "rank=0 death=1\nS nf")
        self.assertIn(("C11", "FAIL"), self._verdicts(self._run(s, "--only", "C11")))
        s = dict(good)   # md5 differs from BUILD_INFO.deadman.md5
        s["deadman"] = good["deadman"].replace("want=b436391f63b7c64f27a10bf3d9d926a7", "want=5982a49d824d21abced819b001a3b0ba")
        self.assertIn(("C11", "FAIL"), self._verdicts(self._run(s, "--only", "C11")))
        s = dict(good)   # an image before 0930dm: no deadman field -> md5 not compared; old deadman + old trees = WARN
        s["deadman"] = ("DM md5=5982a49d824d21abced819b001a3b0ba progress=0 want=-\n"
                        "S 27b sf=none progress=0 rank=0 death=0\nS nf sf=none progress=0 rank=0 death=0")
        self.assertIn(("C11", "WARN"), self._verdicts(self._run(s, "--only", "C11")))
        r = self._run(s, "--only", "C11")
        self.assertIn(("C11", "FAIL"), self._verdicts(subprocess.run(
            ["bash", str(POST), "flliper:0.1.0-cu130", "--only", "C11"], capture_output=True, text=True,
            env=dict(os.environ, DOCKER=str(self.stub), STUB_STATE=str(self.state), FLAT_REQUIRE_DEADMAN="1"))))
        del r

    def test_only(self):
        r = self._run(_good_state(), "--only", "C1 C3")
        self.assertEqual({c for c, _ in self._verdicts(r)}, {"C1", "C3"})


if __name__ == "__main__":
    unittest.main()


class TestPublishGate(unittest.TestCase):
    """B5 (30.09.): host_publish_flliper.sh lock 2 for TWO trees -- its own self-test (fake docker, git fixture with a
    pushed NF head c3 on its own branch) must be all green: one-tree cases unchanged, two-tree cases green/red as
    named, the old time bomb (sed-mutated verdicts outdating the fixed Go file) gone."""

    @unittest.skipUnless(shutil.which("git"), "git missing")
    def test_selftest_all_green(self):
        st = D / "host_publish_flliper_selftest.sh"
        r = subprocess.run(["bash", str(st)], capture_output=True, text=True, timeout=600)
        tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-300:]
        self.assertEqual(r.returncode, 0, r.stdout[-3000:])
        m = re.search(r"== (\d+) passed, (\d+) failed", r.stdout)
        self.assertIsNotNone(m, tail)
        self.assertEqual(m.group(2), "0")
        self.assertGreaterEqual(int(m.group(1)), 63)
        for name in ("L2 two trees: nf pushed on its own branch (all green)", "L2 two trees: nf head unpushed",
                     "L3 two trees: NF verdict names the 27B tree", "PUBLISH two trees all green"):
            self.assertIn(name, r.stdout)


# ---------------------------------------------------------------------------------------------------------------------
# release profiles from the set that ACTUALLY reaches the ranks (docker/flliper/release_profile.py)
# ---------------------------------------------------------------------------------------------------------------------

import importlib.util as _ilu  # noqa: E402

_spec = _ilu.spec_from_file_location("release_profile", D / "release_profile.py")
RP = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(RP)

NF_TARGET = pathlib.Path("/spinning/gpu-arb/docker/profiles/"
                         "nf-h91-dpr-sa-vis-adopt-st-cut-vsync-odx-2b-swr-e2cut-z30y2-arr-wre-ta-dh-ml-ef.env")
NF_DRAFT = D / "profiles_release" / "nf.env"
META = {"PROFILE_NAME", "PROFILE_STATUS", "PROFILE_OWNER", "PROFILE_FORM"}


def _env_dict(args, flag):
    """What the launcher makes of --env-p/--env-d: the last entry of a key wins."""
    out = {}
    for t in RP._env_tokens(args, flag) or []:
        k, _, v = t.partition("=")
        out[k] = v
    return out


def _effective(d, drop=()):
    args = RP.drop_args(d["args"], set(drop))
    return {"args": RP._without_env(args), "env_p": _env_dict(args, "--env-p"), "env_d": _env_dict(args, "--env-d"),
            "form": d["form"], "instr": d["instr"], "exports": d["exports"], "arrays": d["arrays"],
            "vars": {k: v for k, v in d["vars"].items() if k not in META}}


class TestReleaseProfileGenerator(unittest.TestCase):
    """The generator on synthetic profiles: late NF_ENV appends only count once written back, plain assignments do
    not arrive, exports do, instrument/nccl parts become conditional appends, precedence changes are refused."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write(self, body):
        p = self.tmp / "t.env"
        p.write_text(body)
        return p

    def test_effective_set_and_roundtrip(self):
        t = self._write(textwrap.dedent('''\
            PROFILE_NAME=t
            PROFILE_STATUS=experimentell
            PROFILE_MODEL=/m
            E_P="A=1;B=2"
            if [ "${HTSGLANG_INSTRUMENTS:-1}" = 1 ]; then E_P="$E_P;TRACE=1"; fi
            if [ "${HTSGLANG_TRANSPORT:-bar1}" = nccl ]; then E_P="$E_P;B=3"; fi
            PROFILE_ARGS=(--env-p "$E_P" --env-d "D=1" --drop-me x --keep "a b")
            E_P="$E_P;LATE=1"          # never arrives: PROFILE_ARGS is already built
            PLAIN=1                     # a plain assignment does not arrive
            export EXP=1
            profile_form_env() { _form F1 "v ${HTSGLANG_TAG}"; }
            profile_instr_env() { _form I1 "${SGLANG_WEG2_EVIDENCE_DIR:-/x}/c"; }
            '''))
        d = RP.dump(str(t))
        self.assertNotIn("LATE", str(d["args"]))
        self.assertNotIn("PLAIN", d["exports"])
        self.assertEqual(d["exports"], {"EXP": "1"})
        out = self.tmp / "rel.env"
        RP.generate(str(t), str(out), "rel", "experimentell", ["--drop-me"], "", "")
        for instr in ("0", "1"):
            for tr in ("bar1", "nccl"):
                a, b = RP.dump(str(t), instr, tr), RP.dump(str(out), instr, tr)
                self.assertEqual(_effective(a, ["--drop-me"]), _effective(b), (instr, tr))
        self.assertNotIn("--drop-me", RP.dump(str(out))["args"])
        self.assertIn('"${HTSGLANG_TAG}"', out.read_text().replace("v ${HTSGLANG_TAG}", "${HTSGLANG_TAG}"))
        # the release instrument default is OFF: unset HTSGLANG_INSTRUMENTS -> no TRACE
        r = subprocess.run(["bash", "-c", f'source "{out}"; echo "$REL_ENV_P"'], capture_output=True, text=True,
                           env={"PATH": "/usr/bin:/bin"})
        self.assertNotIn("TRACE", r.stdout)

    def test_precedence_change_is_refused(self):
        t = self._write(textwrap.dedent('''\
            PROFILE_NAME=t
            E_P="A=1"
            if [ "${HTSGLANG_TRANSPORT:-bar1}" = nccl ]; then E_P="$E_P;B=3"; fi
            E_P="$E_P;B=2"              # a base token of key B AFTER the nccl one: appending B=3 at the end would flip it
            PROFILE_ARGS=(--env-p "$E_P")
            '''))
        with self.assertRaises(SystemExit) as cm:
            RP.generate(str(t), str(self.tmp / "o.env"), "rel", "experimentell", [], "", "")
        self.assertIn("order would change", str(cm.exception))


@unittest.skipUnless(NF_TARGET.exists(), "NF target profile not present")
class TestReleaseProfileNF(unittest.TestCase):
    """docker/flliper/profiles_release/nf.env (draft for NF's release) against NF's Soll profile -ml-ef: the set that
    reaches the ranks is identical under HTSGLANG_INSTRUMENTS 0/1 x transport bar1/nccl -- --d-kv-token-cut
    INCLUDED (NF correction 30.09.: every NF metal boot since -st-cut ran `owned`, the launcher default is off,
    uneven-DCP KV is a release feature; "not pinned" meant no fixed token cut, not the mode) -- except
    PROFILE_NAME=nf, the draft status/owner/form and the release instrument default."""

    def test_same_effective_set_as_the_target(self):
        for instr in ("0", "1"):
            for tr in ("bar1", "nccl"):
                a, b = RP.dump(str(NF_TARGET), instr, tr), RP.dump(str(NF_DRAFT), instr, tr)
                ea, eb = _effective(a), _effective(b)
                for k in ea:
                    self.assertEqual(ea[k], eb[k], f"{k} differs under instr={instr} transport={tr}")

    def test_named_decisions(self):
        d = RP.dump(str(NF_DRAFT))
        # the uneven-DCP KV mode arrives exactly as on NF's metal: --d-kv-token-cut owned
        self.assertIn("--d-kv-token-cut", d["args"], "--d-kv-token-cut missing (NF metal runs it)")
        self.assertEqual(d["args"][d["args"].index("--d-kv-token-cut") + 1], "owned")
        self.assertEqual(d["vars"]["PROFILE_NAME"], "nf")
        self.assertIn(d["vars"]["PROFILE_STATUS"], ("experimentell", "abgenommen"))
        # NF 30.09.: release value ARRIVAL_SEAT_RULE=1 (A/B in the noise), on front and D
        self.assertIn(["SGLANG_WEG2_ENABLE_ARRIVAL_SEAT_RULE", "1"], d["form"])
        self.assertEqual(_env_dict(d["args"], "--env-d").get("SGLANG_WEG2_ENABLE_ARRIVAL_SEAT_RULE"), "1")
        # the late -ef appends DO arrive (written back into PROFILE_ARGS by the target)
        self.assertEqual(_env_dict(d["args"], "--env-d").get("SGLANG_WEG2_ENABLE_WAKE_READ_EARLY"), "1")

    def test_draft_is_current(self):
        """Regenerating from the target gives the checked-in draft (modulo the timestamp line) -- a changed target
        shows up here, not silently at the freeze."""
        out = pathlib.Path(tempfile.mkdtemp()) / "nf.env"
        RP.generate(str(NF_TARGET), str(out), "nf", "experimentell", [],
                    "NF-Sitz (Release-Entwurf aus -ml-ef, 27B-Implementierer 30.09.; Freigabe NF ausstehend)", "+release")
        strip = lambda s: [l for l in s.splitlines() if "GENERATED by" not in l]
        self.assertEqual(strip(out.read_text()), strip(NF_DRAFT.read_text()))
