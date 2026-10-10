"""H88-E (plan PLAN-H88-W4A8-1007 D3, row H88-E): the switch ``--moe-act-int8 {on,off}`` / ``SGLANG_MOE_ACT_INT8``, the profile
``docker/profiles/nf-int4-w4a8.env``, the catalog entries + edges, and the planner pass-through.  GPU-free, Docker-free.

* ``TestRegistration``   env ``SGLANG_MOE_ACT_INT8`` = EnvBool default False; ServerArgs field ``moe_act_int8`` (choices on/off, default
                         off) parsed by the real argument parser; both visible in the catalog harvest.
* ``TestSwitchReader``   ``layers/quantization/moe_act_int8``: flag OR env switches it on; nothing else does.
* ``TestDispatch``       the compressed-tensors WNA16 MoE scheme dispatch: switch on + no W4A8 scheme in this tree = the named
                         RuntimeError ``MOE-ACT-INT8 requested but no W4A8 MoE scheme in this tree``; switch off = the W4A16 scheme as
                         before (default path unchanged).
* ``TestProfile``        ``nf-int4-w4a8.env`` = ``nf-int4-h6-abl.env`` + the switch in both groups and the launcher env, nothing else.
* ``TestCatalog``        two CURATED entries (text from the sources, benefit/cost 'unbelegt'), three edges with anchors on the NF line.
* ``TestPlannerPassThrough``  ``propose()`` keeps the switch as a profile scalar with its origin; the release profile gets no record.
* ``TestDryRun``         the launcher dry run of the abl profile = the NF golden (diff 0 is ``test_planer_referenz_n3_1006``); the dry run
                         of ``nf-int4-w4a8.env`` differs from it ONLY by lines that carry the switch, and the plan dump shows it in the
                         P and D group environment.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import re
import sys
import tempfile
import unittest

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

try:
    from sglang.srt.weg2 import launcher
    from sglang.srt.weg2 import model_profile as MP
    from sglang.srt.weg2 import propose as P
    from sglang.srt.weg2 import propose_oracle as O
    from sglang.srt.weg2 import propose_rules as R
except Exception as exc:  # pragma: no cover - no weg2 launcher in this build
    pytest.skip(f"weg2 launcher unavailable: {exc}", allow_module_level=True)

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = str(pathlib.Path(launcher.__file__).resolve().parents[4])
SRT = os.path.join(TREE, "python", "sglang", "srt")
WEG2 = os.path.join(SRT, "weg2")
FIX = os.path.join(HERE, "fixtures", "planer_1006")
PROFILES = os.path.join(FIX, "profiles")
CKPT = os.path.join(FIX, "checkpoints")
GOLDEN = os.path.join(FIX, "golden")
REPLAY_REF = os.path.join(HERE, "fixtures", "xchg_launch_replay_0911", "nvml_devices_1378.json")
W4A8_PROFILE = os.path.join(TREE, "docker", "profiles", "nf-int4-w4a8.env")
ABL_PROFILE = os.path.join(PROFILES, "nf-int4-h6-abl.env")
MSG = "MOE-ACT-INT8 requested but no W4A8 MoE scheme in this tree"
_NF = ("Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist-abl-wxp", "Qwen3.8-Flash-Next-MTP-INT4-g32-albucino-abl-wxp")
_LINE = O.launcher_line()
MEASURED_RATES = {"RTX 5090": 203.42, "RTX 3080 20GB": 50.97}


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


PC = _load("t_h88e_profile_catalog", os.path.join(WEG2, "profile_catalog.py"))
CU = _load("t_h88e_profile_catalog_curated", os.path.join(WEG2, "profile_catalog_curated.py"))


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


class _SwitchEnv(unittest.TestCase):
    """Start every test with the switch unset and put the process environment back."""

    def setUp(self):
        self._before = os.environ.pop("SGLANG_MOE_ACT_INT8", None)

    def tearDown(self):
        os.environ.pop("SGLANG_MOE_ACT_INT8", None)
        if self._before is not None:
            os.environ["SGLANG_MOE_ACT_INT8"] = self._before


class TestRegistration(_SwitchEnv):
    def test_env_is_a_bool_default_false(self):
        from sglang.srt.environ import EnvBool, envs

        self.assertIsInstance(type(envs).__dict__["SGLANG_MOE_ACT_INT8"], EnvBool)
        self.assertIs(envs.SGLANG_MOE_ACT_INT8.get(), False)
        os.environ["SGLANG_MOE_ACT_INT8"] = "1"
        self.assertIs(envs.SGLANG_MOE_ACT_INT8.get(), True)
        os.environ["SGLANG_MOE_ACT_INT8"] = "0"
        self.assertIs(envs.SGLANG_MOE_ACT_INT8.get(), False)

    def test_flag_is_on_off_default_off_in_the_real_parser(self):
        import argparse

        from sglang.srt.server_args import ServerArgs

        ap = argparse.ArgumentParser()
        ServerArgs.add_cli_args(ap)
        base = ["--model-path", "x"]
        self.assertEqual(ap.parse_known_args(base)[0].moe_act_int8, "off")
        self.assertEqual(ap.parse_known_args(base + ["--moe-act-int8", "on"])[0].moe_act_int8, "on")
        self.assertEqual(ap.parse_known_args(base + ["--moe-act-int8", "off"])[0].moe_act_int8, "off")
        with self.assertRaises(SystemExit):
            ap.parse_args(base + ["--moe-act-int8", "maybe"])

    def test_the_catalog_harvest_sees_both(self):
        server = PC.server_args_flags(os.path.join(SRT, "server_args.py"))
        envs = PC.environ_fields(os.path.join(SRT, "environ.py"))
        self.assertEqual(server["--moe-act-int8"]["choices"], ["on", "off"])
        self.assertEqual(server["--moe-act-int8"]["default"], "off")
        self.assertIn("W4A8", server["--moe-act-int8"]["help"])
        self.assertIn("SGLANG_MOE_ACT_INT8", envs)
        self.assertIn("W4A8", envs["SGLANG_MOE_ACT_INT8"]["comment"])

    def test_the_launcher_has_no_flag_of_its_own_and_is_not_edited(self):
        """The switch reaches both groups through the existing --env-p/--env-d and --extra-p/--extra-d; the launcher's
        argument parser neither names it nor needs to (R1: no launcher change for a pass-through)."""
        self.assertNotIn("--moe-act-int8", PC.launcher_flags(os.path.join(WEG2, "launcher.py")))


class TestSwitchReader(_SwitchEnv):
    def setUp(self):
        super().setUp()
        from sglang.srt.layers.quantization import moe_act_int8 as M

        self.M = M

    def test_default_is_off(self):
        self.assertFalse(self.M.moe_act_int8_requested())

    def test_env_switches_it_on(self):
        os.environ["SGLANG_MOE_ACT_INT8"] = "1"
        self.assertTrue(self.M.moe_act_int8_requested())

    def test_flag_switches_it_on(self):
        class SA:
            moe_act_int8 = "on"

        class SAoff:
            moe_act_int8 = "off"

        self.assertTrue(self.M.moe_act_int8_requested(SA()))
        self.assertFalse(self.M.moe_act_int8_requested(SAoff()))
        self.assertFalse(self.M.moe_act_int8_requested(object()))          # a server-args object without the field: off

    def test_either_one_is_enough(self):
        class SAoff:
            moe_act_int8 = "off"

        os.environ["SGLANG_MOE_ACT_INT8"] = "1"
        self.assertTrue(self.M.moe_act_int8_requested(SAoff()))

    def test_this_tree_has_the_w4a8_scheme_since_h88b(self):
        self.assertTrue(self.M.HAS_W4A8_MOE_SCHEME)
        os.environ["SGLANG_MOE_ACT_INT8"] = "1"
        self.M.require_w4a8_moe_scheme()                                   # on + scheme in the tree: returns

    def test_a_tree_without_the_scheme_names_itself(self):
        # H88-B set HAS_W4A8_MOE_SCHEME = True; the guard itself is still what a tree WITHOUT the scheme hits
        self.assertEqual(self.M.NO_SCHEME_MESSAGE, MSG)
        old = self.M.HAS_W4A8_MOE_SCHEME
        self.M.HAS_W4A8_MOE_SCHEME = False
        try:
            self.M.require_w4a8_moe_scheme()                               # off: returns
            os.environ["SGLANG_MOE_ACT_INT8"] = "1"
            with self.assertRaises(RuntimeError) as cm:
                self.M.require_w4a8_moe_scheme()
            self.assertEqual(str(cm.exception), MSG)
        finally:
            self.M.HAS_W4A8_MOE_SCHEME = old

    def test_a_tree_with_the_scheme_passes(self):
        os.environ["SGLANG_MOE_ACT_INT8"] = "1"
        old = self.M.HAS_W4A8_MOE_SCHEME
        self.M.HAS_W4A8_MOE_SCHEME = True
        try:
            self.M.require_w4a8_moe_scheme()
        finally:
            self.M.HAS_W4A8_MOE_SCHEME = old


class TestDispatch(_SwitchEnv):
    """``CompressedTensorsConfig.get_moe_scheme`` for an int4 group-quantised expert layer (the WNA16 branch)."""

    def setUp(self):
        super().setUp()
        import types

        from sglang.srt.layers.quantization.compressed_tensors import compressed_tensors as CT

        self.CT = CT
        self.sentinel = object()
        self._patched = []

        def patch(obj, name, value):
            self._patched.append((obj, name, getattr(obj, name)))
            setattr(obj, name, value)

        patch(CT, "_is_npu", False)
        patch(CT, "_is_hip", False)
        patch(CT, "get_moe_runner_backend", lambda: types.SimpleNamespace(is_triton=lambda: False, is_flashinfer_trtllm=lambda: False))
        patch(CT, "CompressedTensorsWNA16MoE", lambda cfg, weight_quant=None: self.sentinel)
        wq = types.SimpleNamespace(strategy="group", dynamic=False, num_bits=4, type="int", symmetric=True, group_size=128)
        cfg = object.__new__(CT.CompressedTensorsConfig)
        cfg._add_fused_moe_to_target_scheme_map = lambda: None
        cfg.get_scheme_dict = lambda layer, name: {"weights": wq, "input_activations": None}
        self.cfg = cfg

    def tearDown(self):
        for obj, name, old in reversed(self._patched):
            setattr(obj, name, old)
        super().tearDown()

    def test_switch_off_is_the_w4a16_scheme_as_before(self):
        self.assertIs(self.cfg.get_moe_scheme(object(), "model.layers.0.mlp.experts"), self.sentinel)

    def test_switch_on_without_a_w4a8_scheme_stops_with_the_named_error(self):
        # since H88-B the tree has the scheme (HAS_W4A8_MOE_SCHEME = True); the dispatch still stops
        # with the named error in a tree that has none, so the flag is patched back here
        from sglang.srt.layers.quantization import moe_act_int8 as M

        os.environ["SGLANG_MOE_ACT_INT8"] = "1"
        old = M.HAS_W4A8_MOE_SCHEME
        M.HAS_W4A8_MOE_SCHEME = False
        try:
            with self.assertRaises(RuntimeError) as cm:
                self.cfg.get_moe_scheme(object(), "model.layers.0.mlp.experts")
        finally:
            M.HAS_W4A8_MOE_SCHEME = old
        self.assertEqual(str(cm.exception), MSG)

    def test_the_flag_form_stops_the_same_way(self):
        import types

        from sglang.srt.layers.quantization import moe_act_int8 as M

        real = M.moe_act_int8_requested
        old = M.HAS_W4A8_MOE_SCHEME
        M.HAS_W4A8_MOE_SCHEME = False
        M.moe_act_int8_requested = lambda server_args=None: real(types.SimpleNamespace(moe_act_int8="on"))
        try:
            with self.assertRaises(RuntimeError) as cm:
                self.cfg.get_moe_scheme(object(), "model.layers.0.mlp.experts")
        finally:
            M.moe_act_int8_requested = real
            M.HAS_W4A8_MOE_SCHEME = old
        self.assertEqual(str(cm.exception), MSG)

    def test_switch_on_with_the_scheme_on_a_non_cuda_box_stays_w4a16(self):
        # H88-B: _is_cuda is False on this box, so the W4A8 branch is not taken (the CUDA case is
        # test_ct_wna16a8_moe_h88b_1007.TestDispatch)
        os.environ["SGLANG_MOE_ACT_INT8"] = "1"
        self.assertIs(self.cfg.get_moe_scheme(object(), "model.layers.0.mlp.experts"), self.sentinel)


class TestProfile(unittest.TestCase):
    def test_the_profile_is_abl_plus_the_switch_and_nothing_else(self):
        abl = _read(ABL_PROFILE).split("\n")
        new = _read(W4A8_PROFILE).split("\n")
        import difflib

        added, removed = [], []
        for ln in difflib.unified_diff(abl, new, lineterm="", n=0):
            if ln.startswith(("+++", "---", "@@")):
                continue
            (added if ln.startswith("+") else removed).append(ln[1:])
        # removed: exactly the NAME and STATUS lines (replaced); everything else of abl is kept word for word
        self.assertEqual(sorted(x.split("=")[0] for x in removed), ["PROFILE_NAME", "PROFILE_STATUS"])
        body = [x for x in added if not x.startswith("#") and not x.startswith("PROFILE_NAME=") and not x.startswith("PROFILE_STATUS=")]
        self.assertEqual(body, ["export SGLANG_MOE_ACT_INT8=1",
                                'NF_ENV_P="$NF_ENV_P;SGLANG_MOE_ACT_INT8=1"',
                                'NF_ENV_D="$NF_ENV_D;SGLANG_MOE_ACT_INT8=1"'])
        self.assertTrue(any(x.startswith("PROFILE_NAME=nf-int4-w4a8") for x in added))
        self.assertTrue(any(x.startswith("PROFILE_STATUS=experimentell") for x in added))

    def test_the_profile_header_says_identity_and_census(self):
        head = "\n".join(x for x in _read(W4A8_PROFILE).split("\n")[:16] if x.startswith("#"))
        for needle in ("L3", "Census", "W4A8", "abl"):
            self.assertIn(needle, head)

    def test_launch_input_carries_the_switch_into_both_groups_and_the_launcher_env(self):
        a = O.profile_launch_input(ABL_PROFILE)
        b = O.profile_launch_input(W4A8_PROFILE)
        self.assertNotIn("SGLANG_MOE_ACT_INT8", a.env)
        self.assertEqual(b.env.get("SGLANG_MOE_ACT_INT8"), "1")
        la, lb = P.LaunchArgv(a.argv, a.env), P.LaunchArgv(b.argv, b.env)
        for g in ("p", "d"):
            self.assertIsNone(la.env_get(g, "SGLANG_MOE_ACT_INT8"), g)
            self.assertEqual(lb.env_get(g, "SGLANG_MOE_ACT_INT8"), "1", g)
            # nothing else of the group environment moved
            la.env_del(g, "SGLANG_MOE_ACT_INT8")
            lb.env_del(g, "SGLANG_MOE_ACT_INT8")
            self.assertEqual(la.gtext("env", g), lb.gtext("env", g), g)
        self.assertEqual(la.t, lb.t)                                                # the whole argv is byte-identical
        rest_a = {k: v for k, v in a.env.items()}
        rest_b = {k: v for k, v in b.env.items() if k != "SGLANG_MOE_ACT_INT8"}
        self.assertEqual(rest_a, rest_b)
        self.assertEqual(b.vars.get("PROFILE_NAME"), "nf-int4-w4a8")
        self.assertEqual(b.vars.get("PROFILE_STATUS"), "experimentell")
        self.assertEqual(a.vars.get("PROFILE_NAME"), "nf-int4-abl")


class TestCatalog(unittest.TestCase):
    NAMES = ("--moe-act-int8", "SGLANG_MOE_ACT_INT8")

    def test_the_two_curated_entries_are_sourced_and_the_benefit_is_unbelegt(self):
        for n in self.NAMES:
            c = CU.CURATED[n]
            self.assertTrue(c.get("satz_quelle"), n)
            self.assertEqual(c["depends"], [], n)                       # edges live only in the edge catalog (with evidence)
            self.assertIn("unbelegt", c["gain"], n)
            self.assertIn("unbelegt", c["cost"], n)
            self.assertIn(MSG, c["text"], n)
            self.assertIn("W4A8", c["text"], n)
        self.assertEqual(CU.CURATED["--moe-act-int8"]["kind"], "flag")
        self.assertEqual(CU.CURATED["SGLANG_MOE_ACT_INT8"]["kind"], "env")

    def test_the_text_says_nothing_the_sources_do_not(self):
        """The sentence is built from the help= of server_args.py and the environ.py comment: every number-like claim of it
        (W4A8, W4A16, int8, the RuntimeError text) is in one of them."""
        server = PC.server_args_flags(os.path.join(SRT, "server_args.py"))["--moe-act-int8"]["help"]
        comment = PC.environ_fields(os.path.join(SRT, "environ.py"))["SGLANG_MOE_ACT_INT8"]["comment"]
        for needle in ("W4A8", "W4A16", "int8", MSG, "SGLANG_MOE_ACT_INT8"):
            self.assertIn(needle, server + " " + comment, needle)

    def test_the_edges(self):
        with open(os.path.join(WEG2, "kantenkatalog_1004.json"), encoding="utf-8") as fh:
            edges = {k["id"]: k for k in json.load(fh)["kanten"]}
        want = {"K132": ("--moe-act-int8", "--quantization", "braucht"),
                "K133": ("SGLANG_MOE_ACT_INT8", "--quantization", "braucht")}
        for i, (von, nach, rel) in want.items():
            e = edges[i]
            self.assertEqual((e["von"], e["nach"], e["rel"]), (von, nach, rel), i)
            self.assertEqual(e["baeume"], ["nf"], i)                    # this code is on the NF line only until the 27B line takes it
            self.assertIsNone(e["wert"])
        self.assertIn("compressed-tensors", edges["K132"]["satz"])
        self.assertIn("Dual", edges["K132"]["satz"])                    # 'Dual not affected' is in the words, with its plan source
        self.assertIn("nicht betroffen", edges["K132"]["satz"])
        # H88-F (08.10.): the L3-identity edge is proven now that H88-D is merged: K134 anchors at the line that
        # appends the field to the storage identity; offen.txt Nr. 17 (the question for that anchor) is gone
        k134 = edges["K134"]
        self.assertEqual((k134["von"], k134["nach"], k134["rel"]), ("--moe-act-int8", "--hicache-storage-backend", "skaliert_mit"))
        self.assertEqual(k134["baeume"], ["nf"])
        self.assertEqual(k134["beleg"]["datei"], "python/sglang/srt/mem_cache/hicache_storage.py")
        # planer decision 1008: the rank part carries the scale encoding ("int8;s16" = moe_act_switch.IDENTITY_VALUE)
        self.assertEqual(k134["beleg"]["anker"], 'identity_parts.append(f"moe_act={_moe_act_switch.IDENTITY_VALUE}")')
        self.assertIn("launcher.py:7709", k134["satz"])
        with open(os.path.join(WEG2, "offen.txt"), encoding="utf-8") as fh:
            offen = fh.read()
        self.assertNotRegex(offen, r"(?m)^17\. --moe-act-int8")

    def test_the_edge_anchors_resolve_in_this_tree(self):
        with open(os.path.join(WEG2, "kantenkatalog_1004.json"), encoding="utf-8") as fh:
            edges = [k for k in json.load(fh)["kanten"] if k["id"] in ("K132", "K133", "K134")]
        res = PC.resolve_edge_belege(edges, TREE, "nf")
        for i, r in res.items():
            self.assertIn(r["status"], PC.ANKER_OK, (i, r))
        res27 = PC.resolve_edge_belege(edges, TREE, "27b")
        self.assertTrue(all(r["status"] in PC.ANKER_FREMD for r in res27.values()))

    def test_the_shipped_catalog_carries_entries_and_edges(self):
        path = os.path.join(TREE, "tools", "rig_dashboard", "rigdash", "profil_data", "catalog.json")
        with open(path, encoding="utf-8") as fh:
            cat = json.load(fh)
        for n in self.NAMES:
            e = cat["entries"][n]
            self.assertEqual(e["status"], "kuratiert", n)
            self.assertEqual(e["baeume"], ["nf"], n)
            self.assertTrue(e["satz_quelle"] if "satz_quelle" in e else True)
            self.assertIn("unbelegt", e["gain"])
        deps = {d["to"]: d for d in cat["entries"]["--moe-act-int8"]["depends"]}
        self.assertEqual(deps["--quantization"]["rel"], "braucht")
        self.assertEqual(deps["--quantization"]["kante"], "K132")
        self.assertEqual((deps["--hicache-storage-backend"]["rel"], deps["--hicache-storage-backend"]["kante"]), ("skaliert_mit", "K134"))
        self.assertEqual({d["to"]: d["kante"] for d in cat["entries"]["SGLANG_MOE_ACT_INT8"]["depends"]}, {"--quantization": "K133"})
        self.assertTrue(all(d["belegt"] for n in self.NAMES for d in cat["entries"][n]["depends"]))
        self.assertEqual((cat["kanten"]["kanten_gesamt"], cat["kanten"]["kanten_belegt"]), (135, 135))  # int24: + K135 (X-SUM-PRICE)

    def test_no_text_says_the_tree_has_no_w4a8_scheme_yet(self):
        """H88-F fix round 1: since H88-B the tree HAS the scheme (HAS_W4A8_MOE_SCHEME = True), so the help=, the environ.py
        comment, the curated text and the shipped catalog must not tell the user the switch aborts the start for lack of one."""
        stale = ("no W4A8 MoE scheme yet", "noch kein W4A8-MoE-Schema")
        from sglang.srt.layers.quantization import moe_act_int8 as M

        self.assertTrue(M.HAS_W4A8_MOE_SCHEME)
        server = PC.server_args_flags(os.path.join(SRT, "server_args.py"))["--moe-act-int8"]["help"]
        comment = PC.environ_fields(os.path.join(SRT, "environ.py"))["SGLANG_MOE_ACT_INT8"]["comment"]
        texts = [server, comment] + [CU.CURATED[n]["text"] for n in self.NAMES]
        path = os.path.join(TREE, "tools", "rig_dashboard", "rigdash", "profil_data", "catalog.json")
        with open(path, encoding="utf-8") as fh:
            cat = json.load(fh)
        for n in self.NAMES:
            e = cat["entries"][n]
            texts += [str(e.get("text", "")), str(e.get("help", ""))]
        for t in texts:
            for needle in stale:
                self.assertNotIn(needle, t)
        for t in (server, CU.CURATED["--moe-act-int8"]["text"]):
            self.assertIn("CompressedTensorsWNA16A8MoE", t)


# ---------------------------------------------------------------------------
# the planner
# ---------------------------------------------------------------------------

_TMP = None
_MODEL = None


def setUpModule():
    global _TMP, _MODEL
    _TMP = tempfile.TemporaryDirectory(prefix="h88e-models-")
    os.environ["SGLANG_CARD_LIBRARY"] = os.path.join(_TMP.name, "no-card-library.json")
    for name in _NF:
        O.materialize_checkpoint(os.path.join(CKPT, name), _TMP.name)
    mp = MP.estimate_or_state(os.path.join(_TMP.name, _NF[0]))
    assert mp["ok"], mp
    _MODEL = (mp["profile"], MP.estimate_draft(os.path.join(_TMP.name, _NF[1])))


def tearDownModule():
    os.environ.pop("SGLANG_CARD_LIBRARY", None)
    if _TMP is not None:
        _TMP.cleanup()


def _propose(profile_path):
    from sglang.srt.planner.card_library import CardLibrary

    rows = O.read_replay(REPLAY_REF)
    modell, draft = _MODEL
    return P.propose(rows, modell, "flip", {}, basis=O.profile_launch_input(profile_path), draft=draft, rates=MEASURED_RATES,
                     library=CardLibrary())


class TestPlannerPassThrough(unittest.TestCase):
    def test_rules_parse_both_spellings(self):
        for v in ("1", "true", "yes", "y", "on", "ON", "'on'"):
            self.assertIs(R.parse_switch(v), True, v)
        for v in ("0", "false", "no", "n", "off"):
            self.assertIs(R.parse_switch(v), False, v)
        self.assertIsNone(R.parse_switch(None))
        self.assertIsNone(R.parse_switch("maybe"))
        self.assertEqual(R.MOE_ACT_INT8_ENV, "SGLANG_MOE_ACT_INT8")
        self.assertEqual(R.MOE_ACT_INT8_FLAG, "--moe-act-int8")

    def test_the_w4a8_profile_keeps_the_switch_with_its_origin(self):
        v = _propose(W4A8_PROFILE)
        w = {x["key"]: x for x in v["werte"]}
        for key in ("--env-p SGLANG_MOE_ACT_INT8", "--env-d SGLANG_MOE_ACT_INT8", "env SGLANG_MOE_ACT_INT8"):
            self.assertIn(key, w, sorted(w))
            x = w[key]
            self.assertEqual((x["alt"], x["wert"], x["geaendert"]), ("1", "1", False), key)
            self.assertEqual(x["zustand"], R.VORGESCHLAGEN, key)
            self.assertTrue(x["herkunft"].startswith("vom Profil nf-int4-w4a8"), x)
            self.assertIn("unbelegt", x["grund"], key)
            self.assertEqual(x["policy"], "scalar", key)
        self.assertTrue(any("MOE-ACT-INT8" in h for h in v["hinweise"]), v["hinweise"])
        # a pass-through: the argv and the environment are the profile's, byte for byte
        li = O.profile_launch_input(W4A8_PROFILE)
        self.assertEqual(v["argv"], list(li.argv))
        self.assertEqual(v["env"], dict(li.env))
        self.assertTrue(v["vektoren_ok"])
        self.assertEqual(v["blocker"], [])

    def test_the_release_profile_has_no_record_of_it(self):
        v = _propose(ABL_PROFILE)
        self.assertFalse([x for x in v["werte"] if "MOE_ACT_INT8" in x["key"] or "moe-act-int8" in x["key"]])
        self.assertFalse([h for h in v["hinweise"] if "MOE-ACT-INT8" in h])

    def test_the_flag_in_the_group_extras_is_kept_too(self):
        b = O.profile_launch_input(ABL_PROFILE)
        la = P.LaunchArgv(b.argv, b.env)
        la.extra_set("p", "--moe-act-int8", "on")
        la.extra_set("d", "--moe-act-int8", "off")
        basis = {"argv": list(la.t), "env": dict(b.env), "vars": dict(b.vars), "name": "nf-int4-h6-abl.env"}
        from sglang.srt.planner.card_library import CardLibrary

        modell, draft = _MODEL
        v = P.propose(O.read_replay(REPLAY_REF), modell, "flip", {}, basis=basis, draft=draft, rates=MEASURED_RATES, library=CardLibrary())
        w = {x["key"]: x for x in v["werte"]}
        self.assertEqual(w["--extra-p --moe-act-int8"]["wert"], "on")
        self.assertEqual(w["--extra-d --moe-act-int8"]["wert"], "off")
        self.assertTrue(any("MOE-ACT-INT8" in h for h in v["hinweise"]))        # one place says on

    def test_an_unreadable_value_is_named_not_guessed(self):
        b = O.profile_launch_input(ABL_PROFILE)
        la = P.LaunchArgv(b.argv, b.env)
        la.env["SGLANG_MOE_ACT_INT8"] = "maybe"
        basis = {"argv": list(la.t), "env": dict(la.env), "vars": dict(b.vars), "name": "x.env"}
        from sglang.srt.planner.card_library import CardLibrary

        modell, draft = _MODEL
        v = P.propose(O.read_replay(REPLAY_REF), modell, "flip", {}, basis=basis, draft=draft, rates=MEASURED_RATES, library=CardLibrary())
        x = {w["key"]: w for w in v["werte"]}["env SGLANG_MOE_ACT_INT8"]
        self.assertEqual(x["wert"], "maybe")
        self.assertIn("unlesbar", x["grund"])
        self.assertFalse([h for h in v["hinweise"] if "MOE-ACT-INT8" in h])


class TestHwSimNote(unittest.TestCase):
    """``hw_sim`` (the hardware simulation grid) passes the switch as a profile scalar with its origin: a note, never a refusal."""

    @staticmethod
    def _model(path):
        from sglang.srt.weg2 import hw_sim

        li = O.profile_launch_input(path)
        m = hw_sim.MODELS["NF"]
        return hw_sim, hw_sim.SimModel(m.key, m.profile, m.weight_format, m.ckpt_mib, tuple(li.argv), dict(li.env), path + " (PROFILE_ARGS)", m.kv_heads)

    def test_the_w4a8_profile_gets_one_note_with_its_origin(self):
        hw_sim, m = self._model(W4A8_PROFILE)
        notes = hw_sim._moe_act_int8_notes(m)
        self.assertEqual(len(notes), 1, notes)
        self.assertIn("MOE-ACT-INT8", notes[0])
        self.assertIn("nf-int4-w4a8.env", notes[0])
        self.assertIn("--env-p SGLANG_MOE_ACT_INT8=1", notes[0])
        self.assertIn("--env-d SGLANG_MOE_ACT_INT8=1", notes[0])
        self.assertIn("unbelegt", notes[0])

    def test_the_release_profile_gets_none(self):
        hw_sim, m = self._model(ABL_PROFILE)
        self.assertEqual(hw_sim._moe_act_int8_notes(m), [])
        self.assertEqual(hw_sim._moe_act_int8_notes(hw_sim.MODELS["NF"]), [])        # the embedded release model: nothing


# ---------------------------------------------------------------------------
# the launcher dry run
# ---------------------------------------------------------------------------

def _snapshots():
    out = {}
    for n in sorted(os.listdir(CKPT)):
        if os.path.isfile(os.path.join(CKPT, n, "manifest.json")):
            out[O.read_snapshot_manifest(os.path.join(CKPT, n))["name"]] = os.path.join(CKPT, n)
    return out


class TestDryRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not all(n in _snapshots() for n in _NF):
            raise unittest.SkipTest("NF header snapshots missing")
        cls.abl = O.run_profile(ABL_PROFILE, O.read_replay(REPLAY_REF), tree=TREE, snapshots=_snapshots())
        cls.w4a8 = O.run_profile(W4A8_PROFILE, O.read_replay(REPLAY_REF), tree=TREE, snapshots=_snapshots())

    def test_both_plan_without_a_refusal(self):
        for run in (self.abl, self.w4a8):
            self.assertIsNone(run.result.exc_type, "%s: %s" % (run.result.exc_type, run.result.exc_msg[:300]))
            self.assertEqual(run.result.rc, 0)
            self.assertEqual(run.result.forced, [])

    def test_the_release_form_is_the_golden(self):
        want = _read(O.golden_path(GOLDEN, "plan_nf_abl_n3.txt", _LINE))
        d = O.diff_lines(want, self.abl.result.dump())
        self.assertEqual(d, [], "\n".join(x[:200] for x in d[:8]))
        self.assertNotIn("SGLANG_MOE_ACT_INT8", self.abl.result.dump())

    def test_the_switch_form_differs_from_the_golden_only_by_the_switch_and_what_follows_from_it(self):
        """H88-F (08.10.): with H88-B/C/D merged the switch has consequences in the dump, all named and nothing else:
        (1) the switch itself in WEG2-GROUP-ENV P/D (H88-E), (2) the L3 store directory/identity carries ``moe_act=int8;s16``
        (H88-D: dir suffix and the front's --store-dir), (3) the expert-store identity carries the layout (H88-C: H2c
        STORE-IDENTITY id + ' layout=marlin_w4a8') and (4) ONE new line 'H88C MOE-LAYOUT layout=marlin_w4a8 P=.. D=..'.
        Take those four out of the new lines and they ARE the golden lines."""
        want = _read(O.golden_path(GOLDEN, "plan_nf_abl_n3.txt", _LINE))
        d = [x for x in O.diff_lines(want, self.w4a8.result.dump()) if x[:1] in "+-" and x[:3] not in ("+++", "---")]
        plus = [x[1:] for x in d if x[0] == "+"]
        minus = [x[1:] for x in d if x[0] == "-"]
        self.assertTrue(plus)
        layout_lines = [x for x in plus if "WEG2-LAUNCH H88C MOE-LAYOUT layout=marlin_w4a8 P=marlin_w4a8 D=marlin_w4a8" in x]
        self.assertEqual(len(layout_lines), 1, plus)
        rest = [x for x in plus if x not in layout_lines]
        self.assertEqual(len(rest), len(minus))
        for x in minus:
            self.assertNotIn("SGLANG_MOE_ACT_INT8", x)
            self.assertNotIn("moe_act", x)
            self.assertNotIn("marlin_w4a8", x)
        # the L3 dir suffix: the golden's suffix (03a5fb6b55) is replaced by the identity hash that carries moe_act
        m_old = re.search(r"-abl-wxp-([0-9a-f]{10})\b", "\n".join(minus))
        m_new = re.search(r"-abl-wxp-([0-9a-f]{10})\b", "\n".join(rest))
        self.assertTrue(m_old and m_new)
        self.assertNotEqual(m_old.group(1), m_new.group(1))
        self.assertTrue(any('"moe_act": "int8;s16"' in x for x in rest))
        self.assertTrue(any("SGLANG_MOE_ACT_INT8=1" in x for x in rest))
        norm = []
        for x in rest:
            x = x.replace(";SGLANG_MOE_ACT_INT8=1", "").replace("SGLANG_MOE_ACT_INT8=1;", "")
            x = x.replace(m_new.group(1), m_old.group(1)).replace(' "moe_act": "int8;s16",', "")
            x = re.sub(r"(STORE[-_]IDENTITY[=_ ](?:id=)?)[0-9a-f]{24}", r"\1<ID>", x).replace(" layout=marlin_w4a8", "")
            norm.append(x)
        want_minus = [re.sub(r"(STORE[-_]IDENTITY[=_ ](?:id=)?)[0-9a-f]{24}", r"\1<ID>", x) for x in minus]
        self.assertEqual(norm, want_minus)

    def test_the_plan_dump_shows_the_switch_in_both_group_environments(self):
        dump = self.w4a8.result.dump()
        groups = {}
        for line in dump.splitlines():
            m = re.search(r"WEG2-GROUP-ENV ([PD]): (\S+)", line)
            if m:
                groups[m.group(1)] = m.group(2).split(";")
        self.assertEqual(sorted(groups), ["D", "P"])
        for g in ("P", "D"):
            self.assertIn("SGLANG_MOE_ACT_INT8=1", groups[g], g)


if __name__ == "__main__":
    unittest.main()
