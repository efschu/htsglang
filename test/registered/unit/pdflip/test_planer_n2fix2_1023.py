"""planer-n2fix2 1023: the launcher/oracle half of the follow-up to the n2fix review (deskq/done/planer-n2fix-1023.md, "Review 23.10." + Nachtrag).

* A3  ``launcher.card_power_limits_w`` gets the replay lock of ``pcie_lanes``: on a REPLAYED inventory (the planner's dry runs) the live query by NVML
      index read the BOX's card of that index -- since 22.10. NVML 0 is the 5090, so a replayed RTX 3080 was declared at 400 W (Review: dry run 2 x 3080,
      ``PP-CUT CARD-POWER stage 0 card='NVIDIA GeForce RTX 3080' ... limit_w=400``).  Now the recording's own limit is read by UUID through the registry's
      replay seam; a recording without the field is "unknown", never the box's value.
* B11 a propose request with ``model_path`` / ``draft_path`` priced the model from that path (stage A) but the launcher dry run kept the base profile's
      ``--model`` / ``--dflash-draft-path`` (the int4ar seat's call of 23.10. 13:35Z ran the dry run on the NVFP4 weights).  ``propose_verdict.retarget_paths``
      retargets the base launch; the dashboard half (the start profile) is pinned in ``rigdash/tests/test_planer_n2fix2_1023.py``.
* A4  the release profile ``27b-int4ar.env`` evaluates to the form of the int4ar seat's boot profile (deskq/bl/27b-int4ar-1023/27b-int4ar.json).
"""

import json
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import launcher as L  # noqa: E402
from flliper.srt.pdflip import propose_oracle as O  # noqa: E402
from flliper.srt.pdflip import propose_verdict as V  # noqa: E402
from flliper.srt.registry import nvml as R  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
RELEASE = os.path.join(REPO, "docker", "flliper", "profiles_release")
MC = "/spinning/llm_stuff/club-3090/models-cache/"
INT4AR = MC + "Huihui-Qwen3.8-27B-Abliterated-INT4-W4A16-AutoRound-ct"
W8 = MC + "Qwen3.8-27B-DFlash2-W8-lued"
NVFP4 = MC + "Qwen3.8-27B-NVFP4-RadixArk"
NVFP4_DRAFT = MC + "Qwen3.8-27B-DFlash2-NVFP4-RTNcal"

U5090 = "GPU-31d7ef41-f574-4d0e-21ad-e773fd938f6d"
U3080A = "GPU-5c648f96-be1d-42d5-0221-34d11ab137f7"
U3080B = "GPU-62dbbae1-e859-9ccc-f9c2-d9f2443a84f4"


class _NoNvml(types.ModuleType):
    """A pynvml that fails the test on ANY use: under a replay nothing may read the box."""

    def __getattr__(self, name):
        raise AssertionError("live NVML read under a replay: pynvml.%s" % name)


class PowerLimitReplay(unittest.TestCase):
    def _replay(self, rows):
        fd, path = tempfile.mkstemp(suffix=".json", prefix="n2fix2_replay_")
        with os.fdopen(fd, "w") as fh:
            json.dump(rows, fh)
        self.addCleanup(os.unlink, path)
        return path

    def _cards(self):
        # replayed 2 x 3080 renumbered 0..1 (a container's view): NVML index 0 is a 3080 HERE, the 5090 on the live box
        return [L.Card(nvml_index=0, uuid=U3080A, name="NVIDIA GeForce RTX 3080", total_mib=20480),
                L.Card(nvml_index=1, uuid=U3080B, name="NVIDIA GeForce RTX 3080", total_mib=20480)]

    def test_the_recorded_limit_is_read_by_uuid_and_the_box_is_never_asked(self):
        path = self._replay([{"uuid": U3080A, "name": "NVIDIA GeForce RTX 3080", "index": 0, "power_limit_w": 230.0},
                             {"uuid": U3080B, "name": "NVIDIA GeForce RTX 3080", "index": 1, "power_limit_w": 231.0}])
        with mock.patch.dict(os.environ, {R.ENV_NVML_REPLAY: path}), mock.patch.dict(sys.modules, {"pynvml": _NoNvml("pynvml")}):
            self.assertEqual(L.card_power_limits_w(self._cards()), [230.0, 231.0])

    def test_a_recording_without_the_field_is_unknown_not_the_boxs_400(self):
        path = self._replay([{"uuid": U3080A, "name": "NVIDIA GeForce RTX 3080", "index": 0},
                             {"uuid": U5090, "name": "NVIDIA GeForce RTX 5090", "index": 1, "power_limit_w": 400.0}])
        with mock.patch.dict(os.environ, {R.ENV_NVML_REPLAY: path}), mock.patch.dict(sys.modules, {"pynvml": _NoNvml("pynvml")}):
            got = L.card_power_limits_w(self._cards())
        self.assertEqual(got, [None, None])          # 3080 A: no field; 3080 B: not in the recording -- never the 5090's 400 of index 1
        self.assertIn("limit_w=unknown", L.card_power_line(self._cards(), got))

    def test_live_reads_by_index_as_before(self):
        class _H:
            def __init__(self, i):
                self.i = i

        fake = types.SimpleNamespace(nvmlInit=lambda: None, nvmlShutdown=lambda: None, nvmlDeviceGetHandleByIndex=_H,
                                     nvmlDeviceGetEnforcedPowerLimit=lambda h: (400000, 230000)[h.i])
        env = dict(os.environ)
        env.pop(R.ENV_NVML_REPLAY, None)
        with mock.patch.dict(os.environ, env, clear=True), mock.patch.dict(sys.modules, {"pynvml": fake}):
            self.assertEqual(L.card_power_limits_w(self._cards()), [400.0, 230.0])


class HardwareProfileReplayCarriesTheLimit(unittest.TestCase):
    """The planner's replay of THIS rig comes from its hardware profile (``flliper.hardware/1``, NVML ``power.limit``): the replay row carries the
    limit, so the dry run of 2 x RTX 3080 declares 230 W per 3080 (the review saw 400 W, the 5090 of the live NVML 0)."""

    HW = os.path.join(REPO, "tools", "rig_dashboard", "rigdash", "tests", "fixtures", "planer_n2fix_1023", "hwprofil_rig_1023.json")

    def test_rows_and_launcher_read_the_recorded_limit(self):
        with open(self.HW, encoding="utf-8") as fh:
            hw = json.load(fh)
        rows = O.replay_from_hardware_profile(hw)
        self.assertEqual([(r["name"][-4:], r.get("power_limit_w"), r.get("power_limit_default_w")) for r in rows],
                         [("5090", 400.0, 575.0), ("3080", 230.0, 320.0), ("3080", 230.0, 320.0)])
        # two of the three, renumbered 0..1 like a container sees them (planer-n2fix B7): NVML 0 of the replay is a 3080
        two = [dict(r, index=i) for i, r in enumerate(rows[1:])]
        fd, path = tempfile.mkstemp(suffix=".json", prefix="n2fix2_hw_")
        with os.fdopen(fd, "w") as fh:
            json.dump(two, fh)
        self.addCleanup(os.unlink, path)
        cards = [L.Card(nvml_index=r["index"], uuid=r["uuid"], name=r["name"], total_mib=r["total_bytes"] >> 20) for r in two]
        with mock.patch.dict(os.environ, {R.ENV_NVML_REPLAY: path}), mock.patch.dict(sys.modules, {"pynvml": _NoNvml("pynvml")}):
            self.assertEqual(L.card_power_limits_w(cards), [230.0, 230.0])


class RetargetPaths(unittest.TestCase):
    def _li(self):
        argv = ["--model", NVFP4, "--fp8-uniform-marlin", "--pdflip-xchg-census", "/x/census.json", "--dflash-draft-path", NVFP4_DRAFT,
                "--extra-p=--speculative-draft-model-path %s --max-running-requests=2" % NVFP4_DRAFT, "--note", NVFP4 + "-ct/config.json"]
        env = {"FLLIPER_X": NVFP4_DRAFT + "/config.json", "OTHER": "1"}
        vars_ = {"PROFILE_MODEL": NVFP4, "PROFILE_DRAFT": NVFP4_DRAFT, "PROFILE_NAME": "27b-nvfp4", "PROFILE_P_CHUNK_MODEL_NVFP4": "/p/27b-nvfp4.pchunk.json"}
        return O.LaunchInput(argv, env, vars_, [], "/r/27b-nvfp4.env", "0")

    def test_model_and_draft_are_the_requested_ones_everywhere(self):
        li, notes = V.retarget_paths(self._li(), INT4AR, W8 + "/")
        self.assertEqual(li.model, INT4AR)
        self.assertEqual(li.draft, W8)
        self.assertEqual(li.argv[li.argv.index("--model") + 1], INT4AR)
        self.assertEqual(li.argv[li.argv.index("--dflash-draft-path") + 1], W8)
        self.assertIn("--speculative-draft-model-path %s --max" % W8, " ".join(li.argv))
        self.assertEqual(li.env["FLLIPER_X"], W8 + "/config.json")
        self.assertEqual(len(notes), 2)
        self.assertNotIn(NVFP4_DRAFT, " ".join(li.argv) + json.dumps(li.env) + json.dumps(li.vars))
        # a LONGER name that starts with the old directory is another directory: untouched
        self.assertEqual(li.argv[-1], NVFP4 + "-ct/config.json")
        self.assertEqual(li.source, "/r/27b-nvfp4.env")

    def test_nothing_requested_or_the_same_path_is_a_no_op(self):
        base = self._li()
        for m, d in ((None, None), (NVFP4, NVFP4_DRAFT), (NVFP4 + "/", "")):
            li, notes = V.retarget_paths(base, m, d)
            self.assertIs(li, base)
            self.assertEqual(notes, [])

    def test_identity_of_the_retargeted_launch_names_the_requested_model(self):
        li, _ = V.retarget_paths(self._li(), INT4AR, None)
        ident = V.profile_identity(li)
        self.assertEqual(ident["model"], INT4AR)
        self.assertEqual(ident["draft"], NVFP4_DRAFT)
        self.assertNotEqual(ident["input_sha256"], V.profile_identity(self._li())["input_sha256"])

    def test_run_propose_retargets_before_the_model_profile(self):
        """``run_propose`` calls ``retarget_paths`` with the request's paths before anything reads the launch (no model profile here: the call
        ends at "no model profile", which is enough to see the launch it was given)."""
        seen = {}

        def fake_profiles(li, req):
            seen["model"], seen["draft"], seen["argv"] = li.model, li.draft, list(li.argv)
            return None, None, ["stub"]

        with mock.patch.object(V, "_basis_from", lambda req, scratch: self._li()), mock.patch.object(V, "_devices_of", lambda req: []), \
                mock.patch.object(V, "_model_profiles", fake_profiles):
            out = V.run_propose({"form": "flip", "model_path": INT4AR, "draft_path": W8}, tree=REPO)
        self.assertFalse(out["ok"])
        self.assertEqual((seen["model"], seen["draft"]), (INT4AR, W8))
        self.assertEqual(seen["argv"][1], INT4AR)
        self.assertTrue(any(INT4AR in n for n in out["notizen"]), out["notizen"])


class ReleaseProfileInt4ar(unittest.TestCase):
    def setUp(self):
        self.li = O.profile_launch_input(os.path.join(RELEASE, "27b-int4ar.env"))

    def test_the_form_of_the_seats_boot_profile(self):
        a = self.li.argv
        self.assertEqual(a[:2], ["--model", INT4AR])
        self.assertEqual(a[a.index("--dflash-draft-path") + 1], W8)
        self.assertEqual(a[a.index("--pdflip-xchg-census") + 2], "--pdflip-xchg-census-foreign")
        for gone in ("--fp8-uniform-marlin", "--fp4-native-mixed", "--p-chunk-policy", "--p-chunk-max", "--p-chunk-model", "--p-chunk-mscale",
                     "--pp-stage-ratio", "--pp-attn-stage-ratio"):
            self.assertNotIn(gone, a)
        v = self.li.vars
        self.assertEqual((v["PROFILE_NAME"], v["PROFILE_FORMAT"], v["PROFILE_MODEL"], v["PROFILE_DRAFT"], v["PROFILE_P_CHUNK"], v["PROFILE_LINE"]),
                         ("27b-int4ar", "int4", INT4AR, W8, "fixed", "27b"))
        self.assertEqual(v["PROFILE_STATUS"], "experimentell")

    @unittest.skipUnless(os.path.isfile(os.path.join(INT4AR, "config.json")), "the int4ar checkpoint is not on this box")
    def test_the_checkpoint_is_what_the_entrypoint_calls_int4(self):
        with open(os.path.join(INT4AR, "config.json"), encoding="utf-8") as fh:
            q = json.load(fh)["quantization_config"]
        kinds = {(g["weights"]["type"], g["weights"]["num_bits"]) for g in q["config_groups"].values()}
        self.assertEqual((q["quant_method"], kinds), ("compressed-tensors", {("int", 4)}))      # entrypoint detect_format -> int4


if __name__ == "__main__":
    unittest.main()
