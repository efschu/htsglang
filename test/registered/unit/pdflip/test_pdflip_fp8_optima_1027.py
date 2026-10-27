"""FP8-Form (Qwen3.8-27B-FP8, --fp8-uniform-marlin = W8A16-Marlin) 27.10.2026, desk/fp8-optima-1027.

Messbasis: Boot dkr27bfp8bar1drei10271317 (cut 44/11,9, 512er Graph-Chunks, 2812 #PGAP-Forwards). Pins:
  * builtin-fp8 ist ein eigener Record (nicht der int8-Satz) und wird wie builtin-int8 behandelt: eine Schnitt-Abweichung warnt,
    sie verweigert nicht -- der Loeser waehlt fuer fp8 mit dem eigenen Stufenmodell einen anderen Schnitt als 44,11,9;
  * --pp-cut-stage-model auto findet fuer fp8 das fp8-Modell und fuer int8 weiter das int8-Modell;
  * das fp8-Modell rankt den P-Schnitt mit leichterer 5090 (PP0) vorn;
  * das Release-Profil 27b-fp8 faehrt P-Chunk fixed, und unter HTSGLANG_P_CHUNK_POLICY=dynamic genau EINEN fp8-Satz.
"""

import os
import subprocess
import types
import unittest

from flliper.srt.pdflip import card_identity as CI
from flliper.srt.pdflip import form as F
from flliper.srt.pdflip import p_stage_model as M

try:
    from flliper.srt.pdflip import launcher as L
except Exception:  # noqa: BLE001
    L = None

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), *([os.pardir] * 4)))
RELEASE_PROFILE = os.path.join(REPO, "docker", "flliper", "profiles_release", "27b-fp8.env")


@unittest.skipIf(L is None, "launcher not importable on this interpreter")
class TestBuiltinFp8Record(unittest.TestCase):
    def setUp(self):
        self.logged = []
        L._P_CHUNK["spec"] = object()  # a dynamic policy is installed

    def tearDown(self):
        L._P_CHUNK["spec"] = None

    def _cut(self, counts):
        return types.SimpleNamespace(layer_counts=counts, gapped=False, layer_set="")

    def test_builtin_fp8_is_its_own_measurement(self):
        fp8, _ = L.p_chunk_stage_model("builtin-fp8", "flat", 3)
        int8, _ = L.p_chunk_stage_model("builtin-int8", "flat", 3)
        # the 5090 stage: W8A16 Marlin costs ~2.5x the INT8 GEMMs per 512 chunk (112.7 vs 43.9 ms intercept)
        self.assertGreater(fp8[0].points[1][1], 2.0 * int8[0].points[1][1])
        self.assertEqual(L.p_chunk_model_cut("builtin-fp8")[0], (44, 11, 9))

    def test_cut_mismatch_of_the_fp8_builtin_warns_and_does_not_refuse(self):
        # the fp8 stage model moves the solved cut off the fitted 44,11,9 (40,13,11 is -9 % in the model): a refusal here
        # (what a JSON table gets) would kill the first boot on the new cut
        ns = types.SimpleNamespace(p_chunk_model="builtin-fp8")
        self.assertEqual(L.p_chunk_model_cut_check(ns, self._cut((44, 11, 9)), self.logged.append), "match")
        self.assertEqual(L.p_chunk_model_cut_check(ns, self._cut((40, 13, 11)), self.logged.append), "mismatch")
        self.assertIn("builtin-fp8 model was fitted on 44,11,9", self.logged[-1])

    def test_power_limits_come_from_the_fp8_record(self):
        vals, _ = L.p_chunk_model_power_limits("builtin-fp8", 3)
        self.assertEqual(vals, [400.0, 230.0, 230.0])


@unittest.skipIf(L is None, "launcher not importable on this interpreter")
class TestFp8CutStageModel(unittest.TestCase):
    def _auto(self, fmt):
        ckpt = F.profile_row("qwen27b").formats[fmt].checkpoint
        return L.resolve_pp_cut_stage_model("auto", "qwen27b", ckpt, inventory=CI.REFERENCE_INVENTORY)

    def test_auto_picks_the_model_of_the_checkpoint_format(self):
        fp8, why = self._auto("fp8")
        int8, _ = self._auto("int8")
        self.assertIn("27b_fp8_drei1027.json", why)
        # external-source literals: ms per layer and 512-token forward on a 3080 (fp8 W8A16 Marlin 7.27, INT8 W8A8 3.12)
        self.assertAlmostEqual(fp8.cards["RTX3080"].gemm(512, M.MODE_GRAPH), 7.27, delta=0.3)
        self.assertAlmostEqual(int8.cards["RTX3080"].gemm(512, M.MODE_GRAPH), 3.12, delta=0.3)

    def test_fp8_model_prefers_the_lighter_5090_stage(self):
        model, _ = self._auto("fp8")
        mix = ((512, 40960, M.MODE_GRAPH, 1.0), (512, 90112, M.MODE_GRAPH, 1.0))
        incumbent = M.weighted_makespan(model, (44, 11, 9), mix)
        lighter = M.weighted_makespan(model, (40, 13, 11), mix)
        self.assertLess(lighter, 0.95 * incumbent)
        best = min(M.contiguous_cuts(64, 3, 4), key=lambda c: M.weighted_makespan(model, c, mix))
        self.assertLessEqual(best[0], 41)


@unittest.skipUnless(os.path.exists(RELEASE_PROFILE), "release profile not in this tree")
class TestReleaseProfileFp8(unittest.TestCase):
    def _args(self, policy):
        env = dict(os.environ)
        env.pop("HTSGLANG_P_CHUNK_POLICY", None)
        if policy:
            env["HTSGLANG_P_CHUNK_POLICY"] = policy
        out = subprocess.run(["bash", "-c", 'source "$1"; printf "%s\\n" "${PROFILE_ARGS[@]}"', "_", RELEASE_PROFILE],
                             capture_output=True, text=True, env=env, check=True).stdout.split("\n")
        return out

    def test_default_is_fixed(self):
        self.assertNotIn("--p-chunk-policy", self._args(""))

    def test_dynamic_is_one_fp8_set_not_the_int8_set(self):
        args = self._args("dynamic")
        for flag, want in (("--p-chunk-policy", "dynamic"), ("--p-chunk-model", "builtin-fp8"),
                           ("--p-chunk-mscale", "flat"), ("--p-chunk-max", "2048")):
            self.assertEqual(args.count(flag), 1, flag)
            self.assertEqual(args[args.index(flag) + 1], want, flag)


if __name__ == "__main__":
    unittest.main()
