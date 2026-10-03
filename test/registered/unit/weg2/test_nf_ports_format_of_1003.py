"""NF PORTS 1003 (order 980 item 5): ``form.format_of`` knows the NF release
checkpoint (``...-abl-wxp``; finding 2 of the HW-P1ab report).

HARD RULE of the order: the NF abl release form on the reference rig stays
byte-identical. Pinned here: the abl checkpoint resolves to the SAME registry
format as its base, and every launcher default that reads the format
(``_row_format_default``: --p-chunk-policy, --d-token-placement) gives the
code default for NF -- before and after, because the NF row lists no format
for either (``chunk.default_formats == ()``, ``d_token_placement_formats == ()``).
"""

import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import form as F
from sglang.srt.weg2 import hw_sim as HS
from sglang.srt.weg2 import launcher as L
from sglang.srt.weg2 import topology as T

MC = "/spinning/llm_stuff/club-3090/models-cache/"
BASE = MC + "Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist"
ABL = BASE + "-abl-wxp"


def ns_for(*extra, model=ABL):
    return L.build_parser().parse_args(
        ["--tree", "/t", "--tag", "t", "--profile", "nextflash", "--model", model, *extra])


class FormatOfKnowsTheNfReleaseCheckpoint(unittest.TestCase):

    def test_abl_release_checkpoint_is_the_int4_mixed_format(self):
        self.assertEqual(F.format_of("nextflash", BASE), "int4-mixed")
        self.assertEqual(F.format_of("nextflash", ABL), "int4-mixed")
        self.assertEqual(F.format_of("nextflash", ABL + "/"), "int4-mixed")

    def test_only_named_derivatives_match_never_a_suffix(self):
        # a new export / another quantisation of the same family stays unknown
        self.assertEqual(F.format_of("nextflash", BASE + "-g128"), "")
        self.assertEqual(F.format_of("nextflash", BASE + "-abl-other"), "")
        self.assertEqual(F.format_of("nextflash", MC + "Qwen3.8-Flash-Next-NVFP4-nvidia-abl-wxp"), "")
        self.assertEqual(F.format_of("nextflash", MC + "Qwen3.8-Flash-Next-NVFP4-nvidia"), "nvfp4")

    def test_the_27b_row_is_untouched(self):
        for name, wf in F.profile_row("qwen27b").formats.items():
            self.assertEqual(wf.derivatives, (), name)
            self.assertEqual(F.format_of("qwen27b", wf.checkpoint), name)
        self.assertEqual(F.format_of("qwen27b", ABL), "")

    def test_docker_profile_source_is_still_the_base_checkpoint(self):
        wf = F.profile_row("nextflash").formats["int4-mixed"]
        self.assertTrue(wf.checkpoint.endswith("Minachist"))
        self.assertEqual(wf.derivatives, (ABL,))

    def test_nf_launcher_defaults_do_not_move_for_the_abl_checkpoint(self):
        # the only readers of format_of that change a DEFAULT: NF lists no format
        row = F.profile_row("nextflash")
        self.assertEqual(row.chunk.default_formats, ())
        self.assertEqual(row.d_token_placement_formats, ())
        for model in (BASE, ABL):
            ns = ns_for(model=model)
            self.assertEqual(
                L._row_format_default("nextflash", ns, row.chunk.policy, row.chunk.default_formats,
                                      L.P_CHUNK_POLICY_DEFAULT), L.P_CHUNK_POLICY_DEFAULT)
            self.assertEqual(
                L._row_format_default("nextflash", ns, row.d_token_placement,
                                      row.d_token_placement_formats, L.D_TOKEN_PLACEMENT_DEFAULT),
                L.D_TOKEN_PLACEMENT_DEFAULT)

    def test_topology_context_names_the_format_for_the_release_model(self):
        ctx = L.topology_context(ns_for(), {})
        self.assertEqual(ctx.weight_format, "int4-mixed")

    def test_pp_cut_pin_blocker_now_fires_for_the_release_checkpoint(self):
        # before: format '' -> the PP-CUT-PIN probe was silent for the abl model
        ns = ns_for("--pp-stage-ratio", "29,11,8", "--pp-attn-stage-ratio", "7,3,2")
        ctx = L.topology_context(ns, {})
        with self.assertRaises(T.TopologyRefused) as cm:
            T.plan_topology(2, ctx)
        self.assertIn("PP-CUT-PIN", [b.code for b in cm.exception.blockers])
        # N = 3 is proven on metal: nothing is judged anew
        T.plan_topology(3, ctx)


if __name__ == "__main__":
    unittest.main()
