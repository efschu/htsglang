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
"""A silently substituted PP cut must announce itself (#505(a) class).

THE SPECIMEN, 2026-08-17. The speed boot was ordered at cut [31,17,16] and
launched with `--pp-stage-ratio 31,17,16`. The coupled derivation balances both
layer families together and returned [32,16,16], which moved a FULL-ATTENTION
layer onto stage 0 -- 8/4/4 instead of 7/5/4 -- shifting KV mass against a
pinned 550000-token pool. The boot OOM'd inside an NCCL send at 08:53:51 with
PP0 down to 176 MiB free against a 1024 MiB corridor law.

The derived split WAS logged. What was missing is any statement that the
request had not been honoured, so the boot record read as the cut that was
ordered. That is the defect: not a wrong number, an unannounced substitution.

WHAT IS NOT CLAIMED HERE: that the cut is unrealizable. It is realizable --
`--pp-attn-stage-ratio` (#485) decouples the families, and 7,5,4 gives exactly
[31,17,16]. The warning therefore names that remedy instead of refusing, and
computes it from the operator's own ranges rather than hinting at it.
"""

import json
import logging
import unittest

from sglang.srt.distributed.utils import derive_pp_layer_split
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5)

MODEL = "/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-27B-INT8-yarn1.5"


def _kinds():
    with open(f"{MODEL}/config.json") as fh:
        cfg = json.load(fh)["text_config"]
    return [("full" in t) for t in cfg["layer_types"]]


class TheDecouplingExists(unittest.TestCase):
    """Pins the mechanism the warning points at, so the advice cannot rot."""

    def test_coupled_derivation_snaps_the_requested_cut(self):
        kinds = _kinds()
        self.assertEqual([32, 16, 16], derive_pp_layer_split([31, 17, 16], kinds))

    def test_attn_stage_ratio_realizes_it_exactly(self):
        """[31,17,16] IS reachable -- the families just have to be decoupled."""
        kinds = _kinds()
        got = derive_pp_layer_split([31, 17, 16], kinds, attn_scores=[7, 5, 4])
        self.assertEqual([31, 17, 16], got)

    def test_the_advised_vector_is_the_ranges_own_full_attention_counts(self):
        """Why 7,5,4 is computable rather than guessed."""
        kinds = _kinds()
        want, start = [], 0
        for count in (31, 17, 16):
            want.append(sum(1 for k in kinds[start : start + count] if k))
            start += count
        self.assertEqual([7, 5, 4], want)
        self.assertEqual(
            [31, 17, 16], derive_pp_layer_split([31, 17, 16], kinds, attn_scores=want)
        )


class TheSubstitutionIsAnnounced(unittest.TestCase):
    def _warn(self, scores, attn=None):
        from sglang.srt.server_args import ServerArgs

        args = ServerArgs.__new__(ServerArgs)
        args.pp_stage_ratio = list(scores)
        args.pp_attn_stage_ratio = list(attn) if attn else None
        args.pp_layer_ratio = None
        args.pp_size = 3
        args.model_path = MODEL
        with self.assertLogs("sglang.srt.server_args", level=logging.INFO) as cm:
            args._handle_pp_stage_ratio()
        return args, [r for r in cm.output if r.startswith("WARNING")]

    def test_a_snapped_layer_spelling_warns_and_names_the_remedy(self):
        args, warnings = self._warn([31, 17, 16])
        self.assertEqual([32, 16, 16], args.pp_layer_ratio)
        self.assertTrue(warnings, "a substituted cut must warn")
        text = "\n".join(warnings)
        self.assertIn("--pp-attn-stage-ratio 7,5,4", text)
        self.assertIn("31,17,16", text)
        self.assertIn("32,16,16", text)

    def test_an_honoured_request_does_not_warn(self):
        """CAN-FAIL: a warning on every boot would be noise, not a signal."""
        _args, warnings = self._warn([32, 16, 16])
        self.assertFalse(warnings, f"unexpected warning: {warnings}")

    def test_a_true_ratio_does_not_warn(self):
        """14,10,8 is the incumbent and does NOT sum to the depth.

        Scores are capability ratios by contract, so deriving a different layer
        split from them is correct and must stay silent. The warning fires only
        on the case that looks like a spelled-out split.
        """
        args, warnings = self._warn([14, 10, 8])
        self.assertEqual([28, 20, 16], args.pp_layer_ratio)
        self.assertFalse(warnings, f"unexpected warning: {warnings}")

    def test_decoupled_request_does_not_warn(self):
        _args, warnings = self._warn([31, 17, 16], attn=[7, 5, 4])
        self.assertFalse(warnings, f"unexpected warning: {warnings}")


if __name__ == "__main__":
    unittest.main()
