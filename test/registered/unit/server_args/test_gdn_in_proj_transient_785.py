# SPDX-License-Identifier: Apache-2.0
"""The GDN prefill estimate must cover the allocation that actually OOM'd.

SPECIMEN, 2026-08-21 on this rig (boot_735_bal785.log, first real agent load,
inside the first tp_to_pp flip):

    File "sglang/srt/models/qwen3_5.py", line 625, in forward
      projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states)
    File "sglang/srt/layers/linear.py", line 835, in forward
      return F.linear(x, layer.weight, bias)
    torch.OutOfMemoryError: Tried to allocate 256.00 MiB.
    GPU 0 has a total capacity of 31.34 GiB of which 131.69 MiB is free.

The instance ran --chunked-prefill-size 8192 --tp-size 1 --pp-size 3, so the
GDN layer was UNSHARDED (tp_size 1 -> full width) and

    256 MiB == 8192 tokens * 16384 columns * 2 bytes
    16384   == 2*key_dim + 2*value_dim == 2*(16*128) + 2*(48*128)

which is in_proj_qkvz's packed output, z half included. gdn_prefill_scratch_mib
began at the CONV output and so did not count it. That is not a rounding
error in a diagnostic: the prefill-admission guard takes its threshold from
this function, and the log shows it admitting the chunk it should have held --
"corridor law 1024 MiB (prefill admission, 8192 tokens)", "corridor shortfall
of 981 MiB for rank 0", "corridor cannot be restored ahead of this chunk" --
where the honest threshold is ~1280 MiB.
"""

import unittest
from unittest.mock import patch

from sglang.srt.server_args import ServerArgs

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10)

MIB = 1 << 20

#: (num_k_heads, num_v_heads, head_k_dim, head_v_dim, itemsize) of the
#: Qwen3.8-27B checkpoint this rig serves, read from its config.json.
QWEN38_GDN = (16, 48, 128, 128, 2)


class GdnInProjIsCounted(unittest.TestCase):
    def _args(self, tokens):
        # model_path="dummy" short-circuits __post_init__, so this exercises
        # the estimator in isolation without device detection.
        args = ServerArgs(model_path="dummy", enable_vram_ledger=False)
        args.chunked_prefill_size = tokens
        return args

    def _scratch(self, tokens, share=1.0):
        args = self._args(tokens)
        with patch.object(
            ServerArgs, "_gdn_linear_attention_dims", return_value=QWEN38_GDN
        ):
            return args.gdn_prefill_scratch_mib(share)

    def test_the_projection_output_is_part_of_the_estimate(self):
        """The 256 MiB tensor from the specimen, at the specimen's settings."""
        full = self._scratch(8192)
        # 2*(16*128) + 2*(48*128) = 16384 columns, bf16, 8192 tokens.
        # 257.5, not the specimen's flat 256.00: the traceback reports only
        # in_proj_qkvz's own tensor, while this term also carries in_proj_ba's
        # 2*Hv columns allocated beside it on the next line.
        in_proj_mib = 8192 * (2 * 16 * 128 + 2 * 48 * 128 + 2 * 48) * 2 / MIB
        self.assertAlmostEqual(in_proj_mib, 257.5, places=1)
        self.assertGreater(full, 1200.0)
        self.assertLess(full, 1350.0)

    def test_the_estimate_now_exceeds_the_guard_threshold_that_let_it_through(self):
        """CAN-FAIL GUARD, and the whole point of the correction.

        The admission guard held 1024 MiB for an 8192-token chunk and the
        chunk needed more. An estimate that still came in at or under 1024
        would leave the gate exactly as permissive as it was when it let the
        OOM through."""
        self.assertGreater(self._scratch(8192), 1024.0)

    def test_it_scales_with_the_chunk_because_the_chunk_is_the_cap(self):
        """Halving --chunked-prefill-size must halve the transient: that is
        the operator's lever, and it has to show up here or the lever is
        invisible to every consumer of this number."""
        self.assertAlmostEqual(self._scratch(8192) / self._scratch(4096), 2.0, places=1)

    def test_a_sharded_rank_carries_proportionally_less(self):
        """Under the rig's 32,16,16 vector rank 0 owns half the head units and
        therefore twice what ranks 1 and 2 each carry."""
        self.assertAlmostEqual(
            self._scratch(8192, 1.0) / self._scratch(8192, 0.5), 2.0, places=1
        )

    def test_no_gdn_layers_still_means_no_number(self):
        args = self._args(8192)
        with patch.object(ServerArgs, "_gdn_linear_attention_dims", return_value=None):
            self.assertIsNone(args.gdn_prefill_scratch_mib(1.0))


if __name__ == "__main__":
    unittest.main()
