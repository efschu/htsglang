"""fnFL2 H58: SGLANG_FORCE_QSA_ROWS_CONFIG -- a per-arch launch override for
the QSA rows kernel, whose device-name-keyed table (H20 / else L20) hands the
rig's 3080 and 5090 the spilling (BLOCK_N 16, 1 warp, 2 stages) build for
every P prefix chunk (sparse_attn.py, H58 comment). CPU only: the parser and
the selection run against a mocked device; the kernel itself is untouched
(launch parameters only), so its metal pin stays test_qsa_sparse_rows_dcp_wp3b.
"""

import os
import re
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.environ import envs
from sglang.srt.layers.attention.qsa import sparse_attn as sa
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

INF = float("inf")


class ParseTest(unittest.TestCase):
    def test_generic_group_applies_to_every_arch(self):
        t = sa.parse_rows_config("32=32/8/2,inf=32/8/2", 86)
        self.assertEqual(t, [(32, (32, 8, 2)), (INF, (32, 8, 2))])
        self.assertEqual(sa.parse_rows_config("inf=64/8/2", 120), [(INF, (64, 8, 2))])

    def test_arch_group_wins_and_other_arch_falls_through(self):
        raw = "sm86:inf=32/8/2;inf=64/8/2"
        self.assertEqual(sa.parse_rows_config(raw, 86), [(INF, (32, 8, 2))])
        self.assertEqual(sa.parse_rows_config(raw, 120), [(INF, (64, 8, 2))])
        # order does not matter: the arch-specific group still wins
        raw2 = "inf=64/8/2;sm86:inf=32/8/2"
        self.assertEqual(sa.parse_rows_config(raw2, 86), [(INF, (32, 8, 2))])
        # a string that only names sm86 leaves sm120 on the table
        self.assertIsNone(sa.parse_rows_config("sm86:inf=32/8/2", 120))
        self.assertIsNone(sa.parse_rows_config("", 86))

    def test_bad_entries_are_refused_by_name(self):
        for raw in ("inf=24/8/2", "inf=32/3/2", "inf=32/8/0", "512=32/8/2", "inf=32/8"):
            with self.assertRaises(ValueError, msg=raw):
                sa.parse_rows_config(raw, 86)


class SelectionTest(unittest.TestCase):
    def setUp(self):
        sa._ROWS_CONFIG_CACHE.clear()
        self.addCleanup(sa._ROWS_CONFIG_CACHE.clear)

    def _pick(self, total_q, capability, raw):
        with envs.SGLANG_FORCE_QSA_ROWS_CONFIG.override(raw), \
                mock.patch.object(sa.torch.cuda, "get_device_capability",
                                  lambda *a: capability), \
                mock.patch.object(sa.torch.cuda, "get_device_name",
                                  lambda *a: "NVIDIA GeForce RTX 3080"):
            return sa._get_rows_config(total_q)

    def test_default_is_the_l20_table_for_the_rigs_cards(self):
        self.assertEqual(envs.SGLANG_FORCE_QSA_ROWS_CONFIG.get(), "")
        # a 16k P prefix chunk: the spilling one-warp build
        self.assertEqual(self._pick(16384, (8, 6), ""), (16, 1, 2))
        self.assertEqual(self._pick(16384, (12, 0), ""), (16, 1, 2))
        # the D draft decode (1 row) and a 4-row verify: (32, 8, 2)
        self.assertEqual(self._pick(1, (12, 0), ""), (32, 8, 2))
        self.assertEqual(self._pick(4, (12, 0), ""), (32, 8, 2))

    def test_override_moves_only_the_named_arch(self):
        raw = "sm86:inf=32/8/2"
        self.assertEqual(self._pick(16384, (8, 6), raw), (32, 8, 2))
        self.assertEqual(self._pick(16384, (12, 0), raw), (16, 1, 2))
        self.assertEqual(self._pick(700, (8, 6), "sm86:512=32/4/2,inf=32/8/2"), (32, 8, 2))
        self.assertEqual(self._pick(300, (8, 6), "sm86:512=32/4/2,inf=32/8/2"), (32, 4, 2))


class WiringTest(unittest.TestCase):
    def test_only_the_rows_launch_reads_the_override(self):
        with open(sa.__file__) as f:
            text = f.read()
        rows = text[text.index("def sparse_attn_rows_triton("):text.index("def fp8_e4m3_bytes_to_f32_reference(")]
        self.assertIn("block_n, warps, stages = _get_rows_config(total_q)", rows)
        self.assertNotIn("_get_best_config(", rows)
        for fn in ("def sparse_gqa_fwd_interface_triton(", "def sparse_gqa_fwd_interface_triton_ck("):
            body = text[text.index(fn):]
            body = body[: body.index("\ndef ", 1)]
            self.assertIn("_get_best_config(total_q)", body, fn)
            self.assertNotIn("_get_rows_config", body, fn)
        self.assertEqual(len(re.findall(r"_get_rows_config\(total_q\)", text)), 1)


if __name__ == "__main__":
    unittest.main()
