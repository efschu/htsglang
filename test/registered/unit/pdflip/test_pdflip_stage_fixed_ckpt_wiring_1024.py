# SPDX-License-Identifier: Apache-2.0
"""PPCUT-INT4AR 24.10.: the launcher wiring of the stage-fixed checkpoint move.

The arithmetic is tested hermetically in
``test/registered/unit/planner/test_pp_cut_stage_fixed_ckpt_1024.py``.  This
file tests the LAUNCHER half by BEHAVIOUR (review #222 MUSS-1: a source anchor
let a mutant that swapped the embedding and lm_head header bytes survive all
461 tests -- invisible on int4ar, whose two payloads are both 2425 MiB, and
worth -51712 / -88650 priced tokens on NVFP4 duo / dual):

* ``p_stage_fixed_for_checkpoint`` with NVFP4-SHAPED terms (embedding BF16
  2425.0, lm_head 606.2 MiB, the header bytes of Qwen3.8-27B-NVFP4-RadixArk):
  the first stage gains exactly the embedding delta, the last stays put;
* an operator-given ``--pp-cut-stage-fixed-mib`` is not moved, and the flag is
  recorded as given by ``apply_profile_arg_defaults``;
* a profile without a registered reference (nextflash) is not moved;
* ``FLLIPER_GGUF_DENSE_VOCAB=1`` on a GGUF checkpoint prices the dequantized
  BF16 vocab, not the header's quant bytes (SOLL-2);
* TRIPWIRE (SOLL-1): the reference table is bound to the record it describes
  -- a refit of ``P_PP_STAGE_FIXED_MIB`` turns this file red until the table is
  re-pointed at the refit's checkpoint;
* ``solve_p_cut`` routes through the helper before the pool model is built.

Hermetic: no GPU, no NVML, no checkpoint, no launcher run.
"""

import inspect
import json
import os
import pathlib
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import launcher as L
from flliper.test.test_utils import CustomTestCase

MIB = 1024.0 * 1024.0
RECORD_3 = (1418.4, 148.8, 4960.6)
RECORD_2 = (1418.4, 4960.6)
EMB_DELTA = 2425.0 - 1212.97


def _terms(emb_mib, head_mib):
    return SimpleNamespace(
        embedding_weight_bytes=emb_mib * MIB, lm_head_weight_bytes=head_mib * MIB
    )


NVFP4 = _terms(2425.0, 606.2)  # Qwen3.8-27B-NVFP4-RadixArk headers
INT4AR = _terms(2425.0, 2425.0)
VOCABEMBED = _terms(1212.97, 2425.0)


def _ns(**kw):
    base = dict(profile=L.PROFILE_QWEN27B, env_p="")
    base.update(kw)
    return SimpleNamespace(**base)


class TheHelperByBehaviour(CustomTestCase):
    def test_nvfp4_two_stage_moves_the_embedding_delta_onto_stage0(self):
        out, line = L.p_stage_fixed_for_checkpoint(_ns(), RECORD_2, NVFP4)
        self.assertAlmostEqual(out[0], 1418.4 + EMB_DELTA, places=3)
        self.assertEqual(out[1], 4960.6)
        self.assertIn("NOT credited", line)
        self.assertIn("lm_head -1818.8 MiB", line)

    def test_nvfp4_three_stage_touches_only_the_first(self):
        out, _ = L.p_stage_fixed_for_checkpoint(_ns(), RECORD_3, NVFP4)
        self.assertAlmostEqual(out[0], 1418.4 + EMB_DELTA, places=3)
        self.assertEqual(out[1:], (148.8, 4960.6))

    def test_int4ar(self):
        out, _ = L.p_stage_fixed_for_checkpoint(_ns(), RECORD_2, INT4AR)
        self.assertAlmostEqual(out[0], 1418.4 + EMB_DELTA, places=3)
        self.assertEqual(out[1], 4960.6)

    def test_reference_checkpoint_is_unchanged(self):
        out, _ = L.p_stage_fixed_for_checkpoint(_ns(), RECORD_3, VOCABEMBED)
        for got, want in zip(out, RECORD_3):
            self.assertAlmostEqual(got, want, places=6)

    def test_given_flag_is_not_moved(self):
        out, line = L.p_stage_fixed_for_checkpoint(
            _ns(pp_cut_stage_fixed_given=True), RECORD_2, NVFP4)
        self.assertEqual(out, RECORD_2)
        self.assertIn("given on the command line", line)

    def test_not_given_is_moved(self):
        out, _ = L.p_stage_fixed_for_checkpoint(
            _ns(pp_cut_stage_fixed_given=False), RECORD_2, NVFP4)
        self.assertGreater(out[0], RECORD_2[0] + 1200.0)

    def test_missing_given_attribute_moves(self):
        # apply_profile_arg_defaults did not run: the safe direction is to move.
        out, _ = L.p_stage_fixed_for_checkpoint(_ns(), RECORD_2, NVFP4)
        self.assertGreater(out[0], RECORD_2[0] + 1200.0)

    def test_nextflash_has_no_reference_and_is_not_moved(self):
        self.assertNotIn(L.PROFILE_NEXTFLASH, L.P_PP_STAGE_FIXED_REF_NONLAYER_MIB)
        out, line = L.p_stage_fixed_for_checkpoint(
            _ns(profile=L.PROFILE_NEXTFLASH), RECORD_3, NVFP4)
        self.assertEqual(out, RECORD_3)
        self.assertIn("registers no reference", line)


class TheGgufDenseVocab(CustomTestCase):
    CFG = {"vocab_size": 248320, "hidden_size": 5120}

    def test_dense_vocab_prices_the_bf16_dequantization(self):
        q8 = _terms(1288.0, 1288.0)  # a Q8_0-sized vocab in the headers
        with mock.patch.object(L, "_p_gguf_dense_vocab", return_value=True):
            out, line = L.p_stage_fixed_for_checkpoint(
                _ns(), RECORD_2, q8, text_cfg=self.CFG, model="/x.gguf")
        self.assertAlmostEqual(out[0], 1418.4 + EMB_DELTA, places=3)
        self.assertIn("dequantized to BF16", line)

    def test_quantized_resident_vocab_keeps_header_bytes(self):
        q8 = _terms(1288.0, 1288.0)
        with mock.patch.object(L, "_p_gguf_dense_vocab", return_value=False):
            out, _ = L.p_stage_fixed_for_checkpoint(
                _ns(), RECORD_2, q8, text_cfg=self.CFG, model="/x.gguf")
        self.assertAlmostEqual(out[0], 1418.4 + (1288.0 - 1212.97), places=3)

    def test_detector_reads_env_p_on_a_gguf_file_only(self):
        with tempfile.TemporaryDirectory() as d:
            g = os.path.join(d, "m.gguf")
            with open(g, "wb") as fh:
                fh.write(b"GGUF\x00\x00\x00\x00")
            with mock.patch.dict(os.environ, {L.GGUF_DENSE_VOCAB_ENV: "0"}):
                self.assertTrue(L._p_gguf_dense_vocab(
                    _ns(env_p="FLLIPER_GGUF_DENSE_VOCAB=1"), g))
                self.assertFalse(L._p_gguf_dense_vocab(_ns(env_p=""), g))
                self.assertFalse(L._p_gguf_dense_vocab(
                    _ns(env_p="FLLIPER_GGUF_DENSE_VOCAB=1"), d))  # a directory

    def test_env_name_is_the_loaders_own(self):
        src = (pathlib.Path(L.__file__).resolve().parents[1]
               / "model_loader" / "gguf_qwen35.py").read_text()
        self.assertIn('os.environ.get("%s", "0") == "1"' % L.GGUF_DENSE_VOCAB_ENV, src)


class TheReferenceTripwire(CustomTestCase):
    """SOLL-1: the table describes ONE fit of the record; a refit must re-point it."""

    def test_reference_table_is_bound_to_the_record_it_describes(self):
        path = (pathlib.Path(L.__file__).resolve().parent
                / "profile_records_data" / "qwen27b.json")
        data = json.loads(path.read_text())
        rows = data if isinstance(data, list) else None
        if rows is None:
            rows = [r for v in data.values() if isinstance(v, list) for r in v
                    if isinstance(r, dict)]
        rec = [r for r in rows if r.get("name") == "P_PP_STAGE_FIXED_MIB"]
        self.assertEqual(len(rec), 1, "record P_PP_STAGE_FIXED_MIB not found once")
        rec = rec[0]
        msg = ("P_PP_STAGE_FIXED_MIB was refit -- re-point "
               "launcher.P_PP_STAGE_FIXED_REF_NONLAYER_MIB['qwen27b'] at the refit's "
               "checkpoint (its embedding/lm_head header MiB), then update this tripwire. "
               "A stale table under-books stage 0 whenever the new reference has the "
               "smaller embedding (OOM direction).")
        self.assertEqual(rec.get("value"), "1418.4,148.8,4960.6", msg)
        self.assertEqual(sorted(rec.get("boots") or []), sorted([
            "dkr27browauthoritybar1fs09291750",
            "dkr27browauthoritycds6bar1fs09291831",
            "dkr27browauthorityz30ybar1fs09291331",
        ]), msg)
        label, emb, head = L.P_PP_STAGE_FIXED_REF_NONLAYER_MIB["qwen27b"]
        self.assertEqual(label, "Qwen3.8-27B-INT8-gdncov-vocabembed", msg)
        self.assertAlmostEqual(emb, 1212.97, places=2)
        self.assertAlmostEqual(head, 2425.0, places=2)


class TheGivenFlag(CustomTestCase):
    def _ns(self, words):
        argv = ["--tree", "/", "--tag", "x"] + list(words)
        ns = L.build_parser().parse_args(argv)
        ns.profile = L.PROFILE_QWEN27B
        L.apply_profile_arg_defaults(ns, argv)
        return ns

    def test_unset_flag_is_not_given(self):
        self.assertFalse(self._ns([]).pp_cut_stage_fixed_given)

    def test_stated_flag_is_given(self):
        ns = self._ns(["--pp-cut-stage-fixed-mib", "2700,5000"])
        self.assertTrue(ns.pp_cut_stage_fixed_given)
        self.assertEqual(ns.pp_cut_stage_fixed_mib, "2700,5000")


class TheOrderInSolvePCut(CustomTestCase):
    def test_vector_is_moved_before_the_pool_model_is_built(self):
        src = inspect.getsource(L.solve_p_cut)
        move = src.index("p_stage_fixed_for_checkpoint(")
        build = src.index("stage_fixed_mib=_stage_fixed")
        self.assertLess(move, build)
        self.assertIn("_stage_fixed, _sf_line = p_stage_fixed_for_checkpoint(", src)


if __name__ == "__main__":
    unittest.main()
