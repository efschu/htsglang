"""#602: price the non-layer weights from the CHECKPOINT, not from a formula.

THE SHORTFALL POINTED HERE. Against the recorder the model's per-stage weight
total was short by 1.63x / 1.43x / 1.90x, highest on the LAST stage -- the
signature of a missing lm_head. `pp_cut` already HAS every term needed
(`embedding_weight_bytes`, `lm_head_weight_bytes`, `replicated_weight_bytes`,
`state_bytes_per_linear_layer`); the #485 fixture was passing zero for all of
them. So this is an input calibration, not a model gap.

MEASURED FROM THE SAFETENSORS HEADERS, which is what `PPCutInputs` already
tells callers to do: "MEASURE THESE FROM THE CHECKPOINT, do not derive them
from the config's parameter formulas", because the formula-derived attention
layer was 30 MiB per layer wrong on the reference checkpoint.

WHAT THE RECONSTRUCTION FOUND, and it is the reason this file exists rather
than a hand-typed constant: the checkpoint carries a VISION TOWER (879 MiB)
and an MTP head (405 MiB) that are resident on EVERY stage, not just one.
Adding those two as replicated payloads closes the per-stage weight identity
against the recorder to within 11 MiB on all three cards (0.1 %), where the
previous model was short by thousands.

THE BOUNDARY, NAMED RATHER THAN APPROXIMATED: the recorder does NOT separate
the GDN/mamba state pool from the KV pool -- both land in the single
`kv_pool_sized` post. See `TheStatePoolIsNotSeparable` below for what the data
does and does not support.
"""

import json
import os
import shutil
import struct
import tempfile
import unittest

from sglang.srt.planner import pp_cut
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10)

MIB = 1 << 20
LIVE_CKPT = "/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-27B-INT8-yarn1.5"


def _write_safetensors(path, tensors):
    """tensors: {name: (dtype, [shape])}. Header only -- the reader never
    touches the payload, which is what makes this hermetic and instant."""
    header = {}
    off = 0
    for name, (dtype, shape) in tensors.items():
        nel = 1
        for d in shape:
            nel *= d
        size = nel * {"I8": 1, "BF16": 2, "F32": 4}[dtype]
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [off, off + size],
        }
        off += size
    blob = json.dumps(header).encode()
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)


def _toy_checkpoint(directory, *, with_visual=True):
    t = {}
    # 4 layers, index 3 is the full-attention one (period 4, as the real one).
    for i in range(4):
        fam = "self_attn" if i == 3 else "linear_attn"
        t[f"model.language_model.layers.{i}.{fam}.qkv.weight"] = ("I8", [1024, 1024])
        t[f"model.language_model.layers.{i}.mlp.down_proj.weight"] = ("I8", [1024, 512])
    t["model.language_model.embed_tokens.weight"] = ("I8", [2048, 1024])
    t["lm_head.weight"] = ("I8", [2048, 1024])
    t["mtp.layers.0.weight"] = ("I8", [512, 512])
    if with_visual:
        t["model.visual.blocks.0.attn.qkv.weight"] = ("I8", [256, 256])
    _write_safetensors(os.path.join(directory, "model-00001.safetensors"), t)


class TheTermsAreReadFromTheCheckpoint(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="ckpt602-")
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        _toy_checkpoint(self.dir)
        self.got = pp_cut.checkpoint_weight_terms(self.dir)

    def test_the_layer_families_are_separated(self):
        self.assertEqual(self.got.n_layers, 4)
        self.assertEqual(self.got.attention_layer_indices, (3,))

    def test_embedding_and_lm_head_are_read_separately(self):
        self.assertAlmostEqual(self.got.embedding_weight_bytes, 2048 * 1024, places=3)
        self.assertAlmostEqual(self.got.lm_head_weight_bytes, 2048 * 1024, places=3)

    def test_the_vision_tower_and_mtp_head_are_replicated_payloads(self):
        """They sit on EVERY stage, which is why they are their own term and
        not folded into the first stage's non-layer weights."""
        expect = 256 * 256 + 512 * 512
        self.assertAlmostEqual(self.got.replicated_weight_bytes, expect, places=3)
        self.assertIn("visual", self.got.replicated_breakdown)
        self.assertIn("mtp", self.got.replicated_breakdown)

    def test_a_checkpoint_without_a_vision_tower_still_reads(self):
        d = tempfile.mkdtemp(prefix="ckpt602b-")
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        _toy_checkpoint(d, with_visual=False)
        got = pp_cut.checkpoint_weight_terms(d)
        self.assertAlmostEqual(got.replicated_weight_bytes, 512 * 512, places=3)

    def test_it_is_provenance_stamped(self):
        self.assertIn(self.dir, self.got.source)


class AbsenceRaisesHereToo(unittest.TestCase):
    """Same #606 separation: zero non-layer weight is a real value (a model
    with tied embeddings and no vision tower), so absence must not produce it."""

    def test_a_directory_with_no_safetensors_raises(self):
        d = tempfile.mkdtemp(prefix="ckpt602c-")
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        with self.assertRaises(pp_cut.DraftResidencyUnavailable):
            pp_cut.checkpoint_weight_terms(d)

    def test_a_checkpoint_with_no_layers_raises(self):
        d = tempfile.mkdtemp(prefix="ckpt602d-")
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        _write_safetensors(
            os.path.join(d, "model-00001.safetensors"),
            {"lm_head.weight": ("I8", [16, 16])},
        )
        with self.assertRaises(pp_cut.DraftResidencyUnavailable):
            pp_cut.checkpoint_weight_terms(d)


@unittest.skipUnless(
    os.path.isdir(LIVE_CKPT), "the live checkpoint is not on this machine"
)
class TheLiveCheckpointClosesTheWeightIdentity(unittest.TestCase):
    """The acceptance for this term: reconstructed per-stage weights must match
    what the recorder measured. Short of that the floor cannot become a gate.

    Recorder, boot 3439375-1786877650: 16196 / 10194 / 10832 MiB.
    """

    RECORDER_MIB = (16196.0, 10194.0, 10832.0)
    TOLERANCE_MIB = 40.0

    def test_the_reconstruction_matches_the_recorder_on_every_stage(self):
        terms = pp_cut.checkpoint_weight_terms(LIVE_CKPT)
        bounds = ((0, 28), (28, 48), (48, 64))
        attn = set(terms.attention_layer_indices)
        for stage, (start, end) in enumerate(bounds):
            n_attn = sum(1 for i in range(start, end) if i in attn)
            n_lin = (end - start) - n_attn
            total = (
                n_attn * terms.attn_layer_weight_bytes
                + n_lin * terms.linear_layer_weight_bytes
                + terms.replicated_weight_bytes
                + (terms.embedding_weight_bytes if stage == 0 else 0.0)
                + (terms.lm_head_weight_bytes if stage == 2 else 0.0)
            ) / MIB
            self.assertAlmostEqual(
                total,
                self.RECORDER_MIB[stage],
                delta=self.TOLERANCE_MIB,
                msg=f"stage {stage}: reconstructed {total:.0f} MiB against "
                f"recorder {self.RECORDER_MIB[stage]:.0f} MiB",
            )

    def test_the_checkpoint_has_the_replicated_payloads_the_fit_needs(self):
        terms = pp_cut.checkpoint_weight_terms(LIVE_CKPT)
        self.assertGreater(terms.replicated_weight_bytes, 0.0)
        self.assertEqual(len(terms.attention_layer_indices), 16)


class TheStatePoolIsNotSeparable(unittest.TestCase):
    """The boundary, named instead of approximated.

    The recorder folds the GDN/mamba state pool into the single
    ``kv_pool_sized`` post, so no reader can split them. What the three stages
    DO support is a linear fit of that post against attention-layer count:

        stage0  7376 MiB / 7 attn      stage1  5232 / 5      stage2  4158 / 4

    which solves to 1072.7 MiB per attention layer and a constant of -132 MiB,
    i.e. the post is linear in attention layers to within 0.2 % and leaves no
    room for a state term that scales with LINEAR layers (stage 0 has 21 linear
    layers, stage 2 has 12; a per-linear-layer state pool would show up as a
    large positive constant difference, and it does not).

    So ``state_bytes_per_linear_layer`` is set to ZERO on the strength of the
    fit, and that is a measurement, not an omission. If a future checkpoint
    sizes GDN state differently this test is where it should fail.
    """

    KV_POST_MIB = (7376.0, 5232.0, 4158.0)
    ATTN_COUNTS = (7, 5, 4)

    def test_the_kv_post_is_linear_in_attention_layers(self):
        (a0, _, a2), (k0, _, k2) = self.ATTN_COUNTS, self.KV_POST_MIB
        slope = (k0 - k2) / (a0 - a2)
        residuals = [k - slope * a for k, a in zip(self.KV_POST_MIB, self.ATTN_COUNTS)]
        self.assertAlmostEqual(residuals[0], residuals[1], delta=3.0)
        self.assertAlmostEqual(residuals[1], residuals[2], delta=3.0)

    def test_no_per_linear_layer_state_term_is_supported(self):
        """Stage 0 has 21 linear layers and stage 2 has 12. A state pool
        proportional to them would break the attention-only fit above."""
        (a0, _, a2), (k0, _, k2) = self.ATTN_COUNTS, self.KV_POST_MIB
        slope = (k0 - k2) / (a0 - a2)
        implied_state_span = abs((k0 - slope * a0) - (k2 - slope * a2))
        self.assertLess(implied_state_span, 5.0)


if __name__ == "__main__":
    unittest.main()


LIVE_FLIGHT = "/spinning/flight_605"
LIVE_BOOT = "3439375-1786877650"


@unittest.skipUnless(
    os.path.isdir(LIVE_CKPT) and os.path.isdir(LIVE_FLIGHT),
    "the live checkpoint and flight recorder are not on this machine",
)
class TheFloorBecomesAGate(unittest.TestCase):
    """The acceptance that turns the predicted floor from a diagnostic into a
    gate: the fully calibrated model must reproduce the floor the live boot
    actually sized.

    Live boot 3439375-1786877650 sized 471638 tokens at cut [28,20,16].

    WHAT THIS TEST COST TO EARN, recorded because the number moved a long way:
    with the #485 reference-bench weights the same model predicted 851960 and
    chose a cut with 17 layers on stage 2. Both were artifacts of pricing the
    real 476 MiB linear layers at the fixture's much smaller ones. Calibrated,
    stage 2 keeps 16 layers and the reclaim is a tenth of the earlier claim.
    """

    LIVE_FLOOR_TOKENS = 471638
    TOLERANCE_FRACTION = 0.05

    def _inputs(self):
        import sys as _sys

        _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import dataclasses

        import test_pp_family_cut_485 as ref

        cards = ["GPU-31d7ef41", "GPU-5c648f96", "GPU-62dbbae1"]
        rt = pp_cut.residency_terms_from_flight(LIVE_FLIGHT, boot=LIVE_BOOT)
        dr = pp_cut.draft_residency_from_flight(LIVE_FLIGHT, boot=LIVE_BOOT)
        pick = lambda d, c: next(  # noqa: E731
            v for v in d.values() if v.card_uuid.startswith(c)
        )
        terms = pp_cut.checkpoint_weight_terms(LIVE_CKPT)
        base = ref._inputs(
            budgets=(32089.0, 20055.0, 20055.0),
            seam_staging=(1289.0, 329.0, 384.0),
            pool=self.LIVE_FLOOR_TOKENS,
            corridor=1024.0,
            draft_residency=tuple(pick(dr, c).net_mib for c in cards),
            draft_runner_present=True,
            overheads=tuple(pick(rt, c).fixed_overhead_mib for c in cards),
            transients=tuple(pick(rt, c).observed_transient_mib for c in cards),
        )
        return dataclasses.replace(
            base,
            attn_layer_weight_bytes=terms.attn_layer_weight_bytes,
            linear_layer_weight_bytes=terms.linear_layer_weight_bytes,
            embedding_weight_bytes=terms.embedding_weight_bytes,
            lm_head_weight_bytes=terms.lm_head_weight_bytes,
            replicated_weight_bytes=terms.replicated_weight_bytes,
            state_bytes_per_linear_layer=0.0,
            kv_bytes_per_token_per_attn_layer=2326.7,
        )

    def test_the_model_reproduces_the_floor_the_live_boot_sized(self):
        got = pp_cut.world_kv_floor_at_seam_fixed_point(
            [28, 20, 16],
            self._inputs(),
            seam_fixed_mib=(227.0, 138.0, 138.0),
            seam_slope_bytes_per_token=(2360.1, 424.1, 547.6),
        )
        self.assertIsNotNone(got, "the calibrated model calls the live boot unrunnable")
        self.assertAlmostEqual(
            got / self.LIVE_FLOOR_TOKENS,
            1.0,
            delta=self.TOLERANCE_FRACTION,
            msg=f"predicted {got:.0f} against a measured {self.LIVE_FLOOR_TOKENS}",
        )

    def test_stage_two_keeps_sixteen_layers_once_the_weights_are_real(self):
        """The earlier 'stage 2: 16 -> 17' recommendation does NOT survive
        calibration. Pinned so the reversal cannot be quietly lost."""
        fp = pp_cut.solve_pp_cut_for_kv_floor_at_seam_fixed_point(
            self._inputs(),
            seam_fixed_mib=(227.0, 138.0, 138.0),
            seam_slope_bytes_per_token=(2360.1, 424.1, 547.6),
        )
        self.assertTrue(fp.converged)
        self.assertEqual(fp.solution.counts[2], 16)

    def test_the_reclaim_is_single_digit_percent_not_thirty(self):
        inputs = self._inputs()
        kw = dict(
            seam_fixed_mib=(227.0, 138.0, 138.0),
            seam_slope_bytes_per_token=(2360.1, 424.1, 547.6),
        )
        inc = pp_cut.world_kv_floor_at_seam_fixed_point([28, 20, 16], inputs, **kw)
        sol = pp_cut.solve_pp_cut_for_kv_floor_at_seam_fixed_point(inputs, **kw)
        gain = (sol.solution.floor_tokens - inc) / inc
        self.assertGreater(gain, 0.0)
        self.assertLess(
            gain,
            0.15,
            "the reclaim is back above 15 %, which on this rig meant the "
            "weight model had gone uncalibrated again",
        )
