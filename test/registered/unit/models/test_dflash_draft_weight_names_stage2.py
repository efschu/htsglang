"""DFLASH2 draft weight loading after the stage-2 shard (task #37).

Stage 2 changed the SHAPE of three parameter groups and the NAME of one:

  * ``candidate_selector.{predecessor,successor}_codebook`` were bare
    nn.Parameters and are VocabParallelEmbedding modules now, so the model's
    parameter is ``....codebook.weight`` while the checkpoint tensor keeps the
    old, module-less name.
  * ``fc.weight`` and ``layers.*.{attention,mlp}_conv.kernel_projection.weight``
    are row-parallel now, so under TP>1 this rank's parameter is NARROWER than
    the checkpoint tensor.

Both are the classic way a draft loads silently wrong: a name that resolves to
nothing is skipped by design (HF rotary caches), and a parameter that nothing
ever wrote runs on ``torch.empty`` and only shows up as a bad accept rate
(#290). ``load_weights`` already raises on unloaded parameters -- these tests
drive the REAL loader with synthetic checkpoint tensors of the real shapes and
pin that the count comes out at zero, and that the bytes land where they
belong.

CPU only, no distributed group: the model is built under an injected parallel
context and the TP-group handle is stubbed (no collective is ever issued --
loading is rank-local).
"""

import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang.srt.layers.linear as linear_mod
import sglang.srt.layers.vocab_parallel_embedding as vpe_mod
from sglang.srt.distributed.utils import set_tp_partition_ratios
from sglang.srt.models.dflash import DFlash2DraftModel
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

FP = torch.float32
# A 3-rank plan whose kv-head and mlp splits this tiny config survives.
PLAN = [2, 1, 1]
TP = len(PLAN)

HIDDEN = 64
LAYERS = 2
VOCAB = 1152
HEAD_DIM = 8
Q_HEADS = 8
KV_HEADS = 4
INTERMEDIATE = 128
SELECTOR_RANK = 8
GROUP_SIZE = 8
TAPS = 2
TARGET_LAYER_IDS = [0, 1, 2]
FEATURES = len(TARGET_LAYER_IDS)
NUM_GROUPS = HIDDEN // GROUP_SIZE


def _config():
    return SimpleNamespace(
        hidden_size=HIDDEN,
        num_hidden_layers=LAYERS,
        vocab_size=VOCAB,
        num_attention_heads=Q_HEADS,
        num_key_value_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        intermediate_size=INTERMEDIATE,
        hidden_act="silu",
        attention_bias=False,
        rms_norm_eps=1e-6,
        max_position_embeddings=128,
        rope_theta=10000.0,
        rope_scaling=None,
        layer_types=["full_attention"] * LAYERS,
        dflash_config={
            "selector_rank": SELECTOR_RANK,
            "selector_top_k": 4,
            "block_size": 4,
            "conv_group_size": GROUP_SIZE,
            "conv_kernel_size": TAPS,
            "target_layer_ids": TARGET_LAYER_IDS,
        },
    )


def _checkpoint():
    """Synthetic DFLASH2 checkpoint: the tensor NAMES and SHAPES the exporter
    writes, values distinct per tensor so a mix-up is visible."""
    weights = {
        "fc.weight": torch.randn(HIDDEN, FEATURES * HIDDEN, dtype=FP),
        "hidden_norm.weight": torch.randn(HIDDEN, dtype=FP),
        "norm.weight": torch.randn(HIDDEN, dtype=FP),
        "candidate_selector.predecessor_codebook": torch.randn(
            VOCAB, SELECTOR_RANK, dtype=FP
        ),
        "candidate_selector.successor_codebook": torch.randn(
            VOCAB, SELECTOR_RANK, dtype=FP
        ),
        "candidate_selector.hidden_projection.weight": torch.randn(
            SELECTOR_RANK, HIDDEN, dtype=FP
        ),
    }
    for layer in range(LAYERS):
        p = f"layers.{layer}"
        weights[f"{p}.input_layernorm.weight"] = torch.randn(HIDDEN, dtype=FP)
        weights[f"{p}.post_attention_layernorm.weight"] = torch.randn(
            HIDDEN, dtype=FP
        )
        weights[f"{p}.self_attn.q_proj.weight"] = torch.randn(
            Q_HEADS * HEAD_DIM, HIDDEN, dtype=FP
        )
        weights[f"{p}.self_attn.k_proj.weight"] = torch.randn(
            KV_HEADS * HEAD_DIM, HIDDEN, dtype=FP
        )
        weights[f"{p}.self_attn.v_proj.weight"] = torch.randn(
            KV_HEADS * HEAD_DIM, HIDDEN, dtype=FP
        )
        weights[f"{p}.self_attn.o_proj.weight"] = torch.randn(
            HIDDEN, Q_HEADS * HEAD_DIM, dtype=FP
        )
        weights[f"{p}.self_attn.q_norm.weight"] = torch.randn(HEAD_DIM, dtype=FP)
        weights[f"{p}.self_attn.k_norm.weight"] = torch.randn(HEAD_DIM, dtype=FP)
        weights[f"{p}.mlp.gate_proj.weight"] = torch.randn(
            INTERMEDIATE, HIDDEN, dtype=FP
        )
        weights[f"{p}.mlp.up_proj.weight"] = torch.randn(
            INTERMEDIATE, HIDDEN, dtype=FP
        )
        weights[f"{p}.mlp.down_proj.weight"] = torch.randn(
            HIDDEN, INTERMEDIATE, dtype=FP
        )
        for conv in ("attention_conv", "mlp_conv"):
            weights[f"{p}.{conv}.base_kernel"] = torch.randn(
                2, TAPS, HIDDEN, dtype=FP
            )
            weights[f"{p}.{conv}.kernel_projection.weight"] = torch.randn(
                2 * TAPS * NUM_GROUPS, HIDDEN, dtype=FP
            )
    return weights


class _LoaderCase(CustomTestCase):
    def setUp(self):
        set_tp_partition_ratios(None)
        self._server_args = get_context().override_server_args()
        self._server_args.install()

    def tearDown(self):
        self._server_args.restore()
        set_tp_partition_ratios(None)

    @contextlib.contextmanager
    def _build(self, tp_rank, tp_size):
        with contextlib.ExitStack() as stack:
            for mod in (vpe_mod, linear_mod):
                stack.enter_context(patch.object(mod, "get_tp_group", lambda: None))
            stack.enter_context(
                get_parallel().override(
                    tp_size=tp_size,
                    tp_rank=tp_rank,
                    attn_tp_size=tp_size,
                    attn_tp_rank=tp_rank,
                )
            )
            yield DFlash2DraftModel(_config())


class TestStage2WeightNames(_LoaderCase):
    def test_tp1_loads_every_parameter_and_places_the_bytes(self):
        weights = _checkpoint()
        with self._build(0, 1) as model:
            # Raises if ANY parameter is left unwritten.
            model.load_weights(list(weights.items()))
            params = dict(model.named_parameters())

        # The renamed pair: checkpoint `...codebook` -> param `...codebook.weight`.
        for side in ("predecessor", "successor"):
            self.assertTrue(
                torch.equal(
                    params[f"candidate_selector.{side}_codebook.weight"],
                    weights[f"candidate_selector.{side}_codebook"],
                )
            )
        self.assertTrue(torch.equal(params["fc.weight"], weights["fc.weight"]))
        for layer in range(LAYERS):
            for conv in ("attention_conv", "mlp_conv"):
                name = f"layers.{layer}.{conv}.kernel_projection.weight"
                self.assertTrue(torch.equal(params[name], weights[name]))

    def test_tp3_shards_the_three_stage2_groups_and_loads_them_all(self):
        set_tp_partition_ratios(PLAN)
        weights = _checkpoint()
        per_rank = {}
        for rank in range(TP):
            with self._build(rank, TP) as model:
                model.load_weights(list(weights.items()))
                per_rank[rank] = {
                    name: param.detach().clone()
                    for name, param in model.named_parameters()
                }

        # fc and the conv projections: the contraction axis is cut, and the
        # rank shards concatenate back to the checkpoint tensor.
        row_parallel = ["fc.weight"] + [
            f"layers.{layer}.{conv}.kernel_projection.weight"
            for layer in range(LAYERS)
            for conv in ("attention_conv", "mlp_conv")
        ]
        for name in row_parallel:
            shards = [per_rank[r][name] for r in range(TP)]
            full = weights[name]
            self.assertLess(int(shards[0].shape[1]), int(full.shape[1]))
            self.assertTrue(torch.equal(torch.cat(shards, dim=1), full))

        # Codebooks: the vocab axis is cut into contiguous EVEN bands.
        for side in ("predecessor", "successor"):
            name = f"candidate_selector.{side}_codebook.weight"
            shards = [per_rank[r][name] for r in range(TP)]
            full = weights[f"candidate_selector.{side}_codebook"]
            self.assertEqual(int(shards[0].shape[0]), VOCAB // TP)
            self.assertTrue(torch.equal(torch.cat(shards, dim=0), full))

        # base_kernel and hidden_projection stay replicated.
        for name in (
            "layers.0.attention_conv.base_kernel",
            "candidate_selector.hidden_projection.weight",
        ):
            for rank in range(1, TP):
                self.assertTrue(
                    torch.equal(per_rank[0][name], per_rank[rank][name])
                )

    def test_a_missing_codebook_is_reported_not_silently_left_empty(self):
        """The #290 guard has to keep covering the codebooks now that they are
        reached through a renamed path -- a rename that resolves to nothing
        would otherwise leave them on torch.empty."""
        weights = _checkpoint()
        del weights["candidate_selector.successor_codebook"]
        with self._build(0, 1) as model:
            with self.assertRaises(ValueError) as ctx:
                model.load_weights(list(weights.items()))
        self.assertIn("successor_codebook.weight", str(ctx.exception))

    def test_only_the_codebook_names_get_the_weight_suffix(self):
        """The `.weight` fallback is scoped: an unrelated checkpoint tensor
        named after a module must not be absorbed by that module's weight."""
        weights = _checkpoint()
        weights["hidden_norm"] = torch.full((HIDDEN,), 7.0, dtype=FP)
        with self._build(0, 1) as model:
            model.load_weights(list(weights.items()))
            loaded = dict(model.named_parameters())["hidden_norm.weight"]
        self.assertTrue(torch.equal(loaded, weights["hidden_norm.weight"]))


class TestFcShapeCheckUsesTheLogicalSize(_LoaderCase):
    def test_project_target_hidden_accepts_the_full_feature_width_under_tp(self):
        """The caller hands in the REPLICATED [N, K*hidden] context features on
        every rank; the layer cuts them itself. A shape check against
        ``param.shape`` would reject that on every rank but one."""
        set_tp_partition_ratios(PLAN)
        weights = _checkpoint()
        for rank in range(TP):
            with self._build(rank, TP) as model:
                model.load_weights(list(weights.items()))
                self.assertEqual(int(model.fc.input_size), FEATURES * HIDDEN)
                self.assertLess(
                    int(model.fc.weight.shape[1]), FEATURES * HIDDEN
                )
                with patch.object(
                    linear_mod, "tensor_model_parallel_all_reduce", lambda x: x
                ), patch.object(linear_mod, "get_tp_group", lambda: None):
                    out = model.project_target_hidden(
                        torch.randn(3, FEATURES * HIDDEN, dtype=FP)
                    )
                self.assertEqual(tuple(out.shape), (3, HIDDEN))

    def test_a_wrong_feature_width_names_the_logical_size(self):
        set_tp_partition_ratios(PLAN)
        with self._build(1, TP) as model:
            with self.assertRaises(ValueError) as ctx:
                model.project_target_hidden(
                    torch.randn(3, FEATURES * HIDDEN - 8, dtype=FP)
                )
        self.assertIn(f"[N, {FEATURES * HIDDEN}]", str(ctx.exception))

    def test_load_weights_rejects_a_wrong_K_against_the_logical_shape(self):
        """Under TP>1 the parameter is narrower than the checkpoint tensor, so
        the guard has to compare against (output_size, input_size) -- against
        ``param.shape`` it would fire on every correct checkpoint instead."""
        set_tp_partition_ratios(PLAN)
        weights = _checkpoint()
        weights["fc.weight"] = torch.randn(
            HIDDEN, (FEATURES + 1) * HIDDEN, dtype=FP
        )
        with self._build(2, TP) as model:
            with self.assertRaises(ValueError) as ctx:
                model.load_weights(list(weights.items()))
        message = str(ctx.exception)
        self.assertIn("fc.weight shape mismatch", message)
        self.assertIn(f"({HIDDEN}, {FEATURES * HIDDEN})", message)


if __name__ == "__main__":
    unittest.main()
