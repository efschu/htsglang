"""Accuracy falsifier for the CPU expert lane on REAL checkpoint weights.

Random Gaussian weights are a convenient proxy but not evidence: real trained
expert weights are not Gaussian, and quantisation error depends on the actual
distribution (outlier channels in particular). This suite therefore reads one
genuine expert out of the shipped Qwen3.5-35B-A3B-GPTQ-Int4 checkpoint,
dequantises it ONCE to fp32 as the reference, and checks the lane's int8 shard
against it.

The dequantisation here is test scaffolding for building a reference, NOT a
model of the runtime path. The lane itself never dequantises -- that is the
whole point of the int8 route (see docs/dev/DESIGN_CPULANE.md §0).

CPU-only and read-only. Skips cleanly when the checkpoint is absent.
"""

from __future__ import annotations

import glob
import json
import os
import unittest

import torch

from sglang.srt.layers.moe.cpu_expert_lane import (
    MODE_W8A8,
    MODE_W8A32,
    Int8ExpertShard,
)

CKPT = "/spinning/llm_stuff/club-3090/models-cache/Qwen3.5-35B-A3B-GPTQ-Int4"


def _have_ckpt() -> bool:
    return os.path.isdir(CKPT) and bool(glob.glob(os.path.join(CKPT, "*.safetensors")))


def _load_gptq_expert(layer: int = 5, expert: int = 0):
    """Dequantise one expert's three projections from GPTQ-Int4 to fp32.

    Standard GPTQ layout: ``qweight`` int32 packs 8 nibbles along dim 0,
    ``qzeros`` packs 8 zero-points along dim 1, ``scales`` is
    [n_groups, out_features].
    """
    from safetensors import safe_open

    index_path = os.path.join(CKPT, "model.safetensors.index.json")
    weight_map = json.load(open(index_path))["weight_map"]
    prefix = f"model.language_model.layers.{layer}.mlp.experts.{expert}."

    def get(proj: str, suffix: str) -> torch.Tensor:
        key = prefix + proj + "." + suffix
        shard = os.path.join(CKPT, weight_map[key])
        with safe_open(shard, framework="pt", device="cpu") as f:
            return f.get_tensor(key)

    def dequant(proj: str) -> torch.Tensor:
        qweight = get(proj, "qweight")          # int32 [in/8, out]
        qzeros = get(proj, "qzeros")            # int32 [n_groups, out/8]
        scales = get(proj, "scales").float()    # [n_groups, out]

        shifts = torch.arange(0, 32, 4, dtype=torch.int32)
        # [in/8, out] -> [in, out]
        w = (qweight.unsqueeze(1) >> shifts.view(1, 8, 1)) & 0xF
        w = w.reshape(-1, qweight.shape[1])
        z = (qzeros.unsqueeze(-1) >> shifts.view(1, 1, 8)) & 0xF
        z = z.reshape(qzeros.shape[0], -1)[:, : scales.shape[1]]

        n_groups = scales.shape[0]
        group_size = w.shape[0] // n_groups
        gidx = torch.arange(w.shape[0]) // group_size
        deq = (w.float() - (z.float()[gidx] + 1)) * scales[gidx]
        # GPTQ stores [in_features, out_features]; Linear wants [out, in].
        return deq.t().contiguous()

    return dequant("gate_proj"), dequant("up_proj"), dequant("down_proj")


def _silu(x):
    return x * torch.sigmoid(x)


def _reference(gate_w, up_w, down_w, x):
    g = torch.nn.functional.linear(x, gate_w)
    u = torch.nn.functional.linear(x, up_w)
    return torch.nn.functional.linear(_silu(g) * u, down_w)


def _rel_err(got, ref):
    return ((got - ref).norm() / ref.norm().clamp_min(1e-12)).item()


@unittest.skipUnless(_have_ckpt(), f"checkpoint not present at {CKPT}")
class TestRealCheckpointAccuracy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gate_w, cls.up_w, cls.down_w = _load_gptq_expert()
        cls.shard = Int8ExpertShard(cls.gate_w, cls.up_w, cls.down_w)

    def test_shapes_match_the_documented_geometry(self):
        """Guards the design's payload arithmetic against a shape drift."""
        inter, hidden = self.gate_w.shape
        self.assertEqual(hidden, 2048, "hidden changed; DESIGN_CPULANE tables assume 2048")
        self.assertEqual(inter, 512, "moe_intermediate changed; tables assume 512")
        self.assertEqual(tuple(self.down_w.shape), (hidden, inter))

    def test_reference_is_not_degenerate(self):
        """A dequant bug that produced zeros would make every error test pass."""
        for name, w in (("gate", self.gate_w), ("up", self.up_w), ("down", self.down_w)):
            self.assertTrue(torch.isfinite(w).all(), f"{name} has non-finite values")
            self.assertGreater(w.abs().mean().item(), 1e-5, f"{name} is ~zero")
            self.assertGreater(w.std().item(), 1e-5, f"{name} has no variance")

    def test_w8a32_on_real_weights(self):
        torch.manual_seed(0)
        for m in (1, 2, 4, 8):
            x = torch.randn(m, self.gate_w.shape[1])
            err = _rel_err(
                self.shard.forward(x, mode=MODE_W8A32),
                _reference(self.gate_w, self.up_w, self.down_w, x),
            )
            self.assertLess(err, 2.0e-2, f"real-weight W8A32 M={m} drifted {err:.4f}")

    def test_w8a8_on_real_weights(self):
        torch.manual_seed(0)
        for m in (1, 4, 8, 32):
            x = torch.randn(m, self.gate_w.shape[1])
            err = _rel_err(
                self.shard.forward(x, mode=MODE_W8A8),
                _reference(self.gate_w, self.up_w, self.down_w, x),
            )
            self.assertLess(err, 8.0e-2, f"real-weight W8A8 M={m} drifted {err:.4f}")

    def test_control_rejects_a_wrong_expert(self):
        """The real-weight tolerances must still reject a genuine defect."""
        torch.manual_seed(0)
        x = torch.randn(8, self.gate_w.shape[1])
        perturbed = self.down_w + torch.randn_like(self.down_w) * self.down_w.std()
        err = _rel_err(
            self.shard.forward(x, mode=MODE_W8A32),
            _reference(self.gate_w, self.up_w, perturbed, x),
        )
        self.assertGreater(err, 2.0e-2, "real-weight tolerance accepted a wrong expert")


if __name__ == "__main__":
    unittest.main(verbosity=2)
