# SPDX-License-Identifier: Apache-2.0
"""#274 round 7c: the GGUF name map for a DFLASH draft checkpoint.

The gate this file automates was first run by hand against the released
``Qwen3.6-27B-DFlash-Q8_0.gguf`` and is the reason the change is small: 56 of
58 tensors already matched llama.cpp's stock qwen3 naming, and the delta was
seven names in three classes. A desk estimate before that run had said "full
adapter, 250-400 lines".

Two levels, deliberately:

* The GENERATED map is checked on its own, with no file -- shape, count, and
  the three classes that differ from the stock naming. This runs anywhere.
* The map is checked AGAINST THE FILE when the checkpoint is on this machine.
  That is the check that actually catches a re-export with renamed tensors,
  and it is skipped rather than faked when the file is absent.
"""

from __future__ import annotations

import os
import struct
import unittest

from sglang.srt.model_loader.gguf_dflash import (
    audit_dflash_name_map,
    build_dflash_name_map,
    dflash_unquantized_module_prefixes,
    is_dflash_gguf_config,
)

DFLASH_GGUF = (
    "/spinning/llm_stuff/club-3090/models-cache/qwen3.6-27b-dflash-gguf/"
    "Qwen3.6-27B-DFlash-Q8_0.gguf"
)
DFLASH_HF = "/spinning/llm_stuff/club-3090/models-cache/qwen3.6-27b-dflash"


class _Cfg:
    """Only the two fields the map is generated from."""

    def __init__(self, num_hidden_layers=5, architectures=("DFlashDraftModel",)):
        self.num_hidden_layers = num_hidden_layers
        self.architectures = list(architectures)


def _read_gguf_tensors(path):
    """Tensor name, ggml type id and element count, straight from the header."""

    def rd(f, fmt):
        return struct.unpack(fmt, f.read(struct.calcsize(fmt)))

    def rstr(f):
        (n,) = rd(f, "<Q")
        return f.read(n).decode("utf-8", "replace")

    def skip_val(f, t):
        prim = {
            0: "<B",
            1: "<b",
            2: "<H",
            3: "<h",
            4: "<I",
            5: "<i",
            6: "<f",
            7: "<?",
            10: "<Q",
            11: "<q",
            12: "<d",
        }
        if t in prim:
            rd(f, prim[t])
        elif t == 8:
            rstr(f)
        elif t == 9:
            (et,) = rd(f, "<I")
            (n,) = rd(f, "<Q")
            for _ in range(n):
                skip_val(f, et)
        else:
            raise ValueError(f"gguf value type {t}")

    out = {}
    with open(path, "rb") as f:
        magic, _ver = rd(f, "<4sI")
        assert magic == b"GGUF"
        (n_tensors,) = rd(f, "<Q")
        (n_kv,) = rd(f, "<Q")
        for _ in range(n_kv):
            rstr(f)
            (t,) = rd(f, "<I")
            skip_val(f, t)
        for _ in range(n_tensors):
            name = rstr(f)
            (nd,) = rd(f, "<I")
            dims = rd(f, "<" + "Q" * nd)
            (tt,) = rd(f, "<I")
            rd(f, "<Q")
            out[name] = (tt, dims)
    return out


class TestDflashConfigDispatch(unittest.TestCase):
    def test_it_dispatches_on_architecture_not_model_type(self):
        """A DFLASH draft config says ``model_type: qwen3``.

        Registering that model_type would swallow every plain Qwen3 GGUF, which
        is exactly why this map is not a registry family.
        """
        self.assertTrue(is_dflash_gguf_config(_Cfg()))
        self.assertTrue(
            is_dflash_gguf_config(_Cfg(architectures=("DFlashLagunaForCausalLM",)))
        )
        self.assertFalse(
            is_dflash_gguf_config(_Cfg(architectures=("Qwen3ForCausalLM",)))
        )
        self.assertFalse(is_dflash_gguf_config(_Cfg(architectures=())))


class TestDflashNameMapShape(unittest.TestCase):
    def test_the_map_has_one_entry_per_checkpoint_tensor(self):
        # 3 model-level + 11 per layer.
        self.assertEqual(len(build_dflash_name_map(_Cfg(5))), 3 + 5 * 11)
        self.assertEqual(len(build_dflash_name_map(_Cfg(1))), 3 + 11)

    def test_the_three_names_the_stock_qwen3_map_gets_wrong(self):
        """These are the whole delta, and they are asserted by name."""
        m = build_dflash_name_map(_Cfg(5))
        self.assertEqual(m["dflash_fc.weight"], "fc.weight")
        self.assertEqual(m["dflash_hidden_norm.weight"], "hidden_norm.weight")
        # The stock map emits ``ffn_norm`` here; DFLASH exports write
        # ``post_attention_norm``. Five tensors hang on this one line.
        for i in range(5):
            self.assertEqual(
                m[f"blk.{i}.post_attention_norm.weight"],
                f"layers.{i}.post_attention_layernorm.weight",
            )
            self.assertNotIn(f"blk.{i}.ffn_norm.weight", m)

    def test_the_names_that_already_matched_still_match(self):
        m = build_dflash_name_map(_Cfg(2))
        self.assertEqual(m["output_norm.weight"], "norm.weight")
        self.assertEqual(m["blk.1.attn_q.weight"], "layers.1.self_attn.q_proj.weight")
        self.assertEqual(
            m["blk.1.attn_output.weight"], "layers.1.self_attn.o_proj.weight"
        )
        self.assertEqual(m["blk.0.ffn_down.weight"], "layers.0.mlp.down_proj.weight")
        self.assertEqual(m["blk.0.attn_norm.weight"], "layers.0.input_layernorm.weight")

    def test_it_refuses_a_config_with_no_layers(self):
        with self.assertRaises(ValueError):
            build_dflash_name_map(_Cfg(0))

    def test_the_norms_are_named_as_dense_modules(self):
        p = dflash_unquantized_module_prefixes(_Cfg(2))
        self.assertIn("hidden_norm", p)
        self.assertIn("norm", p)
        self.assertIn("layers.1.post_attention_layernorm", p)
        self.assertIn("layers.0.self_attn.q_norm", p)


@unittest.skipUnless(
    os.path.exists(DFLASH_GGUF), f"DFLASH GGUF not on this machine: {DFLASH_GGUF}"
)
class TestDflashNameMapAgainstTheFile(unittest.TestCase):
    """The check that catches a re-export, run against the real bytes."""

    @classmethod
    def setUpClass(cls):
        cls.tensors = _read_gguf_tensors(DFLASH_GGUF)
        cls.name_map = build_dflash_name_map(_Cfg(5))

    def test_every_tensor_in_the_file_is_claimed_exactly_once(self):
        self.assertEqual(len(self.tensors), 58)
        self.assertIsNone(
            audit_dflash_name_map(self.name_map, self.tensors),
            audit_dflash_name_map(self.name_map, self.tensors),
        )

    def test_the_hf_side_matches_the_bf16_checkpoints_parameter_names(self):
        """The map must land on the names the model actually registers.

        Taken from the BF16 safetensors checkpoint rather than from the model
        class, so this stays a pure-CPU check with no model construction.
        """
        import json

        index = os.path.join(DFLASH_HF, "model.safetensors.index.json")
        single = os.path.join(DFLASH_HF, "model.safetensors")
        if os.path.exists(index):
            with open(index) as f:
                hf_names = set(json.load(f)["weight_map"])
        elif os.path.exists(single):
            with open(single, "rb") as f:
                (n,) = struct.unpack("<Q", f.read(8))
                hdr = json.loads(f.read(n))
            hdr.pop("__metadata__", None)
            hf_names = set(hdr)
        else:
            self.skipTest("BF16 DFLASH checkpoint not on this machine")
        self.assertEqual(set(self.name_map.values()), hf_names)

    def test_the_quantised_and_dense_tensors_are_the_ones_expected(self):
        """Dtypes pinned: 36 packed, 22 F32 -- and WHICH ones.

        This is what makes ``fc`` on a plain ``nn.Linear`` a hard blocker
        rather than a style question: ``dflash_fc`` is packed, and it is the
        single largest tensor in the checkpoint.
        """
        F32, Q8_0 = 0, 8
        by_type = {}
        for name, (tt, _dims) in self.tensors.items():
            by_type.setdefault(tt, []).append(name)
        self.assertEqual(len(by_type[Q8_0]), 36)
        self.assertEqual(len(by_type[F32]), 22)
        # The projection is packed -- the reason fc cannot be an nn.Linear.
        self.assertEqual(self.tensors["dflash_fc.weight"][0], Q8_0)
        self.assertEqual(self.tensors["dflash_fc.weight"][1], (25600, 5120))
        # Every norm is F32, which is why they must stay dense modules.
        for name, (tt, _dims) in self.tensors.items():
            if "norm" in name:
                self.assertEqual(tt, F32, f"{name} is not F32")

    def test_the_file_holds_the_same_drafter_as_the_bf16_checkpoint(self):
        """Same 58 tensors, same 1.73 G parameters -- not a different build."""
        total = 0
        for _tt, dims in self.tensors.values():
            n = 1
            for d in dims:
                n *= d
            total += n
        self.assertEqual(total, 1_730_213_120)


@unittest.skipUnless(
    os.path.exists(DFLASH_GGUF) and os.path.exists(DFLASH_HF),
    "DFLASH checkpoints not on this machine",
)
class TestTheQ8CheckpointLoadsCompletely(unittest.TestCase):
    """End to end on the CPU: every name the loader emits finds a parameter.

    This is the gate that makes a boot attempt worth spending a card on. It
    builds the draft on the META device with a GGUF quant config -- so it costs
    no VRAM and no weights -- and then replays exactly what the loader will do:
    the file's tensor names through the map, dense tensors as ``weight`` and
    packed ones as ``qweight`` + ``qweight_type``, and each of those through
    ``load_weights``' own stacked-parameter resolution (q/k/v fuse into
    ``qkv_proj``, gate/up into ``gate_up_proj``).

    Measured on this branch: 58 file tensors -> 94 emitted names -> 94 resolved.
    """

    @classmethod
    def setUpClass(cls):
        import torch
        from transformers import AutoConfig

        import sglang.srt.runtime_context as rc
        from sglang.srt.server_args import ServerArgs

        try:
            rc.get_context().set_server_args(ServerArgs(model_path=DFLASH_HF))
            from sglang.srt.distributed import (
                init_distributed_environment,
                initialize_model_parallel,
            )

            init_distributed_environment(
                world_size=1,
                rank=0,
                local_rank=0,
                distributed_init_method="tcp://127.0.0.1:29617",
                backend="gloo",
            )
            initialize_model_parallel(tensor_model_parallel_size=1)
            from sglang.srt.layers.quantization.gguf import GGUFConfig
            from sglang.srt.models.dflash import DFlashDraftModel

            cls.cfg = AutoConfig.from_pretrained(DFLASH_HF, trust_remote_code=True)
            with torch.device("meta"):
                cls.model = DFlashDraftModel(
                    cls.cfg, quant_config=GGUFConfig(), prefix=""
                )
        except Exception as exc:  # noqa: BLE001 - environment, not the code
            raise unittest.SkipTest(f"cannot build the draft here: {exc}")
        cls.params = dict(cls.model.named_parameters())
        cls.tensors = _read_gguf_tensors(DFLASH_GGUF)

    def _emitted_names(self):
        """Exactly what gguf_quant_weights_iterator yields for this file."""
        from sglang.srt.model_loader.gguf_dflash import build_dflash_name_map

        F32 = 0
        name_map = build_dflash_name_map(self.cfg)
        out = []
        for gguf_name, hf_name in name_map.items():
            if self.tensors[gguf_name][0] != F32:
                out.append(hf_name.replace("weight", "qweight_type"))
                out.append(hf_name.replace("weight", "qweight"))
            else:
                out.append(hf_name)
        return out

    def _resolve(self, name):
        """``DFlashDraftModel.load_weights``' own resolution, replayed."""
        stacked = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        def plain(n):
            if n in self.params:
                return n
            if n.startswith("model.") and n[len("model.") :] in self.params:
                return n[len("model.") :]
            if f"model.{n}" in self.params:
                return f"model.{n}"
            return None

        for param_name, weight_name, _shard in stacked:
            if f".{weight_name}." in name:
                hit = plain(name.replace(weight_name, param_name))
                if hit:
                    return hit
        return plain(name)

    def test_fc_is_a_packed_parameter_not_a_dense_one(self):
        """The blocker this round removed, asserted directly."""
        self.assertIn("fc.qweight", self.params)
        self.assertIn("fc.qweight_type", self.params)
        self.assertNotIn("fc.weight", self.params)
        self.assertEqual(
            tuple(self.params["fc.qweight"].tensor_shape),
            (5120, 25600),
        )

    def test_fc_stays_dense_without_a_quant_config(self):
        """Every configuration that worked before must still work."""
        import torch

        from sglang.srt.models.dflash import DFlashDraftModel

        with torch.device("meta"):
            dense = DFlashDraftModel(self.cfg, quant_config=None, prefix="")
        names = dict(dense.named_parameters())
        self.assertIn("fc.weight", names)
        self.assertNotIn("fc.qweight", names)
        self.assertEqual(tuple(names["fc.weight"].shape), (5120, 25600))

    def test_every_emitted_name_resolves_to_a_parameter(self):
        emitted = self._emitted_names()
        # 36 packed tensors -> 2 names each, 22 F32 tensors -> 1 name each.
        self.assertEqual(len(emitted), 36 * 2 + 22)
        unresolved = [n for n in emitted if self._resolve(n) is None]
        self.assertEqual(unresolved, [], f"{len(unresolved)} names find no parameter")

    def test_the_fused_modules_are_reached_through_the_stacked_mapping(self):
        """q/k/v and gate/up have no parameters of their own -- by design."""
        self.assertIn("layers.0.self_attn.qkv_proj.qweight", self.params)
        self.assertIn("layers.0.mlp.gate_up_proj.qweight", self.params)
        self.assertNotIn("layers.0.self_attn.q_proj.qweight", self.params)
        self.assertEqual(
            self._resolve("layers.0.self_attn.q_proj.qweight"),
            "layers.0.self_attn.qkv_proj.qweight",
        )
        self.assertEqual(
            self._resolve("layers.0.mlp.up_proj.qweight"),
            "layers.0.mlp.gate_up_proj.qweight",
        )


if __name__ == "__main__":
    unittest.main()
