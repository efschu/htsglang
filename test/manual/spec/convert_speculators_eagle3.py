#!/usr/bin/env python3
"""Convert a speculators-format EAGLE3 draft head (vLLM ecosystem, e.g.
RedHatAI/*-speculator.eagle3) into the SpecForge-style layout that sglang's
LlamaForCausalLMEagle3 loads.

Two convention differences are handled here so the runtime needs no special
cases beyond the `norm_before_residual` flag:

1. AUX LAYER IDS (off-by-one): speculators/vLLM `eagle_aux_hidden_state_layer_ids`
   follow vLLM's capture convention where id i means the hidden state at the
   INPUT of decoder layer i (= output of layer i-1); sglang's convention is
   "output of layer i" (sglang then adds +1 internally to place the capture
   before layer i+1). We therefore write ids-1 into the converted config.
   Feeding the raw ids loses ~0.2 accept probability per drafted token
   (measured: overlap@1 0.36 raw vs 0.56 translated, T101).

2. RESIDUAL NORM ORDER: speculators heads are trained with
   `norm_before_residual=True` (residual = hidden_norm(hidden)); the flag is
   passed through and honored by sglang's llama_eagle3 input layer.

Weight names (layers.0.*, fc, lm_head, norm, embed_tokens, d2t, t2d) already
match what LlamaForCausalLMEagle3.load_weights resolves; the safetensors file
is hardlinked (or copied across filesystems) unchanged.

Usage:
  python convert_speculators_eagle3.py <src_dir> <dst_dir>
"""
import json
import os
import shutil
import sys


def convert(src: str, dst: str) -> None:
    with open(os.path.join(src, "config.json")) as f:
        cfg = json.load(f)

    if cfg.get("speculators_model_type") != "eagle3":
        raise SystemExit(f"not a speculators eagle3 checkpoint: {src}")

    layer_cfg = cfg["transformer_layer_config"]
    if layer_cfg.get("model_type") != "llama":
        raise SystemExit(
            f"unsupported transformer_layer_config model_type: "
            f"{layer_cfg.get('model_type')}"
        )

    aux_ids = cfg["eagle_aux_hidden_state_layer_ids"]

    out = dict(layer_cfg)
    out.update(
        {
            "architectures": ["LlamaForCausalLMEagle3"],
            "model_type": "llama",
            "draft_vocab_size": cfg["draft_vocab_size"],
            "target_hidden_size": cfg.get("target_hidden_size")
            or layer_cfg["hidden_size"],
            # speculators id i = INPUT of layer i = output of layer i-1;
            # sglang expects "output of layer i" -> translate by -1.
            "eagle_config": {
                "eagle_aux_hidden_state_layer_ids": [i - 1 for i in aux_ids],
                "use_aux_hidden_state": True,
            },
            "norm_before_residual": cfg.get("norm_before_residual", False),
            "dtype": cfg.get("dtype", "bfloat16"),
            "tie_word_embeddings": False,
        }
    )

    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(dst, "config.json"), "w") as f:
        json.dump(out, f, indent=2)

    src_st = os.path.join(src, "model.safetensors")
    dst_st = os.path.join(dst, "model.safetensors")
    if not os.path.exists(dst_st):
        try:
            os.link(src_st, dst_st)
        except OSError:
            shutil.copy2(src_st, dst_st)

    print(f"converted {src} -> {dst}")
    print(f"  aux ids (speculators) {aux_ids} -> (sglang) {[i-1 for i in aux_ids]}")
    print(f"  norm_before_residual = {out['norm_before_residual']}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    convert(sys.argv[1], sys.argv[2])
