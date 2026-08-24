#!/usr/bin/env python
"""#651: prove the NEXTN draft's router gates load DENSE, on the real file, no GPU.

This is the desk gate for speculation. It exists because the failure it catches
is silent: the shared GGUF weight iterator renames every non-F32 tensor's
`.weight` leaf to `.qweight`, which equates "not F32" with "destination module is
quantized". A MoE router gate is never quantized, so the renamed tensor lands on
a parameter no module has, is dropped with one logger.warning, and the gate keeps
its uninitialized values. A garbage router still routes every token to SOME
expert -- the model stays fluent and is quietly wrong, and the only outward sign
is that speculation stops accepting.

Qwen3.6-35B-A3B-UD-Q4_K_XL is the checkpoint that exposes it: its MTP block
(blk.40) stores ffn_gate_inp and ffn_gate_inp_shexp as BF16, and those two are
the ONLY non-F32 dense tensors among its 753 tensors. All 40 base-layer gates
are F32 and unaffected -- so a target-only smoke test cannot see this at all.

Expected values are the ones 0155ff2c00 measured on this exact checkpoint:

    mtp.layers.0.mlp.gate.weight                bfloat16 (256, 2048)  std ~0.0096
    mtp.layers.0.mlp.shared_expert_gate.weight  bfloat16 (1, 2048)    std ~0.0020

A std near byte-noise (or a `.qweight` spelling, or a missing tensor) is the
defect. Run:

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=<tree>/python \\
        python mtp_gate_probe.py <model_dir_or_gguf>
"""

from __future__ import annotations

import os
import sys

import torch


def _resolve(src: str) -> tuple[str, str]:
    if os.path.isdir(src):
        ggufs = [f for f in os.listdir(src) if f.endswith(".gguf")]
        if len(ggufs) != 1:
            raise SystemExit(f"expected exactly one .gguf in {src}, found {ggufs}")
        return os.path.join(src, ggufs[0]), src
    return src, os.path.dirname(src)


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    gguf_file, cfg_dir = _resolve(sys.argv[1])

    import sglang

    print(f"tree:  {sglang.__file__}")
    print(f"gguf:  {gguf_file}")

    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(cfg_dir, trust_remote_code=True)

    # The NEXTN draft reuses the same adapter, but the model loader rewrites the
    # draft ModelConfig's architecture so the adapter emits the MTP-block name
    # map instead of the base map. Reproduce that rewrite here, or we would be
    # gating the target's 40 F32 gates -- the ones that were never at risk.
    text_cfg = getattr(config, "text_config", config)
    draft_cfg = AutoConfig.from_pretrained(cfg_dir, trust_remote_code=True)
    draft_cfg.architectures = ["Qwen3_5ForCausalLMMTP"]
    n_layers = getattr(text_cfg, "num_hidden_layers", None)
    print(f"config: num_hidden_layers={n_layers} -> MTP block is blk.{n_layers}")

    from sglang.srt.model_loader.gguf_registry import create_gguf_adapter

    adapter = create_gguf_adapter(draft_cfg, gguf_file)
    if adapter is None:
        # Returning None means "no bespoke family, use the generic GGUF path" --
        # which for this checkpoint would silently gate the wrong thing, since
        # the MTP name map lives only in the bespoke adapter.
        print(
            f"FAIL  no bespoke GGUF adapter for model_type="
            f"{getattr(draft_cfg, 'model_type', None)!r}; the MTP name map "
            "this probe exists to check would not be used."
        )
        print("VERDICT: DEFECT")
        return 1
    print(f"adapter: {type(adapter).__name__} arch={adapter.arch} is_draft={getattr(adapter, 'is_draft', None)}")

    name_map = adapter.build_name_map()

    from sglang.srt.model_loader.weight_utils import gguf_quant_weights_iterator

    targets = {
        "mtp.layers.0.mlp.gate.weight": (256, 2048),
        "mtp.layers.0.mlp.shared_expert_gate.weight": (1, 2048),
    }
    seen: dict[str, torch.Tensor] = {}
    stray: list[str] = []

    stream = gguf_quant_weights_iterator(gguf_file, name_map)
    if hasattr(adapter, "transform_stream"):
        stream = adapter.transform_stream(stream)
    n = 0
    for name, tensor in stream:
        n += 1
        if name in targets:
            seen[name] = tensor
        # The exact failure mode: the dense gate arriving under a packed name.
        base = name.rsplit(".", 1)[0]
        if name.endswith((".qweight", ".qweight_type")) and (
            base + ".weight" in targets
        ):
            stray.append(name)

    print(f"stream: {n} tensors")

    ok = True
    for want, shape in targets.items():
        t = seen.get(want)
        if t is None:
            print(f"FAIL  {want}: ABSENT from the stream")
            ok = False
            continue
        finite = bool(torch.isfinite(t.float()).all())
        std = float(t.float().std())
        shape_ok = tuple(t.shape) == shape
        # Byte noise reinterpreted as bf16 lands orders of magnitude off a
        # trained router's scale; a real gate sits in the low 1e-3..1e-2 band.
        scale_ok = 1e-4 < std < 1.0
        status = "ok  " if (finite and shape_ok and scale_ok) else "FAIL"
        ok &= finite and shape_ok and scale_ok
        print(
            f"{status}  {want}: dtype={t.dtype} shape={tuple(t.shape)} "
            f"finite={finite} std={std:.4f} (want shape {shape}, std 1e-4..1)"
        )

    for s in stray:
        print(f"FAIL  {s}: dense router gate arrived under a PACKED name")
        ok = False

    print("VERDICT:", "MTP ROUTER GATES DENSE AND SANE" if ok else "DEFECT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
