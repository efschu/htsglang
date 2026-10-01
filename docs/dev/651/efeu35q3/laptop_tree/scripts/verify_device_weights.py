#!/usr/bin/env python
"""#651: byte-verify DEVICE-RESIDENT GGUF weights against the file itself.

Reconstruction and generalization of the laptop's deleted `_host_verify.py`
(#644), whose .pyc survived: independent oracle = the GGUF file read with
``gguf.GGUFReader``; MoE expert parameters hold PACKED ggml bytes, so the
comparison is byte-exact and needs no dequantization. A premature free or a
scrambled expert order shows up directly. The sample always includes the LAST
expert of the LAST layer (a premature free corrupts whatever was copied last).

This is the falsifier for the "device-side load/assembly scrambles or
corrupts weights" hypothesis (HANDOFF §12.10): the CPU-side stream is proven
byte-identical to the on-card-verified tree, so if device-resident bytes
match the file too, the load path is exonerated end-to-end and the defect
must be in forward compute (GDN triton being the prime remaining block).

Runs a weight-only in-process load (no server, no memory pool, no graphs).

  HSA_OVERRIDE_GFX_VERSION=11.0.0 python verify_device_weights.py \
      /root/651-p2/models/Qwen3.6-35B-A3B-UD-Q4KM-noQ6K.gguf /root/lh/models

Exit 0 = every sampled tensor byte-identical; 1 = mismatch (prints where).
"""

import sys

import numpy as np
import torch


def main() -> int:
    gguf_file, tokenizer_path = sys.argv[1], sys.argv[2]

    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.server_args import PortArgs, ServerArgs

    server_args = ServerArgs(
        model_path=gguf_file,
        tokenizer_path=tokenizer_path,
        load_format="gguf",
        quantization="gguf",
        device="cuda",
        tp_size=1,
        mem_fraction_static=0.90,
        disable_cuda_graph=True,
        disable_radix_cache=True,
        mamba_radix_cache_strategy="no_buffer",
        disable_overlap_schedule=True,
        attention_backend="triton",
        sampling_backend="pytorch",
        context_length=2048,
        max_total_tokens=2048,
    )
    port_args = PortArgs.init_new(server_args)
    model_config = ModelConfig.from_server_args(server_args)
    runner = ModelRunner(
        model_config=model_config,
        mem_fraction_static=server_args.mem_fraction_static,
        gpu_id=0,
        tp_rank=0,
        tp_size=1,
        moe_ep_rank=0,
        moe_ep_size=1,
        pp_rank=0,
        pp_size=1,
        nccl_port=port_args.nccl_port,
        server_args=server_args,
    )
    model = runner.model

    from gguf import GGUFReader

    reader = GGUFReader(gguf_file)
    file_tensors = {t.name: t for t in reader.tensors}

    params = dict(model.named_parameters())
    num_layers = int(model_config.hf_text_config.num_hidden_layers)

    # Sampled layers: first, middle, the former-Q6_K layers, and the LAST.
    layers = sorted({0, 20, 34, 38, 39, num_layers - 1})
    failures = 0
    checked = 0

    def cmp(label: str, dev_bytes: torch.Tensor, ref: np.ndarray) -> None:
        nonlocal failures, checked
        checked += 1
        dev = dev_bytes.cpu().numpy()
        if dev.shape != ref.shape:
            print(f"FAIL {label}: shape {dev.shape} vs file {ref.shape}")
            failures += 1
            return
        if not np.array_equal(dev, ref):
            bad = int((dev != ref).sum())
            print(f"FAIL {label}: {bad} of {ref.size} bytes differ")
            failures += 1
        else:
            print(f"ok   {label}: {ref.size} bytes identical")

    for li in layers:
        gate = file_tensors.get(f"blk.{li}.ffn_gate_exps.weight")
        up = file_tensors.get(f"blk.{li}.ffn_up_exps.weight")
        down = file_tensors.get(f"blk.{li}.ffn_down_exps.weight")
        w13 = params.get(f"model.layers.{li}.mlp.experts.w13_qweight")
        w2 = params.get(f"model.layers.{li}.mlp.experts.w2_qweight")
        if gate is None or w13 is None:
            print(f"note: layer {li}: expert params not found "
                  f"(gate={gate is not None} w13={w13 is not None}); "
                  f"names sample: "
                  f"{[n for n in params if f'layers.{li}.mlp' in n][:4]}")
            continue
        E = gate.data.shape[0]
        gate_rows = gate.data.shape[1]
        # Experts sampled per layer: first, a middle one, and the LAST.
        for e in (0, E // 2, E - 1):
            cmp(f"L{li} exp{e} gate",
                w13.data[e, :gate_rows], np.asarray(gate.data[e]))
            cmp(f"L{li} exp{e} up",
                w13.data[e, gate_rows:], np.asarray(up.data[e]))
            cmp(f"L{li} exp{e} down", w2.data[e], np.asarray(down.data[e]))

    # A couple of non-expert quantized tensors (plain GGUFLinearMethod path).
    for gguf_name, param_name in (
        ("blk.0.attn_q.weight", "model.layers.0.self_attn.q_proj.qweight"),
        ("blk.0.ffn_down_shexp.weight",
         "model.layers.0.mlp.shared_expert.down_proj.qweight"),
        ("output.weight", "lm_head.qweight"),
    ):
        ft = file_tensors.get(gguf_name)
        pt = params.get(param_name)
        if ft is None or pt is None:
            print(f"note: skip {gguf_name} -> {param_name} "
                  f"(file={ft is not None} param={pt is not None})")
            continue
        cmp(gguf_name, pt.data.reshape(-1), np.asarray(ft.data).reshape(-1))

    print(f"\nchecked {checked} tensors, failures {failures}")
    print("VERDICT:", "DEVICE BYTES MATCH FILE" if failures == 0
          else "DEVICE-SIDE CORRUPTION/MISASSIGNMENT")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
