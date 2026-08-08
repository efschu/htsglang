#!/usr/bin/env python
"""#651: the registry refactor's byte-identity gate, run for the 35B (qwen35moe).

d68d8075cd's gate covered qwen35 only via the dense 27B (851-entry map) and
gemma4; the 35B MoE arm was never byte-gated, and no on-card 35B GGUF run
exists after 2026-07-18 (pre-refactor). This reproduces the gate for the 35B:
name map, unquantized-prefix set, and the ORDERED transform_stream digest over
every yielded (name, dtype, shape, bytes).

Run once per tree (PYTHONPATH selects the tree -- verify the printed
sglang.__file__!), then diff the per-tensor digest files.

    CUDA_VISIBLE_DEVICES=99 PYTHONPATH=<tree>/python python stream_digest_gate.py \
        <model_dir_or_gguf> <out.digest>
"""

import hashlib
import os
import sys

import torch  # noqa: F401  (adapter imports expect torch present)


def main() -> int:
    src, out_path = sys.argv[1], sys.argv[2]
    if os.path.isdir(src):
        ggufs = [f for f in os.listdir(src) if f.endswith(".gguf")]
        assert len(ggufs) == 1, ggufs
        gguf_file = os.path.join(src, ggufs[0])
        cfg_dir = src
    else:
        gguf_file = src
        cfg_dir = os.path.dirname(src)

    import sglang

    print(f"tree: {sglang.__file__}")

    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(cfg_dir, trust_remote_code=True)

    # Adapter construction: registry (post-refactor) or direct class (pre).
    try:
        from sglang.srt.model_loader.gguf_registry import create_gguf_adapter

        adapter = create_gguf_adapter(config, gguf_file)
        how = "registry"
    except ImportError:
        from sglang.srt.model_loader.gguf_qwen35 import Qwen35GGUFAdapter

        adapter = Qwen35GGUFAdapter(config, gguf_file)
        how = "direct"
    print(f"adapter: {type(adapter).__name__} via {how}, arch {adapter.arch}")

    name_map = adapter.build_name_map()
    unq = sorted(adapter.unquantized_module_prefixes())
    nm_digest = hashlib.sha256(
        "\n".join(f"{k}\t{v}" for k, v in sorted(name_map.items())).encode()
    ).hexdigest()[:12]
    unq_digest = hashlib.sha256("\n".join(unq).encode()).hexdigest()[:12]
    print(f"name_map {len(name_map)} sha {nm_digest}")
    print(f"unquantized prefixes {len(unq)} sha {unq_digest}")

    from sglang.srt.model_loader.weight_utils import gguf_quant_weights_iterator

    overall = hashlib.sha256()
    n = 0
    with open(out_path, "w") as fh:
        fh.write(f"# name_map {len(name_map)} sha {nm_digest}\n")
        fh.write(f"# unq {len(unq)} sha {unq_digest}\n")
        for name, tensor in adapter.transform_stream(
            gguf_quant_weights_iterator(gguf_file, name_map)
        ):
            t = tensor.contiguous()
            meta = f"{name}\t{t.dtype}\t{tuple(t.shape)}"
            h = hashlib.sha256()
            h.update(meta.encode())
            h.update(t.cpu().numpy().tobytes())
            d = h.hexdigest()[:16]
            fh.write(f"{meta}\t{d}\n")
            overall.update(d.encode())
            n += 1
            if n % 8192 == 0:
                print(f"  {n} tensors...", flush=True)
    print(f"stream tensors {n}  ORDERED-DIGEST {overall.hexdigest()[:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
