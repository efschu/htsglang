#!/usr/bin/env python3
"""Which construction-time allocations ignore `torch.device("meta")`?

Register #1036. Building the model on `meta` should cost no device memory, but the
load contract still needed ~1 GiB of VRAM and therefore raced the standing serving
job for it on an occupied rig. Anything that allocates for real inside a `meta`
context does so by naming a device explicitly, which is a bug for any desk workflow.

This hooks the tensor factories, records every allocation whose device is not `meta`
during construction, and prints the biggest ones with the frame that asked. It is
diagnosis only: nothing is patched, nothing is fixed here.
"""

from __future__ import annotations

import collections
import json
import os
import os.path as osp
import sys
import traceback

sys.path.insert(0, "python")

import torch  # noqa: E402

_HITS: collections.Counter = collections.Counter()
_BYTES: collections.Counter = collections.Counter()
_ARMED = False


def _frame() -> str:
    """The first frame inside sglang that is not this file or torch internals."""
    for fr in reversed(traceback.extract_stack()[:-2]):
        if "/sglang/" in fr.filename and "qwen4_exp_meta_leaks" not in fr.filename:
            return f"{fr.filename.split('/python/')[-1]}:{fr.lineno} in {fr.name}"
    return "<no sglang frame>"


def _wrap(mod, name):
    orig = getattr(mod, name)

    def probe(*a, **kw):
        out = orig(*a, **kw)
        if _ARMED and isinstance(out, torch.Tensor) and out.device.type != "meta":
            # Device type is part of the key: a `cpu` allocation inside a `meta`
            # context is virtual and mostly harmless (torch.empty never touches the
            # pages), while a `cuda` one is what races a serving job for VRAM. Sum
            # them together and the report is dominated by the harmless kind -- the
            # first version of this probe reported 175 GiB on a 20 GiB card.
            key = f"[{out.device.type}] {name}  <- {_frame()}"
            _HITS[key] += 1
            _BYTES[key] += out.numel() * out.element_size()
        return out

    setattr(mod, name, probe)
    return orig


def main() -> int:
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "/spinning/qwen38-flash-next/ckpt"

    for fn in ("empty", "zeros", "ones", "arange", "full", "tensor", "eye", "rand"):
        _wrap(torch, fn)

    from sglang.srt.configs.load_config import LoadConfig
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.srt.layers.dp_attention import initialize_dp_attention
    from sglang.srt.model_loader.loader import (
        _get_quantization_config,
        _initialize_model,
    )
    from sglang.srt.model_loader.utils import set_default_torch_dtype
    from sglang.srt.runtime_context import lane_scope
    from sglang.srt.server_args import ServerArgs

    import tempfile

    rdzv = osp.join(tempfile.mkdtemp(prefix="qwen4_leaks_"), "rdzv")
    init_distributed_environment(
        world_size=1, rank=0, distributed_init_method=f"file://{rdzv}",
        local_rank=0, backend="gloo",
    )
    initialize_model_parallel(tensor_model_parallel_size=1)

    server_args = ServerArgs(model_path=ckpt)
    model_config = ModelConfig.from_server_args(server_args)
    initialize_dp_attention(server_args, model_config)
    load_config = LoadConfig()

    global _ARMED
    with lane_scope(None, server_args):
        quant_config = _get_quantization_config(model_config, load_config)
        with set_default_torch_dtype(model_config.dtype):
            with torch.device("meta"):
                _ARMED = True
                print(f"ambient default device at construction: "
                      f"{torch.get_default_device()}")
                try:
                    _initialize_model(model_config, load_config, quant_config)
                finally:
                    _ARMED = False

    per_dev: collections.Counter = collections.Counter()
    for key, nbytes in _BYTES.items():
        per_dev[key.split("]")[0].lstrip("[")] += nbytes
    print(f"\nnon-meta allocations during a meta construction: {sum(_HITS.values())}")
    for dev, nbytes in per_dev.most_common():
        note = (
            "  <-- THIS is what races the serving job for VRAM"
            if dev == "cuda"
            else "  (virtual; torch.empty never faults the pages in)"
        )
        print(f"  {dev:5s} {nbytes / 2**30:8.2f} GiB{note}")
    print()
    for key, nbytes in _BYTES.most_common(20):
        print(f"  {nbytes / 2**20:9.2f} MiB  x{_HITS[key]:<6d} {key}")
    print("\nCUDA-only, which is the number that matters here:")
    cuda = {k: v for k, v in _BYTES.items() if k.startswith("[cuda]")}
    if not cuda:
        print("  none -- construction is VRAM-free apart from the CUDA context")
    for key, nbytes in sorted(cuda.items(), key=lambda kv: -kv[1])[:15]:
        print(f"  {nbytes / 2**20:9.2f} MiB  x{_HITS[key]:<6d} {key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
