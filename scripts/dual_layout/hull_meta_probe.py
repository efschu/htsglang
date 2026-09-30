#!/usr/bin/env python3
"""DUAL-TP3PP3 1b desk probe: does the P STAGE hull assemble for the real 27B
NVFP4 checkpoint under PP3 -- on META, no GPU, no weights?

Three processes = three P stages (tp 1, pp 3, cut --pp-stage-ratio). Each builds
D's three TP3 part trees for its stage (lane_geometry_override(3, r) + D's
vectors), the TP1 hull, and runs the SAME assembly the boot runs
(assemble_lane_shells, _finalize_hull_params, _fill_hull_buffers,
_refresh_captured_linear_attention_tensors). What it proves: every parallel
linear of the stage has a counterpart in every part, every remaining hull
parameter is aliasable or a known composed vector -- the structural half of
build_dual_stage_model. What it cannot prove: bytes, kernels, processing.

  python hull_meta_probe.py [--model PATH] [--cut 49,8,7] [--tp 58,25,25] [--mlp 98,19,19]
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys

MODEL = "/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-27B-NVFP4-RadixArk"


def _worker(rank: int, world: int, port: str, a) -> None:
    import torch

    import flliper.srt.server_args as SA
    from flliper.srt.configs.load_config import LoadConfig
    from flliper.srt.configs.model_config import ModelConfig
    from flliper.srt.distributed import parallel_state as ps
    from flliper.srt.distributed.dual_group import NestedGroupPlan
    from flliper.srt.distributed.utils import scoped_tp_partition_ratios
    from flliper.srt.layers.dp_attention import initialize_dp_attention
    from flliper.srt.model_executor import dual_group_lane as L
    from flliper.srt.model_loader.loader import (
        _get_quantization_config,
        _initialize_model,
        set_default_torch_dtype,
    )
    from flliper.srt.runtime_context import get_context
    from flliper.srt.server_args import ServerArgs

    SA.is_cuda = lambda: True
    torch.cuda.get_device_capability = lambda *x, **k: (12, 0)
    ps.should_build_pynccl = lambda *x, **k: False
    sa = ServerArgs(model_path=a.model, trust_remote_code=True, dtype="bfloat16", tp_size=1, pp_size=world,
                    page_size=1, disable_overlap_schedule=True, kv_cache_dtype="fp8_e4m3", context_length=98304,
                    device="cuda", pp_stage_ratio=[int(x) for x in a.cut.split(",")])
    get_context().set_server_args(sa)
    ps.init_distributed_environment(world_size=world, rank=rank, local_rank=rank,
                                    distributed_init_method=f"tcp://127.0.0.1:{port}", backend="gloo")
    ps.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=world, backend="gloo")
    mc = ModelConfig(model_path=a.model, trust_remote_code=True, dtype="bfloat16")
    initialize_dp_attention(server_args=sa, model_config=mc)
    lc = LoadConfig()
    qc = _get_quantization_config(mc, lc)
    tp = [int(x) for x in a.tp.split(",")]
    fams = {"mlp": [int(x) for x in a.mlp.split(",")]} if a.mlp else None
    plan = NestedGroupPlan(big_ratio=tuple(tp), segments=((0,), (1,), (2,)),
                           family_ratios=tuple((k, tuple(v)) for k, v in (fams or {}).items()))
    parts = []
    for r in range(3):
        with scoped_tp_partition_ratios(list(plan.fast_ratio), {k: list(v) for k, v in plan.fast_family_ratios} or None), \
                L.lane_geometry_override(3, r), set_default_torch_dtype(mc.dtype), torch.device("meta"):
            parts.append(_initialize_model(mc, lc, qc).eval())
    with L.lane_geometry_override(1, 0), set_default_torch_dtype(mc.dtype), torch.device("meta"):
        hull = _initialize_model(mc, lc, qc).eval()
    local = rank  # d_rank_of_stage identity (both groups on rank-gpu-id 0,1,2)
    counts = L.assemble_lane_shells(hull, parts)
    fill = L._finalize_hull_params(hull, parts[local], parts)
    fill["buffers"] = L._fill_hull_buffers(hull, parts[local])
    fill["captured"] = L._refresh_captured_linear_attention_tensors(hull)
    n_layers = sum(1 for n, _ in hull.named_modules() if n.count(".") and n.split(".")[-1].isdigit())
    print("HULLPROBE " + json.dumps({"rank": rank, "counts": counts, "fill": fill,
                                     "hull_params": sum(1 for _ in hull.named_parameters()),
                                     "part_params": [sum(1 for _ in p.named_parameters()) for p in parts],
                                     "layer_modules": n_layers}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--cut", default="49,8,7")
    ap.add_argument("--tp", default="58,25,25")
    ap.add_argument("--mlp", default="98,19,19")
    ap.add_argument("--_rank", type=int, default=-1)
    ap.add_argument("--_port", default="")
    a = ap.parse_args()
    if a._rank >= 0:
        _worker(a._rank, 3, a._port, a)
        return
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = str(s.getsockname()[1])
    s.close()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", MASTER_ADDR="127.0.0.1", MASTER_PORT=port)
    args = [sys.executable, __file__, "--model", a.model, "--cut", a.cut, "--tp", a.tp, "--mlp", a.mlp, "--_port", port]
    ps = [subprocess.Popen(args + ["--_rank", str(r)], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           text=True) for r in range(3)]
    ok = True
    for r, p in enumerate(ps):
        out, err = p.communicate(timeout=1200)
        lines = [l for l in out.splitlines() if l.startswith("HULLPROBE ")]
        if p.returncode != 0 or not lines:
            ok = False
            print(f"rank {r} rc={p.returncode}\n{err[-3000:]}")
        for l in lines:
            print(l)
    print("HULL META PROBE", "OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
