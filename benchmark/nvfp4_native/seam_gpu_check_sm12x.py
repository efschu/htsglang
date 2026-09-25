import os, sys, json, torch
os.environ["SGLANG_FP4_SM12X_W4A16_MAX_M"] = "16"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_fi_next_sm12x as B
from sglang.srt.layers.quantization import fp4_utils
from sglang.srt.layers.quantization import nvfp4_sm12x_w4a16 as S
from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp4Config, ModelOptFp4LinearMethod
from flashinfer.autotuner import autotune
m = ModelOptFp4LinearMethod(ModelOptFp4Config(is_checkpoint_nvfp4_serialized=True, group_size=16))
res = []
for name, N, K in [("D.gate_up", 18688, 5120), ("D.down", 5120, 9344)]:
    L, raw = B.make_layer(m, N, K, "cutlass", name.endswith("gate_up"))
    fp4_utils.FP4_GEMM_RUNNER_BACKEND = fp4_utils.Fp4GemmRunnerBackend("cutlass")
    for M in (1, 8, 16, 17, 32):
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        ref = B.reference(x, raw)
        path = S.choose_sm12x_fp4_kernel(M, S.sm12x_w4a16_max_m(), 12)
        with autotune(True):
            m.apply(L, x)
        o = m.apply(L, x).float()[:, :B.REF_ROWS]
        # graph capture of the same call
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            m.apply(L, x)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            og = m.apply(L, x)
        x.copy_(torch.randn_like(x)); g.replay(); torch.cuda.synchronize()
        refg = B.reference(x, raw)
        r = dict(shape=name, M=M, path=path, rel_err=float((o - ref).norm() / ref.norm()),
                 graph_rel_err=float((og.float()[:, :B.REF_ROWS] - refg).norm() / refg.norm()))
        print(json.dumps(r), flush=True); res.append(r)
ok = all((r["rel_err"] < 5e-3 and r["graph_rel_err"] < 5e-3) if r["path"] == "w4a16_native" else r["rel_err"] < 0.2 for r in res)
print("SEAM_GPU_CHECK", "PASS" if ok else "FAIL")
