"""#190: is the GDN prefill chunk op run-to-run deterministic?

Calls chunk_gated_delta_rule with exactly the convention
gdn_backend.forward_extend uses (see layers/attention/linear/kernels/
gdn_triton.py::extend), which is the part the first #187 attempt got wrong.

A naive repeat loop is blind to uninitialized-memory reads: inside a tight loop the
caching allocator hands the same physical block back to every `torch.empty` /
`new_empty`, so garbage is *constant* across reps and the op looks clean.

Here every rep is preceded by a poison pass: allocate-and-free a set of buffers
filled with a rep-dependent pattern, so the allocator's free blocks carry
different bytes each time. Any output change is then a read of memory the
kernels never wrote.
"""

import argparse
import os
import sys

import torch

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python")
)

from sglang.srt.layers.attention.fla.chunk import chunk_gated_delta_rule  # noqa: E402
from sglang.srt.layers.attention.fla.fused_gdn_gating import (  # noqa: E402
    fused_gdn_gating,
)


def build_inputs(T, Hg, H, Kdim, Vdim, seed, device, dtype, n_slots=4, slot=1):
    gcpu = torch.Generator(device="cpu").manual_seed(seed)

    def r(*shape, dt=dtype):
        return torch.randn(*shape, generator=gcpu, dtype=torch.float32).to(
            device=device, dtype=dt
        )

    q = r(1, T, Hg, Kdim)
    k = r(1, T, Hg, Kdim)
    v = r(1, T, H, Vdim)
    A_log = r(H, dt=torch.float32)
    dt_bias = r(H, dt=torch.float32)
    a = r(T, H)
    b = r(T, H)
    g, beta = fused_gdn_gating(A_log, a, b, dt_bias)
    g = g.view(1, T, H)
    beta = beta.view(1, T, H)
    state = (
        torch.randn(n_slots, H, Vdim, Kdim, generator=gcpu, dtype=torch.float32).to(
            device
        )
        * 0.1
    )
    idx = torch.tensor([slot], device=device, dtype=torch.int32)
    cu = torch.tensor([0, T], device=device, dtype=torch.int32)
    return q, k, v, g, beta, state, idx, cu


def poison(device, value, mb=192):
    """Dirty the caching allocator's free blocks with a rep-dependent pattern."""
    bufs = []
    # a spread of sizes so blocks of every bucket get touched
    for nbytes in (1 << 20, 1 << 22, 1 << 24, mb * (1 << 20) // 4):
        n = max(nbytes // 4, 1)
        t = torch.empty(n, dtype=torch.float32, device=device)
        t.fill_(value)
        bufs.append(t)
    del bufs
    torch.cuda.synchronize()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lens", type=str, default="64,109,128,129,133,157,205,257,512")
    p.add_argument("--reps", type=int, default=6)
    p.add_argument("--hg", type=int, default=8)
    p.add_argument("--h", type=int, default=24)
    p.add_argument("--kdim", type=int, default=128)
    p.add_argument("--vdim", type=int, default=128)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--no-poison", action="store_true")
    a = p.parse_args()

    dev = "cuda"
    dtype = torch.bfloat16
    print(
        f"# device={torch.cuda.get_device_name(0)} Hg={a.hg} H={a.h} "
        f"K={a.kdim} V={a.vdim} reps={a.reps} poison={not a.no_poison}"
    )
    print(f"{'T':>6} {'chunks':>7} {'o_cls':>6} {'h_cls':>6} {'st_cls':>7} "
          f"{'o_maxabs':>12} {'h_maxabs':>12} {'st_maxabs':>12}")

    for T in [int(x) for x in a.lens.split(",")]:
        q, k, v, g, beta, state0, idx, cu = build_inputs(
            T, a.hg, a.h, a.kdim, a.vdim, a.seed, dev, dtype
        )
        os_, hs_, sts_ = [], [], []
        for rep in range(a.reps):
            if not a.no_poison:
                poison(dev, float(rep + 1) * 3.7e3)
            st = state0.clone()
            o, _l, h = chunk_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=st,
                initial_state_indices=idx,
                cu_seqlens=cu,
                head_first=False,
                use_qk_l2norm_in_kernel=True,
            )
            torch.cuda.synchronize()
            os_.append(o.clone())
            hs_.append(h.clone())
            sts_.append(st.clone())

        def nclass(lst):
            seen = []
            for t in lst:
                bb = t.detach().float().cpu().numpy().tobytes()
                if bb not in seen:
                    seen.append(bb)
            return len(seen)

        def mx(lst):
            return max((lst[0] - x).abs().max().item() for x in lst[1:])

        print(
            f"{T:>6} {(T + 63)//64:>7} {nclass(os_):>6} {nclass(hs_):>6} "
            f"{nclass(sts_):>7} {mx(os_):>12.3e} {mx(hs_):>12.3e} {mx(sts_):>12.3e}"
        )


if __name__ == "__main__":
    main()
