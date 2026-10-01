#!/usr/bin/env python
"""#651: layer-wise activation comparison — llama.cpp reference vs our dump.

Reference side: `llama-eval-callback` output (common/debug.cpp format): per
tensor a header `common_debug_cb_eval: <name> = (<type>) <op>(...) = {ne}`
followed by a value preview and a full-tensor `sum = <float>` line. The
layer-output tensors are named `l_out-<i>`.

Our side: a `PassNNNNN.pt` file from `--debug-tensor-dump-output-folder`
(tensor_dump_forward_hook): dict of module-name -> output tensor(s). A
qwen3_5 decoder layer returns (hidden, residual); both are summed, since
llama.cpp's `l_out` is the post-residual hidden state.

Sums are float-accumulation-order dependent (CPU fp32 reference vs GPU fp16),
so agreement is judged RELATIVELY: a healthy layer differs by ~1e-2 relative;
a broken one diverges by orders of magnitude, flips sign, or goes non-finite.
The output is the per-layer table and the FIRST suspicious layer.

    python compare_activations.py <eval_callback.log> <PassNNNNN.pt> [--layers 40]
"""

import re
import sys

import torch

HEADER = re.compile(
    r"common_debug_cb_eval:\s+(\S+)\s+=\s+\((\S+)\)\s+(\S+)\(.*=\s*\{([0-9, ]+)\}"
)
SUM_RE = re.compile(r"sum\s*=\s*([-0-9.einfa]+)")


def parse_reference(path):
    """-> {tensor_name: (op, shape, sum)}"""
    out = {}
    cur = None
    for line in open(path, errors="replace"):
        m = HEADER.search(line)
        if m:
            cur = (m.group(1), m.group(3), m.group(4).strip())
            continue
        m = SUM_RE.search(line)
        if m and cur is not None:
            try:
                out[cur[0]] = (cur[1], cur[2], float(m.group(1)))
            except ValueError:
                out[cur[0]] = (cur[1], cur[2], float("nan"))
            cur = None
    return out


def tensor_sum(obj) -> float:
    if isinstance(obj, torch.Tensor):
        return float(obj.detach().float().sum().item())
    if isinstance(obj, (tuple, list)):
        return sum(tensor_sum(t) for t in obj if t is not None)
    return float("nan")


def main() -> int:
    ref_path, dump_path = sys.argv[1], sys.argv[2]
    n_layers = 40
    if "--layers" in sys.argv:
        n_layers = int(sys.argv[sys.argv.index("--layers") + 1])

    ref = parse_reference(ref_path)
    dump = torch.load(dump_path, map_location="cpu", weights_only=False)
    print(f"reference tensors: {len(ref)}; dump entries: {len(dump)}")

    # Our layer outputs: module names ending in `model.layers.<i>` (the whole
    # decoder layer module). Fall back to any name containing `layers.<i>`
    # with the shortest suffix.
    by_layer = {}
    for name, val in dump.items():
        m = re.search(r"(?:^|\.)layers\.(\d+)$", name)
        if m:
            by_layer[int(m.group(1))] = (name, val)

    print(f"{'layer':>5} {'ref l_out sum':>16} {'our sum':>16} {'ratio':>10}  flag")
    first_bad = None
    for i in range(n_layers):
        r = ref.get(f"l_out-{i}")
        o = by_layer.get(i)
        if r is None or o is None:
            print(f"{i:>5} {'-' if r is None else f'{r[2]:16.3f}'} "
                  f"{'-' if o is None else f'{tensor_sum(o[1]):16.3f}'}   (missing side)")
            continue
        rs = r[2]
        os_ = tensor_sum(o[1])
        ratio = os_ / rs if rs not in (0.0,) else float("inf")
        bad = (
            not (0.5 < ratio < 2.0)
            or rs != rs
            or os_ != os_
        )
        flag = "  <-- DIVERGES" if bad else ""
        if bad and first_bad is None:
            first_bad = i
        print(f"{i:>5} {rs:16.3f} {os_:16.3f} {ratio:10.3f}{flag}")

    if first_bad is None:
        print("\nno layer-level divergence at sum granularity "
              "(defect may be below sum sensitivity — compare previews next)")
    else:
        print(f"\nFIRST DIVERGENT LAYER: {first_bad} "
              f"(inspect its site class: GDN vs full-attn vs MoE)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
