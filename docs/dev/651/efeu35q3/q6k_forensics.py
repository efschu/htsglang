#!/usr/bin/env python
"""efeu-TP14: value forensics of faulty Q6_K dequantize output.

For each wrong fp16 element (vs the consensus of all launches) ask where the
wrong value could have come from:
  H_perm   it equals the CORRECT value of another element of the same 256-block
           (a misplaced store / wrong lane index)
  H_neigh  it equals the correct value of the same offset in a neighbouring block
           (wrong block index / wrong scale header)
  H_raw    its two bytes occur at the same byte offset of the raw input
           (aliasing: the output buffer was read back with input bytes in it)
  H_prev   it equals what an EARLIER launch wrote to the same element (stale)
and report the fraction explained by each, plus the bit-level XOR histogram
(single flipped bits => transport/electrical; random => wrong data).

    python q6k_forensics.py --src <gguf> --tensor output.weight --rows 2048 --launches 40
"""

import argparse
import importlib
import json
import os

import numpy as np
import torch
from gguf import GGUFReader
import gguf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--tensor", default="output.weight")
    ap.add_argument("--rows", type=int, default=2048)
    ap.add_argument("--launches", type=int, default=40)
    ap.add_argument("--module", default=os.environ.get("GGUF_MODULE", "sglang_gguf_rocm"))
    ap.add_argument("--fresh-input", action="store_true", help="re-upload W before every launch")
    ap.add_argument("--json")
    a = ap.parse_args()
    K = importlib.import_module(a.module)
    r = GGUFReader(a.src)
    t = [x for x in r.tensors if x.name == a.tensor][0]
    qt = t.tensor_type
    raw = np.array(t.data.reshape(-1, t.data.shape[-1])[: a.rows])
    cols = raw.shape[1] // gguf.GGML_QUANT_SIZES[qt][1] * gguf.GGML_QUANT_SIZES[qt][0]
    rows = raw.shape[0]
    W = torch.from_numpy(raw).cuda()
    outs = []
    for _ in range(a.launches):
        if a.fresh_input:
            W = torch.from_numpy(raw).cuda()
        o = K.ggml_dequantize(W, int(qt), rows, cols, torch.float16, None)
        torch.cuda.synchronize()
        outs.append(o.cpu().numpy().reshape(-1).view(np.uint16).copy())
        del o
    st = np.stack(outs)
    # consensus = per-element mode via median on uint16 is wrong for floats; use
    # majority: the value most launches agree on (cheap: compare to launch-wise median of float view)
    f = st.view(np.float16).astype(np.float32)
    cons = np.median(f, axis=0).astype(np.float16).view(np.uint16)
    ref = gguf.quants.dequantize(raw, qt).astype(np.float16).reshape(-1)
    raw_u16 = raw.reshape(-1).view(np.uint8)
    stats = {"launches": a.launches, "faulty_launches": 0, "wrong_elems": 0,
             "H_perm": 0, "H_neigh": 0, "H_raw": 0, "H_prev": 0, "unexplained": 0,
             "xor_popcount_hist": {}, "nonfinite": 0, "blocks_hit": {}}
    for li in range(a.launches):
        bad = np.flatnonzero(st[li] != cons)
        if bad.size == 0:
            continue
        stats["faulty_launches"] += 1
        for e in bad:
            v = st[li, e]
            stats["wrong_elems"] += 1
            if not np.isfinite(np.uint16(v).view(np.float16)):
                stats["nonfinite"] += 1
            blk = e // 256
            stats["blocks_hit"][int(blk)] = stats["blocks_hit"].get(int(blk), 0) + 1
            x = int(v) ^ int(cons[e])
            pc = bin(x).count("1")
            stats["xor_popcount_hist"][pc] = stats["xor_popcount_hist"].get(pc, 0) + 1
            if pc == 1:
                bit = x.bit_length() - 1
                stats.setdefault("single_bit_pos", {})
                stats["single_bit_pos"][bit] = stats["single_bit_pos"].get(bit, 0) + 1
            stats.setdefault("off_in_block_hist", {})
            ob = int((e % 256) // 32)
            stats["off_in_block_hist"][ob] = stats["off_in_block_hist"].get(ob, 0) + 1
            expl = False
            b0 = blk * 256
            if (cons[b0:b0 + 256] == v).any():
                stats["H_perm"] += 1
                expl = True
            off = e % 256
            for nb in (blk - 2, blk - 1, blk + 1, blk + 2):
                if 0 <= nb < cons.size // 256 and cons[nb * 256 + off] == v:
                    stats["H_neigh"] += 1
                    expl = True
                    break
            lo, hi = v & 0xFF, v >> 8
            byte_pos = e * 2
            # raw bytes around the same byte address of the input buffer
            if byte_pos + 1 < raw_u16.size and raw_u16[byte_pos] == lo and raw_u16[byte_pos + 1] == hi:
                stats["H_raw"] += 1
                expl = True
            if li > 0 and (st[:li, e] == v).any():
                stats["H_prev"] += 1
                expl = True
            if not expl:
                stats["unexplained"] += 1
    stats["consensus_vs_oracle_maxabs"] = float(np.abs(cons.view(np.float16).astype(np.float32) - ref.astype(np.float32)).max())
    stats["n_blocks_hit"] = len(stats["blocks_hit"])
    top = sorted(stats["blocks_hit"].items(), key=lambda x: -x[1])[:10]
    stats["blocks_hit"] = dict(top)
    stats["variant"] = {"HSA": os.environ.get("HSA_OVERRIDE_GFX_VERSION"), "module": K.__file__,
                        "fresh_input": a.fresh_input}
    print(json.dumps(stats, indent=1, default=str))
    if a.json:
        json.dump(stats, open(a.json, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
