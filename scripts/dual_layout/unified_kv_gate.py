#!/usr/bin/env python3
"""DUAL-TP3PP3 unified KV gate (D) -- the GPU half the CPU tests cannot see.

Run under the patched torch_memory_saver preload (the release image's
SGLANG_WEG2_TMS_PRELOAD_SO on LD_PRELOAD), one card, ~1.5 GiB:

  LD_PRELOAD=$SGLANG_WEG2_TMS_PRELOAD_SO python unified_kv_gate.py --device 0

Checks, each printed PASS/FAIL:
 G1 born: a KV-shaped buffer allocated in a saver region at the TOP size and
    trimmed to 0 tokens keeps its address; cudaMemGetInfo gets the top back.
 G2 capture on the trimmed VA (risk 3): a CUDA graph that writes and reads
    slots [0, 512) captures and replays on the trimmed buffer.
 G3 live grow: set_spans(now) to 12288 tokens -- the address is unchanged,
    rows 4096..12287 are writable, the captured graph still replays and its
    rows kept their bytes (a kept lattice cell is never remapped fresh).
 G4 live shrink back to 0: the grown bytes return to the driver, the graph
    still replays on the kept page.
"""
from __future__ import annotations

import argparse
import sys

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--top", type=int, default=196608)
    ap.add_argument("--row-bytes", type=int, default=2048)
    a = ap.parse_args()
    dev = torch.device("cuda", a.device)
    torch.cuda.set_device(dev)
    from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter
    from sglang.srt.weg2 import d_seat_vram as sv
    from sglang.srt.weg2 import dual_p_kv_stage as pk

    spans = sv.tms()
    if not spans.available:
        print("FAIL: tms_set_spans/tms_alloc_info not in the preloaded hook (stock wheel?)")
        return 2
    saver = TorchMemorySaverAdapter.create(enable=True)
    page, step, g = 64, 4096, sv.granule_for(dev)
    rows = a.top + page
    ok = True
    torch.cuda.synchronize()
    free0 = torch.cuda.mem_get_info(dev)[0]
    with saver.region("kv_cache"):
        t = torch.zeros(rows, a.row_bytes // 2, dtype=torch.bfloat16, device=dev)
    ptr0 = t.data_ptr()
    info = spans.info(ptr0)
    geom = pk._geom_for(t, a.top, page, "k0", info.size)
    cuts = [geom.slots_for(k) for k in pk.lattice(a.top, step)]
    torch.cuda.synchronize()
    rc = spans.set_spans(ptr0, sv.slot_spans(geom.geom, geom.slots_for(0), g, cuts=cuts), now=True)
    torch.cuda.synchronize()
    free1 = torch.cuda.mem_get_info(dev)[0]
    g1 = rc == 0 and t.data_ptr() == ptr0 and (free1 - free0) > -(64 << 20)
    print("G1 born+trim rc=%d addr_same=%s free_delta_mib=%.0f -> %s" % (
        rc, t.data_ptr() == ptr0, (free1 - free0) / 2**20, "PASS" if g1 else "FAIL"))
    ok &= g1
    idx = torch.arange(512, device=dev)
    val = torch.arange(512, device=dev, dtype=torch.bfloat16).unsqueeze(1).expand(512, t.shape[1]).contiguous()
    out = torch.empty(512, t.shape[1], dtype=torch.bfloat16, device=dev)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        t.index_copy_(0, idx, val)
        out.copy_(t.index_select(0, idx))
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        t.index_copy_(0, idx, val)
        out.copy_(t.index_select(0, idx))
    graph.replay()
    torch.cuda.synchronize()
    g2 = torch.equal(out, val)
    print("G2 capture+replay on the trimmed VA -> %s" % ("PASS" if g2 else "FAIL"))
    ok &= g2
    rc = spans.set_spans(ptr0, sv.slot_spans(geom.geom, geom.slots_for(12288), g, cuts=cuts), now=True)
    torch.cuda.synchronize()
    hi = torch.arange(4096, 12288, device=dev)
    t.index_fill_(0, hi, 3.0)
    graph.replay()
    torch.cuda.synchronize()
    kept = torch.equal(out, val)
    grown = bool((t.index_select(0, hi) == 3.0).all())
    g3 = rc == 0 and t.data_ptr() == ptr0 and kept and grown
    print("G3 live grow rc=%d addr_same=%s kept_rows=%s grown_rows=%s -> %s" % (
        rc, t.data_ptr() == ptr0, kept, grown, "PASS" if g3 else "FAIL"))
    ok &= g3
    free_g = torch.cuda.mem_get_info(dev)[0]
    torch.cuda.synchronize()
    rc = spans.set_spans(ptr0, sv.slot_spans(geom.geom, geom.slots_for(0), g, cuts=cuts), now=True)
    torch.cuda.synchronize()
    free_s = torch.cuda.mem_get_info(dev)[0]
    graph.replay()
    torch.cuda.synchronize()
    g4 = rc == 0 and free_s > free_g and torch.equal(out, val)
    print("G4 live shrink rc=%d freed_mib=%.0f graph_ok=%s -> %s" % (
        rc, (free_s - free_g) / 2**20, torch.equal(out, val), "PASS" if g4 else "FAIL"))
    ok &= g4
    print("UNIFIED-KV GATE", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
