#!/usr/bin/env python3
"""#441(a): GPU falsifier for the transfer_kv_all_layer_lf_ph segfault. FILED, NOT RUN.

Desk analysis (test_page_head_offset_alignment_441.py) proved a real missing
invariant -- ``head_size_bytes = item_size / head_num`` is never required to be
8-byte aligned, while the copy helper uses ``.b64`` PTX -- but ALSO proved that
the reported crash shapes are aligned, so that defect is not its cause.
Attribution needs metal. This script is that experiment, ready to run.

RUN IT UNDER A gpu-arb CLAIM. Every arm crashes the process by design, so run
one arm per invocation and read the exit status; a segfault is the RESULT, not
a failure of the script.

    python3 tools/441/falsify_lf_ph_441.py --arm repro
    python3 tools/441/falsify_lf_ph_441.py --arm alignment
    python3 tools/441/falsify_lf_ph_441.py --arm bisect

ARMS

  repro      The reported shapes (page_size=4, float32, head_num=1,
             head_dim=2), the ones test_minimax_sparse_pool_host_unit uses.
             EXPECTED per the ticket: segfault on both wheels. If it does NOT
             crash here, the crash is not in the kernel at these shapes and
             the pool wrapper is the next suspect.

  alignment  head_num=2, head_dim=2, fp16 -> item_size=8 (passes the launcher
             guard), head_size=4 (misaligned). EXPECTED: crash or
             `misaligned address`. This is the desk finding put on metal; a
             clean run here REFUTES the alignment analysis and that must be
             recorded, not explained away.

  bisect     Walks head_num over a fixed item_size, reporting which values
             fault. Isolates whether the fault tracks head subdivision (the
             alignment story) or something else entirely.

WHY NOT A PYTEST: it segfaults the interpreter, so it cannot report through
pytest. Exit code is the observation channel.
"""

import argparse
import sys

import torch


def _kernel():
    from sgl_kernel.kvcacheio import transfer_kv_all_layer_lf_ph

    return transfer_kv_all_layer_lf_ph


def _build(head_num, head_dim, dtype, layers, page_size, pages):
    """Device layer-first source, host page-head destination, and the tables.

    Mirrors pool_host/mha.py:135-144: the pointer TABLE is a device tensor of
    uint64 holding each layer buffer's address.
    """
    dev = torch.device("cuda:0")
    item_elems = head_num * head_dim
    tokens = pages * page_size

    src_k = [torch.arange(tokens * item_elems, dtype=dtype, device=dev).reshape(tokens, item_elems) for _ in range(layers)]
    src_v = [torch.zeros(tokens, item_elems, dtype=dtype, device=dev) for _ in range(layers)]
    k_tbl = torch.tensor([t.data_ptr() for t in src_k], dtype=torch.uint64, device=dev)
    v_tbl = torch.tensor([t.data_ptr() for t in src_v], dtype=torch.uint64, device=dev)

    # page_head destination: [page_num, head_num, page_size, layer_num, head_dim]
    dst_k = torch.zeros(pages, head_num, page_size, layers, head_dim, dtype=dtype, device="cpu", pin_memory=True)
    dst_v = torch.zeros_like(dst_k)

    item_size = item_elems * dtype.itemsize
    layout_dim = item_size * layers
    return src_k, src_v, k_tbl, v_tbl, dst_k, dst_v, item_size, layout_dim


def _fire(head_num, head_dim, dtype, layers, page_size, pages, label):
    fn = _kernel()
    src_k, src_v, k_tbl, v_tbl, dst_k, dst_v, item_size, layout_dim = _build(
        head_num, head_dim, dtype, layers, page_size, pages
    )
    n = pages * page_size
    src_idx = torch.arange(n, dtype=torch.int64, device="cuda:0")
    dst_idx = torch.arange(n, dtype=torch.int64, device="cuda:0")

    print(
        f"[{label}] head_num={head_num} head_dim={head_dim} dtype={dtype} "
        f"layers={layers} page_size={page_size} item_size={item_size} "
        f"head_size={item_size // head_num} "
        f"guard_item%8={item_size % 8} head%8={(item_size // head_num) % 8}",
        flush=True,
    )
    # Flush before the launch: if the process dies, the line above is the record.
    fn(
        src_k_layers=k_tbl,
        dst_k=dst_k,
        src_v_layers=v_tbl,
        dst_v=dst_v,
        src_indices=src_idx,
        dst_indices=dst_idx,
        item_size=item_size,
        dst_layout_dim=layout_dim,
        num_layers=layers,
        page_size=page_size,
        head_num=head_num,
    )
    torch.cuda.synchronize()
    print(f"[{label}] SURVIVED (no fault)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=("repro", "alignment", "bisect"))
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("no CUDA: this falsifier needs a card and a gpu-arb claim", file=sys.stderr)
        return 2

    if args.arm == "repro":
        _fire(1, 2, torch.float32, 4, 4, 2, "repro")
    elif args.arm == "alignment":
        _fire(2, 2, torch.float16, 3, 4, 2, "alignment")
    else:
        # Fixed item_size (16 bytes at fp16, 8 elems), head_num walked.
        for head_num in (1, 2, 4, 8):
            head_dim = 8 // head_num
            if head_dim == 0:
                continue
            _fire(head_num, head_dim, torch.float16, 3, 4, 2, f"bisect hn={head_num}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
