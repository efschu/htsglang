"""#201 slice 2 -- what one pipeline stage boundary costs on the wire.

The in-server counter (SGLANG_PP_BOUNDARY_STATS) can only report a blocking
recv, which is pipeline bubble plus wire. This measures the wire alone: two
processes, the same two transports the PP path actually uses, and the exact
payload shapes ``PPProxyTensors`` carries.

  - NCCL isend/irecv on the device group, for ``hidden_states`` + ``residual``
    (2 tensors of ``[num_tokens, hidden_size]`` in the model dtype)
  - gloo for the pickled metadata, which ``send_tensor_dict`` sends ahead of
    every crossing as a size tensor plus payload -- two extra messages per
    crossing that no size argument makes cheaper

Reported per shape: bytes on the wire, and one-way microseconds taken as half
a round trip, so no clock has to be shared between the hosts.

Run one process per host, same as a two-node boot:

    NODE=0 MASTER=10.10.10.1 python pp_link_pingpong.py
    NODE=1 MASTER=10.10.10.1 python pp_link_pingpong.py
"""

import os
import statistics
import time

import torch
import torch.distributed as dist


def _pct(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def main():
    rank = int(os.environ["NODE"])
    master = os.environ.get("MASTER", "10.10.10.1")
    port = os.environ.get("PORT", "31975")
    hidden = int(os.environ.get("HIDDEN", "2560"))
    dtype = getattr(torch, os.environ.get("DTYPE", "float16"))
    iters = int(os.environ.get("ITERS", "300"))
    warmup = int(os.environ.get("WARMUP", "50"))

    torch.cuda.set_device(0)
    dist.init_process_group(
        backend="cpu:gloo,cuda:nccl",
        init_method=f"tcp://{master}:{port}",
        world_size=2,
        rank=rank,
    )
    peer = 1 - rank
    gloo = dist.new_group(ranks=[0, 1], backend="gloo")

    if rank == 0:
        print(f"device: {torch.cuda.get_device_name(0)}  dtype: {dtype}", flush=True)
        print(
            f"{'tokens':>8} {'payload':>12} {'nccl 1-way us':>16} "
            f"{'p90 us':>9} {'gloo meta us':>14}",
            flush=True,
        )

    # bs=1 decode is the case the pipeline runs in most of the time; the larger
    # counts are prefill chunks.
    for num_tokens in (1, 4, 64, 512, 2048, 8192):
        tensors = [
            torch.zeros((num_tokens, hidden), dtype=dtype, device="cuda")
            for _ in range(2)
        ]
        nbytes = sum(t.numel() * t.element_size() for t in tensors)
        meta = [
            {"hidden_states": (num_tokens, hidden), "residual": (num_tokens, hidden)}
        ]

        samples = []
        meta_samples = []
        for i in range(warmup + iters):
            dist.barrier()
            started = time.perf_counter()
            if rank == 0:
                works = [dist.isend(t, peer) for t in tensors]
                for w in works:
                    w.wait()
                works = [dist.irecv(t, peer) for t in tensors]
                for w in works:
                    w.wait()
            else:
                works = [dist.irecv(t, peer) for t in tensors]
                for w in works:
                    w.wait()
                works = [dist.isend(t, peer) for t in tensors]
                for w in works:
                    w.wait()
            torch.cuda.synchronize()
            rtt = time.perf_counter() - started

            dist.barrier()
            meta_started = time.perf_counter()
            if rank == 0:
                dist.broadcast_object_list(meta, src=0, group=gloo)
                dist.broadcast_object_list(meta, src=1, group=gloo)
            else:
                dist.broadcast_object_list(meta, src=0, group=gloo)
                dist.broadcast_object_list(meta, src=1, group=gloo)
            meta_rtt = time.perf_counter() - meta_started

            if i >= warmup:
                samples.append(rtt / 2 * 1e6)
                meta_samples.append(meta_rtt / 2 * 1e6)

        if rank == 0:
            print(
                f"{num_tokens:>8} {nbytes / 1024:>9.1f} KiB "
                f"{statistics.median(samples):>16.1f} "
                f"{_pct(samples, 0.9):>9.1f} "
                f"{statistics.median(meta_samples):>14.1f}",
                flush=True,
            )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
