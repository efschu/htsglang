"""#757 liveness measurement: does a gloo isend+wait stall the sender until
the receiver posts its recv?

Models the exact in-tree shape: the upstream posts async sends
(_pp_send_dict_to_next_stage, async_send=True) and BLOCKS at its commit
(_pp_commit_comm_work: work.wait()). Under the disarm-time #757 form the
armed downstream posts NO recv until disarm -- so the sender's stall equals
the remaining armed window IF gloo wait() needs the peer recv, and ~0 if
gloo buffers. Two sizes: proxy-shaped (4 KiB) and output-shaped (8 MiB).
"""

import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

DELAY_S = 3.0


def run(rank, world, size_bytes, recv_delay_s, port, results):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    n = size_bytes // 4
    if rank == 0:
        t = torch.arange(n, dtype=torch.float32)
        dist.barrier()  # both ranks ready; timing starts clean
        t0 = time.perf_counter()
        work = dist.isend(t, dst=1)
        post_dt = time.perf_counter() - t0
        work.wait()
        wait_dt = time.perf_counter() - t0
        results[0] = (post_dt, wait_dt)
    else:
        buf = torch.empty(n, dtype=torch.float32)
        dist.barrier()
        if recv_delay_s > 0:
            time.sleep(recv_delay_s)  # the ARMED window: no recv posted
        dist.recv(buf, src=0)
        results[1] = float(buf[-1].item())
    dist.barrier()
    dist.destroy_process_group()


def main():
    mp.set_start_method("spawn", force=True)
    port = 29631
    for label, size in (("proxy-4KiB", 4 << 10), ("output-8MiB", 8 << 20)):
        for arm, delay in (("armed-window-no-recv", DELAY_S), ("immediate-recv", 0.0)):
            mgr = mp.Manager()
            results = mgr.dict()
            ps = [
                mp.Process(target=run, args=(r, 2, size, delay, port, results))
                for r in range(2)
            ]
            for p in ps:
                p.start()
            for p in ps:
                p.join(60)
            port += 1
            post_dt, wait_dt = results[0]
            print(
                f"{label:12s} {arm:22s} sender: post={post_dt*1000:7.1f} ms "
                f"wait-complete={wait_dt*1000:8.1f} ms  (recv delayed {delay:.0f}s)"
            )


if __name__ == "__main__":
    main()
