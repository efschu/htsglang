#!/usr/bin/env python3
"""Diagnose: HTCCLBar1Transport direkt bauen, vollen Traceback zeigen.

Zwei Prozesse via mp.spawn, gloo-Gruppe, dann HTCCLBar1Transport(...)
OHNE die Fabrik baue_bar1 -- damit die Ausnahme nicht in ein logger.info
uebersetzt wird, sondern samt Traceback sichtbar bleibt.
"""
import os
import sys
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def arbeiter(lokal: int, devs: list, port: str) -> None:
    rank = lokal
    welt = len(devs)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(welt)
    dist.init_process_group("gloo", rank=rank, world_size=welt)

    import logging
    logging.basicConfig(
        level=logging.DEBUG,
        format=f"[r{rank}] %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
        force=True,
    )

    torch.cuda.set_device(devs[rank])
    device = torch.device("cuda", devs[rank])
    torch.cuda.init()
    torch.zeros(1, device=device)

    from sglang.srt.distributed.device_communicators.htccl_bar1 import (
        HTCCLBar1Transport,
    )
    from sglang.srt.distributed.device_communicators.htccl_matrix_transport import (
        _fenster_bytes,
    )

    gruppe = dist.group.WORLD
    fb = _fenster_bytes()
    print(f"[r{rank}] dev={devs[rank]} fenster_bytes={fb}", flush=True)
    print(f"[r{rank}] patchstand={HTCCLBar1Transport.patchstand()}", flush=True)

    t = None
    try:
        t = HTCCLBar1Transport(gruppe, device, fb)
        print(f"[r{rank}] AUFBAU OK, fenster_minimum={t.fenster_minimum()}",
              flush=True)
    except BaseException:
        sys.stderr.write(f"\n===== [r{rank}] AUFBAU FEHLGESCHLAGEN =====\n")
        traceback.print_exc()
        sys.stderr.flush()
        dist.destroy_process_group()
        os._exit(1)

    try:
        res = t.byte_beleg_alle()
        print(f"[r{rank}] BYTE-BELEG: {res}", flush=True)
    except BaseException:
        sys.stderr.write(f"\n===== [r{rank}] BYTE-BELEG FEHLGESCHLAGEN =====\n")
        traceback.print_exc()
        sys.stderr.flush()
        try:
            t.close()
        except Exception:
            pass
        dist.destroy_process_group()
        os._exit(2)

    dist.barrier()
    t.close()
    dist.destroy_process_group()
    print(f"[r{rank}] FERTIG", flush=True)


if __name__ == "__main__":
    devs = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "1,2").split(",")]
    port = sys.argv[2] if len(sys.argv) > 2 else "29591"
    mp.spawn(arbeiter, args=(devs, port), nprocs=len(devs), join=True)
