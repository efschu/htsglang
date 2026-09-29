"""BOOTZEIT 5 (29.09.): the model load runs with the boot's objects frozen.

The expert presplit ends every MoE layer with a full ``gc.collect()``
(``expert_offload.presplit_host_reclaim``, 29 layers on NF PP0, 48 per D
rank). A full collection walks EVERY tracked object of the process -- and a
scheduler process at load time carries the whole import graph: measured on
the rig's CPU, the imports alone (torch, scheduler, model_runner, qwen4_exp)
are 801522 tracked objects and one ``gc.collect()`` takes 0.26-0.29 s, under
the GIL. z30w-park prints exactly that shape per layer: the presplit's
"repack" posten (process_weights_after_loading minus store open/write) is
0.43 s on PP0 (512 rows repacked per layer) and 0.49 s on D TP0 (29 of 201
rows) -- a fixed cost per layer that does not scale with the rows, i.e. not
the Marlin repack (marlin=0.01 s) but the collect. The LOAD-PROFILE sampler
cannot see it: a GIL-holding C call yields about ONE sample per call however
long it runs (TP0: 41 samples at the collect line for 48 calls, PP0: 26 for
29), which is why BOOTZEIT 4 read it as 7.8-9.6 %.

``gc.freeze()`` moves every object tracked at that moment into the permanent
generation, which no collection walks. Frozen before the model is built, it
covers the import graph and nothing of the model: every module, parameter and
tensor the load creates -- the [E] host stacks included -- is born after the
freeze and stays fully collectible, cycles included. Measured after the
freeze: ``gc.collect()`` 1.4-1.6 ms.

WHERE IT APPLIES -- ONLY WHERE THE PER-LAYER COLLECT EXISTS. The collect runs
in the expert presplit, and the presplit runs only under expert offload
(resident fraction < 1.0 somewhere in the group, ``offload_active``). A dense
model (27B) or a MoE without offload never collects per layer, so it freezes
nothing: the loader passes ``expert_presplit=False`` and this is a no-op --
no freeze, no collect, no sampler; that load is the one it was before.

WHAT CAN BE HELD BACK, AND FOR HOW LONG. Frozen objects are not walked during
the load, so a reference cycle made ONLY of pre-load objects that becomes
unreachable during the load waits until the load ends. Two steps bound that:

* one ``gc.collect()`` right before the freeze, so nothing that is ALREADY
  garbage gets frozen -- everything frozen was reachable at that moment;
* at the end ``gc.unfreeze()`` and one ``gc.collect()`` that frees whatever
  waited, before the process goes on (graph capture, KV sizing, the flip).
  That collect runs with ``DEBUG_SAVEALL`` so the load-end line can name what
  it found -- objects, tensors, and their bytes by device -- then frees it.

The load-end line also carries the per-layer reclaim cost (``gc_s`` summed
over the presplit reclaims) and the container's non-reclaimable host memory
(cgroup ``anon + shmem``, the currency of bootzeit_eval's HOST-LADESPITZE)
at the start and its peak during the load, sampled at 5 Hz.

A permanent generation that someone else filled (``utils.common.freeze_gc``,
``weg2/gc_instrument`` -- both run after the load, so this is a guard, not a
case) is left alone, because ``gc.unfreeze()`` cannot give back only its own
part. "Filled" is a THRESHOLD, not ``> 0``: CPython 3.12 parks the immortal
objects it meets in the permanent generation on every collection -- measured
on the rig, a bare interpreter reports ``gc.get_freeze_count() == 375`` after
any ``gc.collect()`` without anyone calling ``gc.freeze()``. A ``> 0`` test
therefore skipped the freeze in every real process (c5d2599676; caught before
any boot). A real freeze of a scheduler process is ~800k objects.
"""

from __future__ import annotations

import gc
import logging
import threading
import time
from contextlib import contextmanager
from typing import Callable, Dict, Iterator, Optional

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_CGROUP_STAT = "/sys/fs/cgroup/memory.stat"
_SAMPLE_S = 0.2
#: Above this many objects the permanent generation holds someone's freeze;
#: below it, only the immortals CPython 3.12 parks there itself (375 measured).
_FOREIGN_FREEZE_MIN = 20000


def expert_presplit_runs() -> bool:
    """Will this load run the expert presplit (and its per-layer collect)?

    ``offload_active``: a resident fraction < 1.0 on some rank of the group --
    the precondition of ``presplit_expert_offload_after_repack``. False on
    anything unreadable: not freezing is the load as it always was."""
    try:
        from sglang.srt.layers.moe.resident_fraction import offload_active

        return bool(offload_active())
    except Exception as e:  # noqa: BLE001 -- a guess must never turn the freeze on
        logger.info("BOOTZEIT5 LOAD-GC-FREEZE: offload state unreadable (%s) -- no freeze", e)
        return False


def cgroup_nonreclaim_bytes(path: str = _CGROUP_STAT) -> Optional[int]:
    """``anon + shmem`` of this cgroup (bootzeit_eval's nonreclaim), or None."""
    anon = shmem = None
    try:
        with open(path) as f:
            for line in f:
                if line.startswith("anon "):
                    anon = int(line.split()[1])
                elif line.startswith("shmem "):
                    shmem = int(line.split()[1])
    except (OSError, ValueError, IndexError):
        return None
    if anon is None or shmem is None:
        return None
    return anon + shmem


class _PeakSampler:
    """Max of ``read()`` while the load runs (a daemon thread, 5 Hz)."""

    def __init__(self, read: Callable[[], Optional[int]], period_s: float = _SAMPLE_S):
        self._read = read
        self._period = period_s
        self._stop = threading.Event()
        self.start_value = read()
        self.peak = self.start_value
        self.samples = 0 if self.start_value is None else 1
        self._t = threading.Thread(target=self._run, name="load-gc-peak", daemon=True)
        self._t.start()

    def _take(self) -> None:
        v = self._read()
        if v is None:
            return
        self.samples += 1
        if self.peak is None or v > self.peak:
            self.peak = v

    def _run(self) -> None:
        while not self._stop.wait(self._period):
            self._take()

    def stop(self) -> None:
        self._stop.set()
        self._t.join(timeout=2.0)
        self._take()


def _collect_and_account() -> Dict[str, float]:
    """One full collection that says what it freed: objects, and the tensors
    among them by device (bytes of their storages). Frees them before return."""
    saved = gc.get_debug()
    garbage = []
    try:
        gc.set_debug(saved | gc.DEBUG_SAVEALL)
        found = gc.collect()
        garbage = list(gc.garbage)
        del gc.garbage[:]
    finally:
        gc.set_debug(saved)
    out = {"found": float(found), "tensors": 0.0, "host_mib": 0.0, "cuda_mib": 0.0}
    try:
        import torch

        seen = set()
        for obj in garbage:
            if not isinstance(obj, torch.Tensor):
                continue
            out["tensors"] += 1
            try:
                st = obj.untyped_storage()
                key = (st.data_ptr(), st.nbytes())
            except Exception:  # noqa: BLE001 -- a meta/sparse tensor has no storage
                continue
            if key in seen:
                continue
            seen.add(key)
            dev = "cuda_mib" if obj.is_cuda else "host_mib"
            out[dev] += key[1] / 2**20
    except ImportError:
        pass
    garbage.clear()
    gc.collect()  # the saved objects are unreachable again: free them now
    return out


def _gib(v: Optional[int]) -> str:
    return "?" if v is None else "%.2f" % (v / 2**30)


@contextmanager
def load_gc_frozen(*, what: str, expert_presplit: bool) -> Iterator[Optional[int]]:
    """Run the block with the objects that exist now frozen -- only when the
    block runs the expert presplit. Yields the frozen object count, or None
    when nothing was frozen."""
    if not expert_presplit or not envs.SGLANG_OPT_LOAD_GC_FREEZE.get():
        yield None
        return
    already = gc.get_freeze_count()
    if already >= _FOREIGN_FREEZE_MIN:
        logger.info(
            "BOOTZEIT5 LOAD-GC-FREEZE skipped for %s: %d objects already frozen "
            "by someone else (unfreeze would release theirs too)",
            what,
            already,
        )
        yield None
        return
    from sglang.srt.layers.moe.expert_offload import expert_store_clock

    t0 = time.perf_counter()
    pre_found = gc.collect()  # nothing that is already garbage gets frozen
    t1 = time.perf_counter()
    gc.freeze()
    frozen = gc.get_freeze_count()
    logger.info(
        "BOOTZEIT5 LOAD-GC-FREEZE %s: pre-collect %d objects in %.0f ms, %d objects "
        "frozen in %.1f ms; every collection during the load walks only what the "
        "load creates",
        what,
        pre_found,
        (t1 - t0) * 1e3,
        frozen,
        (time.perf_counter() - t1) * 1e3,
    )
    clock0 = expert_store_clock()
    sampler = _PeakSampler(cgroup_nonreclaim_bytes)
    try:
        yield frozen
    finally:
        sampler.stop()
        # Someone inside may have unfrozen already (freeze_gc does on exit);
        # unfreezing an empty permanent generation is a no-op either way.
        gc.unfreeze()
        t2 = time.perf_counter()
        freed = _collect_and_account()
        clock1 = expert_store_clock()
        n = int(clock1.get("reclaims", 0) - clock0.get("reclaims", 0))
        gc_s = clock1.get("gc_s", 0.0) - clock0.get("gc_s", 0.0)
        logger.info(
            "BOOTZEIT5 LOAD-GC-FREEZE end %s: reclaim gc=%.2f s over %d layers "
            "(%.3f s/layer) found=%d | nonreclaim(anon+shmem) start=%s peak=%s GiB "
            "(%d samples) | load-end collect %.0f ms freed %d objects, %d tensors "
            "host=%.1f MiB cuda=%.1f MiB",
            what,
            gc_s,
            n,
            gc_s / n if n else 0.0,
            int(clock1.get("gc_found", 0) - clock0.get("gc_found", 0)),
            _gib(sampler.start_value),
            _gib(sampler.peak),
            sampler.samples,
            (time.perf_counter() - t2) * 1e3,
            int(freed["found"]),
            int(freed["tensors"]),
            freed["host_mib"],
            freed["cuda_mib"],
        )
