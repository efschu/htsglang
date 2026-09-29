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
freeze: ``gc.collect()`` 1.4-1.6 ms. The collection semantics for anything
the load allocates are unchanged; only the walk over the pre-load objects is
gone. ``gc.unfreeze()`` at the end hands them back to the oldest generation,
so the process leaves the load exactly as it entered it.

A permanent generation that is already populated belongs to someone else
(``utils.common.freeze_gc``, ``weg2/gc_instrument``); then this does nothing,
because ``gc.unfreeze()`` cannot give back only its own part.
"""

from __future__ import annotations

import gc
import logging
import time
from contextlib import contextmanager
from typing import Iterator, Optional

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


@contextmanager
def load_gc_frozen(*, what: str = "model load") -> Iterator[Optional[int]]:
    """Run the block with the objects that exist now frozen.

    Yields the frozen object count, or None when nothing was frozen (switch
    off, or the permanent generation already in use)."""
    if not envs.SGLANG_OPT_LOAD_GC_FREEZE.get():
        yield None
        return
    already = gc.get_freeze_count()
    if already > 0:
        logger.info(
            "BOOTZEIT5 LOAD-GC-FREEZE skipped for %s: %d objects already frozen "
            "by someone else (unfreeze would release theirs too)",
            what,
            already,
        )
        yield None
        return
    t0 = time.perf_counter()
    gc.freeze()
    frozen = gc.get_freeze_count()
    logger.info(
        "BOOTZEIT5 LOAD-GC-FREEZE %s: %d objects frozen in %.1f ms; every "
        "collection during the load walks only what the load creates",
        what,
        frozen,
        (time.perf_counter() - t0) * 1e3,
    )
    try:
        yield frozen
    finally:
        # Someone inside may have unfrozen already (freeze_gc does on exit);
        # unfreezing an empty permanent generation is a no-op either way.
        gc.unfreeze()
