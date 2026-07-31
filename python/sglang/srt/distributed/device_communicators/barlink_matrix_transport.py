# SPDX-License-Identifier: Apache-2.0
"""barlink ``matrix`` transport: planner plus BAR1 direct path.

The seam this module closes
----------------------------
``barlink_matrix.py`` plans, ``barlink_bar1.py`` transports. Between the two,
the piece that was missing until now is the one that

1. builds the direct path (``build_bar1``),
2. feeds its **actual** capacity into the planner as ``window_bytes`` --
   the minimum across all destinations and all ranks, not the raw size
   from sysfs and not the requested size,
3. calls ``plan()`` and logs ``plan.explanation()`` on rank 0,
4. hands the plan back to the direct path, so that ``handles`` and the
   per-size kernel choice come from the same source.

The order here is not arbitrary. The planner excludes algorithms whose
window requirement would overrun the mapping (``plan_collective(...,
window_bytes=)``); that number is only known once the transport has been
built. Conversely, the transport needs the plan to choose mesh or ring per
size. Hence: build first, then plan, then feed the plan back in.

Why the direct path doubles as the measurement sensor
------------------------------------------------------
``BarlinkBar1Transport`` implements the ``Sensor`` protocol
(``name``/``self_load``/``self_load_duplex``/``pair``/``pair_receive``).
When it is up, the planner measures **real directed edges** instead of
estimating from self-load. When it is not, the planner falls back to the
self-load estimate and records that in the explanation -- it does not
pretend to have measured the edge.

What happens when the direct path is unavailable
---------------------------------------------------
Then this transport reports ``handles(...) == False`` for everything. The
planner still runs and logs its explanation -- the choice of roles and
algorithm is useful information for the other transports too. But nothing
is sent over a path that does not exist.

Turning it on and off
-----------------------
None of this happens without an explicit choice:

* ``SGLANG_BARLINK_TRANSPORT=matrix`` selects this transport,
  ``SGLANG_BARLINK_TRANSPORT=bar1`` selects the bare direct path without a
  planner. The default remains ``device``; nothing changes unless you
  switch it.
* ``SGLANG_BARLINK_MATRIX_DIRECT=0`` disables the direct path; the planner
  keeps running.
* ``SGLANG_BARLINK_BAR1_WINDOW_MIB`` (default 96) is the **requested**
  region size per rank. 96 MiB, because the measured region is 90.69 MiB
  and fits inside the 256 MiB BAR1 of an RTX 3080.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from sglang.srt.distributed.device_communicators import (
    barlink_env_guard,  # noqa: F401  (rejects retired SGLANG_HTCCL* vars)
)

logger = logging.getLogger(__name__)

#: Requested receive-region size per rank, in MiB.
WINDOW_MIB_DEFAULT = 96

#: What stays untouched in BAR1. RM itself occupies part of the aperture,
#: and the number from sysfs is GROSS. 32 MiB is a default, not a
#: measurement -- whoever has NVML doesn't need it at all, because then the
#: real free size is already known.
RESERVE_MIB_DEFAULT = 32

#: Window ledger: per device ordinal, the list of ``(group, bytes)``
#: entries this PROCESS has already pinned in BAR1.
#:
#: It exists because the first group used to simply take whatever it
#: wanted. With ``SGLANG_UNEVEN_DCP=1`` there are two communicator groups;
#: ``tp`` grabbed 96 MiB, and ``dcp`` got a bare ``[Errno 12]`` from the
#: holder. An ENOMEM from an ioctl doesn't say WHO holds the space. This
#: table does.
_LEDGER: dict[int, list[tuple[str, int]]] = {}


def _ordinal(device) -> int:
    idx = getattr(device, "index", None)
    if idx is not None:
        return int(idx)
    import torch

    return int(torch.cuda.current_device())


def _group_key(group: str) -> str:
    """``dcp`` -> ``DCP``, ``tp:0`` -> ``TP_0``. For the variable name."""
    return "".join(c if c.isalnum() else "_" for c in group).upper()


def _requested(group: str) -> tuple[int, str]:
    """The requested region size for this group, and where it came from.

    **Independently configurable per group, and that's the whole point.**
    Different groups carry different messages: the tp group carries 20 MiB
    during prefill (chunked_prefill_size 2048 x hidden 5120 x 2 B), while
    the dcp group carries something else entirely. Giving them the same
    region means giving one too little or the other too much -- and on a
    3080 with 256 MiB gross, BAR1 is too tight to afford "too much".

    ``SGLANG_BARLINK_BAR1_WINDOW_MIB_DCP=16`` turns the dcp group's window
    down without touching the tp group's.
    """
    own = f"SGLANG_BARLINK_BAR1_WINDOW_MIB_{_group_key(group)}"
    if group and own in os.environ:
        return int(os.environ[own]) * 1024 * 1024, own
    return (
        int(os.environ.get("SGLANG_BARLINK_BAR1_WINDOW_MIB",
                           str(WINDOW_MIB_DEFAULT))) * 1024 * 1024,
        "SGLANG_BARLINK_BAR1_WINDOW_MIB",
    )


def bar1_free(device) -> tuple[Optional[int], int, str]:
    """``(free, gross, source)`` for this card's BAR1 aperture.

    ``free`` is ``None`` when it could not be determined -- in that case
    the caller must compute from ``gross`` minus reserve, and say so.
    Nothing is guessed here.

    NVML (``nvmlDeviceGetBAR1MemoryInfo``) is the only source that truly
    knows ``used``/``free``; sysfs only knows the aperture's gross size,
    and how much of that RM itself occupies is recorded nowhere. That is
    exactly the gap the holder's ENOMEM falls into.
    """
    gross = 0
    try:
        from sglang.srt.distributed.device_communicators.barlink_bar1 import (
            bar1_window,
        )
        from sglang.srt.distributed.device_communicators.barlink_matrix import (
            bdf_of_card,
        )

        gross = bar1_window(bdf_of_card(device)).size
    except Exception as e:
        logger.debug("barlink-BAR1: could not get BAR1 gross size from sysfs (%r)", e)
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(_ordinal(device))
            info = pynvml.nvmlDeviceGetBAR1MemoryInfo(h)
            return int(info.bar1Free), (gross or int(info.bar1Total)), "nvml"
        finally:
            pynvml.nvmlShutdown()
    except Exception as e:
        logger.debug("barlink-BAR1: NVML did not provide BAR1 usage (%r)", e)
    return None, gross, "sysfs-gross"


def ledger_credit(device, group: str, bytes_: int) -> None:
    _LEDGER.setdefault(_ordinal(device), []).append((group, int(bytes_)))


def ledger_debit(device, group: str) -> None:
    items = _LEDGER.get(_ordinal(device))
    if not items:
        return
    for i, (g, _) in enumerate(items):
        if g == group:
            items.pop(i)
            return


def ledger_balance(device) -> list[tuple[str, int]]:
    return list(_LEDGER.get(_ordinal(device), []))


def window_for(group: str, device) -> int:
    """The region size this group is allowed to request on THIS rank.

    A local PROPOSAL, not the final decision: the group's cards have
    apertures of different sizes (3080 with 256 MiB, 5090 with
    considerably more), and a per-rank-different region would mean a
    per-rank-different slot layout -- i.e. wrong addresses instead of a
    clean error. ``_build_up`` takes the minimum across the group.

    Computed against what is REALLY free, minus what this process has
    already pinned in other groups. Any shrinkage is logged, along with
    the arithmetic behind it -- a silent shrinkage would be worse than the
    ENOMEM: it quietly lowers the largest payload this group can carry,
    and messages about it fall back to the gloo layer without a single
    hint.
    """
    requested, source = _requested(group)
    free, gross, origin = bar1_free(device)
    reserve = int(os.environ.get("SGLANG_BARLINK_BAR1_RESERVE_MIB",
                                 str(RESERVE_MIB_DEFAULT))) * 1024 * 1024
    already = sum(b for _, b in ledger_balance(device))

    if free is None:
        if gross <= 0:
            logger.info(
                "barlink-BAR1: BAR1 size of this card is unknown (neither "
                "NVML nor sysfs). Requesting what %s says (%d MiB); if the "
                "aperture isn't big enough, the holder will report ENOMEM.",
                source, requested // 2**20,
            )
            return requested
        # sysfs only knows the GROSS size. What RM itself occupies isn't
        # recorded there -- which is why our own ledger has to be
        # subtracted here, and why this estimate is optimistic. This is
        # exactly what tripped up the second group with ENOMEM.
        cap = gross - reserve - already
        arithmetic = (
            f"BAR1 gross per sysfs {gross // 2**20} MiB - reserve "
            f"{reserve // 2**20} MiB - already pinned by this process "
            f"{already // 2**20} MiB = {max(cap, 0) // 2**20} "
            f"MiB. NOTE: sysfs only knows the gross aperture size; what RM "
            f"itself occupies is NOT subtracted from it. Without NVML this "
            f"number is an upper bound, not a guarantee."
        )
    else:
        # NVML knows `used` -- that already includes what this process has
        # pinned in other groups. So `already` must NOT be subtracted again
        # here; it only appears below for attribution.
        cap = free - reserve
        arithmetic = (
            f"free per NVML {free // 2**20} MiB - reserve "
            f"{reserve // 2**20} MiB = {max(cap, 0) // 2**20} MiB "
            f"(already includes what this process holds: "
            f"{', '.join(f'{g}: {b // 2**20} MiB' for g, b in ledger_balance(device)) or 'nothing'})"
        )

    if cap >= requested:
        return requested

    logger.warning(
        "barlink-BAR1: group %r requests %d MiB (%s), but only %d MiB is "
        "usable. %s The request is being clipped to %d MiB, which LOWERS "
        "the largest payload this group can carry -- messages about it "
        "fall back to the gloo layer without further notice. Anyone who "
        "doesn't want that should set SGLANG_BARLINK_BAR1_WINDOW_MIB_%s "
        "explicitly, give the other group less, or deliberately let this "
        "group run over NCCL.",
        group or "<unnamed>", requested // 2**20, source,
        max(cap, 0) // 2**20, arithmetic, max(cap, 0) // 2**20,
        _group_key(group),
    )
    return max(cap, 0)


def _window_bytes() -> int:
    """The old, group-less form. Only still used by callers without a
    group name (``benchmark/bar1_diag.py``, ``benchmark/bar1_graph_check.py``):
    there is exactly one group there, so there is nothing to split."""
    return int(os.environ.get("SGLANG_BARLINK_BAR1_WINDOW_MIB",
                              str(WINDOW_MIB_DEFAULT))) * 1024 * 1024


class BarlinkMatrixTransport:
    """Composite transport: plan plus one sub-path per operation and size.

    Today there is exactly **one** sub-path (BAR1) and two operations
    (``all_reduce``, ``all_to_all_single``). This is the honest version:
    mixing NIC and system-RAM edges per directed edge has been designed
    (``ENTWURF_PFADMATRIX.md``) but not measured, and building a selection
    scaffold for paths that don't exist would be a mock-up.

    The choice that genuinely exists is between the **kernels** of
    ``all_reduce``: ``mesh`` or ``ring`` per size, taken from the plan
    instead of a hardcoded number. ``all_to_all`` has no such choice --
    there is exactly one way to do it -- so it is passed through unplanned.
    """

    BARLINK_OPS: frozenset = frozenset(
        {"all_reduce", "all_to_all", "all_to_all_single"}
    )

    def __init__(self, cpu_group, device, group: str = ""):
        import torch.distributed as dist

        from sglang.srt.distributed.device_communicators.barlink_bar1 import (
            build_bar1,
        )
        from sglang.srt.distributed.device_communicators.barlink_matrix import (
            BarlinkMatrixPlanner,
            load_config,
        )

        self.cpu_group = cpu_group
        self.device = device
        self.group = group
        self.rank = dist.get_rank(cpu_group)
        self.world = dist.get_world_size(cpu_group)

        # 1. Direct path. `None` means: this machine can't have it, with a
        #    logged reason. No raising -- the planner is useful even
        #    without it. But the REASON is recorded: a planner without a
        #    direct path answers `handles -> False` to everything, and
        #    then every collective runs over the gloo layer while the log
        #    reports "transport=matrix". Exactly this mix-up once
        #    invalidated a measurement.
        report: dict = {}
        self.bar1 = build_bar1(
            cpu_group, device, window_for(group, device), report,
            group=group,
        )
        # ``holds_space`` is a key ``barlink_bar1.build_bar1`` writes; the
        # spelling is a cross-module contract, renamed on all three sides
        # together in #358.
        if report.get("holds_space") and self.bar1 is not None:
            # It's up, but not carrying anything (byte proof failed). Tear
            # it down instead of leaving it lying around: otherwise it
            # keeps holding the BAR1 pages the next group needs.
            self.bar1.close()
            self.bar1 = None
        self.bar1_reason = report.get("reason", "")
        self.bar1_stage = report.get("stage", "")

        # 2. Capability, fed to the planner. Minimum over ALL destinations
        #    and all ranks; `None` means "unknown" and rules nothing out --
        #    explicitly NOT "unlimited".
        window = self.bar1.window_minimum() if self.bar1 is not None else None

        # 3. Plan. The direct path doubles as the pair sensor when it's
        #    up: then the planner measures real directed edges.
        planner = BarlinkMatrixPlanner(
            cpu_group, device, config=load_config(),
            sensor=self.bar1, window_bytes=window,
        )
        self.plan = planner.plan()

        # 4. Explanation. Mandatory output, not gated behind a debug flag
        #    -- without it, nobody can debug this on unfamiliar hardware.
        #    Rank 0 only, because the plan is identical across the group
        #    (the checksum was reconciled while planning) and R copies of
        #    the same block would make the log unreadable.
        if self.rank == 0:
            logger.info("%s", self.plan.explanation())

        # 5. Feed the plan back into the direct path: from here on it
        #    picks mesh or ring from the same source the planner used to
        #    justify the choice.
        if self.bar1 is not None:
            self.bar1.set_plan(self.plan)
            logger.info(
                "barlink-Matrix: direct path is up. Mapped window "
                "%d KiB group-wide, largest payload %d KiB, ladder: %s.",
                (window or 0) // 1024, self.bar1.max_bytes // 1024,
                ", ".join(
                    (f"up to {s.max_bytes // 1024} KiB" if s.max_bytes > 0
                     else "above that") + f": {s.algorithm}"
                    for s in self.plan.ladder
                ),
            )
        else:
            logger.info(
                "barlink-Matrix: no direct path -- the plan is logged, but "
                "handles() returns False for everything. Nothing is sent "
                "over a path that doesn't exist."
            )

    # -- Transport seam ------------------------------------------------------

    def handles(self, op: str, nbytes: int) -> bool:
        """True exactly when a sub-path can actually run the operation.

        The decision is delegated to the sub-path instead of being
        reconstructed here: a second copy of the same condition would be
        exactly where the two could drift apart -- and a transport that
        says yes and then fails is worse than one that declines up front.
        """
        if op not in self.BARLINK_OPS or self.bar1 is None:
            return False
        return self.bar1.handles(op, nbytes)

    def barlink_all_reduce(self, comm, inp):
        self._must_hold()
        return self.bar1.barlink_all_reduce(comm, inp)

    # -- all_to_all --------------------------------------------------------
    #
    # The planner has NOTHING to say about this: it chooses between the
    # all_reduce decompositions (mesh/ring/star/hierarchical), and
    # all_to_all has no decomposition -- there is exactly one way to do
    # it, everyone writes their block to everyone else. So this is passed
    # straight through here, without consulting the plan. A plan line for
    # a choice that doesn't exist would be a mock-up.

    def supports_a2a(self, largest_block: int) -> bool:
        if self.bar1 is None:
            return False
        return self.bar1.supports_a2a(largest_block)

    def a2a_slot_bytes(self) -> int:
        return 0 if self.bar1 is None else self.bar1.a2a_slot_bytes()

    def barlink_all_to_all_single(self, comm, output, inp, send_bytes,
                                recv_bytes, send_offsets=None,
                                recv_offsets=None, kernel_bytes=None,
                                rounds=None):
        self._must_hold()
        return self.bar1.barlink_all_to_all_single(
            comm, output, inp, send_bytes, recv_bytes,
            send_offsets, recv_offsets, kernel_bytes=kernel_bytes,
            rounds=rounds,
        )

    def a2a_rounds_for(self, largest_block: int) -> int:
        """Passed through the same way as supports_a2a -- a second copy
        would be exactly where the two could drift apart."""
        if self.bar1 is None:
            return 0
        return self.bar1.a2a_rounds_for(largest_block)

    def _must_hold(self) -> None:
        if self.bar1 is None:
            raise NotImplementedError(
                "The matrix transport today has exactly one sub-path "
                "(BAR1), and it isn't up. This line is only reachable if "
                "someone bypassed handles()."
            )

    def close(self) -> None:
        if self.bar1 is not None:
            self.bar1.close()
            self.bar1 = None
