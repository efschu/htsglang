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


class Bar1WindowRefused(RuntimeError):
    """An explicitly requested BAR1 window does not fit into the aperture.

    Deliberately fatal. The alternative -- serving a smaller window than the
    operator asked for -- is the failure this class exists to remove: it
    silently rewrites the input to every downstream size decision.
    """


#: Windows that came out SMALLER than requested, per group. Only the default
#: request can land here; an explicit one raises instead (``window_for``).
#:
#: This exists because the clip is otherwise invisible in the place people
#: actually look. ``report_state``/``state_summary`` report which TRANSPORT a
#: group achieved, and a clipped group still achieves bar1 -- truthfully, but
#: on a smaller window than the configuration asked for. Without this table
#: the two cases print identically, and the run's round counts have no
#: recorded cause.
_CLIPS: dict[str, dict] = {}


def record_clip(group: str, requested: int, granted: int, source: str,
                arithmetic: str) -> None:
    _CLIPS[group or "<unnamed>"] = {
        "group": group or "<unnamed>",
        "requested_bytes": int(requested),
        "granted_bytes": int(granted),
        "source": source,
        "arithmetic": arithmetic,
    }


def window_clips() -> dict[str, dict]:
    """Groups whose BAR1 window was reduced below the requested size."""
    return dict(_CLIPS)


def reset_clips_for_test() -> None:
    _CLIPS.clear()


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


def nvml_card_for_device(device, bdf: str = ""):
    """The physical card behind ``device``, or a named error (#406).

    A torch device carries a CUDA ordinal, and an NVML handle is addressed
    by an NVML index. The two enumerations are different orderings of the
    same cards -- on this rig the 5090 is CUDA ordinal 0 and NVML index 1
    -- so passing the ordinal to ``nvmlDeviceGetHandleByIndex`` reads
    another card's numbers. Resolution therefore goes through the #331
    identity map: over the PCI address when the device could name one
    (the identity that survives ``CUDA_VISIBLE_DEVICES``), otherwise over
    the map's CUDA-ordinal side.

    Raises ``DeviceOrderUnresolvedError`` when neither route places the
    card. There is no fall-back to the index of the same number: this is
    the sizing input for a BAR1 window, and a window sized from a card
    other than the one that will host it is the failure mode the identity
    map exists to remove.
    """
    from sglang.srt.registry.nvml import (
        DeviceOrderUnresolvedError,
        cuda_bridge_diagnosis,
        identity_map,
    )

    # ``allow_cuda_init`` is free here: bar1_free is only ever asked about a
    # device this process has already built a context on.
    imap = identity_map(allow_cuda_init=True)
    card = imap.by_pci_bus_id(bdf) if bdf else None
    ordinal = None
    if card is None:
        try:
            ordinal = _ordinal(device)
        except Exception as e:  # noqa: BLE001 - no torch device, no ordinal
            logger.debug("barlink-BAR1: device carries no usable ordinal (%r)", e)
        if ordinal is not None:
            card = imap.by_cuda_ordinal(ordinal)
    if card is None:
        raise DeviceOrderUnresolvedError(
            "barlink-BAR1 window sizing: the physical card behind "
            f"{device!r} could not be identified (pci address "
            f"{bdf or 'unknown'}, CUDA ordinal "
            f"{'unknown' if ordinal is None else ordinal}). NVML reports:\n  "
            + ("\n  ".join(c.describe() for c in imap) or "no devices")
            + f"\nReason: {cuda_bridge_diagnosis()}"
        )
    return card


def bar1_free(device) -> tuple[Optional[int], int, str]:
    """``(free, gross, source)`` for this card's BAR1 aperture.

    ``free`` is ``None`` when it could not be determined -- in that case
    the caller must compute from ``gross`` minus reserve, and say so.
    Nothing is guessed here.

    NVML (``nvmlDeviceGetBAR1MemoryInfo``) is the only source that truly
    knows ``used``/``free``; sysfs only knows the aperture's gross size,
    and how much of that RM itself occupies is recorded nowhere. That is
    exactly the gap the holder's ENOMEM falls into.

    Both readings are addressed by the card's PCI address rather than by
    the device's ordinal (#406): sysfs already was, NVML now is. When the
    card cannot be identified at all, the NVML read is SKIPPED and the
    refusal is logged by name -- the answer degrades to "free unknown",
    which the caller already handles and states in its arithmetic, rather
    than to another card's free bytes, which it could not detect.
    """
    from sglang.srt.registry.nvml import DeviceOrderUnresolvedError, nvml_session

    bdf = ""
    try:
        from sglang.srt.distributed.device_communicators.barlink_matrix import (
            bdf_of_card,
        )

        bdf = bdf_of_card(device)
    except Exception as e:
        logger.debug("barlink-BAR1: could not resolve the PCI address (%r)", e)

    gross = 0
    if bdf:
        try:
            from sglang.srt.distributed.device_communicators.barlink_bar1 import (
                bar1_window,
            )

            gross = bar1_window(bdf).size
        except Exception as e:
            logger.debug(
                "barlink-BAR1: could not get BAR1 gross size from sysfs (%r)", e
            )

    try:
        card = nvml_card_for_device(device, bdf=bdf)
    except DeviceOrderUnresolvedError as e:
        logger.warning(
            "barlink-BAR1: NVML was NOT asked for this card's free BAR1 "
            "space, because it could not be told which card that is. The "
            "window is sized from the gross aperture instead. %s",
            e,
        )
        return None, gross, "sysfs-gross"
    except Exception as e:
        logger.debug("barlink-BAR1: card identity unavailable (%r)", e)
        return None, gross, "sysfs-gross"

    try:
        with nvml_session() as pynvml:
            h = pynvml.nvmlDeviceGetHandleByIndex(card.nvml_index)
            info = pynvml.nvmlDeviceGetBAR1MemoryInfo(h)
            return int(info.bar1Free), (gross or int(info.bar1Total)), "nvml"
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

    granted = max(cap, 0)

    # THE INVARIANT: a number the operator wrote down is never silently
    # lowered. `_requested` falls back to a default when neither env var is
    # set, and shrinking THAT is an adaptation. Shrinking a value somebody
    # typed is overriding an instruction, and on an explicitly requested
    # direct path that is the C1 shape -- a warning on a promise path. So an
    # explicit request that does not fit refuses here, at group build, where
    # the arithmetic is still on hand, instead of degrading later where only
    # the symptom shows.
    if source in os.environ:
        raise Bar1WindowRefused(
            f"barlink-BAR1: group {group or '<unnamed>'!r} was explicitly "
            f"given {requested // 2**20} MiB via {source}, but only "
            f"{granted // 2**20} MiB of BAR1 is available on this card. "
            f"{arithmetic} This is NOT clipped to fit: an explicit window is "
            f"an instruction, and silently serving a smaller one would make "
            f"every later size decision of this group rest on a number "
            f"nobody chose. Either lower {source}, give the other groups "
            f"less (their windows are listed in the arithmetic above), or "
            f"raise SGLANG_BARLINK_BAR1_RESERVE_MIB's counterpart by freeing "
            f"BAR1 elsewhere."
        )

    # The default was not met. This is recorded rather than merely logged:
    # the group still comes up, and its "ACHIEVED=bar1" line would otherwise
    # read exactly like a group whose window was granted in full.
    #
    # What the clip actually costs, stated precisely -- the previous wording
    # here claimed messages "fall back to the gloo layer without further
    # notice", and that has been wrong since the round decompositions landed:
    # all_reduce (`_handles_all_reduce`) and all_to_all (`_handles_a2a`) both
    # split an oversized payload into rounds instead of declining it. A
    # smaller window therefore costs ROUNDS first; the direct path is only
    # declined once the round caps (SGLANG_BARLINK_BAR1_AR_MAX_ROUNDS /
    # _A2A_MAX_ROUNDS, 16 each) are exceeded. And when that does happen, it is
    # not silent either: outside a capture the dispatcher warns once per
    # operation and size class, and inside a CUDA-graph capture it raises
    # (barlink.py, the graph gate) rather than staging over the host.
    record_clip(group, requested, granted, source, arithmetic)
    logger.warning(
        "barlink-BAR1: group %r requests %d MiB (%s, the default), but only "
        "%d MiB is usable. %s The window is reduced to %d MiB. This lowers "
        "the payload this group carries per ROUND: all_reduce and all_to_all "
        "decompose oversized payloads into rounds, so the direct path is kept "
        "and pays in launches, not in coverage -- until a payload needs more "
        "than the round cap allows, at which point the dispatcher warns "
        "(outside a capture) or raises (inside one). It never degrades "
        "unreported. Set SGLANG_BARLINK_BAR1_WINDOW_MIB_%s explicitly to get "
        "a refusal instead of this reduction, or give the other groups less.",
        group or "<unnamed>", requested // 2**20, source,
        granted // 2**20, arithmetic, granted // 2**20,
        _group_key(group),
    )
    return granted


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
