# SPDX-License-Identifier: Apache-2.0
"""Slice 5a, runtime half: the Form-A allocations actually go under the tags.

``flip_form_a_sleep`` is the TABLE -- which tag each Form-A allocation wears,
per role, and in which order the tags are paused and resumed. This module is
the MECHANISM that uses it: a hook a Form-A rank holds, whose ``region(name)``
is the one door every taggable device allocation goes through, and whose
``sleep()`` / ``wake()`` drive the adapter in the table's order.

WHY A DOOR AND NOT A CALL SITE PER ALLOCATION. The failure this closes is not
"a tag was wrong", it is "an allocation had no tag at all" -- and an
allocation with no tag is INVISIBLE to a call-site review: nothing is written
where nothing was done. Routing every allocation through ``region(name)``
turns the absent case into a positive one: the hook knows what it handed out,
so ``sleep()`` can compare the live inventory against the table and refuse
(W118) instead of pausing a proper subset and grading it as a whole sleep.

THREE HAZARDS FROM THE 27B PATH, CARRIED OVER BECAUSE THEY ARE THE SAME
HAZARDS, NOT BECAUSE THE CODE LOOKS ALIKE:

  1. FLUSH BEFORE THE KV PAUSE (``weight_updater.py:6020-6021``, a MUST_FIX).
     The mamba/GDN anchors live under the ``kv_cache`` tag -- there is no
     separate mamba tag (``memory_pool.py:1017``). Pausing before the flush
     leaves the pool's index tables pointing into pages the tag no longer
     owns.

  2. FIT BEFORE THE KV RESUME (``weg2/wake_kv.kv_resume_fit_refusal``, #1490).
     The saver's resume ABI returns VOID: a resume that fails on a device OOM
     rolls the whole tag back and tells Python nothing. On Form A the host
     rank maps 3.45 GiB of KV onto a 5090 that PP3 released ONE RPC earlier,
     so "the card has not caught up yet" is the normal case, not the rare
     one. Asking first turns a silent zeroing of unmapped pages into a named
     refusal. The corridor floor is REPORTED and never subtracted
     (Memory ``KEINE-KORRIDOR-RESERVE-NIE``).

  3. PAUSE/RESUME UNDER ``pause_polling``
     (``torch_memory_saver_adapter._abort_poll_excluded``). The barlink
     abort-word watchdog reads a device pointer from ANOTHER THREAD every
     10 ms, and between pause and resume those pages are unmapped. This hook
     does NOT re-wrap it: the adapter already holds the exclusion at the one
     chokepoint every tag pause and resume goes through, and a second wrapper
     here would be a claim rather than a mechanism. :meth:`sleep` asserts the
     adapter has it, so a downgraded adapter is a refusal and not a silence.

WHAT FORM A DOES *NOT* INHERIT FROM THE 27B PATH, stated so the absence is a
decision rather than a gap: the DEPOSIT-BEFORE-PAUSE half of the exchange
ordering (``_weg2_xchg_deposit_before_sleep``). Form A exchanges no weight
bytes with the P group -- the experts stay in the shared page-locked HOST pool
across the flip (seam B), and a BAR1 ring would transport what is meant to lie
still. There is therefore nothing to deposit, and :meth:`sleep` says so in its
log line rather than leaving a reader to wonder which of the two orderings was
forgotten.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.flip_form_a_sleep import (
    FORM_A_ALLOCATIONS,
    ROLE_HOST,
    ROLE_WORKER,
    FormAAllocation,
    SleepHookPlan,
    Weg2FlipFormAUntagged,
    plan_form_a_sleep,
    tags_for_role,
)

__all__ = [
    "FormASleepHook",
    "SleepReport",
    "FORM_A_SLEEP_MARKER",
    "FORM_A_WAKE_MARKER",
    "resolve_rpc_tags",
    "assert_group_agrees",
]

FORM_A_SLEEP_MARKER = "FORM-A-SLEEP"
FORM_A_WAKE_MARKER = "FORM-A-WAKE"


@dataclass
class SleepReport:
    """What one sleep or wake actually did -- per tag, with bytes and ms."""

    direction: str
    role: str
    rank: int
    steps: List[Tuple[str, Optional[int], float]] = field(default_factory=list)
    refused: Optional[str] = None

    def line(self) -> str:
        marker = FORM_A_SLEEP_MARKER if self.direction == "sleep" else FORM_A_WAKE_MARKER
        body = " ".join(
            f"{tag}={'?' if b is None else b >> 20}MiB/{ms:.0f}ms"
            for tag, b, ms in self.steps
        )
        tail = f" REFUSED={self.refused}" if self.refused else ""
        return (
            f"{marker} role={self.role} rank={self.rank} "
            f"tags={len(self.steps)} {body}{tail}"
        )

    @property
    def total_bytes(self) -> int:
        return sum(b for _, b, _ in self.steps if b)


class FormASleepHook:
    """One Form-A rank's sleep/wake hook.

    Everything external is INJECTED -- the adapter, the free-memory probe, the
    flush, the clock, the logger -- so the whole thing runs against stubs at
    the desk. That is not a testing convenience: the hazards above are all
    ORDERING hazards, and ordering is exactly what a stub can prove and a GPU
    cannot prove cheaply.
    """

    def __init__(
        self,
        adapter,
        role: str,
        rank: int,
        weight_chunk_tags: Sequence[str] = (),
        free_bytes_fn: Optional[Callable[[], Optional[int]]] = None,
        flush_fn: Optional[Callable[[], bool]] = None,
        corridor_floor_bytes: int = 0,
        logger=None,
        clock: Callable[[], float] = time.perf_counter,
        table: Sequence[FormAAllocation] = FORM_A_ALLOCATIONS,
    ) -> None:
        self.adapter = adapter
        self.role = role
        self.rank = int(rank)
        self.weight_chunk_tags = tuple(weight_chunk_tags)
        self.free_bytes_fn = free_bytes_fn
        self.flush_fn = flush_fn
        self.corridor_floor_bytes = int(corridor_floor_bytes)
        self.logger = logger
        self.clock = clock
        self.table = tuple(table)
        self._by_name: Dict[str, FormAAllocation] = {a.name: a for a in self.table}
        self._registered: List[str] = []
        self._asleep = False
        # Validates the role up front: an unknown role must fail at
        # construction, not at the first flip.
        self._role_tags = tags_for_role(role)

    # ------------------------------------------------------------------
    # The one door
    # ------------------------------------------------------------------
    @contextlib.contextmanager
    def region(self, name: str):
        """Allocate inside ``name``'s memory-saver tag.

        Refuses an unknown name (W118) rather than falling through untagged --
        falling through is the exact failure this module exists to prevent,
        and it would be invisible.
        """
        alloc = self._by_name.get(name)
        if alloc is None:
            raise Weg2FlipFormAUntagged(
                f"W118 Weg2FlipFormAUntagged -- no Form-A allocation named "
                f"{name!r}. Every device allocation of this layout goes "
                f"through a named region so the sleep can prove it covered "
                f"all of them; an unnamed one would survive the sleep and be "
                f"invisible to review. Add it to FORM_A_ALLOCATIONS with its "
                f"role set and the log line that proves it."
            )
        if self.role not in alloc.roles:
            raise Weg2FlipFormAUntagged(
                f"W118 Weg2FlipFormAUntagged -- role {self.role!r} opened "
                f"region {name!r}, which the Form-A plan gives to "
                f"{sorted(alloc.roles)}. An allocation on the wrong side is a "
                f"layout bug, not a tagging one (rank_role.py:459-493)."
            )
        self._registered.append(name)
        with self.adapter.region(tag=alloc.tag):
            yield alloc.tag

    def registered(self) -> Tuple[str, ...]:
        return tuple(self._registered)

    def plan(self) -> SleepHookPlan:
        """The table's verdict on what this rank has actually built."""
        return plan_form_a_sleep(
            self.role, self._registered, self.weight_chunk_tags, self.table
        )

    # ------------------------------------------------------------------
    # Sleep
    # ------------------------------------------------------------------
    def _assert_exclusion(self) -> None:
        """The adapter must hold the watchdog off; this hook does not.

        A second wrapper here would be a CLAIM. Checking that the adapter has
        the mechanism is a MECHANISM -- and a downgraded adapter then refuses
        instead of silently racing the abort-word poll.
        """
        mod = type(self.adapter).__module__ or ""
        if mod.startswith("sglang."):
            from sglang.srt.utils import torch_memory_saver_adapter as tms

            if not hasattr(tms, "_abort_poll_excluded"):
                raise Weg2FlipFormAUntagged(
                    "W118 Weg2FlipFormAUntagged -- the memory-saver adapter has "
                    "no _abort_poll_excluded: pause/resume would run while the "
                    "barlink abort-word watchdog reads device pointers from "
                    "another thread, against pages this call unmaps "
                    "(torch_memory_saver_adapter.py, #1489). Do not wrap it "
                    "here -- restore it at the adapter, the one chokepoint "
                    "every tag pause and resume goes through."
                )

    def sleep(self) -> SleepReport:
        """Give this rank's device memory back, in the table's order."""
        if self._asleep:
            raise Weg2FlipFormAUntagged(
                f"W118 Weg2FlipFormAUntagged -- rank {self.rank} is already "
                f"asleep. A second sleep would pause tags whose allocations "
                f"are unmapped and grade the result against a residency that "
                f"is not there."
            )
        self._assert_exclusion()
        plan = self.plan()  # W118 for an untagged or misplaced live allocation
        report = SleepReport("sleep", self.role, self.rank)

        for tag in plan.sleep_order:
            if tag == GPU_MEMORY_TYPE_KV_CACHE:
                # HAZARD 1: flush BEFORE the pause, never after.
                if self.flush_fn is not None:
                    if not self.flush_fn():
                        report.refused = "flush_before_kv_pause"
                        self._log(report)
                        raise Weg2FlipFormAUntagged(
                            f"W118 Weg2FlipFormAUntagged -- flush_cache() "
                            f"refused before pause({tag}) on rank {self.rank}. "
                            f"The mamba/GDN anchors live under this tag "
                            f"(memory_pool.py:1017); pausing now leaves the "
                            f"pool's index tables pointing into pages the tag "
                            f"no longer owns. The group must be drained first."
                        )
            t0 = self.clock()
            nbytes = self._tag_bytes(tag)
            self.adapter.pause(tag)
            report.steps.append((tag, nbytes, (self.clock() - t0) * 1000.0))

        self._asleep = True
        self._log(report)
        return report

    # ------------------------------------------------------------------
    # Wake
    # ------------------------------------------------------------------
    def wake(self) -> SleepReport:
        """Map this rank's device memory back, graph first and KV last."""
        if not self._asleep:
            raise Weg2FlipFormAUntagged(
                f"W118 Weg2FlipFormAUntagged -- rank {self.rank} is not "
                f"asleep; a wake would resume tags that were never paused."
            )
        self._assert_exclusion()
        plan = self.plan()
        report = SleepReport("wake", self.role, self.rank)

        for tag in plan.wake_order:
            nbytes = self._tag_bytes(tag)
            if tag == GPU_MEMORY_TYPE_KV_CACHE:
                # HAZARD 2: ask the card BEFORE the void-ABI resume.
                from sglang.srt.weg2.wake_kv import kv_resume_fit_refusal

                free = self._free_bytes()
                why = kv_resume_fit_refusal(
                    free, nbytes, self.corridor_floor_bytes
                )
                if why is not None:
                    report.refused = f"kv_fit: {why}"
                    self._log(report)
                    raise Weg2FlipFormAUntagged(
                        f"W118 Weg2FlipFormAUntagged -- rank {self.rank} "
                        f"cannot resume {tag}: {why}. The saver's resume ABI "
                        f"returns VOID, so a failed mapping would roll the tag "
                        f"back and report success, and the wake would then "
                        f"zero an unmapped pool (boot weg2xsn408: TP0 and TP1 "
                        f"died there with no traceback). On Form A the host "
                        f"maps its KV onto the card the P group released one "
                        f"RPC earlier -- if the release has not landed, this "
                        f"is the normal case, and the answer is to wait for "
                        f"the credit, not to resume anyway."
                    )
            t0 = self.clock()
            # The adapter verifies the landing itself and raises
            # Weg2TmsResumeRefused (W119) when free memory did not move.
            self.adapter.resume(tag)
            report.steps.append((tag, nbytes, (self.clock() - t0) * 1000.0))

        self._asleep = False
        self._log(report)
        return report

    # ------------------------------------------------------------------
    def _tag_bytes(self, tag: str) -> Optional[int]:
        try:
            value = self.adapter.tag_bytes(tag)
        except Exception:  # noqa: BLE001 -- an absent probe is not a verdict
            return None
        return None if value is None else int(value)

    def _free_bytes(self) -> Optional[int]:
        if self.free_bytes_fn is None:
            return None
        try:
            value = self.free_bytes_fn()
        except Exception:  # noqa: BLE001
            return None
        return None if value is None else int(value)

    def _log(self, report: SleepReport) -> None:
        if self.logger is None:
            return
        try:
            self.logger.info("%s", report.line())
        except Exception:  # noqa: BLE001 -- an instrument never kills a boot
            pass

    @property
    def asleep(self) -> bool:
        return self._asleep

    def deposit_note(self) -> str:
        """Why there is no deposit step, stated instead of missing."""
        return (
            "Form A deposits nothing before its pause: the experts stay in the "
            "shared page-locked HOST pool across the flip (seam B), so there "
            "are no weight bytes to hand to the peer and a BAR1 ring would "
            "transport what is meant to lie still."
        )


# ==========================================================================
# The Sleep/Wake RPC for the Form-A group -- endpoints as on the P/D side
# ==========================================================================
# The group is an ordinary TP group, so ``ReleaseMemoryOccupationReqInput`` /
# ``ResumeMemoryOccupationReqInput`` already REACH it through the registration
# at ``scheduler.py:2776-2781``. What does not already exist is the ROLE step:
# those requests carry ONE ``tags`` list for the whole group, and under Form A
# the three ranks do not own the same tags. A request that names ``kv_cache``
# arrives identically at the host (which holds 3.45 GiB of it) and at the two
# workers (which hold a 768 B stub). Letting each rank do what it can with the
# list is precisely how the ranks end up disagreeing about what the group just
# did -- Memory ``RAENGE-NIE-UNEINS``: disagreement is a CRASH/STOP, not a
# per-rank best effort.
def resolve_rpc_tags(role: str, requested: Optional[Sequence[str]]) -> List[str]:
    """The tags THIS rank pauses for a group-wide release/resume request.

    ``requested=None`` means "the whole population", which under Form A is
    role-dependent and is resolved here rather than defaulted to
    ``GPU_MEMORY_ALL_TYPES`` -- a worker handed the full list would pause an
    empty ``kv_cache`` region and grade its sleep against a denominator that
    is not there.

    A request that names a tag this ROLE does not own is REFUSED, not
    filtered. Filtering is the silent form: the host would pause three tags,
    the workers two, both answer OK, and the front would read one verdict for
    two different acts. The refusal names which rank and which tag, so the
    caller fixes the request instead of the symptom.
    """
    owned = tags_for_role(role)
    if requested is None:
        return sorted(owned)
    asked = [str(t) for t in requested]
    foreign = sorted({t for t in asked if t not in owned})
    if foreign:
        raise Weg2FlipFormAUntagged(
            f"W118 Weg2FlipFormAUntagged -- a release/resume request named "
            f"tag(s) {foreign} that Form-A role {role!r} does not own (it owns "
            f"{sorted(owned)}). Not filtered away on purpose: a host that "
            f"pauses three tags and a worker that pauses two, both answering "
            f"OK, give the front ONE verdict for TWO different acts. Send the "
            f"role's tags, or send none and let each rank resolve its own "
            f"(resolve_rpc_tags)."
        )
    return sorted({t for t in asked})


def assert_group_agrees(reports: Sequence[Tuple[int, str, Sequence[str]]]) -> None:
    """Every rank did what its ROLE says, and the roles are the expected set.

    ``reports`` is ``(rank, role, tags_acted_on)`` gathered over the group --
    the same shape the P/D fence already gathers
    (``ReleaseMemoryOccupationReqOutput.per_tag`` is reduced by the group
    fence's ``all_gather_object``). Called after the leg, on the front.

    Two ways to disagree, both refused by name: a rank that acted on tags its
    role does not own, and a group that is not exactly one host plus workers.
    """
    if not reports:
        raise Weg2FlipFormAUntagged(
            "W118 Weg2FlipFormAUntagged -- an empty gather is not agreement"
        )
    hosts = [r for r, role, _ in reports if role == ROLE_HOST]
    if len(hosts) != 1:
        raise Weg2FlipFormAUntagged(
            f"W118 Weg2FlipFormAUntagged -- the group reports {len(hosts)} "
            f"attention host(s) {hosts} across {len(reports)} rank(s). Form A "
            f"has exactly one (rank_role.py:459-478); any other count means "
            f"the ranks disagree about the layout they just slept."
        )
    bad = []
    for rank, role, tags in reports:
        owned = tags_for_role(role)
        extra = sorted({str(t) for t in tags} - owned)
        if extra:
            bad.append((rank, role, extra))
    if bad:
        detail = "; ".join(f"rank {r} ({ro}) acted on {ex}" for r, ro, ex in bad)
        raise Weg2FlipFormAUntagged(
            f"W118 Weg2FlipFormAUntagged -- {detail}. The ranks are not "
            f"agreed on what this leg did; serving on a group whose halves "
            f"believe different things about their residency is the failure "
            f"this refusal exists to stop (Memory RAENGE-NIE-UNEINS)."
        )
