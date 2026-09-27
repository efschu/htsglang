# SPDX-License-Identifier: Apache-2.0
"""WEG2-MEMHIST: torch allocation history with stacks, as a BOOKED instrument.

The residue instrument (``SGLANG_WEG2_MEMHIST=1``, weg2_memory_saver) arms
``torch.cuda.memory._record_memory_history(max_entries=200000)`` at rank start
in every group that sees the variable. rc12d (27.09. 02:06Z, set in the
container env, so P and D): P's ``memory.current`` stood at 67.7 GiB instead of
rc12c's 62.3 at the same point -- the history over the whole weight load costs
~5.4 GiB of host for three P ranks (~9.4 KiB per recorded entry and rank) --
and D was refused by its pinned-host riegel before it started. Nobody had
booked it.

The LEAN form (``SGLANG_WEG2_MEMHIST=sleep``) for finding the unnamed untagged
residue of D (rc12c: 2036 -> 2340 MiB untagged_live over 13 sleeps):

* only the groups in ``SGLANG_WEG2_MEMHIST_GROUPS`` (default ``D``);
* armed at the END of the rank's first sleep -- after the load, after the
  first serving phase -- so the load's hundreds of thousands of allocations
  are never recorded; blocks born before still appear in the snapshot with
  their sizes, only without a stack; everything allocated from the first wake
  on (the growth) carries one;
* ``SGLANG_WEG2_MEMHIST_MAX_ENTRIES`` (default 20000) caps the ring.

The launcher books the cost into the host ledger (post ``memhist``): ranks x
entries x the measured per-entry cost, at BOTH moments for the legacy form
(armed from the start) and at the RUN moment only for the lean one (it arms
after the first sleep). Pure: stdlib only, read by the launcher and the ranks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

MEMHIST_ENV = "SGLANG_WEG2_MEMHIST"
GROUPS_ENV = "SGLANG_WEG2_MEMHIST_GROUPS"
MAX_ENTRIES_ENV = "SGLANG_WEG2_MEMHIST_MAX_ENTRIES"

MODE_LOAD = "load"    # "1": from rank start (the pre-existing form)
MODE_SLEEP = "sleep"  # lean: from the end of the first sleep

LEGACY_MAX_ENTRIES = 200000
LEAN_MAX_ENTRIES = 20000
LEAN_GROUPS: Tuple[str, ...] = ("D",)
ALL_GROUPS: Tuple[str, ...] = ("P", "D")

#: MEASURED rc12d (27.09.): P memory.current 67.7 GiB with the history armed
#: from rank start vs 62.3 GiB in rc12c at the same point, three P ranks at a
#: full 200000-entry ring: 5.4 GiB / (3 x 200000) = 9.44 KiB per entry and rank.
HOST_KIB_PER_ENTRY = 5.4 * 1024 * 1024 / (3 * LEGACY_MAX_ENTRIES)
HOST_KIB_PER_ENTRY_SOURCE = "rc12d P memory.current 67.7 vs rc12c 62.3 GiB, 3 ranks x 200000 entries"


@dataclass(frozen=True)
class MemhistPlan:
    mode: str
    max_entries: int
    groups: Tuple[str, ...]

    def armed_for(self, group: str) -> bool:
        return str(group or "").strip().upper() in self.groups

    @property
    def run_moment_only(self) -> bool:
        return self.mode == MODE_SLEEP

    def host_gib(self, ranks_by_group: Mapping[str, int]) -> float:
        ranks = sum(int(n) for g, n in ranks_by_group.items() if self.armed_for(g))
        return ranks * self.max_entries * HOST_KIB_PER_ENTRY / (1024.0 * 1024.0)

    def describe(self) -> str:
        return "mode=%s groups=%s max_entries=%d" % (self.mode, ",".join(self.groups), self.max_entries)


def plan(environ: Mapping[str, str]) -> Optional[MemhistPlan]:
    """The instrument this environment arms, or ``None`` (off)."""
    raw = str(environ.get(MEMHIST_ENV, "") or "").strip().lower()
    if raw in ("1", "true", "on", MODE_LOAD):
        mode, groups, entries = MODE_LOAD, ALL_GROUPS, LEGACY_MAX_ENTRIES
    elif raw == MODE_SLEEP:
        mode, groups, entries = MODE_SLEEP, LEAN_GROUPS, LEAN_MAX_ENTRIES
    else:
        return None
    g_raw = str(environ.get(GROUPS_ENV, "") or "").strip()
    if g_raw:
        groups = tuple(x.strip().upper() for x in g_raw.split(",") if x.strip())
    e_raw = str(environ.get(MAX_ENTRIES_ENV, "") or "").strip()
    if e_raw:
        try:
            entries = max(1, int(e_raw))
        except ValueError:
            pass
    return MemhistPlan(mode=mode, max_entries=int(entries), groups=groups)


def group_plan(environ: Mapping[str, str], group_env: Mapping[str, str]) -> Optional[MemhistPlan]:
    """The plan a group's ranks see: the launcher's environment (container
    env) overlaid by the group's own ``--env-p/--env-d`` entries."""
    merged = dict(environ)
    merged.update(group_env)
    return plan(merged)


def host_charge(
    environ: Mapping[str, str],
    group_envs: Mapping[str, Mapping[str, str]],
    ranks_by_group: Mapping[str, int],
) -> Tuple[float, bool, str]:
    """``(gib, run_moment_only, provenance)`` of the ledger post ``memhist``
    for this boot; ``(0.0, False, "")`` when no group arms it."""
    total = 0.0
    run_only = True
    parts = []
    for g, genv in group_envs.items():
        p = group_plan(environ, genv)
        if p is None or not p.armed_for(g):
            continue
        gib = p.host_gib({g: ranks_by_group.get(g, 0)})
        total += gib
        run_only = run_only and p.run_moment_only
        parts.append("%s %s %.2f GiB" % (g, p.describe(), gib))
    if not parts:
        return 0.0, False, ""
    return total, run_only, "; ".join(parts) + " (%.2f KiB/entry, %s)" % (
        HOST_KIB_PER_ENTRY, HOST_KIB_PER_ENTRY_SOURCE)
