# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# ==============================================================================
"""Named refusals -- the vocabulary the turnkey boot path fails in.

#539's requirement is "no LLM guesswork and no manual reconstruction". The
enforcement of the first half is this module: every way the boot path can
decline to start carries a STABLE NAME and the measured numbers that produced
it. A refusal is never a log line saying something looked wrong; it is a
:class:`Refusal` with a name an operator can grep for, the observed value, the
expected value, and a remedy sentence.

Why names rather than exit codes or messages: the boot path is read by three
audiences with different needs -- a human at 03:00, a systemd journal filter,
and a later agent reconstructing what happened. A free-text message serves
only the first. ``REFUSE_WHEEL_SHADOW`` serves all three and survives
rewording of the human sentence.

The cardinal rule, from #539's spec: a refusal is LOUD and TERMINAL. Where the
turnkey path cannot establish a fact it needs, it refuses. It never
substitutes a default, never rounds, never "probably". A stack that boots on a
guess is the defect this feature exists to remove -- a wrong boot is strictly
worse than no boot, because a wrong boot looks like success.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

__all__ = [
    "Refusal",
    "RefusalError",
    "NAMES",
    "REFUSE_CONFIG_UNPARSABLE",
    "REFUSE_CONFIG_INCOMPLETE",
    "REFUSE_REPO_IS_WORKTREE",
    "REFUSE_PATH_MISSING",
    "REFUSE_WHEEL_SHADOW",
    "REFUSE_CARD_CENSUS",
    "REFUSE_CARD_UNKNOWN_UUID",
    "REFUSE_CARD_BUSY",
    "REFUSE_HOST_HEADROOM",
    "REFUSE_DISK_HEADROOM",
    "REFUSE_PORT_BUSY",
    "REFUSE_PLAN_MISSING",
    "REFUSE_PLAN_STALE",
    "REFUSE_PLAN_UNSOLVABLE",
    "REFUSE_LOG_PATH_SHARED",
]

# --- the vocabulary -------------------------------------------------------
# Kept as module constants rather than an enum so that a unit file, a shell
# script and a grep all spell the name the same way.

REFUSE_CONFIG_UNPARSABLE = "REFUSE_CONFIG_UNPARSABLE"
REFUSE_CONFIG_INCOMPLETE = "REFUSE_CONFIG_INCOMPLETE"
#: The canonical repo path points at a git WORKTREE. Observed on this rig
#: 2026-08-12: the planner daemon on :8780 was running with cwd
#: /spinning/wt-631-routea, i.e. a service whose working directory can be
#: deleted by ``git worktree remove``. A turnkey stack must be rooted in the
#: canonical checkout.
REFUSE_REPO_IS_WORKTREE = "REFUSE_REPO_IS_WORKTREE"
REFUSE_PATH_MISSING = "REFUSE_PATH_MISSING"
#: #384 wheel-shadow: two distributions provide the same import name and a
#: plain ``pip install`` silently drops the INT8 arm. Verified BEFORE boot,
#: because the symptom otherwise appears as a quality regression hours later.
REFUSE_WHEEL_SHADOW = "REFUSE_WHEEL_SHADOW"
#: The set of cards NVML reports does not match the set the config names.
REFUSE_CARD_CENSUS = "REFUSE_CARD_CENSUS"
REFUSE_CARD_UNKNOWN_UUID = "REFUSE_CARD_UNKNOWN_UUID"
#: A card the config claims already carries foreign VRAM -- the orphan
#: container / orphan process trap. Booting on top of it produces an OOM
#: whose cause is invisible in the new process's own logs.
REFUSE_CARD_BUSY = "REFUSE_CARD_BUSY"
REFUSE_HOST_HEADROOM = "REFUSE_HOST_HEADROOM"
REFUSE_DISK_HEADROOM = "REFUSE_DISK_HEADROOM"
REFUSE_PORT_BUSY = "REFUSE_PORT_BUSY"
REFUSE_PLAN_MISSING = "REFUSE_PLAN_MISSING"
#: The pinned plan was solved against a different world than the one booting
#: now. REFUSING is the whole point: a stale plan is exactly the situation
#: where guessing looks harmless and is not.
REFUSE_PLAN_STALE = "REFUSE_PLAN_STALE"
REFUSE_PLAN_UNSOLVABLE = "REFUSE_PLAN_UNSOLVABLE"
#: #375 turnkey defect 3: two serving instances configured onto one boot-log
#: path interleave their output, and the resulting file proves nothing about
#: either. Each instance owns its own log path.
REFUSE_LOG_PATH_SHARED = "REFUSE_LOG_PATH_SHARED"

NAMES = (
    REFUSE_CONFIG_UNPARSABLE,
    REFUSE_CONFIG_INCOMPLETE,
    REFUSE_REPO_IS_WORKTREE,
    REFUSE_PATH_MISSING,
    REFUSE_WHEEL_SHADOW,
    REFUSE_CARD_CENSUS,
    REFUSE_CARD_UNKNOWN_UUID,
    REFUSE_CARD_BUSY,
    REFUSE_HOST_HEADROOM,
    REFUSE_DISK_HEADROOM,
    REFUSE_PORT_BUSY,
    REFUSE_PLAN_MISSING,
    REFUSE_PLAN_STALE,
    REFUSE_PLAN_UNSOLVABLE,
    REFUSE_LOG_PATH_SHARED,
)


@dataclasses.dataclass(frozen=True)
class Refusal:
    """One named reason the stack will not boot.

    ``observed`` and ``expected`` are free-form strings rather than numbers on
    purpose: a card census refusal compares SETS, a headroom refusal compares
    GiB, and forcing both through a numeric field would lose the units that
    make the line readable without the source.
    """

    name: str
    subject: str
    observed: str
    expected: str
    remedy: str = ""

    def __post_init__(self):
        if self.name not in NAMES:
            # A refusal with an unregistered name cannot be grepped for by an
            # operator who only has the vocabulary, so it is itself a defect.
            raise ValueError(
                f"unregistered refusal name {self.name!r}; add it to "
                f"turnkey.refusal.NAMES")

    def line(self) -> str:
        """The single log line. Format is load-bearing -- operators grep it."""
        s = (f"{self.name} subject={self.subject} "
             f"observed={self.observed} expected={self.expected}")
        if self.remedy:
            s += f" remedy={self.remedy}"
        return s

    def to_json(self) -> dict:
        return dataclasses.asdict(self)


class RefusalError(RuntimeError):
    """Raised to abort the boot. Carries the :class:`Refusal` verbatim."""

    def __init__(self, refusal: Refusal):
        super().__init__(refusal.line())
        self.refusal = refusal


def refuse(name: str, subject: str, observed, expected,
           remedy: str = "") -> Refusal:
    """Build a :class:`Refusal`, stringifying observed/expected."""
    return Refusal(name=name, subject=subject, observed=str(observed),
                   expected=str(expected), remedy=remedy)


def raise_refusal(name: str, subject: str, observed, expected,
                  remedy: str = "") -> None:
    raise RefusalError(refuse(name, subject, observed, expected, remedy))


def first_refusal(refusals) -> Optional[Refusal]:
    for r in refusals:
        if r is not None:
            return r
    return None
