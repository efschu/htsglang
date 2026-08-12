# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# ==============================================================================
"""The pinned plan, and the fingerprint that decides whether it still applies.

A turnkey boot must be reproducible, so the cuts and budgets it uses are
PINNED: solved once, written to a file, replayed on every boot. That buys
reproducibility and immediately raises the question this module exists to
answer -- what happens when the machine underneath the pin changes?

The answer is: **refuse, by name, with a diff.** Not re-solve, not adapt, not
warn-and-continue. A stale pin means something changed that the operator has
not accounted for -- a card swapped, the model replaced, the wheel
reinstalled -- and every one of those is a thing a human should see before
production comes back up. Silently re-solving would produce a stack that
boots successfully in a shape nobody chose, which is #539's failure mode
wearing a success message.

The fingerprint covers exactly the inputs a plan is a function of:

* the card set -- UUIDs and their total VRAM, in rank order;
* the model -- path, size and mtime, because a checkpoint swapped in place
  keeps its path (the ship checkpoint on this rig is a ``-yarn1.5`` variant
  that differs from its neighbour only in the suffix);
* the wheel version, because the kernels decide what fits;
* the launch argv, because a plan solved for one context length says nothing
  about another.

It deliberately does NOT cover: free VRAM (transient), host load, the clock.
A fingerprint that changes when nothing structural changed would train
operators to bypass it, which is worse than not having one.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import time
from typing import List, Optional, Sequence, Tuple

from sglang.srt.turnkey.refusal import (
    REFUSE_PLAN_MISSING,
    REFUSE_PLAN_STALE,
    Refusal,
    refuse,
)

__all__ = ["Fingerprint", "PinnedPlan", "fingerprint_of", "load_pinned",
           "save_pinned", "check_staleness"]

_SCHEMA = 1


@dataclasses.dataclass(frozen=True)
class Fingerprint:
    """The world a plan was solved against, reduced to comparable facts."""

    cards: Tuple[Tuple[str, int], ...]   # (uuid, total_mib) in rank order
    model_path: str = ""
    model_size: int = 0
    model_mtime: int = 0
    wheel_version: str = ""
    argv_digest: str = ""

    def to_json(self) -> dict:
        return {
            "cards": [list(c) for c in self.cards],
            "model_path": self.model_path,
            "model_size": self.model_size,
            "model_mtime": self.model_mtime,
            "wheel_version": self.wheel_version,
            "argv_digest": self.argv_digest,
        }

    @staticmethod
    def from_json(d: dict) -> "Fingerprint":
        return Fingerprint(
            cards=tuple((str(u), int(t)) for u, t in d.get("cards", ())),
            model_path=d.get("model_path", ""),
            model_size=int(d.get("model_size", 0)),
            model_mtime=int(d.get("model_mtime", 0)),
            wheel_version=d.get("wheel_version", ""),
            argv_digest=d.get("argv_digest", ""),
        )

    def diff(self, other: "Fingerprint") -> List[str]:
        """Field-by-field difference, phrased for a refusal line."""
        out: List[str] = []
        if self.cards != other.cards:
            out.append(f"cards {_cards_str(other.cards)} -> "
                       f"{_cards_str(self.cards)}")
        for field in ("model_path", "model_size", "model_mtime",
                      "wheel_version", "argv_digest"):
            a, b = getattr(self, field), getattr(other, field)
            if a != b:
                out.append(f"{field} {b!r} -> {a!r}")
        return out


def _cards_str(cards: Sequence[Tuple[str, int]]) -> str:
    return "[" + ", ".join(f"{u[:12]}…={t}MiB" for u, t in cards) + "]"


def argv_digest(argv: Sequence[str]) -> str:
    h = hashlib.sha256()
    for a in argv:
        h.update(a.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def fingerprint_of(cards: Sequence[Tuple[str, int]], argv: Sequence[str],
                   *, model_path: str = "", wheel_version: str = "",
                   stat=os.stat) -> Fingerprint:
    """Measure the current world. ``stat`` is injected for tests."""
    size = mtime = 0
    if model_path:
        try:
            st = stat(model_path)
            size, mtime = int(st.st_size), int(st.st_mtime)
        except OSError:
            # An unreadable model path is a path problem, reported by
            # preflight; here it simply contributes zeros, which will differ
            # from a pin taken when it was readable and so still refuses.
            size = mtime = 0
    return Fingerprint(
        cards=tuple((str(u), int(t)) for u, t in cards),
        model_path=model_path, model_size=size, model_mtime=mtime,
        wheel_version=wheel_version, argv_digest=argv_digest(argv))


@dataclasses.dataclass(frozen=True)
class PinnedPlan:
    fingerprint: Fingerprint
    #: The flags the plan contributes to the launch line. Kept as a list of
    #: already-split tokens so nothing has to re-parse a shell string.
    launch_flags: Tuple[str, ...] = ()
    solved_at: float = 0.0
    solver: str = ""
    note: str = ""

    def to_json(self) -> dict:
        return {"schema": _SCHEMA, "fingerprint": self.fingerprint.to_json(),
                "launch_flags": list(self.launch_flags),
                "solved_at": self.solved_at, "solver": self.solver,
                "note": self.note}

    @property
    def age_days(self) -> float:
        if not self.solved_at:
            return 0.0
        return max(0.0, (time.time() - self.solved_at) / 86400.0)


def save_pinned(path: str, plan: PinnedPlan) -> None:
    """Atomic write -- a half-written plan must never be loadable."""
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(tmp, "w") as fh:
        json.dump(plan.to_json(), fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, path)


def load_pinned(path: str) -> Tuple[Optional[PinnedPlan], Optional[Refusal]]:
    if not os.path.exists(path):
        return None, refuse(
            REFUSE_PLAN_MISSING, path, "absent", "a pinned plan json",
            remedy="solve one with `python -m sglang.srt.turnkey plan-pin`")
    try:
        with open(path, "r") as fh:
            d = json.load(fh)
    except (OSError, ValueError) as e:
        return None, refuse(REFUSE_PLAN_MISSING, path, f"unreadable: {e}",
                            "a valid plan json")
    if int(d.get("schema", 0)) != _SCHEMA:
        return None, refuse(REFUSE_PLAN_STALE, path,
                            f"schema {d.get('schema')}", f"schema {_SCHEMA}",
                            remedy="re-pin the plan")
    return PinnedPlan(
        fingerprint=Fingerprint.from_json(d.get("fingerprint", {})),
        launch_flags=tuple(d.get("launch_flags", ())),
        solved_at=float(d.get("solved_at", 0.0)),
        solver=d.get("solver", ""), note=d.get("note", "")), None


def check_staleness(plan: PinnedPlan, now_fp: Fingerprint,
                    max_age_days: int = 0) -> Optional[Refusal]:
    """The refusal that keeps a pin honest."""
    diffs = now_fp.diff(plan.fingerprint)
    if diffs:
        return refuse(
            REFUSE_PLAN_STALE, "pinned plan", "; ".join(diffs),
            "a fingerprint matching the machine",
            remedy="the world changed since the plan was solved; re-pin "
                   "deliberately rather than booting a plan for another rig")
    if max_age_days and plan.age_days > max_age_days:
        return refuse(REFUSE_PLAN_STALE, "pinned plan",
                      f"{plan.age_days:.1f} days old",
                      f"<= {max_age_days} days")
    return None
