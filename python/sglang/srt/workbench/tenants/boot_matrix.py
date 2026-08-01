# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""The boot matrix as an idle-workbench tenant (#349 / ANALYSE #347 item 4).

A thin wrapper over :data:`sglang.srt.boot_matrix.arms.ARMS`. It does NOT
restate the matrix: the arm list, the composition, the checks and the
verdicts all live in :mod:`sglang.srt.boot_matrix`, which stands alone and is
unit-tested on a card-less host. This module only teaches the workbench
scheduler how to run one arm as one segment, so the same matrix that runs as
the pre-Docker manual sweep also runs as idle work once #347's workbench is up.

WHY ONE ARM PER SEGMENT. A segment is the preemption granularity. Booting a
TP server takes minutes and holds every card; a segment that is one arm loses
at most one boot to a preemption and never leaves a half-booted server holding
a CUDA context (the SubprocessSegment signals the child's whole process group,
which is exactly why a full-server boot -- torch, NCCL, draft workers -- is
safe to preempt).

WHY THE MODEL IS INPUT, NOT CODE. Which model the matrix boots is a fact about
a deployment, not the rig. A hardcoded path would be the rig-only assumption
ANALYSE #347 excludes; the tenant is unavailable, by name, until a model is
configured.

CARDS. Every arm is a full TP boot, so the segment wants every visible card
(``cards_wanted=0``). Under the workbench that means the matrix runs only when
the whole rig is idle -- correct: a cross-feature boot sweep next to a live
serving engine would measure neither.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

from sglang.srt.boot_matrix.arms import ARMS, Arm
from sglang.srt.workbench.tenant import (
    GIB,
    EventSink,
    IdleWorkTenant,
    SubprocessSegment,
    WorkEstimate,
    WorkEvent,
    WorkGrant,
    WorkSegment,
)

logger = logging.getLogger(__name__)

#: A full TP serving boot uses most of a card. Declared as one post the
#: operator can see and argue with (DESIGN #341 D2 forbids an opaque number),
#: not a measured reservation: the arm sweep is a whole-rig-idle job, so this
#: is a floor for "this needs a real card", not a tight budget.
FULL_BOOT_CARD_BYTES = 12 * GIB


def _pythonpath() -> str:
    import os

    from sglang.srt.boot_matrix import arms as _arms_mod

    # repo/python is the import root; derive it from the package location so no
    # path is baked in.
    pkg = Path(_arms_mod.__file__).resolve()
    for parent in pkg.parents:
        if (parent / "sglang").is_dir():
            return str(parent) + os.pathsep + os.environ.get("PYTHONPATH", "")
    return os.environ.get("PYTHONPATH", "")


class BootMatrixTenant(IdleWorkTenant):
    """Run the #349 integration boot matrix as idle work."""

    name = "boot_matrix"
    #: After training, tuning and probing: the matrix is a pre-release net, not
    #: a continuous background consumer, so it runs last when several tenants
    #: contend. One number to change for a deployment that disagrees.
    priority = 70

    def __init__(
        self,
        *,
        artifact_root: Path,
        model_path: str = "",
        python_executable: str = "",
        arms: Sequence[Arm] = ARMS,
        port: int = 31349,
    ) -> None:
        super().__init__()
        self.artifact_root = Path(artifact_root)
        self.model_path = model_path
        self.port = port
        import sys as _sys

        self.python_executable = python_executable or _sys.executable
        self._arms = list(arms)
        #: Arm names already run in this process; a segment pops the next.
        self._done: set[str] = set()

    # -- availability -------------------------------------------------------

    def available(self) -> tuple[bool, str]:
        if not self.model_path:
            return False, (
                "no model configured for the boot matrix; pass a model path "
                "(the matrix boots a real server, and which model is a "
                "deployment fact, not a default)"
            )
        return True, ""

    def describe(self) -> str:
        return (
            f"integration boot matrix: {len(self._remaining())} of "
            f"{len(self._arms)} arms pending, model {self.model_path or '<unset>'}"
        )

    # -- queue --------------------------------------------------------------

    def _remaining(self) -> list[Arm]:
        return [a for a in self._arms if a.name not in self._done]

    def pending(self) -> int:
        return len(self._remaining())

    def enqueue(self, item: Mapping[str, Any]) -> str:
        """Re-arm one arm by name (e.g. after a fix), so a green sweep can be
        re-run selectively without restarting the server."""
        name = str(item.get("arm", "")).strip()
        known = {a.name for a in self._arms}
        if name not in known:
            raise ValueError(
                f"unknown boot-matrix arm {name!r}; known arms are "
                + ", ".join(sorted(known))
            )
        self._done.discard(name)
        return name

    # -- pricing ------------------------------------------------------------

    def estimate(self) -> WorkEstimate:
        remaining = self._remaining()
        if not remaining:
            return WorkEstimate(per_card_bytes=0, cards_wanted=0)
        arm = remaining[0]
        return WorkEstimate(
            per_card_bytes=FULL_BOOT_CARD_BYTES,
            posts={"serving_boot": FULL_BOOT_CARD_BYTES},
            cards_wanted=0,  # every visible card: a full TP boot
            expected_seconds=arm.expected_seconds,
        )

    # -- running ------------------------------------------------------------

    def _out_dir(self) -> Path:
        return self.artifact_root / "boot_matrix"

    def segment_argv(self, arm: Arm) -> list[str]:
        """The launch line for one arm as its own single-arm sweep. Pure, so a
        test can assert the command without spawning a subprocess."""
        return [
            self.python_executable,
            "-m",
            "sglang.srt.boot_matrix.sweep",
            "--run",
            "--only",
            arm.name,
            "--model",
            self.model_path,
            "--out",
            str(self._out_dir()),
            "--port",
            str(self.port),
        ]

    async def start_segment(self, grant: WorkGrant, sink: EventSink) -> WorkSegment:
        remaining = self._remaining()
        if not remaining:
            raise RuntimeError("the boot matrix has no pending arms")
        arm = remaining[0]
        out_dir = self._out_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        argv = self.segment_argv(arm)
        note = f" ({arm.capture_note.split(':')[0]})" if arm.capture_note else ""
        sink(
            WorkEvent(
                "info",
                f"boot-matrix arm {arm.name}{note}: {arm.axis}; "
                f"catches {arm.catches}",
                data={
                    "arm": arm.name,
                    "kind": arm.kind,
                    "expect_s": arm.expected_seconds,
                },
            )
        )
        # Mark done now: a preempted arm is re-armed via enqueue(), matching the
        # fp8 tuner's "requeue on PREEMPTED" contract without this class having
        # to know what a work item is.
        self._done.add(arm.name)
        segment = SubprocessSegment(
            argv=argv,
            cwd=None,
            env={
                "CUDA_VISIBLE_DEVICES": grant.visible_devices,
                "PYTHONPATH": _pythonpath(),
                "BOOT_MATRIX_MODEL": self.model_path,
            },
            sink=sink,
            label=f"boot_matrix {arm.name}",
            artifact_path=str(out_dir / arm.name / "summary.json"),
        )
        return await segment.start()
