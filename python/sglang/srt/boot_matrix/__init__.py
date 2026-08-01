# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Integration boot matrix -- a standing bug net for cross-feature breaks (#349).

ANALYSE #347 item 4. Cross-feature bugs are invisible to git and to
single-feature tests: the #132 x weightless NCCL hang, and the #340 arm
matrix that silently carried ``SGLANG_UNEVEN_DCP=1`` in the shared harness
environment and published a wrong verdict from it. A small, time-boxed matrix
of FEATURE-CROSS boots -- each printing its EFFECTIVE (resolved) configuration
and gated on coherence rather than on identity -- catches this class
systematically.

The package is deliberately split so the checking logic stands alone before it
is wired into anything (the "einzelteil vor verbund" rule):

* :mod:`arms`      -- the arm list AS DATA. One dataclass, one tuple. Shared
                      verbatim between the standalone sweep and the tenant, so
                      there is exactly one place the matrix is defined.
* :mod:`effective` -- ``report_effective``: the resolved configuration of a
                      booted arm, read FROM ITS SERVER LOG, never re-derived
                      from the launch flags. The generalisation of
                      ``scripts/dual_group/dcp_report.sh`` to every axis.
* :mod:`coherence` -- the gate: short byte-exact probes for the byte tier,
                      plus a graded score against an empirical A-vs-A band for
                      anything longer (the #360/#274 house standard). Never
                      byte-identity on long output -- Qwen GDN prefill is not
                      reproducible past ~109 tokens on any backend.
* :mod:`check`     -- ``check_arm``: hermetic, file-only. Turns one arm's
                      collected artifacts into a single PASS / FAIL / STOP
                      verdict. This is the part the unit tests exercise against
                      synthetic logs; it never touches a card or a server.
* :mod:`sweep`     -- the standalone runner (``python -m
                      sglang.srt.boot_matrix.sweep``). The only part that boots
                      servers; the pre-Docker manual sweep.

The idle-workbench tenant
(:mod:`sglang.srt.workbench.tenants.boot_matrix`) is a thin wrapper over the
SAME :data:`arms.ARMS` list -- it does not re-declare the matrix.

Verdict vocabulary is the GPU-battery one, for the same reason:

* PASS -- the arm ran and its artifact says the thing under test is sound.
          A REJECT arm that refused cleanly at boot is a PASS.
* FAIL -- the arm ran and the artifact shows a real cross-feature defect.
* STOP -- a precondition or the environment is wrong; nothing was learned.
"""

from sglang.srt.boot_matrix.arms import ARMS, Arm, arm_by_name
from sglang.srt.boot_matrix.check import Verdict, check_arm
from sglang.srt.boot_matrix.coherence import CoherenceResult, grade_probes
from sglang.srt.boot_matrix.effective import EffectiveConfig, report_effective

__all__ = [
    "ARMS",
    "Arm",
    "arm_by_name",
    "Verdict",
    "check_arm",
    "CoherenceResult",
    "grade_probes",
    "EffectiveConfig",
    "report_effective",
]
