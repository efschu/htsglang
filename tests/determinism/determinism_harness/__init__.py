# SPDX-License-Identifier: Apache-2.0
"""#124 determinism / byte-identity regression harness for the htsglang fork.

Turns the fork's manually-validated byte-identity discipline into an
enforceable, tested contract:

* :mod:`.trajectory`  -- the ``Trajectory`` data model + ``Verdict`` results.
* :mod:`.primitives`  -- pure, CPU-testable comparison primitives.
* :mod:`.classes`     -- the byte-identity CLASS registry (the contract) and
  the per-class assertion bundles.
* :mod:`.matrix`      -- the declarative (feature/config) -> expected-class
  test matrix, plus the pinned seed discipline.
* :mod:`.runner`      -- ``run_case_config`` GPU seam (STUBBED, deferred to
  the r2 GPU window) + the fully-implemented ``evaluate_case`` dispatch.

Everything except the GPU boot/run wiring is complete and unit-tested on
synthetic data (see ``tests/determinism/test_*.py``).
"""

from .classes import CLASS_SPECS, ByteIdentityClass, check_class
from .matrix import EXCLUDED_CASES, PINNED_SEED, TEST_MATRIX, CaseSpec, get_case
from .primitives import (
    check_argmax_clean_trajectory,
    check_delta_band,
    check_machine_zero,
    check_near_tie_only_divergence,
    check_non_compounding,
    check_self_determinism,
    classify_flip,
    per_step_max_abs_delta,
)
from .runner import evaluate_case, gpu_available, requires_gpu, run_case_config
from .trajectory import FlipKind, Trajectory, Verdict

__all__ = [
    "ByteIdentityClass",
    "CLASS_SPECS",
    "CaseSpec",
    "EXCLUDED_CASES",
    "FlipKind",
    "PINNED_SEED",
    "TEST_MATRIX",
    "Trajectory",
    "Verdict",
    "check_argmax_clean_trajectory",
    "check_class",
    "check_delta_band",
    "check_machine_zero",
    "check_near_tie_only_divergence",
    "check_non_compounding",
    "check_self_determinism",
    "classify_flip",
    "evaluate_case",
    "get_case",
    "gpu_available",
    "per_step_max_abs_delta",
    "requires_gpu",
    "run_case_config",
]
