# SPDX-License-Identifier: Apache-2.0
"""The GPU runner seam for the #124 harness.

Everything in this module EXCEPT :func:`run_case_config` is implemented and
CPU-tested. :func:`run_case_config` is the single deliberate stub: it is the
only missing piece between this harness and live gating, and it is deferred
to the GPU window around the r2 integration (no server boots from this
branch until then).

========================  THE SEAM (r2 GPU wiring)  ========================
``run_case_config(case, which)`` must, per boot:

1.  BOOT: launch ``python -m sglang.launch_server`` (or Engine API) with the
    row's server args (``case.test_config`` / ``reference_config``) INCLUDING
    the pinned ``random_seed`` -- never let sglang randomize it -- and the
    row's env baked into the MAIN process environment before launch. Do NOT
    rely on worker-side env toggles: sglang scrubs custom env for scheduler
    TP workers, making them silent no-ops (memory ``full-perf-testen``).
    Model path comes from a ``model_role -> local path`` map resolved at the
    GPU window (dense_lane / moe_fp8 / moe_marlin vehicles).

2.  SERIALIZE: this is a shared box. One boot at a time per GPU set; track
    and kill ONLY your own PIDs on teardown -- never a broad pkill (memory
    ``shared-box-agent-koordination``). Co-located gates (test + rerun) are
    sequential fresh boots with the identical command line, or same-boot
    repeated requests where the gate definition allows.

3.  DRIVE: send the fixed gate prompt with ``temperature=0``,
    ``max_new_tokens=case.num_decode_tokens``, requesting per-step logprobs
    (``return_logprob`` + a large ``top_logprobs_num``). Full-vocab rows are
    preferred; a top-k slice is acceptable ONLY if k is large enough that
    every flip candidate is inside the slice -- margin classification
    degrades to "flip outside top-k = CORRUPTION by default", which is
    conservative but can misclassify a deep near-tie. If the HTTP surface
    cannot return raw logits, add a debug dump env in the fork (server-side
    per-step logits row capture) rather than weakening the class assertions.
    For the prefill row (``num_decode_tokens == 1``) capture the
    last-position prefill logits. For ``chunked_prefill_with_prefix``, first
    warm the radix prefix with a prefix-sharing request, then run the gate
    request. For graph rows, keep the full-perf discipline: graphs ON is the
    test arm, ``--disable-cuda-graph`` ONLY as the designated reference arm.

4.  COLLECT: build ``Trajectory(token_ids, logits, seed=case.seed,
    label=f"{case.case_id}:{which}")`` with logits rows in decode-step order
    and dtype preserved as served (machine-zero rows are dtype-strict).

5.  TEARDOWN: shut down the server (own PID only), verify port free before
    the next boot.

Then the gate is simply::

    ref  = run_case_config(case, "reference")
    test = run_case_config(case, "test")
    rerun = run_case_config(case, "test-rerun") if case.needs_rerun else None
    verdict = evaluate_case(case, ref=ref, test=test, rerun=rerun)

===========================================================================
"""

from __future__ import annotations

from typing import Optional

import pytest
import torch

from .classes import check_class
from .matrix import TEST_MATRIX, CaseSpec, validate_matrix
from .trajectory import Trajectory, Verdict

__all__ = ["gpu_available", "requires_gpu", "run_case_config", "evaluate_case", "run_matrix"]


def gpu_available() -> bool:
    return torch.cuda.is_available()


#: pytest marker for anything that needs the (deferred) GPU wiring.
requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires CUDA + the r2 GPU runner wiring (deferred; see runner.py seam doc)",
)

_STUB_MSG = (
    "run_case_config is the deliberate #124 stub: server boot/run wiring is "
    "deferred to the GPU window around the r2 integration. See the seam "
    "documentation at the top of tests/determinism/determinism_harness/runner.py "
    "for exactly what to implement. No other part of the harness is stubbed."
)


def run_case_config(case: CaseSpec, which: str) -> Trajectory:
    """STUB -- boot the server for one arm of a matrix row and capture its
    trajectory. ``which`` is ``"reference"``, ``"test"`` or ``"test-rerun"``.

    Raises :class:`NotImplementedError` until the r2 GPU window fills in the
    seam documented in this module's docstring.
    """
    raise NotImplementedError(_STUB_MSG)


def evaluate_case(
    case: CaseSpec,
    *,
    ref: Trajectory,
    test: Trajectory,
    rerun: Optional[Trajectory] = None,
) -> Verdict:
    """Fully-implemented pure dispatch: apply the row's expected class to the
    captured trajectories. This is the function the GPU runner calls; it is
    also exercised end-to-end on synthetic trajectories by the CPU suite.
    """
    return check_class(
        case.expected_class,
        ref=ref,
        test=test,
        rerun=rerun,
        band=case.band,
        near_tie_margin=case.near_tie_margin,
    )


def run_matrix() -> "list[tuple[CaseSpec, Verdict]]":
    """Run every matrix row end-to-end (GPU window entry point).

    Validates the matrix first (fail fast before any boot), then boots
    reference/test/(rerun) per row via :func:`run_case_config` -- which
    raises NotImplementedError until the r2 wiring lands.
    """
    if not gpu_available():
        raise RuntimeError("run_matrix needs CUDA; CPU-side use evaluate_case directly")
    validate_matrix()
    results = []
    for case in TEST_MATRIX:
        ref = run_case_config(case, "reference")
        test = run_case_config(case, "test")
        rerun = run_case_config(case, "test-rerun") if case.needs_rerun else None
        results.append((case, evaluate_case(case, ref=ref, test=test, rerun=rerun)))
    return results
