# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Token-slice pipelining of the tensor-parallel all-reduce (task #588).

WHAT THIS IS FOR
----------------
On this rig a prefill's per-layer TP all-reduce is transfer-bound: the bf16
payload is ``new_tokens x hidden`` per layer, which at ~1900 new tokens is
~16 MB per collective and ~1.25 GB over a full prefill on x4/x8 PCIe links.
The measured wait is 2-3x the measured compute, and it scales with the
new-token count. Compressing the payload would be lossy; splitting it is not.

A row-parallel layer computes ``partial = x_shard @ W_shard`` and then sums
``partial`` across the TP ranks. Both halves are independent per TOKEN ROW:
row ``t`` of the GEMM reads only row ``t`` of the input, and the all-reduce
sums row ``t`` across ranks without touching any other row. Splitting the
token axis into K slices therefore leaves every output element as the sum of
exactly the same K-independent set of per-rank partials -- the arithmetic is
unchanged -- while letting slice ``i``'s transfer occupy the wire during
slice ``i+1``'s GEMM.

THE CEILING, STATED UP FRONT
----------------------------
The consumer of a row-parallel output is the immediately following statement
(a layernorm or residual add), so there is NO independent downstream compute
to hide the collective behind -- the only overlap partner a layer offers is
its OWN GEMM. With per-slice compute ``g = G/K`` and per-slice transfer
``a = L + P/(K*B)`` the two-stage pipeline makespan is

    makespan = (K-1) * max(g, a) + g + a

which for the transfer-bound case here reduces to

    makespan = P/B + G/K + K*L        (baseline: P/B + G + L)

so the saving is ``G - G/K - (K-1)*L``: bounded above by the layer's OWN
GEMM time G, never by the transfer term P/B. The best attainable makespan is
``P/B + 2*sqrt(G*L)`` at ``K* = sqrt(G/L)``. If the wire term P/B already
dominates the measured floor, this lever is exhausted at that point and what
remains is pure wire -- that is a finding, not a failure.

That last expression is also the whole K story: the first slice's compute is
never hidden (nothing is in flight yet) and the last slice's transfer is
never hidden (no compute remains). K trades those two exposures, which fall
as 1/K, against the per-collective launch overhead K*L, which grows with K.
:func:`derive_num_slices` minimizes exactly that sum from MEASURED G, L and
B rather than from a tuned constant.

RANK-UNIFORMITY (deadlock safety)
---------------------------------
Every rank must issue the same NUMBER of collectives in the same order, or
the group deadlocks. K is therefore derived only from quantities that are
identical on every rank of the group:

- the payload shape (num_tokens x out_features), which is the FULL output
  width on every rank -- unlike the input shard, which differs under uneven
  TP;
- the calibration scalars, which are max-reduced across the group before
  use, so every rank consumes bit-identical inputs (a max over floats is
  exact) and the slowest rank sets the schedule;
- process-uniform environment variables.

Nothing rank-local (rank id, shard width, free memory, local timings) may
enter the derivation. ``test_tp_ar_pipeline.py`` pins the signature against
exactly that.

SCOPE
-----
Eager prefill only. Collectives issued on a side stream cannot be recorded
into a CUDA graph capture, and a replayed graph would not run this Python at
all, so :func:`plan_num_slices` returns 1 while a capture is active. Decode
falls out by itself: its token count is far below the minimum-token gate.
Off by default behind ``SGLANG_TP_AR_PIPELINE``; when off, callers never
reach this module.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from typing import Callable, List, Optional, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.utils.collective_clock import collective_clock

logger = logging.getLogger(__name__)

__all__ = [
    "Calibration",
    "calibration_from_probe",
    "derive_num_slices",
    "plan_num_slices",
    "reset_tp_ar_pipeline_state",
    "slice_bounds",
    "tp_ar_pipeline_enabled",
    "tp_ar_pipeline_stats",
    "pipelined_row_all_reduce",
]

#: Collective-clock family for the all-reduces this module issues on the side
#: stream. Deliberately NOT ``tp.all_reduce``: those spans are recorded on the
#: comm stream, so their sum is "wire busy time", not exposed wait. Mixing
#: them into the baseline family would make an overlapped run look like an
#: unchanged one.
CLOCK_FAMILY_WIRE = "tp.all_reduce.pipe_wire"

#: Collective-clock family for the join. This IS the exposed wait: the span
#: is recorded on the COMPUTE stream and brackets the point where compute
#: blocks until the last slice's transfer has landed.
CLOCK_FAMILY_JOIN = "tp.ar_pipeline.join"

#: Bytes for the small-message probe used to separate per-collective latency
#: from per-byte wire cost. Small enough to be latency-dominated on any link
#: this runs on, large enough not to hit a zero-length special case.
_PROBE_BYTES = 4096

#: Probe repetitions. Fixed, so every rank issues the same collective count
#: no matter what it measures.
_PROBE_ITERS = 4


@dataclasses.dataclass(frozen=True)
class Calibration:
    """Measured cost model of one collective on one group.

    ``latency_s`` is the size-independent per-collective cost, ``wire_s_per_byte``
    the marginal cost per payload byte, and ``compute_s_per_byte`` the layer
    GEMM cost per byte of OUTPUT produced (bytes, not FLOPs, because that is
    the quantity the caller can compute for any layer without knowing its
    shard width).
    """

    latency_s: float
    wire_s_per_byte: float
    compute_s_per_byte: float

    @property
    def usable(self) -> bool:
        return (
            self.latency_s > 0.0
            and self.wire_s_per_byte > 0.0
            and self.compute_s_per_byte > 0.0
            and math.isfinite(self.latency_s)
            and math.isfinite(self.wire_s_per_byte)
            and math.isfinite(self.compute_s_per_byte)
        )


def calibration_from_probe(
    compute_s: float,
    payload_bytes: int,
    big_all_reduce_s: float,
    small_all_reduce_s: float,
    small_bytes: int = _PROBE_BYTES,
) -> Calibration:
    """Two-point fit of ``T(S) = latency + S * wire_s_per_byte``.

    Pure so the fit can be tested without a group. A degenerate fit (a
    non-positive slope or intercept, which a noisy pair of measurements can
    produce) yields an unusable Calibration rather than a nonsense K; the
    caller then stays on the unsliced path.
    """
    if payload_bytes <= small_bytes:
        return Calibration(0.0, 0.0, 0.0)
    wire_s_per_byte = (big_all_reduce_s - small_all_reduce_s) / float(
        payload_bytes - small_bytes
    )
    latency_s = small_all_reduce_s - small_bytes * wire_s_per_byte
    compute_s_per_byte = compute_s / float(payload_bytes)
    return Calibration(
        latency_s=latency_s,
        wire_s_per_byte=wire_s_per_byte,
        compute_s_per_byte=compute_s_per_byte,
    )


def derive_num_slices(
    payload_bytes: int,
    num_tokens: int,
    calibration: Optional[Calibration],
    max_slices: int,
    slices_override: int = 0,
) -> int:
    """Number of token slices for one all-reduce. Pure and rank-uniform.

    The parameter list is the contract: every argument is identical on every
    rank of the group (see the module docstring). Adding a rank-local
    argument here is what a deadlock looks like before it happens, so the
    signature is pinned by a test.

    Formula: the exposed cost above the unavoidable wire term ``P/B`` is
    ``G/K + K*L`` -- the un-hidden first-slice compute plus the K launch
    latencies. Minimizing over K gives ``K* = sqrt(G/L)`` with ``G`` taken
    from the measured compute-per-output-byte. K is then clamped so that no
    slice falls below the link's half-power size ``n_1/2 = L/B``, below which
    a slice's transfer costs more in latency than it moves in bytes.
    """
    if num_tokens <= 1 or payload_bytes <= 0:
        return 1
    if slices_override > 0:
        num_slices = slices_override
    else:
        if calibration is None or not calibration.usable:
            return 1
        compute_s = payload_bytes * calibration.compute_s_per_byte
        num_slices = int(round(math.sqrt(compute_s / calibration.latency_s)))
    num_slices = min(max(num_slices, 1), max(int(max_slices), 1))
    if calibration is not None and calibration.usable:
        half_power_bytes = calibration.latency_s / calibration.wire_s_per_byte
        if half_power_bytes >= 1.0:
            num_slices = min(num_slices, int(payload_bytes // half_power_bytes))
    num_slices = min(num_slices, num_tokens)
    return max(num_slices, 1)


def slice_bounds(num_tokens: int, num_slices: int) -> List[Tuple[int, int]]:
    """Contiguous, near-equal partition of ``[0, num_tokens)``.

    Contiguous because a contiguous row range of a row-major tensor is itself
    a contiguous tensor, which every all-reduce backend accepts without a
    staging copy. Near-equal because the pipeline is only as balanced as its
    slowest stage.
    """
    num_slices = max(1, min(int(num_slices), int(num_tokens)))
    base, extra = divmod(int(num_tokens), num_slices)
    bounds = []
    start = 0
    for i in range(num_slices):
        end = start + base + (1 if i < extra else 0)
        bounds.append((start, end))
        start = end
    assert start == num_tokens
    return bounds


class _EventPool:
    """CUDA events reused only once the GPU has passed them.

    Re-recording an event that a previously enqueued wait has not yet
    observed would silently redirect that wait, so an event is only handed
    out again after ``query()`` reports it complete.
    """

    def __init__(self) -> None:
        self._pool: List[torch.cuda.Event] = []

    def acquire(self) -> torch.cuda.Event:
        for i, event in enumerate(self._pool):
            if event.query():
                return self._pool.pop(i)
        return torch.cuda.Event()

    def release(self, events) -> None:
        self._pool.extend(events)


@dataclasses.dataclass
class _State:
    calibration: Optional[Calibration] = None
    calibration_done: bool = False
    calls_pipelined: int = 0
    calls_unsliced: int = 0
    slices_issued: int = 0
    logged_shapes: set = dataclasses.field(default_factory=set)
    events: _EventPool = dataclasses.field(default_factory=_EventPool)
    comm_stream: Optional["torch.cuda.Stream"] = None


_STATE = _State()
_ENABLED: Optional[bool] = None


def tp_ar_pipeline_enabled() -> bool:
    """Read once. Callers on the default path pay exactly this bool."""
    global _ENABLED
    if _ENABLED is None:
        _ENABLED = bool(envs.SGLANG_TP_AR_PIPELINE.get())
    return _ENABLED


def reset_tp_ar_pipeline_state() -> None:
    """Drop the cached flag, calibration and counters. Tests only."""
    global _STATE, _ENABLED
    _STATE = _State()
    _ENABLED = None


def tp_ar_pipeline_stats() -> dict:
    """Counters for the runsheet: did the hook actually fire, and with what K.

    An arm that reports ``calls_pipelined == 0`` measured the baseline twice;
    this is the cheapest way to see that before trusting a delta.
    """
    return {
        "calls_pipelined": _STATE.calls_pipelined,
        "calls_unsliced": _STATE.calls_unsliced,
        "slices_issued": _STATE.slices_issued,
        "calibrated": _STATE.calibration_done,
        "calibration": (
            None
            if _STATE.calibration is None
            else dataclasses.asdict(_STATE.calibration)
        ),
    }


def plan_num_slices(num_tokens: int, payload_bytes: int) -> int:
    """K for the next call, or 1 to stay on the unsliced path.

    Every gate here is group-uniform: a token count, a byte count, the
    process environment, and the capture state -- which sglang enters and
    leaves on all ranks together.
    """
    if num_tokens < int(envs.SGLANG_TP_AR_PIPELINE_MIN_TOKENS.get()):
        return 1
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        # A side-stream collective cannot be captured, and inside a replayed
        # graph this function would not run at all. Eager prefill only.
        return 1
    return derive_num_slices(
        payload_bytes=payload_bytes,
        num_tokens=num_tokens,
        calibration=_STATE.calibration,
        max_slices=int(envs.SGLANG_TP_AR_PIPELINE_MAX_SLICES.get()),
        slices_override=int(envs.SGLANG_TP_AR_PIPELINE_SLICES.get()),
    )


def _comm_stream(device: torch.device) -> Optional["torch.cuda.Stream"]:
    if device.type != "cuda":
        return None
    if _STATE.comm_stream is None:
        _STATE.comm_stream = torch.cuda.Stream(device=device)
    return _STATE.comm_stream


def _run_unsliced(
    input_parallel: torch.Tensor,
    apply_fn: Callable[[torch.Tensor], torch.Tensor],
    all_reduce_fn: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Exactly what the caller would have done without this module.

    Kept as one code path so that "pipeline on, K == 1" is bit-identical to
    "pipeline off" by construction rather than by inspection.
    """
    _STATE.calls_unsliced += 1
    return all_reduce_fn(apply_fn(input_parallel))


def _run_sliced(
    input_parallel: torch.Tensor,
    apply_fn: Callable[[torch.Tensor], torch.Tensor],
    all_reduce_fn: Callable[[torch.Tensor], torch.Tensor],
    bounds: List[Tuple[int, int]],
) -> torch.Tensor:
    num_tokens = input_parallel.shape[0]
    clock = collective_clock()
    output: Optional[torch.Tensor] = None
    comm_stream = None
    compute_stream = None
    ready_events: List[torch.cuda.Event] = []
    done_events: List[torch.cuda.Event] = []

    for start, end in bounds:
        part = apply_fn(input_parallel[start:end])
        if output is None:
            output = torch.empty(
                (num_tokens,) + tuple(part.shape[1:]),
                dtype=part.dtype,
                device=part.device,
            )
            if part.is_cuda:
                compute_stream = torch.cuda.current_stream(part.device)
                comm_stream = _comm_stream(part.device)
                # Allocated on the compute stream, read and written on the
                # comm stream: the allocator must not recycle it on the
                # strength of the compute stream's last use alone.
                output.record_stream(comm_stream)
        output[start:end].copy_(part)

        if comm_stream is None:
            # CPU / no-CUDA: same slicing, no overlap. Used by the hermetic
            # tests, and the honest fallback if a device has no side stream.
            view = output[start:end]
            reduced = all_reduce_fn(view)
            if reduced is not view:
                view.copy_(reduced)
            continue

        ready = _STATE.events.acquire()
        ready.record(compute_stream)
        ready_events.append(ready)
        with torch.cuda.stream(comm_stream):
            comm_stream.wait_event(ready)
            view = output[start:end]
            with clock.label_scope(CLOCK_FAMILY_WIRE):
                reduced = all_reduce_fn(view)
            if reduced.data_ptr() != view.data_ptr():
                view.copy_(reduced)
            done = _STATE.events.acquire()
            done.record(comm_stream)
            done_events.append(done)

    if comm_stream is not None:
        # The join is the exposed wait. Timing it on the compute stream is
        # what makes an overlapped run distinguishable from a serial one:
        # the wire spans above move to their own family, this span is what
        # the forward actually paid.
        if clock.armed:
            with clock.span(CLOCK_FAMILY_JOIN):
                for event in done_events:
                    compute_stream.wait_event(event)
        else:
            for event in done_events:
                compute_stream.wait_event(event)
        _STATE.events.release(ready_events)
        _STATE.events.release(done_events)

    _STATE.calls_pipelined += 1
    _STATE.slices_issued += len(bounds)
    return output


def _calibrate(
    input_parallel: torch.Tensor,
    apply_fn: Callable[[torch.Tensor], torch.Tensor],
    all_reduce_fn: Callable[[torch.Tensor], torch.Tensor],
    max_reduce_fn: Optional[Callable[[torch.Tensor], torch.Tensor]],
) -> torch.Tensor:
    """Run this call unsliced, timed, and fit the cost model from it.

    Rank-uniform by construction: a fixed number of collectives with fixed
    shapes, no data-dependent branching, and one max-reduce so that every
    rank ends up with the SAME scalars -- the max picks the slowest rank per
    term, which is the rank that paces the group.
    """
    _STATE.calibration_done = True

    if not input_parallel.is_cuda or not torch.cuda.is_available():
        # No device timers: stay unsliced forever rather than guess.
        return _run_unsliced(input_parallel, apply_fn, all_reduce_fn)

    stream = torch.cuda.current_stream(input_parallel.device)
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(6)]

    ev[0].record(stream)
    partial = apply_fn(input_parallel)
    ev[1].record(stream)

    payload_bytes = partial.numel() * partial.element_size()
    probe_elems = max(1, _PROBE_BYTES // partial.element_size())
    probe = torch.zeros(probe_elems, dtype=partial.dtype, device=partial.device)

    ev[2].record(stream)
    for _ in range(_PROBE_ITERS):
        probe_out = all_reduce_fn(probe)
        if probe_out.data_ptr() != probe.data_ptr():
            probe.copy_(probe_out)
    ev[3].record(stream)

    ev[4].record(stream)
    output = all_reduce_fn(partial)
    ev[5].record(stream)

    ev[5].synchronize()
    compute_s = ev[0].elapsed_time(ev[1]) / 1000.0
    small_s = ev[2].elapsed_time(ev[3]) / 1000.0 / _PROBE_ITERS
    big_s = ev[4].elapsed_time(ev[5]) / 1000.0

    measured = torch.tensor(
        [compute_s, big_s, small_s], dtype=torch.float64, device=partial.device
    )
    if max_reduce_fn is not None:
        measured = max_reduce_fn(measured)
    compute_s, big_s, small_s = (float(v) for v in measured.tolist())

    probe_bytes = probe_elems * partial.element_size()
    calibration = calibration_from_probe(
        compute_s=compute_s,
        payload_bytes=payload_bytes,
        big_all_reduce_s=big_s,
        small_all_reduce_s=small_s,
        small_bytes=probe_bytes,
    )
    _STATE.calibration = calibration
    _STATE.calls_unsliced += 1

    # Every input of the derivation, logged once, so a K that looks wrong in
    # a run can be recomputed by hand from the log line alone.
    logger.info(
        "tp_ar_pipeline calibration: compute=%.3f ms all_reduce=%.3f ms "
        "probe=%.3f ms payload=%d B probe_payload=%d B -> latency=%.1f us "
        "wire=%.3f GB/s compute_per_byte=%.3f ns/B usable=%s",
        compute_s * 1e3,
        big_s * 1e3,
        small_s * 1e3,
        payload_bytes,
        probe_bytes,
        calibration.latency_s * 1e6,
        (
            0.0
            if calibration.wire_s_per_byte <= 0
            else 1.0 / calibration.wire_s_per_byte / 1e9
        ),
        calibration.compute_s_per_byte * 1e9,
        calibration.usable,
    )
    return output


def pipelined_row_all_reduce(
    input_parallel: torch.Tensor,
    apply_fn: Callable[[torch.Tensor], torch.Tensor],
    all_reduce_fn: Callable[[torch.Tensor], torch.Tensor],
    max_reduce_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    out_features: Optional[int] = None,
) -> torch.Tensor:
    """``all_reduce(apply(x))`` with the token axis pipelined against the wire.

    ``apply_fn`` must be row-independent (a row-parallel GEMM is) and
    ``all_reduce_fn`` must be a sum over the group. The first call on a
    process runs unsliced and calibrates; every later call derives K from the
    calibration and slices.
    """
    if input_parallel.dim() != 2:
        return _run_unsliced(input_parallel, apply_fn, all_reduce_fn)

    if not _STATE.calibration_done:
        return _calibrate(input_parallel, apply_fn, all_reduce_fn, max_reduce_fn)

    num_tokens = input_parallel.shape[0]
    if out_features is None:
        return _run_unsliced(input_parallel, apply_fn, all_reduce_fn)
    payload_bytes = num_tokens * int(out_features) * input_parallel.element_size()

    num_slices = plan_num_slices(num_tokens, payload_bytes)
    if num_slices <= 1:
        return _run_unsliced(input_parallel, apply_fn, all_reduce_fn)

    key = (num_tokens, int(out_features), num_slices)
    if key not in _STATE.logged_shapes:
        _STATE.logged_shapes.add(key)
        logger.info(
            "tp_ar_pipeline: tokens=%d out_features=%d payload=%d B -> K=%d",
            num_tokens,
            int(out_features),
            payload_bytes,
            num_slices,
        )

    return _run_sliced(
        input_parallel, apply_fn, all_reduce_fn, slice_bounds(num_tokens, num_slices)
    )
