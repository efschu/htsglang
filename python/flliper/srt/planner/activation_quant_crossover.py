# Copyright 2026 SGLang Team
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
"""#725: measured per-shape activation-quantisation crossovers, as DATA.

The token count ``T`` at which quantising ACTIVATIONS starts to pay, per exact
weight shape, for the Qwen3.6/3.8-27B geometries this rig serves.

READ THIS BEFORE WIRING ANYTHING TO IT
======================================

**Nothing consumes this table yet, and that is deliberate rather than
unfinished.** Three facts have to be understood together before a consumer
would be correct, and each of them was established the hard way:

1. **The planner cost model has no activation-quant axis at all.** It prices
   decode as ``weight_bytes / effective_decode_bw``
   (``key_solver.decode_weight_time``) and prefill as
   ``2 * params / gemm_tflops`` (``key_solver.prefill_compute_time``).
   Quantisation enters those two formulas ONLY as a smaller weight-byte count
   and as which GEMM lane rate is picked (#324). There is no quant STAGE with
   its own cost, so there is no crossover in the planner that could be right
   or wrong -- a bs=1 decode point is priced by bandwidth and never passes
   through a quant stage in the first place. Adding thresholds to the cost
   model would therefore not correct a misprice; it would require first
   introducing the axis.

2. **The real dispatch decision lives in the serving path**, not the planner:
   ``layers/quantization/fp8_dequant_gemv.py:324`` (``fused_gemv_applicable``,
   aspect test ``N >= K``) and the sm120 NVFP4 buckets in
   ``jit_kernel/csrc/gemm/nvfp4/nvfp4_scaled_mm_sm120.cuh:192-201``. That is
   where a threshold would bite, and it is a per-call runtime branch.

3. **#368 measured the small-M quant penalty as largely a LAUNCH CONSTANT that
   CUDA-graph replay removes**: ``int8_quant`` 0.0266 -> 0.0012 ms (~21x), and
   the quant share of the fused op 61% -> 11%. NInfer's thresholds are
   dispatch branches in a C++ header, i.e. taken per call in an eager engine;
   a captured graph cannot branch on ``T`` at replay. So a crossover measured
   under eager dispatch does not transfer unexamined to our graph-captured
   decode path, which is the path full-perf validation actually runs.

Wiring a consumer without (3) would import an eager-mode artefact into a
graph-captured path -- the same shape as pricing a flip on intention rather
than completion.

WHAT THE TABLE IS GOOD FOR TODAY
================================

Corroboration, which is real value and is why it is recorded. Two independent
codebases, two different mechanisms, measured on different GPU classes, agree
on which shape family must not take the small-M quant/fused path: the ``K > N``
shapes. Our own aspect gate separates exactly those, and was measured
independently on sm86:

    N= 5120 K= 6144  N/K=0.83  ours 0.86x LOSE  |  NInfer A16 below T=25
    N= 5120 K=17408  N/K=0.29  ours 0.90x LOSE  |  NInfer A16 below T=25

Five of the six shapes agree in direction. The sixth is a genuine open
question and is flagged as one: see :data:`LM_HEAD_DIVERGENCE`.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, Optional, Tuple

#: Provenance values. Kept apart because they carry different weight, and
#: collapsing them is how a foreign measurement becomes a local law.
MEASURED_NINFER_SM120A = "measured_ninfer_sm120a"
MEASURED_HERE_SM86 = "measured_here_sm86"
ABSENT = "absent"


@dataclasses.dataclass(frozen=True)
class Crossover:
    """One shape's activation-quant crossover, with where it came from.

    ``first_quant_token`` is the smallest ``T`` at which quantised activations
    win. ``None`` means "never quantise at any measured T" -- which is a
    RESULT, not a missing value; the distinction is carried by ``provenance``.
    """

    n: int
    k: int
    role: str
    first_quant_token: Optional[int]
    provenance: str
    note: str = ""

    @property
    def measured(self) -> bool:
        return self.provenance != ABSENT

    def quantises_at(self, tokens: int) -> Optional[bool]:
        """Should activations be quantised at ``tokens``? None when unmeasured.

        None is not "no". A caller that treats an unmeasured shape as "do not
        quantise" has silently adopted a policy the measurement never stated.
        """
        if self.provenance == ABSENT:
            return None
        if self.first_quant_token is None:
            return False
        return int(tokens) >= self.first_quant_token


#: FP8 A16 -> A8, sm120a, measured by NInfer (include/ninfer/ops/linear.h:83-90)
#: on the exact Qwen3.6/3.8-27B shapes this rig serves. Recorded in
#: /spinning/evidence-665-f1/ANALYSE_NINFER.md section 3.1.
#:
#: MLP gate/up is NON-MONOTONIC in their measurement (A8 at T=1, A16 at T=2..4,
#: A8 from T=5). It is stored with first_quant_token=5 and the non-monotonicity
#: in the note rather than smoothed away: a table that hid it would let a
#: caller at T=1 believe A16 was measured to win there, which it was not.
FP8_SM120A: Dict[Tuple[int, int], Crossover] = {
    (14336, 5120): Crossover(
        14336, 5120, "attention input", 12, MEASURED_NINFER_SM120A
    ),
    (16384, 5120): Crossover(16384, 5120, "GDN input", 11, MEASURED_NINFER_SM120A),
    (34816, 5120): Crossover(
        34816,
        5120,
        "MLP gate/up",
        5,
        MEASURED_NINFER_SM120A,
        note="NON-MONOTONIC: A8 at T=1, A16 at T=2..4, A8 from T=5",
    ),
    (5120, 6144): Crossover(
        5120, 6144, "attention/GDN output", 25, MEASURED_NINFER_SM120A
    ),
    (5120, 17408): Crossover(5120, 17408, "MLP down", 25, MEASURED_NINFER_SM120A),
    (248320, 5120): Crossover(
        248320,
        5120,
        "lm_head",
        None,
        MEASURED_NINFER_SM120A,
        note="A16 ALWAYS, even where A8 is permitted",
    ),
}

#: sm86 (this rig's two RTX 3080s) for the ACTIVATION-QUANT axis: NOT MEASURED.
#:
#: NInfer measured sm120a only. Our own sm86 numbers in fp8_dequant_gemv.py
#: are a DIFFERENT quantity -- fused weight-dequant GEMV versus materialise --
#: and may not be read as activation-quant crossovers however similar the
#: shapes look. #368 additionally established that sm86 is GEMM-slow rather
#: than quant-dominant (GEMM 58-99% of the fused op, 2-5x the 5090's), so its
#: crossovers are not merely unmeasured, they are expected to differ in KIND.
#:
#: Left empty on purpose. An estimate here would be a number with no
#: measurement behind it wearing the same type as five that have one.
FP8_SM86: Dict[Tuple[int, int], Crossover] = {}

#: The one row where an outside measurement contradicts our shipped heuristic.
#:
#: Our gate is the aspect test ``N >= K``. For lm_head, N/K = 48.5, so the gate
#: says "fused, decisively". NInfer measured A16 ALWAYS for that shape.
#:
#: The two are not the same quantity -- ours is weight-dequant GEMV, theirs is
#: activation quantisation -- so this is not yet a refutation of our gate. It
#: is the one shape where the cheap corroboration runs out and a measurement of
#: our own would be worth taking. It may also be moot on our serving
#: checkpoint: if lm_head is carried in BF16 (untied), the fp8 gate never sees
#: it. That is the open #724 check.
LM_HEAD_DIVERGENCE = (
    "lm_head [248320, 5120]: our N>=K aspect gate selects the fused lane "
    "(N/K=48.5); NInfer measured A16 always on sm120a. Different mechanisms, "
    "so not a refutation -- but the only shape of the six where the two "
    "sources point opposite ways, and the only one worth a measurement."
)


def crossover_for(n: int, k: int, *, arch: str = "sm120a") -> Optional[Crossover]:
    """The recorded crossover for a shape, or None when none was measured.

    None means "no measurement for this shape on this arch". It never means
    "do not quantise": see :meth:`Crossover.quantises_at`.
    """
    table = FP8_SM120A if arch == "sm120a" else FP8_SM86
    return table.get((int(n), int(k)))
