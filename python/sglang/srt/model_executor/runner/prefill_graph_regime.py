# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""#613: the per-batch regime gate for the captured prefill graph.

WHY A REGIME GATE AND NOT A FLAG. Window 4 on 2026-08-05 (``3b4526c4ac``)
measured prefill-eager against prefill-graphs under barlink -- production's
transport -- interleaved, ms per fixed unit of work, each point scored against
its own A-vs-A floor:

===================  =========  =========  ==============  =======  ==========
point                eager      graphs     paired delta    floor    verdict
===================  =========  =========  ==============  =======  ==========
1900 single-stream   1196.3 ms  1320.6 ms  **+10.25%**     0.12%    REPORTABLE
256 x4 concurrent     230.1 ms   219.0 ms  **-4.62%**      1.99%    REPORTABLE
===================  =========  =========  ==============  =======  ==========

All three pairs agreed in sign at both points; the deltas cleared their own
floors by 85x and 2.3x. The captured prefill pays where the work is launch-train
bound (several short prefills in flight, where barlink's device-side collectives
run inside the captured graph) and loses where it is GEMM bound (one long
prefill, no launch overhead to recover, paying bucket padding and capture cost
instead). The conclusion recorded with the measurement is that "the answer is
not on/off but workload-dependent, and a blanket flag is the wrong shape", and
that ``can_run_graph`` -- already a per-batch decision -- is where the gate
belongs. This module is that gate.

WHAT THIS MODULE REFUSES TO GUESS. Two points are not a curve. The sign
anywhere between 256 and ~1974 tokens per request is **unmeasured**, and a gate
that interpolated a crossover would be inventing the number that decides every
borderline batch. So the graph is permitted only inside the regime that was
measured to WIN, and everything else is refused BY NAME -- the unmeasured middle
as unmeasured, not as slow. Narrowing that refusal is a measurement, not an
edit.

DEFAULT OFF. With ``SGLANG_PREFILL_GRAPH_REGIME`` unset every batch is
permitted and ``can_run_graph`` behaves exactly as it does today. This gate
cannot change a boot that did not ask for it.

NOT WHAT KEEPS PREFILL GRAPHS OFF IN PRODUCTION. That is an upstream,
transport-agnostic rule -- ``multimodal model`` in
``_disable_breakable_cudagraph_if_incompatible`` (``server_args.py:9546``) --
which fires because the served checkpoint is a
``Qwen3_5ForConditionalGeneration`` carrying a vision subconfig, and which has
no barlink term in it. This gate only decides eligibility once a boot has a
prefill graph at all.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

#: The lever. Unset/0 -> this module permits everything (today's behaviour).
PREFILL_GRAPH_REGIME_ENV = "SGLANG_PREFILL_GRAPH_REGIME"

#: Tokens per request at the point measured to WIN (-4.62%, 256 x4 concurrent).
MEASURED_WIN_TOKENS_PER_REQ = 256

#: Tokens per request at the point measured to LOSE (+10.25%, the 1900 point;
#: 1974 is the real prefill length behind that label).
MEASURED_LOSS_TOKENS_PER_REQ = 1974

#: Concurrency at the measured win. Below this there is no launch train to
#: recover, which is the mechanism the measurement attributes the win to.
MEASURED_WIN_MIN_BATCH = 2


@dataclass(frozen=True)
class RegimeVerdict:
    """Whether this batch is in the measured-win regime, and why."""

    permits: bool
    reason: str


def regime_enabled() -> bool:
    """Whether the per-batch regime gate is active at all.

    Parsed here rather than through a shared helper because this module is
    imported on the prefill hot path and deliberately depends on nothing.
    """
    return os.environ.get(PREFILL_GRAPH_REGIME_ENV, "").strip().lower() in ("1", "true")


def regime_permits_graph(batch_size: int, num_tokens: int) -> RegimeVerdict:
    """May this batch use the captured prefill graph, on regime grounds?

    Called from ``PrefillCudaGraphRunner.can_run_graph`` AFTER every
    correctness eligibility check, because this is a performance decision and
    must never be able to admit a batch that correctness refused.
    """
    if not regime_enabled():
        return RegimeVerdict(True, f"regime gate disabled ({PREFILL_GRAPH_REGIME_ENV})")

    if batch_size < MEASURED_WIN_MIN_BATCH:
        return RegimeVerdict(
            False,
            "single-stream prefill: measured +10.25% SLOWER captured "
            f"(1196.3 -> 1320.6 ms at {MEASURED_LOSS_TOKENS_PER_REQ} tokens, "
            "floor 0.12%). No launch train to recover.",
        )

    tokens_per_req = num_tokens // max(1, batch_size)
    if tokens_per_req > MEASURED_WIN_TOKENS_PER_REQ:
        return RegimeVerdict(
            False,
            f"{tokens_per_req} tokens/request is above the measured win point "
            f"({MEASURED_WIN_TOKENS_PER_REQ}, -4.62%) and below the measured "
            f"loss point ({MEASURED_LOSS_TOKENS_PER_REQ}, +10.25%). The sign "
            "in between is UNMEASURED, so the graph is refused here rather "
            "than admitted on an interpolated crossover. Narrowing this needs "
            "a measurement, not an edit.",
        )

    return RegimeVerdict(
        True,
        f"launch-train regime: {batch_size} in flight at {tokens_per_req} "
        f"tokens/request, at or below the measured win point "
        f"({MEASURED_WIN_TOKENS_PER_REQ}, -4.62%, floor 1.99%).",
    )
