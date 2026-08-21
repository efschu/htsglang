# SPDX-License-Identifier: Apache-2.0
"""#794 — the corridor's ACTUATOR: how wide a prefill chunk the card can fund.

WHY THIS MODULE EXISTS
----------------------
``corridor_admission.PrefillAdmissionGate`` measures the corridor before a
prefill chunk and spends a relief ladder to restore it. On 2026-08-21 that
measurement was exactly right and the instance died anyway. From the boot log,
in this order:

    corridor law 1024 MiB (prefill admission, 8192 tokens)
    corridor shortfall of 981 MiB for rank 0 (was 953 MiB)
    corridor cannot be restored ahead of this chunk
    ... F.linear in in_proj_qkvz: 256.00 MiB requested, 131.69 MiB free

The gate priced the chunk, named the shortfall to the MiB, concluded that the
corridor could not be restored ahead of this chunk -- and the chunk ran. The
gate is ADVISORY: it returns evidence, and the caller admits either way. That
is the #683 counter-vs-actuator family: a mechanism that measures perfectly
and actuates nothing is indistinguishable, at the moment it matters, from one
that was never wired.

WHY THE ACTUATOR IS A SPLIT AND NOT A REFUSAL
---------------------------------------------
``corridor_admission``'s docstring refuses a rank-local REFUSAL, and it is
right to: the guard reads THIS rank's NVML free column, so a refusal taken on
it is a rank-local answer to a question that decides how much work the GROUP
takes on -- the desync that left a scheduler not heartbeating with every rank
alive (HANDOFF_675 1a). It also names the prerequisite for a refusing gate:
"a reduction on this path".

A SPLIT needs no such reduction to be SAFE, only to be UNIFORM, and those are
different requirements met in different places:

  * SAFE, because the transient is linear in the chunk width (the terms of
    ``ServerArgs.gdn_prefill_scratch_mib`` are all first-order in ``T`` bar a
    ``ceil(T/64)``), so a chunk that does not fit has a narrower prefix that
    does. Nothing is refused and no request is starved -- the same tokens are
    admitted over more passes.
  * UNIFORM, because the width is applied through the SAME reduction the pool
    budget already takes (see ``corridor_admission.uniform_corridor_width``);
    every rank cuts to the group minimum or none of them cuts at all.

This module is the arithmetic half, and it is deliberately pure: no torch, no
NVML, no scheduler, no rig constant. It takes a budget in bytes and a price
function and returns a width. It can therefore be tested without a GPU, which
is the property the mechanism it replaces never had.

THE PRICE IS SUPPLIED, NEVER INVENTED
-------------------------------------
There are three estimators in this fork and they disagree by orders of
magnitude: the forward-peak probe (measured, needs ``SGLANG_FORWARD_PEAK_PATH``),
the metrics-reporter geometry (a movement proxy, needs ``--enable-mfu-metrics``)
and ``ServerArgs.gdn_prefill_scratch_mib`` (config-derived, always available,
and the one whose omission of ``in_proj_qkvz`` set the threshold that let the
OOM through). Choosing between them is the caller's job. This module takes
whatever the caller resolved as a callable and never substitutes a literal --
an unpriced chunk is returned UNCUT, because a cut taken on an invented number
is a throughput loss with no safety behind it.
"""

from __future__ import annotations

from typing import Callable, Optional

#: Smallest width this actuator will ever cut to, in tokens.
#:
#: THE FLOOR IS A LIVENESS PROPERTY, NOT A TUNING KNOB. A width of 0 admits
#: nothing, and a prefill that admits nothing this pass admits nothing next
#: pass either -- the corridor does not recover on its own while the request
#: that would free it is the one being refused. That is a deadlock, and a
#: deadlock is the failure this actuator's own siblings (corridor_guard.py:739
#: on the arming floor, 411 abandons and 0 completions in 6 minutes) were
#: written to avoid.
#:
#: So the cut saturates: below this width the actuator stops narrowing, admits,
#: and lets the ladder and the SHORT counter speak -- which is exactly the
#: pre-#794 behaviour, now reached only after the width has already been cut as
#: far as it can go instead of at full width.
MIN_CHUNK_TOKENS = 64


def fundable_chunk_tokens(
    *,
    requested_tokens: int,
    budget_bytes: float,
    price_bytes: Optional[Callable[[int], Optional[float]]],
    granularity_tokens: int = 1,
    min_tokens: int = MIN_CHUNK_TOKENS,
) -> int:
    """The widest chunk <= ``requested_tokens`` whose transient fits the budget.

    ``budget_bytes`` is what the card may spend on the forward's transient
    WITHOUT crossing the arming floor -- i.e. free-minus-floor, measured after
    the relief ladder has run, so the actuator narrows only what relief could
    not fund. ``price_bytes(width)`` returns the transient that width costs, or
    None when it cannot be priced.

    NOT PRICED MEANS NOT CUT. A None price (no estimator, an unreadable config,
    a raising probe) returns ``requested_tokens`` unchanged. The alternative --
    assuming a price -- would narrow the chunk on every boot where the probe is
    off, which is most of them, and would trade a measured OOM for an unmeasured
    throughput loss.

    The price is assumed MONOTONE NON-DECREASING in the width, which every
    estimator here is by construction (each term is a positive multiple of
    ``T`` or of ``ceil(T / FLA_CHUNK_SIZE)``). The search is a bisection rather
    than a division so a price with a step in it -- the ``ceil`` term is one --
    is inverted exactly instead of linearly approximated.
    """
    requested = int(requested_tokens or 0)
    if requested <= 0:
        return requested
    if price_bytes is None:
        return requested
    floor = max(1, int(min_tokens))
    grain = max(1, int(granularity_tokens))
    if requested <= floor:
        # Already at or below the liveness floor: there is nothing to cut that
        # would not be a stall. Admit and let the ladder speak.
        return requested

    def _price(width: int) -> Optional[float]:
        try:
            value = price_bytes(int(width))
        except Exception:  # noqa: BLE001 -- an estimator must never break admission
            return None
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    full = _price(requested)
    if full is None:
        return requested
    budget = float(budget_bytes)
    if full <= budget:
        # The corridor funds the chunk as asked. THE COMMON CASE, and it must
        # cost one price evaluation and no cut: an actuator that trims a chunk
        # the card could carry is a throughput regression wearing a safety
        # jacket.
        return requested
    # Bisect for the widest FUNDABLE width. `lo` is always fundable-or-floor,
    # `hi` is always known-unfundable, so the loop terminates on lo+1 == hi
    # with lo the answer.
    lo, hi = floor, requested
    while hi - lo > 1:
        mid = (lo + hi) // 2
        priced = _price(mid)
        if priced is None:
            # The estimator answered for the full width and refuses a narrower
            # one. Cutting on a half-answer is worse than not cutting.
            return requested
        if priced <= budget:
            lo = mid
        else:
            hi = mid
    # Align DOWN to the caller's granularity, but never below the floor: an
    # unaligned width is legal here, it is merely wasteful downstream.
    aligned = (lo // grain) * grain
    return max(floor, aligned if aligned > 0 else lo)


def width_was_cut(requested_tokens: int, granted_tokens: int) -> bool:
    """Whether the actuator actually narrowed this pass.

    Exists so callers count the EVENT and not the call: the gate this replaces
    logged on every admission and actuated on none, and the counter that would
    have caught that is one that only moves when the width changes.
    """
    return int(granted_tokens) < int(requested_tokens)
