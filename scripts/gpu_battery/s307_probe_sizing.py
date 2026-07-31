#!/usr/bin/env python3
"""Pure sizing helpers for the #307 arm B raise probe (s307_raise_probe.py).

Split out of the probe so the pressure-load arithmetic can be tested without
a server, a thread, or a socket -- see the TestArmBPressureSizing tests in
test/registered/unit/model_executor/test_ceiling_mamba_fit.py.

Why this exists: the pressure phase must occupy enough of the pool to cross
--admission-throttle-high (default 0.30) with headroom. Getting there took two
corrections, both of the same family -- a load sized against a PREDICTED
quantity instead of a MEASURED one -- and both are recorded here because each
one silently scored throttle_count 0 rather than failing loudly.

FIRST correction (#317, request count). A FIXED request count calibrated
against a predicted pool size stops proving anything the moment the fitted
pool comes out a different size. The #307-Beleg card run (2026-07-31) hit
exactly this: 24 requests were sized against the ~70-slot pool the fit's
arithmetic predicted, but the pool that actually got fitted was 90-94 slots,
and 0.30 * 90 = ~27 occupied slots was never reached by 24 concurrent
requests. Fix: read the pool the server reports, size as a fraction of it.

SECOND correction (#320 arm B, prompt length -- and it is the decisive one).
With the first fix in place the probe still scored throttle_count 0 on the
card. The reason is that the controller's pressure sample is TOKEN occupancy,
not slot occupancy: ``scheduler.py`` feeds ``replicated_pool_usage(sum(
req.seqlen for req in batch.reqs), max_total_num_tokens)``. Two consequences
the slot-based sizing above cannot see:

  * The denominator is ``max_total_num_tokens`` (437,463 on the #320 boot),
    not the 90-slot mamba pool. Crossing 0.30 needs ~131k HELD TOKENS.
  * The numerator is bounded by ``current * prompt_tokens``, because only
    ``current`` requests are admitted at a time (start 8). Extra clients
    QUEUE; they hold no tokens. At the historical 900-repeat prompt
    (~11.7k tokens) that ceiling is 8 * 11,700 = 93,600 = 0.216 of the pool
    -- short of 0.30 no matter how many clients pile up behind it.

So the binding variable is PROMPT LENGTH, and concurrency only has to keep the
admitted slots full. ``default_prompt_repeat`` sizes the prompt from the token
pool the server reports, the admitted count it reports, and the context length
it reports -- measured, not predicted, in all three.
"""

import math

PRESSURE_FRACTION = 0.4

#: Target token occupancy for the pressure phase. Above --admission-throttle-high
#: (0.30) with margin, and low enough to stay clear of the retract path -- the
#: arm's whole point is that throttling comes BEFORE retraction.
PRESSURE_TOKEN_FRACTION = 0.38

#: Tokens per repeat of the probe's filler sentence. A deliberate
#: UNDER-estimate: too few tokens per repeat means too many repeats, which
#: overshoots occupancy (safe, the context cap catches it), while an
#: over-estimate would undershoot the throttle mark and reproduce the bug this
#: helper exists to prevent.
TOKENS_PER_REPEAT = 12

#: Leave the top of the context window alone: the request also carries the
#: generated tokens, and a prompt sized to the exact limit is rejected.
CONTEXT_SAFETY = 0.75


def pool_from_info(info):
    """The fitted mamba pool size (``max_mamba_cache_size``), wherever the
    /get_server_info payload puts it -- same lookup shape as the admission
    limiter in s307_ceiling_fit.py's ``_limiter``."""
    if not isinstance(info, dict):
        return None
    for st in info.get("internal_states") or []:
        if isinstance(st, dict) and isinstance(st.get("max_mamba_cache_size"), int):
            return st["max_mamba_cache_size"]
    size = info.get("max_mamba_cache_size")
    return size if isinstance(size, int) else None


def token_pool_from_info(info):
    """``max_total_num_tokens`` -- the DENOMINATOR of the controller's pressure
    sample, and a different number from the mamba pool above."""
    if not isinstance(info, dict):
        return None
    size = info.get("max_total_num_tokens")
    if isinstance(size, int):
        return size
    for st in info.get("internal_states") or []:
        if isinstance(st, dict) and isinstance(st.get("max_total_num_tokens"), int):
            return st["max_total_num_tokens"]
    return None


def admitted_from_info(info):
    """How many requests the limiter admits concurrently right now.

    This is the multiplier on the held-token total, so it is read rather than
    assumed: ``current`` if the limiter reports one, else its ``start``."""
    if not isinstance(info, dict):
        return None
    for st in info.get("internal_states") or []:
        if not isinstance(st, dict):
            continue
        lim = st.get("admission_limiter")
        if isinstance(lim, dict):
            for key in ("current", "start"):
                if isinstance(lim.get(key), int) and lim[key] > 0:
                    return lim[key]
        if isinstance(st.get("max_running_requests_start"), int):
            return st["max_running_requests_start"]
    return None


def context_tokens_from_info(info):
    """The server's context length, the hard cap on one prompt."""
    if not isinstance(info, dict):
        return None
    if isinstance(info.get("context_length"), int):
        return info["context_length"]
    for st in info.get("internal_states") or []:
        if isinstance(st, dict) and isinstance(st.get("context_length"), int):
            return st["context_length"]
    return None


def default_prompt_repeat(
    token_pool,
    admitted,
    context_tokens,
    fraction=PRESSURE_TOKEN_FRACTION,
    tokens_per_repeat=TOKENS_PER_REPEAT,
    fallback=900,
):
    """Repeats of the filler sentence so ``admitted`` live requests hold
    ``fraction`` of ``token_pool``.

    Returns ``(repeats, note)``. The note names the binding constraint, so a
    run that CANNOT reach the throttle mark says so in its own output instead
    of reporting a quiet throttle_count of 0 -- the failure mode this whole
    module exists to convert into a fact.
    """
    if not (isinstance(token_pool, int) and token_pool > 0):
        return fallback, "no max_total_num_tokens reported; using the fixed fallback"
    if not (isinstance(admitted, int) and admitted > 0):
        return fallback, "no admitted count reported; using the fixed fallback"

    needed_per_request = (token_pool * fraction) / admitted
    repeats = max(1, math.ceil(needed_per_request / tokens_per_repeat))

    if isinstance(context_tokens, int) and context_tokens > 0:
        cap_tokens = context_tokens * CONTEXT_SAFETY
        cap_repeats = max(1, int(cap_tokens // tokens_per_repeat))
        if repeats > cap_repeats:
            reachable = admitted * cap_repeats * tokens_per_repeat / token_pool
            return (
                cap_repeats,
                f"context-capped at {cap_repeats} repeats; {admitted} admitted "
                f"requests can hold at most ~{reachable:.3f} of the token pool, "
                f"so a throttle mark above that is UNREACHABLE on this boot",
            )
    return repeats, ""


def default_concurrency(pool, fraction=PRESSURE_FRACTION, fallback=24):
    """ceil(fraction * pool), comfortably above --admission-throttle-high.

    Falls back to the historical fixed count only when no pool was reported
    at all (e.g. queried before any mamba-fit boot, so the field is absent).
    That fallback is not a calibration -- it is just "do not crash" when the
    server has nothing to size against.
    """
    if not isinstance(pool, int) or pool <= 0:
        return fallback
    return max(1, math.ceil(pool * fraction))
