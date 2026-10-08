# SPDX-License-Identifier: Apache-2.0
"""#536 -- a KV-headroom reserve the fast lane can always admit into.

THE PREMISE, ESTABLISHED AT CODE RATHER THAN ASSUMED.

The fast lane is ALREADY first in the queue. ``PrefillAdder``'s ordering
(``schedule_policy.py:385-432``) sorts fast requests ahead of heavy ones, and
#552's aging promotes an aged heavy request only to ``fast_lane_priority - 1``
-- BELOW the fast tier, never above it. Its docstring says why in its own
words: *"a heavy req cannot preempt, so promoting it ABOVE fast would only
wedge the admission loop and starve the fast lane"*.

So the 34.5 s first-token under a co-tenant's prefill is **not an ordering
defect**. Being first in a queue does not produce memory. The #536 mechanism
note puts it exactly: *"priority orders the queue; it cannot release memory
another request holds"* -- a heavy prefill's chunks are unpreemptible, so the
fast request sits at the head of the queue with nothing to be admitted into
until the heavy prefill finishes.

That is why the remedy on record is a **reserve**, not chunk preemption: give
the fast lane a token budget the heavy lane structurally cannot consume, so
"first in the queue" becomes "first AND admissible".

DARK BY DEFAULT. The operator decision gated this on a lane-ON/OFF live
observation, and no such observation exists on record -- the only ``fast_lane``
strings in the #665-F1 boot logs are the ``server_args`` echo, not a
measurement. So this module ships default-off: ``solve`` returns a zero reserve
unless a reserve is explicitly asked for, and the caller draws nothing. The
live reproduction is filed as the window item.

NO SILENT SHRINK (#584-conform). If the requested reserve cannot be funded from
the pool at boot, this refuses BY NAME with the numbers. A reserve quietly
rounded down to what fits is worse than none: it reports protection it does not
provide.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

#: Provenance labels. A reserve without one is not judgeable -- the whole point
#: of pricing it is that a later reader can tell a measurement from a guess.
PROV_DECLARED = "declared"  # operator stated the fast-lane prompt ceiling
PROV_CHUNK = "chunk_derived"  # derived from the admission chunk size
PROV_ABSENT = "absent"  # no reserve: the default, dark path


class FastLaneReserveError(ValueError):
    """A reserve that cannot be honoured. Never downgraded to a warning."""


@dataclasses.dataclass(frozen=True)
class ReservePlan:
    """What the fast lane may draw on, and where the number came from."""

    tokens: int
    provenance: str
    #: The quantity the reserve was priced against, for the log line.
    priced_against: str
    enabled: bool

    @property
    def is_dark(self) -> bool:
        return not self.enabled or self.tokens <= 0


def solve_fast_lane_reserve(
    *,
    enabled: bool,
    pool_total_tokens: int,
    fast_lane_max_prompt_tokens: Optional[int],
    chunked_prefill_size: int,
    min_pool_tokens_after_reserve: int,
) -> ReservePlan:
    """Price the reserve, or refuse by name.

    THE SIZE IS NOT A MAGIC NUMBER. A fast request needs enough room to reach
    its FIRST TOKEN while a heavy prefill holds the rest of the pool. That is
    one admission's worth of prompt: the operator's declared fast-lane prompt
    ceiling when there is one, and otherwise one admission chunk -- because a
    chunk is the largest slice the admission path will take in one pass, and
    sizing below it cannot admit even a single chunk.

    Refuses rather than shrinking when the reserve would drive the remaining
    pool under the corridor floor the deployment must still serve. A reserve
    that eats the floor trades one starvation for another.
    """
    if not enabled:
        return ReservePlan(0, PROV_ABSENT, "disabled", False)

    if fast_lane_max_prompt_tokens is not None:
        if int(fast_lane_max_prompt_tokens) <= 0:
            raise FastLaneReserveError(
                f"declared fast-lane prompt ceiling {fast_lane_max_prompt_tokens} "
                "is not positive; a reserve of nothing is not a reserve."
            )
        tokens = int(fast_lane_max_prompt_tokens)
        prov, against = PROV_DECLARED, "fast_lane_max_prompt_tokens"
    else:
        if int(chunked_prefill_size) <= 0:
            raise FastLaneReserveError(
                "no fast-lane prompt ceiling was declared and the chunk size is "
                f"{chunked_prefill_size}; there is nothing to price the reserve "
                "against. Declare --fast-lane-max-prompt-tokens rather than "
                "letting the reserve be guessed."
            )
        tokens = int(chunked_prefill_size)
        prov, against = PROV_CHUNK, "chunked_prefill_size"

    remaining = int(pool_total_tokens) - tokens
    if remaining < int(min_pool_tokens_after_reserve):
        raise FastLaneReserveError(
            f"fast-lane reserve of {tokens:,} tokens (priced against {against}) "
            f"leaves {remaining:,} of {int(pool_total_tokens):,} pool tokens, "
            f"under the {int(min_pool_tokens_after_reserve):,} the deployment "
            "must still serve. REFUSED rather than shrunk: a reserve rounded "
            "down to what fits reports protection it does not provide, and "
            "trading the corridor floor for the fast lane is one starvation "
            "for another."
        )
    return ReservePlan(tokens, prov, against, True)


def admissible_tokens(
    *,
    is_fast_lane: bool,
    pool_free_tokens: int,
    reserve: ReservePlan,
    reserve_in_use_tokens: int = 0,
) -> int:
    """How many tokens THIS request may draw. The whole mechanism, in one rule.

    A heavy request sees the pool MINUS the untouched part of the reserve, so
    it structurally cannot consume the fast lane's headroom however long its
    prefill runs. A fast request sees the whole free pool: the reserve is a
    floor under it, not a ceiling on it.

    Byte-identical to the pre-#536 path when the reserve is dark -- the heavy
    subtraction is zero and both lanes see ``pool_free_tokens``.
    """
    free = int(pool_free_tokens)
    if reserve.is_dark:
        return free
    if is_fast_lane:
        return free
    withheld = max(0, reserve.tokens - int(reserve_in_use_tokens))
    return max(0, free - withheld)


def describe(reserve: ReservePlan) -> str:
    if reserve.is_dark:
        return "fast-lane KV reserve: DARK (no tokens withheld from the heavy lane)"
    return (
        f"fast-lane KV reserve: {reserve.tokens:,} tokens withheld from the "
        f"heavy lane, priced against {reserve.priced_against} "
        f"[{reserve.provenance}]"
    )


# ---------------------------------------------------------------------------
# The combined invariant with #552's aging, stated because the two mechanisms
# look like they should fight and do not.
#
#   #552 (ordering)  an aged heavy request is promoted to
#                    fast_lane_priority - 1: ahead of un-aged heavy work,
#                    NEVER ahead of a fast request.
#   #536 (memory)    the heavy lane's admissible tokens exclude the untouched
#                    reserve, whatever its queue position.
#
# So an aged heavy request can win the next freed slot ahead of fresher heavy
# work -- which is what #552 exists to do -- and still cannot take the fast
# lane's headroom, because the reserve is subtracted from what it may draw
# rather than from where it sits in the queue. One acts on ORDER, the other on
# MEMORY, and they never contend for the same quantity.
#
# The framing "aged heavy beats the fast tier for ONE admission" would be a
# DIFFERENT and more dangerous design: #552 deliberately does not do that, and
# its docstring says promoting a heavy request above the fast tier "would only
# wedge the admission loop and starve the fast lane" -- the very defect #536 is
# about.
# ---------------------------------------------------------------------------
