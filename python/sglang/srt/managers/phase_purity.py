"""#631 STRICT PHASE PURITY -- each layout runs only the work it is for.

THE USER RULE THIS ENCODES (2026-08-09, hard, default on this rig)
------------------------------------------------------------------
- Decode in the PP layout is COMPLETELY FORBIDDEN. Decode work is DEFERRED
  and executed BATCHED in the TP layout after the flip.
- NOT A SINGLE TOKEN may be prefilled in the TP layout. Prefill is deferred
  and executed ONLY in the PP layout.

The server therefore alternates:

    accumulate prefill -> flip to PP -> run ALL pending prefill
    (new decode work queues) -> flip to TP -> run ALL deferred decode,
    batched, with graphs and speculation (new prefill queues) -> repeat

WHY, in one measurement
-----------------------
Without this rule the server pins itself in the layout that is wrong for
the work it is doing. Metal, 2026-08-09 21:15:25-21:16:15Z, live agent
traffic:

    PHASE-POLICY holding in pp: prefilling in pp (302757 tok pending,
    running bs 2)                                        x 6 samples
    same window: 87 Decode batch records, 6 Prefill batch records,
    cuda graph: False, gen throughput 35.4 tok/s

87 decode batches executed in the PREFILL layout -- no speculation, no
decode CUDA graphs, 35 tok/s -- while the policy refused to leave PP
because prefill was pending, and prefill barely advanced because decode
was taking the rounds. Each half starved the other in the same layout.
Purity removes the interleaving that makes that state reachable.

THE COST, accepted by the ordering user and stated plainly
-----------------------------------------------------------
A request mid-decode when prefill pressure arrives is PAUSED: it stays
resident, is carried across the cutover by the existing resident-carry and
draft-bootstrap machinery, and resumes -- batched -- in the next TP window.
Its inter-token latency therefore contains an entire PP window. That is a
deliberate trade of tail latency for layout-correct throughput on both
sides, not an oversight.

MODES
-----
``--phase-flip-purity``:

    strict          (DEFAULT) no decode in PP, no prefill in TP.
    threshold:<n>   ESCAPE HATCH: decode may still run in the PP layout
                    while at most <n> requests are decoding. Above <n> the
                    strict rule applies again. n=0 is exactly ``strict``.
                    The prefill-in-TP prohibition is NOT relaxed by this --
                    prefill in TP is a 4.3x throughput loss with no
                    latency argument on the other side, so it has no
                    threshold form.
    off             both prohibitions lifted: the pre-purity interleaving.
                    Kept reachable so the defect above can be reproduced
                    on demand for an A/B, not because it is a supported
                    way to run.

Anything else is a loud parse error. A silently-ignored purity setting
would serve decode from the prefill layout while the operator believes it
cannot happen, which is the exact failure this module exists to make
impossible.

THE DEADLOCK THIS OPENS, and what closes it
--------------------------------------------
Purity makes a new state reachable: the PP phase has pending prefill it
CANNOT admit (no free mamba/GDN state slot, or the KV pool is held by
paused decodes), and decode -- which would release those slots -- is
forbidden here. Nothing progresses, and the load-triggered PP->TP rule
(``pending <= N``) is false, so nothing flips either.

The closer is ``phase_policy``'s bounded PP window (``pp_window_s``): after
a bounded time in PP with decode work waiting, PP->TP arms REGARDLESS of
pending prefill. Under purity that window is not a fairness nicety -- it is
the deadlock breaker, and disabling it (0) while purity is strict is
therefore refused by ``validate_purity_policy_pair``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

LOG_PREFIX = "PHASE-PURITY"

MODE_STRICT = "strict"
MODE_THRESHOLD = "threshold"
MODE_OFF = "off"

ENV_PURITY = "SGLANG_PHASE_FLIP_PURITY"
DEFAULT_PURITY = MODE_STRICT


class PhasePurityError(ValueError):
    """An unusable purity configuration. Raised, never defaulted around."""


@dataclass(frozen=True)
class PhasePurity:
    """The parsed rule. Immutable; built once at boot."""

    mode: str = DEFAULT_PURITY
    #: Only meaningful for ``threshold``: how many requests may decode in
    #: the PP layout before the strict prohibition applies again.
    decode_in_pp_threshold: int = 0

    @property
    def strict(self) -> bool:
        return self.mode == MODE_STRICT

    @property
    def enforced(self) -> bool:
        return self.mode != MODE_OFF

    def decode_allowed_in_pp(self, running_bs: int) -> bool:
        """May a decode batch execute in the PP layout right now?

        ``running_bs`` is the number of requests that would decode. The
        threshold is on the BATCH, not on a per-request count, because the
        cost being traded is one decode step in the slow layout.
        """
        if self.mode == MODE_OFF:
            return True
        if self.mode == MODE_STRICT:
            return False
        return int(running_bs) <= self.decode_in_pp_threshold

    def prefill_allowed_in_tp(self) -> bool:
        """May a prefill batch be BUILT in the TP layout right now?

        No threshold form on purpose -- see the module docstring.
        """
        return self.mode == MODE_OFF

    def describe(self) -> str:
        if self.mode == MODE_THRESHOLD:
            return f"threshold:{self.decode_in_pp_threshold}"
        return self.mode


def parse_purity(raw: Optional[str]) -> PhasePurity:
    """Parse ``strict`` / ``threshold:<n>`` / ``off``. Loud on anything else."""
    if raw is None or raw == "":
        return PhasePurity(mode=DEFAULT_PURITY)
    value = str(raw).strip().lower()
    if value == MODE_STRICT:
        return PhasePurity(mode=MODE_STRICT)
    if value == MODE_OFF:
        return PhasePurity(mode=MODE_OFF)
    if value.startswith(MODE_THRESHOLD):
        rest = value[len(MODE_THRESHOLD) :]
        if not rest.startswith(":"):
            raise PhasePurityError(
                f"{LOG_PREFIX} purity {raw!r} is missing its count; the "
                f"threshold mode is written 'threshold:<n>', e.g. "
                f"'threshold:2' to allow decode in the PP layout while at "
                f"most 2 requests are decoding"
            )
        try:
            n = int(rest[1:])
        except ValueError:
            raise PhasePurityError(
                f"{LOG_PREFIX} purity {raw!r} has a non-integer count "
                f"{rest[1:]!r}; write 'threshold:<n>'"
            )
        if n < 0:
            raise PhasePurityError(
                f"{LOG_PREFIX} purity threshold {n} is negative; use 0 (which "
                f"is exactly 'strict') or a positive count"
            )
        if n == 0:
            return PhasePurity(mode=MODE_STRICT)
        return PhasePurity(mode=MODE_THRESHOLD, decode_in_pp_threshold=n)
    raise PhasePurityError(
        f"{LOG_PREFIX} unknown purity {raw!r}; valid values are "
        f"'{MODE_STRICT}' (default), 'threshold:<n>', '{MODE_OFF}'"
    )


def purity_from_server_args(server_args) -> PhasePurity:
    """Boot resolution: the server arg wins, the environment is the
    fallback so an A/B needs no boot-script edit."""
    raw = getattr(server_args, "phase_flip_purity", None)
    if raw in (None, ""):
        raw = os.environ.get(ENV_PURITY)
    purity = parse_purity(raw)
    logger.warning(
        "%s mode=%s -- decode in the PP layout: %s; prefill in the TP "
        "layout: %s",
        LOG_PREFIX,
        purity.describe(),
        "forbidden"
        if purity.strict
        else (
            "allowed"
            if purity.mode == MODE_OFF
            else f"allowed up to bs {purity.decode_in_pp_threshold}"
        ),
        "allowed" if purity.prefill_allowed_in_tp() else "forbidden",
    )
    return purity


def validate_purity_policy_pair(purity: PhasePurity, policy_cfg) -> None:
    """Refuse the one combination that deadlocks.

    With purity enforced, the PP phase can reach a state where it may not
    decode and cannot admit prefill (no free state slot). The only exit is
    the policy's bounded PP window; without it the instance parks forever
    with both queues non-empty. Caught at boot rather than at 03:00.
    """
    if not purity.enforced:
        return
    window = float(getattr(policy_cfg, "pp_window_s", 0.0) or 0.0)
    if window > 0:
        return
    raise PhasePurityError(
        f"{LOG_PREFIX} purity={purity.describe()} requires a positive "
        f"phase-policy PP window, but pp_window_s is {window!r}. Under "
        f"purity the PP phase may not decode, so a PP phase that cannot "
        f"admit its pending prefill (no free mamba/GDN slot, KV held by "
        f"paused decodes) has NO exit except the bounded window: the "
        f"load-triggered rule needs prefill to drain, and prefill cannot "
        f"drain. Set SGLANG_PHASE_POLICY_PP_WINDOW_S > 0, or run "
        f"--phase-flip-purity off."
    )


def _active_phase(scheduler) -> Optional[str]:
    """The layout this scheduler is currently running, or None when the
    flip is not enabled at all (then nothing is gated)."""
    if not getattr(scheduler.server_args, "enable_phase_flip", False):
        return None
    return getattr(scheduler, "phase_flip_active_stack", None)


def prefill_blocked_here(scheduler) -> bool:
    """True when a prefill batch must NOT be built this iteration.

    Rank-uniform by construction, which is the load-bearing property: the
    purity rule is static boot config and the active layout is replicated
    (a cutover commits on every rank or on none). A rank-local input here
    would split the group across branches with mismatched collectives --
    the failure family documented at ``_update_uniform_pool_budget``.
    """
    from sglang.srt.managers.phase_policy import PHASE_TP

    if _active_phase(scheduler) != PHASE_TP:
        return False
    return not purity_of(scheduler).prefill_allowed_in_tp()


def decode_blocked_here(scheduler, running_bs: int) -> bool:
    """True when a decode step must NOT execute this iteration."""
    from sglang.srt.managers.phase_policy import PHASE_PP

    if _active_phase(scheduler) != PHASE_PP:
        return False
    return not purity_of(scheduler).decode_allowed_in_pp(running_bs)


def purity_of(scheduler) -> PhasePurity:
    """The scheduler's purity rule, resolved once and cached."""
    cached = getattr(scheduler, "_phase_purity", None)
    if cached is not None:
        return cached
    purity = purity_from_server_args(scheduler.server_args)
    scheduler._phase_purity = purity
    return purity
