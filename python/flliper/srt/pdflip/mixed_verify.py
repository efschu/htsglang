# SPDX-License-Identifier: Apache-2.0
"""MIXED-VERIFY (Stufe 1, 29.10.): the running decode rows keep their speculative round in a mixed step.

Order (user 29.10. evening): "ja implementieren, oder das PR/Fork nehmen, oder abschauen" -- mixed chunk WITH spec verify on the 27B
(DFLASH). The 1-token-tail mix (PR #36933, ``ScheduleBatch.mix_with_running``) gives a running decode row ONE token per mixed forward
and drops its draft state; a row's speculative round yields E tokens (about 3.4 at k=8, order figure, not measured here).

What this file is: the arm / eligibility / census side of MIXED-VERIFY stage 1. Stage 1 does not fuse the two forwards (the 27B target is
hybrid GDN: a single forward holding prefill rows AND verify rows needs a GDN split per layer, and PR #41893 -- the upstream
verify-merged mixed step -- refuses hybrid linear-attention targets itself; GDN extend vs decode kernels are not bit-identical, SGLang
issue #46446). Instead the scheduler runs, where the MIXED-D gate (``mixed_d_short.py``) says yes, the DECODE batch (its normal
draft + verify + accept + commit round, the same worker code as a separate run) and then the PREFILL batch in the SAME scheduler pass,
one stream, fixed order. The decode rows are bit-identical to the separate run by construction. Cost: one decode round more per mixed
step than a single fused forward would take (stage 2, not built: deskq/bl/mixed_verify_entwurf_1029.md section 6).

Rank agreement (R11): every term below is either a process constant (:func:`armed`) or rides the ONE group MIN the gate already takes
(``Scheduler._mixed_d_agreed``: ``[mine, mv_local]`` in a single all_reduce); no rank-local batch shape.
Switch ``FLLIPER_PDFLIP_MIXED_VERIFY`` (default ON, ``=0`` = the 1-token-tail mix byte for byte).
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)
MARK = "MIXED-VERIFY"

#: slots of the overlap keep-alive ring per scheduler pass: two forwards per pass, each pinned for the one-pass result offset
RING_SLOTS_ARMED = 4
RING_SLOTS_DEFAULT = 2

_STATE = {"steps": 0, "fallbacks": {}}


def enabled() -> bool:
    """Default ON (new feature); ``FLLIPER_PDFLIP_MIXED_VERIFY=0`` = the old 1-token-tail mix."""
    from flliper.srt.environ import envs

    return bool(envs.FLLIPER_PDFLIP_MIXED_VERIFY.get())


def armed(
    *,
    mixed_d_short: bool,
    is_dflash: bool,
    pp_size: int,
    dp_attention: bool,
    two_batch_overlap: bool,
    pdmux: bool,
) -> bool:
    """Process constant, decided once at scheduler init: may this scheduler ever run a mixed step as decode round + prefill?

    Needs the MIXED-D gate (the verdict is the gate's), the DFLASH worker (the only worker whose decode round this stage was
    reviewed against), a single pipeline stage (the PP loops build their batches elsewhere), and none of the loops that
    synchronise or split the batch before the forward (DP attention, two-batch overlap, pdmux)."""
    if not enabled():
        return False
    if not (mixed_d_short and is_dflash):
        return False
    if int(pp_size) != 1:
        return False
    return not (dp_attention or two_batch_overlap or pdmux)


def refusal_reason(*, n_running: int, has_grammar: bool, spec_state_ok: bool, pp_size: int = 1) -> str:
    """Census name of the rank-local eligibility (first matching term); "" = this rank says yes.

    ``has_grammar``: a running request holds a grammar (overlap + spec + grammar syncs the loop per decode batch,
    ``Scheduler.is_disable_overlap_for_batch``: the 1-token-tail path keeps its old behaviour there).
    ``spec_state_ok``: the running batch carries a ``DFlashDraftInputV2`` (the worker's decode branch raises on anything else).
    ``pp_size``: the pipeline width NOW (a phase-flip process changes layout at run time; :func:`armed` read it once at init)."""
    if int(n_running) <= 0:
        return "no_running"
    if int(pp_size) != 1:
        return "pp_layout"
    if has_grammar:
        return "grammar"
    if not spec_state_ok:
        return "no_draft_state"
    return ""


def decide(gate_verdict: int, mv_verdict: int, armed_here: bool) -> bool:
    """Combine the two group-MIN flags: mixed-verify runs only when the MIXED-D gate agreed AND every rank is eligible."""
    return bool(armed_here and gate_verdict and mv_verdict)


def note_step(n_decode_rows: int, n_prefill_tokens: int) -> None:
    """One line per mixed-verify step (cumulative count rides on it)."""
    _STATE["steps"] += 1
    logger.info(
        "%s step n=%d n_decode_rows=%d n_prefill_tok=%d (decode round + prefill forward in one scheduler pass)",
        MARK, _STATE["steps"], n_decode_rows, n_prefill_tokens,
    )


def note_fallback(reason: str) -> None:
    """Count a gate-yes step that took the 1-token-tail mix instead (named reason; census every 64 on the line)."""
    fb = _STATE["fallbacks"]
    fb[reason] = fb.get(reason, 0) + 1
    n = sum(fb.values())
    if n % 64 == 1:
        logger.info(
            "%s fallback n=%d reasons=%s (a gate-yes step that kept the 1-token-tail mix; reason = first blocking term)",
            MARK, n, dict(sorted(fb.items())),
        )

