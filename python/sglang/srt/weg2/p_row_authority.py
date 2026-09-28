"""Fix B: PP0 decides, the followers of group P execute its row (#631 ROW AUTHORITY, re-armed).

WHY IT WAS OFF. fb3631c434 (#1233 S0, 07.09.) un-wove the phase flip and pinned
``pp_flip_counters`` and ``pp_chain_receiver`` to None -- the value they had with the flip flag
off. Not a crash, leak or crawl finding. On the carrierless P form every follower then plans from
its own pool (``#631 ROW AUTHORITY DISABLED``), the class W27, #1004, TF and SF closed one term at
a time. The PP0 terms keyed on the carrier (#1066/#1175 withhold, #1039 floor clamp, #794
corridor) were disabled on purpose by 5c97a2286b (#973 ring commit timeout).

WHAT THIS RE-ARMS (switch ``SGLANG_WEG2_P_ROW_AUTHORITY``; default = the model registry row
(weg2/form.py ModelProfile.p_row_authority): qwen27b ON since the 28.09. registry flip, before it
per profile / --env-p in the proof boot only (operator 27.09.); nextflash OFF; no form OFF;
off = every image byte for byte as before):
  * the message counters (``phase_flip_counters.PhaseFlipCounters``, /dev/shm, swept at boot):
    the follower's non-blocking frame probe ``_pp_proxy_frame_pending`` -> receive-before-plan,
    PP0's admission row executed (#791 scheduled extents); no frame = PP0 planned nothing (#969J);
  * the chain receiver in its #1180 form (``pp_chain_receiver.PpChainReceiver``): the loop is
    the pacemaker (631row5: a per-slot wait crawled 30 s per hop; without the receiver the
    follower blocks in the chain recv and falls a slot behind, the misalignment the stub test
    pins). Its receive is bounded here (``STALL_S``, default 120 s): expiry raises the named
    ``PpChainRecvStalled`` -- a group stop, never a local decision.

NOT re-armed: ``pp_row_carrier_present`` stays False for every PP0 term unless that term's own
switch is on (all DEFAULT OFF): ``SGLANG_WEG2_P_ROW_WITHHOLD`` (#1066/#1175),
``SGLANG_WEG2_P_ROW_FLOOR_CLAMP`` (#1039), ``SGLANG_WEG2_P_ROW_CORRIDOR`` (#794). With no term
the predicate answers False -- so the told carrier (#1400/#1416e/TF/H91, ``weg2_store_told.armed``)
stays armed and the #1245 drop keeps its current answer.

START CHECK: switch on on a PP group, but the counters (any rank) or the chain receiver (a
follower) not built -> ``W-P-ROW Weg2RowAuthorityIncomplete`` at startup, by name.

INSTRUMENT: ``P-ROW-COST`` (followers, every 60 s): plan ms of passes executed from a row
(p50/p90 over the window), frames delivered, and the probe's drained/delivered/vanish/quiet/
head_other counters.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)

ENV = "SGLANG_WEG2_P_ROW_AUTHORITY"
STALL_ENV = "SGLANG_WEG2_P_ROW_CHAIN_STALL_S"
STALL_S_DEFAULT = 120.0
TERM_ENVS = {
    "withhold": "SGLANG_WEG2_P_ROW_WITHHOLD",
    "floor_clamp": "SGLANG_WEG2_P_ROW_FLOOR_CLAMP",
    "corridor": "SGLANG_WEG2_P_ROW_CORRIDOR",
}
#: scheduler attribute: True while this rank runs the re-armed row form.
ROW_ONLY_ATTR = "_weg2_p_row_only"
COST_EVERY_S = 60.0


def _on(name: str, default: bool, env=None) -> bool:
    e = os.environ if env is None else env
    raw = (e.get(name, "") or "").strip().lower()
    if not raw:
        return default
    return raw not in ("0", "false", "no", "off")


def enabled(env=None) -> bool:
    """Explicit env wins; unset -> the published form's profile default
    (ModelProfile.p_row_authority: qwen27b True since the 28.09. registry flip,
    nextflash False); no form -> off."""
    e = os.environ if env is None else env
    if (e.get(ENV, "") or "").strip():
        return _on(ENV, False, e)
    try:
        from sglang.srt.weg2.form import profile_switch_default

        return bool(profile_switch_default(ENV, False, e))
    except Exception:  # noqa: BLE001 -- no form module / no form: off
        return False


def term_on(term: str, env=None) -> bool:
    name = TERM_ENVS.get(term)
    return bool(name) and _on(name, False, env)


def stall_s(env=None) -> float:
    e = os.environ if env is None else env
    try:
        v = float(e.get(STALL_ENV, "") or STALL_S_DEFAULT)
    except ValueError:
        v = STALL_S_DEFAULT
    return v if v > 0 else STALL_S_DEFAULT


def applies(scheduler) -> bool:
    ps = getattr(scheduler, "ps", None)
    return bool(enabled() and ps is not None and int(getattr(ps, "pp_size", 1) or 1) > 1)


def build_counters(scheduler):
    """The flip-era builder, minus the flip gate (fb3631c434^ _build_pp_flip_counters)."""
    if not applies(scheduler):
        return None
    ps = scheduler.ps
    if getattr(ps, "attn_tp_rank", 0) != 0 or getattr(ps, "attn_cp_rank", 0) != 0:
        return None
    from sglang.srt.managers.phase_flip_counters import PhaseFlipCounters
    from sglang.srt.managers.phase_flip_presence import DEFAULT_PRESENCE_DIR, resolve_instance_tag

    counters = PhaseFlipCounters(
        n_ranks=ps.pp_size, rank=ps.pp_rank, directory=DEFAULT_PRESENCE_DIR,
        instance=resolve_instance_tag(),
    )
    # a previous boot's counts on this instance tag would read as messages in
    # flight and send this rank into a blocking recv for nothing (boot 15)
    counters.sweep()
    return counters


def build_chain_receiver(scheduler, counters):
    """The flip-era builder (fb3631c434^ _build_pp_chain_receiver) with a finite stall bound."""
    if not applies(scheduler) or counters is None:
        return None
    ps = scheduler.ps
    if ps.pp_rank == 0:
        return None
    if getattr(ps, "attn_tp_rank", 0) != 0 or getattr(ps, "attn_cp_rank", 0) != 0:
        return None
    from sglang.srt.managers.phase_flip_counters import CHAN_REQ
    from sglang.srt.managers.pp_chain_receiver import PpChainReceiver

    dp_offset = ps.attn_dp_rank * ps.attn_cp_size * ps.attn_tp_size
    return PpChainReceiver(
        group=scheduler.world_group.cpu_group,
        src=(ps.pp_rank - 1) * ps.tp_size + dp_offset,
        dst=ps.pp_rank * ps.tp_size + dp_offset,
        on_consumed=lambda _n: counters.bump_consumed(CHAN_REQ),
        on_blocked=getattr(scheduler, "_note_pp_chain_blocked", None),
        abort_check=getattr(scheduler, "_pp_chain_abort_check", None),
        stall_timeout_s=stall_s(),
    )


def start_check(scheduler) -> None:
    """Named refusal when the switch is on but a piece is missing."""
    if not applies(scheduler):
        return
    ps = scheduler.ps
    if getattr(ps, "attn_tp_rank", 0) != 0 or getattr(ps, "attn_cp_rank", 0) != 0:
        return
    missing = []
    if getattr(scheduler, "pp_flip_counters", None) is None:
        missing.append("pp_flip_counters")
    if ps.pp_rank != 0 and getattr(scheduler, "pp_chain_receiver", None) is None:
        missing.append("pp_chain_receiver")
    if missing:
        raise RuntimeError(
            f"W-P-ROW Weg2RowAuthorityIncomplete: {ENV}=1 on pp_rank={ps.pp_rank} "
            f"(pp_size={ps.pp_size}) but {', '.join(missing)} not built -- the row form "
            f"would run without its frame probe / pacemaker (631row5 crawl, followers "
            f"planning for themselves). Refusing to start instead of running without it."
        )
    setattr(scheduler, ROW_ONLY_ATTR, True)
    logger.info(
        "P-ROW-AUTHORITY armed pp_rank=%d pp_size=%d chain_stall_s=%.0f terms=%s "
        "(followers receive PP0's row before planning; told stays armed)",
        ps.pp_rank, ps.pp_size, stall_s(),
        {t: term_on(t) for t in TERM_ENVS})


def carrier_for(scheduler, term: Optional[str]) -> Optional[bool]:
    """pp_row_carrier_present's split: None = not the row form (the old answer
    applies); else the term's own switch (no term -> False)."""
    if not getattr(scheduler, ROW_ONLY_ATTR, False):
        return None
    return term_on(term) if term else False


# ---------------------------------------------------------------------------
# P-ROW-COST
# ---------------------------------------------------------------------------


def note_plan(scheduler, plan_ms: float) -> None:
    st = getattr(scheduler, "_p_row_cost", None)
    if st is None:
        st = scheduler._p_row_cost = {"plans": [], "t0": time.monotonic(), "frames": 0}
    st["plans"].append(float(plan_ms))
    st["frames"] += 1
    now = time.monotonic()
    if now - st["t0"] < COST_EVERY_S:
        return
    xs = sorted(st["plans"])
    n = len(xs)
    p50 = xs[n // 2] if n else 0.0
    p90 = xs[min(n - 1, int(n * 0.9))] if n else 0.0
    probe = getattr(scheduler, "_pp_row_probe_stats", None) or {}
    logger.info(
        "P-ROW-COST pp_rank=%s window_s=%.0f frames=%d plan_ms_p50=%.1f plan_ms_p90=%.1f "
        "probe calls=%s drained=%s delivered=%s vanish=%s quiet=%s head_other=%s "
        "(follower plan time AFTER the frame arrived: the cost of receive-before-plan)",
        getattr(getattr(scheduler, "ps", None), "pp_rank", "?"), now - st["t0"], n, p50, p90,
        probe.get("calls"), probe.get("drained"), probe.get("delivered"),
        probe.get("vanish_probes", 0), probe.get("quiet"), probe.get("head_other"))
    st["plans"] = []
    st["t0"] = now
