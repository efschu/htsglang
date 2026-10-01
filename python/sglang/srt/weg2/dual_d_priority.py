"""DUAL-TP3PP3 D PRIORITY UNDER KV PRESSURE (user decision 01.10., verbatim):
"das darf nie passieren. davor muss P aufhören zu prefillen und seinen kv
freigeben und wenn es dann immer noch nicht reicht, muss P schlafen und seine
gewichte im systemram parken. es geht um unterbrechungsfreies decode. wenn ein
sitz fertig decoded hat wird sein platz ja frei und P kann zurückkehren und
weitermachen".

Metal gmps7 (boot dkr27bnvfp4dual1mbar1fs10011748, D 17:53:15): "KV cache pool is
full. Retract requests. #retracted_reqs: 4" -> four W50 re-routes -> P's grant for
four contexts at once -> PP0 died of cuMemCreate OOM. In the dual layout a
retracted D decode is an interrupted decode; the stages that keep D's pool from
running full act BEFORE it (P stops and frees its KV, then P sleeps), and a
retract that still comes is a NAMED STOP here, never a silent path.

This module holds the D-side tripwire (no retract in the dual layout).
"""

from __future__ import annotations

import os

MARK = "W-DUAL-D-RETRACT"


class Weg2DualDRetract(RuntimeError):
    """W-DUAL-D-RETRACT: group D of the dual layout was about to retract a
    running decode. Decode is never interrupted in the dual layout; the pool
    must be kept clear by the P stages (P stops and frees its KV, P sleeps)."""


def d_retract_forbidden(env=None) -> bool:
    """True on a group-D rank of the dual layout (both groups carry
    SGLANG_WEG2_DUAL_LAYOUT=1)."""
    e = os.environ if env is None else env
    return ((e.get("SGLANG_WEG2_DUAL_LAYOUT", "") or "").strip() == "1"
            and (e.get("SGLANG_WEG2_GROUP", "") or "").strip().upper() == "D")


def refuse_d_retract(batch, *, kv_full: bool, reason, env=None) -> None:
    """Raise W-DUAL-D-RETRACT on a dual D rank instead of retracting. Group-
    uniform: the decision to retract is uniform over the D ranks (#583/#603),
    and this predicate reads only the env, the same on every rank."""
    if not d_retract_forbidden(env):
        return
    reqs = list(getattr(batch, "reqs", None) or ())
    rids = [str(getattr(r, "rid", "?"))[:16] for r in reqs]
    raise Weg2DualDRetract(
        "%s: group D of the dual layout was about to retract %s (kv_full=%s, reason=%s) -- decode is "
        "never interrupted here; P must have stopped, freed its KV or slept before D's pool ran full"
        % (MARK, rids, bool(kv_full), reason or "-"))
