"""EZ-S2a: the glue of ``phase_policy.decide(kind="set")`` (plan nf-entscheidungszentrale-plan-1019, S2a).

``DeciderState`` is the set of THIS D epoch (the dict ``Front._set_st`` always kept, now a named type: set, go_p, t_open,
seq). ``granted_unstarted`` is the S49 input the old ``_set_xsum_hold`` read inside the front. ``flip_held`` is the one
record of "work waits, no flip" (SOLL R7) that replaces the second set (``_set_flip_justified``'s queue sum).
Everything here is a pure helper or a record; the front keeps its calls (R11: the front decides, the ranks get finished
sets)."""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from flliper.srt.environ import envs
from flliper.srt.pdflip import decision_record as _dr
from flliper.srt.pdflip import phase_policy


def enabled() -> bool:
    """``FLLIPER_PDFLIP_DECIDER`` (default ON). Off = the S1 code path."""
    try:
        return bool(envs.FLLIPER_PDFLIP_DECIDER.get())
    except Exception:  # noqa: BLE001 -- a partial env: the switch reads ON (new features on)
        return True


class DeciderState(dict):
    """The set state of one D epoch. ``set_id`` = seq of the set, ``t_open`` = stamp of its FIRST member (R3: the timer
    starts there once), ``go_p`` = rids committed to a decided flip, ``shorts``/``released`` = the halb SHORTs on the seat
    path. A new epoch starts empty (no epoch rule, R7)."""

    def reset(self, epoch: Any) -> "DeciderState":
        self.clear()
        self.update(epoch=epoch, set_id=0, t_open=None, shorts={}, released={}, go_p=set(), seen=0, logged=None)
        return self


def granted_unstarted(front: Any, running: int, *, beacon: bool, d_pf_unstarted: int, d_pf_pending: int) -> int:
    """S49 (plan 2.6): the tokens D has been GRANTED but not begun (``d_granted_unstarted`` of ``GroupNumbers``). With the
    beacon (S56) the book's unstarted rows; without it the book's pending minus what D already runs. 0 when the X-SUM
    switch is off (nothing is held)."""
    try:
        if not envs.FLLIPER_PDFLIP_ENABLE_DECODE_COLLECT_PREFILL_BUSY.get():
            return 0
    except Exception:  # noqa: BLE001
        return 0
    return max(0, int(d_pf_unstarted) if beacon else int(d_pf_pending) - int(running))


def flip_held(front: Any, *, epoch: int, n: int, s: int, x: int, why: str) -> None:
    """SOLL R7 (plan A9, B3): queued work that is no set (kurz, or a set under X) is no flip reason. The old FLIP-HELD line
    stays (alias, monitors read it) and the same fact is ONE edge-triggered ``DECISION rule=R7 action=FLIP_HELD`` record."""
    try:
        _dr_rec = front._dec()
        _dr_rec.record(_dr.Decision(kind="set", action="FLIP_HELD", rule="R7", why=why, s_tokens=int(s), s_p_tokens=int(s),
                                    x=int(x), **front._dec_group()), edge=True)
    except Exception:  # noqa: BLE001 -- an instrument never breaks a verdict
        pass


def rider_under_x(front: Any, rid: str, u: int, x_own: int, why: str) -> None:
    """A5/R7: a request D refused although its exact u is under X (the 140-474 class, a requeue/carrier/deferred under X): a
    DEFECT report, never a flip reason. One edge record per rid."""
    try:
        front._dec().record(_dr.Decision(kind="set", action="WAIT_SEAT", rule="R7", rid=str(rid),
                                         why="d_refused_under_x u=%d X=%d %s" % (int(u), int(x_own), why),
                                         x=int(x_own), **front._dec_group()), edge=True)
    except Exception:  # noqa: BLE001
        pass
