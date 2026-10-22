"""TOPUP (20.10.): an open lend grows back to what the card holds -- free VRAM = experts.

THE METAL. rc27 D (boot ..._893d09fa3f_1020_145331): TP0 bs2 rows_on p50 13 against rc26 24, lent in 13 % of
the bs2 rounds by the old tracker (rc26 98 %); at the SAME KV stage rc27 decodes slower (bs2 S5 33.5 vs 31.2 ms,
S6 34.4 vs 32.1, S7 37.9 vs 29.9, S8 40.1 vs 34.5). Same requests, adjacent rounds: every lend made the
decode faster (TP0 123 lends, gpu-ms p50 -18.0, 0 of 123 slower; TP1 -19.8, TP2 -16.4; control splits without a
move -0.6 ... -5.6). Cause: after S106 no stage move drops the lend any more, and nothing let it grow back --
the gate tops up only after a partial return, and the extend-time KEEP-SHORT ``keep`` cleared that very flag.
The lend only ever shrank (KEEP-SHORT returns, stage moves paying KV with lent rows).

What must hold (desk rank of test_s78_d_lend_keep_short_1019):
(1) at the settle of every decode streak an open lend maps the difference the card now holds (boundary rule);
(2) a phase move that KEEPS the lend arms a top-up, also mid-streak;
(3) a ``keep`` (nothing moved) never cancels a top-up a partial return armed;
(4) no churn: within one streak a full lend is not touched again; a top-up with nothing to add is no move.
"""
from __future__ import annotations

import importlib.util
import os
import types
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import d_transient_lend as dtl  # noqa: E402

_HERE = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("_t_topup_s78", os.path.join(_HERE, "test_s78_d_lend_keep_short_1019.py"))
S78 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S78)
h, STEP, NEED = S78.h, S78.STEP_BYTES, S78.NEED
_fresh_trim_cache = S78._fresh_trim_cache  # noqa: F401 -- autouse


def _modes():
    from flliper.srt.model_executor.forward_batch_info import ForwardMode

    return (types.SimpleNamespace(forward_mode=ForwardMode.DECODE),
            types.SimpleNamespace(forward_mode=ForwardMode.EXTEND, extend_num_tokens=100))


class _Free:
    def __init__(self, v):
        self.v = v

    def __call__(self):
        return self.v


def _ctx(tms, free):
    from flliper.srt.environ import envs

    return [h._armed(tms)] + S78._env() + [envs.FLLIPER_PDFLIP_D_LEND_SETTLE_ROUNDS.override(2),
                                           mock.patch.object(dtl, "_card_free_bytes", free)]


class _Stack:
    def __init__(self, cms):
        self.cms = cms

    def __enter__(self):
        for c in self.cms:
            c.__enter__()

    def __exit__(self, *a):
        for c in reversed(self.cms):
            c.__exit__(*a)


def test_rc27_the_lend_grows_back_at_the_next_settle():
    """(1) a full lend that never returned anything (topup not armed): a short extend keeps it, the extend
    runs, and at the settle of the next decode streak the card's headroom becomes rows again."""
    decode, extend = _modes()
    tms = h.CoreTms()
    free = _Free(NEED)
    with _Stack(_ctx(tms, free)):
        r, ctl = S78._live_lent(tms)
        assert ctl.unlend_to(36) == 4 and ctl.rows_on == 36       # an earlier return shrank it ...
        dtl._gate(r.sched).topup = False                          # ... and nothing armed a top-up
        assert dtl.round_start(r.sched) == "keep"                 # the short extend fits: no move
        assert dtl.on_batch(r.sched, extend) == "keep" and ctl.rows_on == 36
        r.sched.waiting_queue = []
        free.v = 10 * STEP
        assert dtl.on_batch(r.sched, decode) is None              # streak 1: settling
        assert dtl.on_batch(r.sched, decode) == "lend"            # streak 2 = settle: the top-up
        assert ctl.rows_on == 40 and ctl.lent_from == 32
        for _ in range(5):                                        # (4) the same streak: no further move
            assert dtl.on_batch(r.sched, decode) is None
        assert ctl.rows_on == 40


def test_a_kept_lend_arms_a_top_up_mid_streak():
    """(2) a stage move that kept the whole lend arms a top-up -- the card changed (a shrink frees KV)."""
    decode, _extend = _modes()
    tms = h.CoreTms()
    free = _Free(10 * STEP)
    with _Stack(_ctx(tms, free)):
        r, ctl = S78._live_lent(tms)
        r.sched.waiting_queue = []
        assert ctl.unlend_to(36) == 4
        g = dtl._gate(r.sched)
        g.topup, g.streak = False, 50                             # deep in a decode streak, nothing armed
        assert dtl.on_batch(r.sched, decode) is None and ctl.rows_on == 36
        dtl.after_phase_move(r.sched, ctl, int(ctl.rows_on))       # a stage move kept the lend whole
        assert g.topup is True
        assert dtl.on_batch(r.sched, decode) == "lend" and ctl.rows_on == 40


def test_a_keep_never_cancels_an_armed_top_up():
    """(3) KEEP-SHORT gave rows back (top-up armed); later passes only KEEP (a held request re-prices);
    the request leaves without an extend -- mid-streak, the armed top-up still maps the rows back."""
    decode, _extend = _modes()
    tms = h.CoreTms()
    free = _Free(NEED - STEP // 2)
    with _Stack(_ctx(tms, free)):
        r, ctl = S78._live_lent(tms)
        g = dtl._gate(r.sched)
        assert dtl.round_start(r.sched) == "return" and ctl.rows_on == 36 and g.topup is True
        free.v = NEED
        assert dtl.round_start(r.sched) == "keep" and g.topup is True     # the keep moved nothing
        r.sched.waiting_queue = []                                        # the request left (abort / P)
        g.streak = 50
        free.v = 10 * STEP
        assert dtl.on_batch(r.sched, decode) == "lend" and ctl.rows_on == 40


def test_a_top_up_with_nothing_to_add_moves_nothing():
    """(4) the card holds no further lattice step: the settle's top-up is no move and no line."""
    decode, extend = _modes()
    tms = h.CoreTms()
    free = _Free(STEP // 2)
    with _Stack(_ctx(tms, free)):
        r, ctl = S78._live_lent(tms)
        r.sched.waiting_queue = []
        before = ctl.lent_rows
        dtl.on_batch(r.sched, extend)                             # (full return: rows 32)
        ctl.lend(STEP)                                            # 36 by hand
        free.v = STEP // 2
        assert dtl.on_batch(r.sched, decode) is None
        assert dtl.on_batch(r.sched, decode) is None              # settle: nothing fits -> no move
        assert ctl.rows_on == 36 and ctl.lent_rows == before + 4
