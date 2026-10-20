"""Review probes NF-VISION-CACHED (desk/nf-vision-cached-flip-1020 @ 73c172d901), 20.10.

Each probe states the CORRECT behaviour; they were xfail(strict) on 73c172d901 and are green since the A1/A2 fix (the
markers removed). Hermetic: no GPU, no model, no network.

B1  n - store depth (u_p) is an UPPER bound of the class number (D covers at least the store: u_d <= u_p). The EZ-S1
    ladder (``decision_numbers.class_basis``) lets u_p say ONE thing: u_p < halbX => kurz. The fix classes halb/gross on
    it (``_rvp_text_u`` u_src=p_store, ``_rvp_reprice``, ``_wall_extent_to_u`` u_src=p_store). Front logs 20.10.: of 3563
    exact D probes, 391 had u_d < X while u_p >= X (a flip under X if u_p were the class, SOLL R7).
B2  the arrival's store depth is not the depth at the decision: 33 of 185 S63 re-probes on 20.10. found the store
    deeper than at the arrival (median +4544, max +17216 tokens) -- n_ctx - arrival depth overprices the leg.
B3  the S63 re-probe answers for the ARRIVAL ids (the prompt): applied to the RESUME-VIA-P leg it drops the streamed
    output the leg's price counted (``extra``).
"""
from __future__ import annotations

import asyncio
import os
import sys
import time

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flliper.srt.pdflip import d_match_probe as dmp  # noqa: E402
from flliper.srt.pdflip import front as F  # noqa: E402
from flliper.srt.pdflip import phase_policy as pp  # noqa: E402
from flliper.srt.pdflip import resume_via_p as rvp  # noqa: E402

from test_nf_vision_cached_flip_1020 import (  # noqa: E402
    N, RID, STORE, U, X, _answer, _front, _late, _take, d_env)  # noqa: F401 -- d_env is a fixture

LAG = 20000  # store depth the arrival read lies LAG tokens below what D's tree (and later the store) held


def test_b1_b2_p_store_upper_bound_must_not_class_gross(d_env):
    """No D price (probe gave up), arrival store depth N-LAG; D's tree held all but U. The leg must not be a P trigger
    on the upper bound u_p = LAG >= X (D's u is U = 379, kurz: R7 no flip reason)."""
    f = _front()
    f.__dict__["_za_presence"] = {RID: (N - LAG, N)}
    rvp.write_request(RID, list(range(N)), N, X, rvp.REASON_WALL)
    assert _take(f) == 1
    p = f.queue[-1]
    assert pp.immediate_park_trigger([p], X) is None


def test_b1_reprice_on_u_p_must_not_class_gross(d_env):
    """Unpriced leg; the store holds less than D's tree (it lags). u_p >= halbX is no class number (class_basis)."""
    f = _front()

    async def _store(rid, ids, timeout_s, extra_key=None):
        return N - LAG, "l3_index"

    f._store_probe_depth = _store
    f.store_probe = object()
    rvp.write_request(RID, list(range(N)), N, X, rvp.REASON_WALL)
    assert _take(f, await_reprice=True) == 1
    p = f.queue[-1]
    assert pp.immediate_park_trigger([p], X) is None


def test_b1_fresh_wall_p_store_upper_bound_is_no_extent():
    """A1: without a D price the fresh path's u_p (here LAG >= X) is no class number -- unpriced (0), D's number NOW
    follows (was: the extent became u_p, pinned on 73c172d901)."""
    f = _front()
    f.__dict__["_za_presence"] = {RID: (N - LAG, N)}
    body = b"W50 ... extent after prefix matching is 226683. D-WALL-HEAD: the extent is the device rows"
    assert F.Front._wall_extent_to_u(f, RID, body, N, n_prompt=N) == 0
    assert LAG >= X


def test_b3_reprobe_keeps_the_streamed_output_of_the_leg(d_env, monkeypatch):
    """A2: the arrival-key round (prompt ids) lands after the leg is priced on its context ids: the output stays. (On
    73c172d901 the leg's interim u was U + out; since A1 an old price >= halbX is no class -- D's exact number on the
    context ids NOW prices the leg, here as U + out.)"""
    from test_nf_vision_cached_flip_1020 import _ans, _fake_d, _store
    f = _front()
    _late(f, _answer(N, 0, STORE))
    e = f._reprobe_book[RID]                      # the arrival's key, captured before the reroute
    out = 2 * X                                   # streamed output D generated before the wall refusal
    _fake_d(monkeypatch, [_ans(N + out, 0, STORE)])
    _store(f, STORE)
    rvp.write_request(RID, list(range(N + out)), N + out, X, rvp.REASON_WALL)
    assert _take(f, await_reprice=True) == 1
    p = f.queue[-1]
    assert p.est_uncached == U + out              # the leg's price counts the output (gross: a member)
    ans = dmp.parse_answer(__import__("json").loads(_answer(N, 0, STORE)))
    F.Front._set_reprobe_apply(f, RID, e, ans, STORE, time.monotonic(), "review")
    assert p.est_uncached >= U + out
