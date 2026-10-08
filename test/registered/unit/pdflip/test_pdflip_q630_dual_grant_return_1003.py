# SPDX-License-Identifier: Apache-2.0
"""Q-630 DUAL-GRANT-RETURN (27B dual y8u, boot dkr27bnvfp4dual1mpsleepbar1fs10031206, cfb873b0e8).

METAL (front + P log):

  12:09:17  PP0 ``PP0 GRANT rid=pdflip-0-1 tokens=90112 on all 3 cards``
  12:09:23  PP0 ``PP0 GRANT rid=pdflip-0-10 tokens=147456 on all 3 cards``
  12:09:23  front ``DUAL P-PAUSE`` aborts both before their told left PP0;
            PP0 ``Q-580 TOLD-FORGET rid=pdflip-0-1 pp=0 why=abort dropped=held``
  12:09:24  every P stage ``P-KV RELEASE ... -> 0`` -- yet ``LEDGER-PHYS`` on PP1
            keeps ``committed={'P': 973078528}`` = (90112 + 147456) x 4096 B, PP2
            1459617792 = (90112 + 147456) x 6144 B: the front's RESUME-WAIT never
            saw all zeros (max 210.8 s, then 1441 s), long1/long2 timed out (503).

pp0_grant charges every follower card at the intake; a follower takes the charge
over only when the told reaches it (on_told -> map_granted). A request that left
PP0's hold before its told went out left the charge on the follower cards with no
owner. RED on cfb873b0e8: the follower cards stay committed after the abort, and a
re-intake charges them a second time. GREEN: PP0 gives back the follower charge of
a told that never left; a told on the wire is the followers' to adopt (no return).
"""
from __future__ import annotations

import inspect
import json
import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from flliper.srt.managers import schedule_batch as SB  # noqa: E402
from flliper.srt.managers import pdflip_store_told as ST  # noqa: E402
from flliper.srt.pdflip import card_kv_ledger as K  # noqa: E402
from flliper.srt.pdflip import dual_p_kv_stage as S  # noqa: E402
from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

MIB = 1 << 20
STEP = 4096
# bytes per lattice level k (k x STEP tokens): PP0 / PP1 / PP2 -- the metal's
# per-token unit on the followers (4096 / 6144 B), PP0 its own
PER_TOKEN = (2048, 4096, 6144)


def _cards(tmp_path, monkeypatch):
    paths = []
    for r, per in enumerate(PER_TOKEN):
        path = str(tmp_path / ("card%d" % r))
        K.CardKvLedger(path, "D").contribute(4096 * MIB, committed=0)
        K.CardKvLedger(path, "P").contribute(0)
        stage = {"ledger": path, "step": STEP, "top": 196608,
                 "bytes": [k * STEP * per for k in range(196608 // STEP + 1)]}
        with open(str(tmp_path / ("stage%d" % r)), "w") as f:
            json.dump(stage, f)
        paths.append(path)
    monkeypatch.setattr(S, "stage_file", lambda tag, r, root="/dev/shm": str(tmp_path / ("stage%d" % r)))
    actor = types.SimpleNamespace(page=64, _committed=0, map_granted=lambda lvl, charged=None: None)
    monkeypatch.setattr(S, "_actor", lambda sched: actor)
    return paths


def _sched():
    return types.SimpleNamespace(ps=types.SimpleNamespace(pp_rank=0, pp_size=3), waiting_queue=[],
                                 _pdflip_store_told_armed=True, _pdflip_store_held={})


def _req(rid="pdflip-0-1", tokens=90000):
    return types.SimpleNamespace(rid=rid, origin_input_ids=list(range(tokens)), _dual_grant_untold=None)


def _p(path):
    return K.peek(path).committed["P"]


def test_a_grant_whose_told_never_left_goes_back_to_the_follower_cards(tmp_path, monkeypatch):
    paths = _cards(tmp_path, monkeypatch)
    sched, req = _sched(), _req()
    lvl = S.pp0_grant(sched, req)
    assert lvl == 90112
    assert _p(paths[1]) == lvl * 4096 and _p(paths[2]) == lvl * 6144, "the metal's follower charge"
    sched._pdflip_store_held[req.rid] = req                      # PP0 holds it, told not published
    dropped = ST.forget_left_queue(sched, req, "abort")         # the front's P-PAUSE
    assert "held" in dropped
    assert _p(paths[1]) == 0 and _p(paths[2]) == 0, "the follower cards keep an owner-less grant"
    assert _p(paths[0]) == lvl * 2048, "PP0's own card is PP0's mapping -- not returned here"
    assert req._dual_grant_untold is None


def test_a_told_on_the_wire_is_the_followers_to_adopt_nothing_is_returned(tmp_path, monkeypatch):
    paths = _cards(tmp_path, monkeypatch)
    sched, req = _sched(), _req()
    lvl = S.pp0_grant(sched, req)
    told = S.with_dual_kv(types.SimpleNamespace(rid=req.rid), req)   # the told leaves PP0
    assert getattr(told, S.WIRE_DUAL_KV) == lvl
    sched._pdflip_store_held[req.rid] = req
    ST.forget_left_queue(sched, req, "abort")
    assert _p(paths[1]) == lvl * 4096 and _p(paths[2]) == lvl * 6144, \
        "a follower adopts this charge from the told -- returning it too would free it twice"


def test_a_re_intake_does_not_charge_the_follower_cards_twice(tmp_path, monkeypatch):
    paths = _cards(tmp_path, monkeypatch)
    sched, req = _sched(), _req()
    first = S.pp0_grant(sched, req)
    second = S.pp0_grant(sched, req)                            # intake again, no told in between
    assert first == second == 90112
    assert _p(paths[1]) == second * 4096 and _p(paths[2]) == second * 6144


def test_every_real_request_carries_the_untold_field():
    # forget_left_queue reads it on PP0 for every held drop, dual or not
    src = inspect.getsource(SB.Req.__init__)
    assert "self._dual_grant_untold: Optional[list] = None" in src


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
