"""z30y14 (27B, image 02adfaadee, 01.10. 03:15:38Z): the P group died on PP1
with ``#1400 STORE-TOLD MISMATCH rid=weg2-64-273 told=97631 own_prefix=93858``.

The specimen, from the boot's P log:
  * PP0  ``WEG2-READ-STAGES pages=97631 l3fill_pages=1888`` -> told 97631;
  * PP1  ``pages=95684 l3fill_pages=88``, ``#257 PREFETCH BELOW-ANCHOR read=95684
    of 97631 anchored=93858`` -> own prefix 93858 (``tail_release=1826``);
  * PP2  ``pages=95716 l3fill_pages=88``.
The three P ranks reach three different store depths for one told: PP0's L3
view (the shared #706 canonical store, ``WEG2-STORE-RESCAN ... SIBLING GROUP's
pages`` > 0 B on PP0 only, 0 B on PP1/PP2 at every wake) holds a tail the
followers cannot read at all. Their reads TERMINATED short (``#1157 PREFETCH
REAPED ... elapsed=2.09s budget=97.34s``), so no admission wait can close it.

The group rescue for exactly this case exists in the tree -- PF
(``SGLANG_WEG2_TOLD_GROUP_FALLBACK``, weg2_told_fallback): each follower acks
its own prefix, PP0 answers ``own != told`` with ``Admit(0, fallback)`` for
EVERY rank -- but the qwen27b registry row had it off, so the paced told was
admitted on the window / idle path and PP1 refused by name. These tests pin
the specimen through the 27B row's PROFILE default (no explicit env): the
group agrees on told=0, nobody raises, no read leaks.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _told_ring_pf as R  # noqa: E402

from sglang.srt.managers import weg2_store_told as m  # noqa: E402
from sglang.srt.managers import weg2_told_fallback as fb  # noqa: E402
from sglang.srt.weg2 import form as FM  # noqa: E402

RID = "weg2-64-273"
TOLD = 97631
#: per-rank store depth the told-bounded read reaches (PP1 after the #257
#: anchor cut, PP2 as read; both short of PP0's verdict).
DEPTH = {0: TOLD, 1: 93858, 2: 95716}
#: read durations (s): PP0 2.11 (read_ms=2111), followers ~1.07.
READ_S = {0: 2.11, 1: 1.07, 2: 1.06}


def _form_env(profile: str) -> str:
    arch, experts, draft, kv = (("dense", "none", "dflash", "paged_dcp") if profile == "qwen27b"
                                else ("moe", "offload", "mtp", "qsa_forma"))
    return FM.Weg2Form(arch=arch, experts=experts, draft=draft, p_draft="none", kv=kv,
                       flip="family", vision="off", profile=profile, model="m").env_value()


def _ring(monkeypatch, profile="qwen27b", pf_env=None):
    for k in list(os.environ):
        if k.startswith("SGLANG_WEG2_TOLD") or k == "SGLANG_WEG2_P_TWIN_DEFER":
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv(FM.FORM_ENV, _form_env(profile))
    # the ring double models the plain span told: the other registry switches
    # (TK/absolute, TW) are pinned off explicitly; PF stays UNSET so the
    # profile row decides it -- the boot's own situation.
    monkeypatch.setenv("SGLANG_WEG2_TOLD_PROBE_TREE_KEY", "0")
    monkeypatch.setenv("SGLANG_WEG2_TOLD_ABSOLUTE", "0")
    monkeypatch.setenv("SGLANG_WEG2_P_TWIN_DEFER", "0")
    if pf_env is not None:
        monkeypatch.setenv(fb.ENV_FALLBACK, pf_env)
    reads = {r: {RID: READ_S[r]} for r in range(3)}
    ring = R.Ring(m, monkeypatch, {RID: TOLD}, reads)
    for s in ring.stages:
        s.prompts = {RID: DEPTH[s.ps.pp_rank]}
    return ring


def test_the_27b_row_arms_pf_by_profile_default(monkeypatch):
    monkeypatch.delenv(fb.ENV_FALLBACK, raising=False)
    monkeypatch.setenv(FM.FORM_ENV, _form_env("qwen27b"))
    assert fb.env_on() is True
    assert FM.PROFILES["qwen27b"].told_group_fallback is True
    # NF keeps its own row (released by the NF seat with a boot tag only)
    monkeypatch.setenv(FM.FORM_ENV, _form_env("nextflash"))
    assert fb.env_on() is False


def test_z30y14_specimen_is_a_rank_agreed_told_zero_not_a_death(monkeypatch):
    """The specimen under the 27B row: PP1 acks 93858, PP2 95716, PP0 sends
    Admit(0, fallback); all three ranks admit the rid in the same PP0 pass at
    prefix cap 0 / credit 0, every read (PP0's own included) is released, no
    row or reader reference stays behind, and no stage busy-waits."""
    ring = _ring(monkeypatch)
    ring.arrive(RID)
    ring.run(200)  # 10 s of passes, inside the 12 s Frist cap
    plans = ring.plans(RID)
    assert all(len(p) == 1 for p in plans), plans
    assert plans[0] == plans[1] == plans[2]
    assert plans[0][0][2] == 0  # told 0 on every rank: P recomputes the prefix
    assert [a[3] for s in ring.stages for a in s.admitted] == [0, 0, 0]
    objs = ring.wire_objs()
    assert [(type(o).__name__, o.told) for o in objs] == [
        ("Weg2StoreTold", TOLD),
        ("Weg2StoreAdmit", 0),
    ]
    assert getattr(objs[0], fb.WIRE_ACK) == 1
    assert getattr(objs[1], fb.WIRE_FALLBACK) == 1
    # decided on the acks (a short read), not at the Frist
    assert ring.stages[0]._pf_fallback_n == 1
    assert plans[0][0][0] * R.DT < fb.frist_s(m.pace_window_s(READ_S[0], TOLD)) + READ_S[0]
    for s in ring.stages:
        assert RID in s.tree_cache.released, s.ps.pp_rank
        assert s.tree_cache.op_refs == 0
        assert RID not in s.tree_cache.ongoing
        assert RID not in s.tree_cache.completed and RID not in s.tree_cache.loaded
    assert ring.sleeps == {0: 0.0, 1: 0.0, 2: 0.0}


def test_z30y14_specimen_without_pf_is_the_boot_death(monkeypatch):
    """Characterisation: PF explicitly off (the pre-fix 27B row) reproduces
    the boot's death by name on PP1 -- the refusal is never hidden."""
    ring = _ring(monkeypatch, pf_env="0")
    ring.arrive(RID)
    with pytest.raises(m.Weg2StoreToldMismatch, match=r"told=97631 own_prefix=93858"):
        ring.run(200)


def test_followers_that_reach_told_still_admit_told(monkeypatch):
    """No regression of the good case under the 27B row: equal depths on all
    ranks -> Admit(told) on the acks, nothing released."""
    ring = _ring(monkeypatch)
    for s in ring.stages:
        s.prompts = {RID: TOLD}
    ring.arrive(RID)
    ring.run(200)
    plans = ring.plans(RID)
    assert all(len(p) == 1 for p in plans) and plans[0] == plans[1] == plans[2]
    assert plans[0][0][2] == TOLD
    assert ring.wire_objs()[-1].told == TOLD
    assert getattr(ring.wire_objs()[-1], fb.WIRE_FALLBACK, None) is None
    assert all(not s.tree_cache.released for s in ring.stages)
