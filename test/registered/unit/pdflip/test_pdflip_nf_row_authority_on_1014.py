# SPDX-License-Identifier: Apache-2.0
"""NF-RANG-AUDIT 1014, Fix 1: #631 row authority (Fix B) is ON for nextflash, and
the LB-VERDICT gate yields to a load-back that executes PP0's row.

S5 of nf-rang-audit-1014.md: on the carrierless P form every follower planned from
its OWN pool through every rank-local gate of ``add_one_req``; #1233/#1004 caught
the split only after the launch. The mechanism "PP0 decides, the followers execute
its row" exists (``pdflip/p_row_authority.py``) and was off on NF for want of a metal
proof. One seam has to yield for it: ``execute_scheduled_prefix`` materialises the
scheduled prefix through ``load_back``, whose LB-VERDICT gate would hold a
follower for a verdict PP0 never sent (the follower's host hit is not PP0's on the
layer-split form) and stop it at the bound while PP0 proceeds -- a split made by
the gate itself.

RANK-EQUALITY MUTANT (``test_rank_equality_mutant_the_gate_held_the_row_and_split``):
``FLLIPER_PDFLIP_LBV_ROW_YIELD=0`` = the gate of before: PP0 (no deficit) proceeds, the
follower dies at the materialisation bound. On: both end with the scheduled prefix.
"""
from __future__ import annotations

import inspect
import os
import types

import pytest
import torch

from flliper.srt.managers import pp_admission_congruence as PAC
from flliper.srt.pdflip import form as FM
from flliper.srt.pdflip import p_row_authority as PR
from flliper.srt.pdflip import pp_loadback_verdict as LBV
from flliper.srt.pdflip.pp_slot_fidelity import FLOOR_LOCAL_PP_ATTR

RID = "pdflip-0-10"


# ---------------------------------------------------------------------------
# the profile row
# ---------------------------------------------------------------------------


def _form_env(profile):
    arch, experts, draft, kv = (("dense", "none", "dflash", "paged_dcp") if profile == "qwen27b"
                                else ("moe", "offload", "mtp", "qsa_forma"))
    return FM.PdFlipForm(arch=arch, experts=experts, draft=draft, p_draft="none", kv=kv,
                       flip="family", vision="off", profile=profile, model="m").env_value()


def test_nextflash_row_turns_row_authority_on():
    assert FM.PROFILES["nextflash"].p_row_authority is True
    assert FM.PROFILE_SWITCH_DEFAULTS["nextflash"][PR.ENV] is True
    assert FM.PROFILES["qwen27b"].p_row_authority is True  # the 27B row unchanged


@pytest.mark.parametrize("profile,explicit,want", [
    ("nextflash", None, True),   # default ON on the NF row now
    ("nextflash", "0", False),   # an explicit off still wins
    ("qwen27b", None, True),
    (None, None, False),         # no form: off, unchanged
])
def test_enabled_follows_the_row(profile, explicit, want):
    env = {}
    if profile is not None:
        env[FM.FORM_ENV] = _form_env(profile)
    if explicit is not None:
        env[PR.ENV] = explicit
    assert PR.enabled(env) is want


# ---------------------------------------------------------------------------
# the gate yields to an executed row
# ---------------------------------------------------------------------------


class _Tree:
    """A follower's tree at the LB-VERDICT seam of ``UnifiedRadixCache.load_back``
    (URC: ``_lbv_gated = ... _lbv.gated(self, req)``; gated and not released ->
    ``hold`` and nothing is loaded). Pinned against the source below."""

    enable_storage = False
    prefetch_progress_is_collective_free = None

    def __init__(self, pp0: bool = False):
        setattr(self, LBV._STATE_ATTR, LBV._State(pp0))
        setattr(self, FLOOR_LOCAL_PP_ATTR, True)
        self.holds = 0
        self.loads = 0

    def init_load_back(self, params):
        req = params.req
        if LBV.gated(self, req) and not LBV.released(self, req.rid):
            self.holds += 1
            LBV.hold(self, req.rid, int(params.host_hit_length), lambda: True)
            return torch.empty(0, dtype=torch.int64), req.last_node
        self.loads += 1
        n = int(params.host_hit_length)
        return torch.arange(n, dtype=torch.int64), "host-node"


class _Req:
    def __init__(self, local: int):
        self.rid = RID
        self.prefix_indices = torch.arange(local, dtype=torch.int64)
        self.best_match_node = "best"
        self.last_node = "last"
        self.host_hit_length = 0
        self.full_untruncated_fill_ids = list(range(256))
        self.cache_protected_len = local
        self.mamba_loadback_anchor_adopted = False

    def init_next_round_input(self, tree=None):
        pass


@pytest.fixture
def fast_bound(monkeypatch):
    monkeypatch.setattr(PAC, "store_read_bound_s", lambda deficit: 0.05)
    monkeypatch.setattr(PAC, "MATERIALISE_POLL_S", 0.005)
    monkeypatch.delenv(LBV.ROW_YIELD_ENV, raising=False)
    monkeypatch.delenv(LBV.ENV, raising=False)


def _follower_executes(local: int, scheduled: int):
    tree = _Tree(pp0=False)
    req = _Req(local)
    got = PAC.execute_scheduled_prefix(req, tree, scheduled)
    return tree, req, got


def test_a_follower_executes_the_row_through_the_gate(fast_bound):
    tree, req, got = _follower_executes(local=0, scheduled=64)
    assert got == 64 and len(req.prefix_indices) == 64
    assert tree.loads == 1 and tree.holds == 0
    # the flag lived only for the station: cleared on exit
    assert LBV.row_executed(req) is False


def test_pp0_without_a_deficit_is_a_no_op_either_way(fast_bound):
    tree, req, got = _follower_executes(local=64, scheduled=64)
    assert got == 0 and tree.loads == 0 and tree.holds == 0


def test_rank_equality_mutant_the_gate_held_the_row_and_split(fast_bound, monkeypatch):
    """The old gate (switch off): PP0 (local == scheduled) proceeds, the follower
    (deficit 64, no verdict ever sent for this rid) is held until its bound and
    stops -- two states for one pass. With the yield: both hold the scheduled prefix."""
    monkeypatch.setenv(LBV.ROW_YIELD_ENV, "0")
    pp0_tree, pp0_req, pp0_got = _follower_executes(local=64, scheduled=64)
    assert pp0_got == 0
    with pytest.raises(RuntimeError, match="PREFIX MATERIALISATION SHORTFALL"):
        _follower_executes(local=0, scheduled=64)
    # the detector: the mutant split the ranks (PP0 whole, the follower short at the bound)
    monkeypatch.delenv(LBV.ROW_YIELD_ENV, raising=False)
    tree, req, got = _follower_executes(local=0, scheduled=64)
    assert got == 64 and tree.holds == 0 and len(req.prefix_indices) == len(pp0_req.prefix_indices)


def test_an_own_plan_load_back_stays_gated():
    """Only a row-executed load-back passes; a stage's own admission plan still
    waits for PP0's verdict (the LB-VERDICT law of 853534a9ff is intact)."""
    tree = _Tree(pp0=False)
    req = _Req(0)
    assert LBV.gated(tree, req) is True
    setattr(req, LBV.ROW_ATTR, True)
    assert LBV.gated(tree, req) is False
    setattr(req, LBV.ROW_ATTR, False)
    assert LBV.gated(tree, req) is True
    # unarmed tree / no rid: never gated
    assert LBV.gated(types.SimpleNamespace(), req) is False
    assert LBV.gated(tree, types.SimpleNamespace(rid=None)) is False


def test_the_station_clears_the_flag_on_a_stop(fast_bound, monkeypatch):
    monkeypatch.setenv(LBV.ROW_YIELD_ENV, "0")
    req = _Req(0)
    with pytest.raises(RuntimeError):
        PAC.execute_scheduled_prefix(req, _Tree(), 64)
    assert LBV.row_executed(req) is False


def test_defaults_on():
    assert LBV.row_yield_enabled() is True if not os.environ.get(LBV.ROW_YIELD_ENV) else True


def test_source_pins():
    """The seam in URC reads the predicate (not ``armed`` alone), and the station
    sets the flag around exactly its materialisation."""
    from flliper.srt.mem_cache import unified_radix_cache as URC

    src = inspect.getsource(URC.UnifiedRadixCache.load_back)
    assert "_lbv.gated(self, req)" in src
    assert "_lbv.armed(self)\n" not in src.split("_lbv.gated")[0][-200:]
    st = inspect.getsource(PAC.execute_scheduled_prefix)
    assert "setattr(req, _lbv_row.ROW_ATTR, True)" in st and "setattr(req, _lbv_row.ROW_ATTR, False)" in st
    assert st.index("ROW_ATTR, True") < st.index("_materialise_scheduled_prefix(") < st.index("ROW_ATTR, False")
