# SPDX-License-Identifier: Apache-2.0
"""X-CREDIT-FINISHED-1002: a FINISHED D leg 2 credited only its admission hit.

NF boot dkrnfint4bar1dauer10020634 (5b46b8842e, #49 agent span OFF on the
nextflash row), front log ..._1002_063419.front.log:

  06:42:29,440 'PDFLIP-SERVED group=D leg=2 rid=pdflip-26-44 ... prompt_tokens=53748
               cached_tokens=50688 ... epoch=26' ('PRESENCE-OWN-TEXT-CLAMP ...
               depth_d=53760 credited=53696')
  06:42:29,485 'X-EXACT-PRICE rid=pdflip-26-45 pending=4582 tokens=55270
               credit=50688 src=d_leg2_cached' (SESSION-PREFIX common=53748)
               -> LONG at X=3647; 'SERVED group=P leg=1 ... cached_tokens=53760'
               -- P prefilled 1510.
  pdflip-24-39 (47809, cached 45824, depth 47872) -> pdflip-24-40 (50163) priced
               credit=45824 pending 4339 > X=3602 -> LONG, P prefilled 2291.

Fix: (1) the nextflash row runs #49 (agent_span on): a served leg holds
min(prompt, D's resumable depth) for its epoch (d_served_epoch); (2) after the
epoch the leg keeps min(prompt, depth, page floor) -- a store load, not a
prefill -- (d_served_anchor) until ANCHOR-LOST, which now also matches D's
unclamped depth.
"""

from __future__ import annotations

import collections
import os

import numpy as np
import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import form as FM  # noqa: E402
from flliper.srt.pdflip import front as F  # noqa: E402
from flliper.srt.pdflip.front_tokens import TokenSpans  # noqa: E402

SWITCH = "FLLIPER_PDFLIP_ENABLE_AGENT_SPAN"


def _form_env(profile):
    arch, experts, draft, kv = (("dense", "none", "dflash", "paged_dcp") if profile == "qwen27b"
                                else ("moe", "offload", "mtp", "qsa_forma"))
    return FM.PdFlipForm(arch=arch, experts=experts, draft=draft, p_draft="none", kv=kv,
                       flip="family", vision="off", profile=profile, model="m").env_value()


@pytest.fixture
def nf(monkeypatch):
    monkeypatch.delenv(SWITCH, raising=False)
    monkeypatch.setenv(FM.FORM_ENV, _form_env("nextflash"))
    return monkeypatch


class _Tok:
    def __init__(self):
        self.m = {}

    def ids_for(self, text):
        return self.m.get(text)


def _front(epoch):
    """The front's X-EXACT state as Front.__init__ builds it (the span read
    from the published form)."""
    f = object.__new__(F.Front)
    f.counters = collections.Counter()
    f.x_exact = True
    f.epoch = epoch
    f.spans = F.SpanLRU()
    f.tspans = TokenSpans(agent_span=f.spans.agent_span)
    f.ftok = _Tok()
    f._x_exact_rid = collections.OrderedDict()
    f._x_exact_reprice_queue = lambda why: 0
    return f


def _ids(n):
    return np.arange(n, dtype=np.int32)


def _served(f, rid, n, pt, ct, depth, epoch):
    """leg2's finish of a 200 D serve: held_epoch = the epoch (``_held``)."""
    f.ftok.m[rid] = _ids(n)
    f._x_exact_record(rid, rid, pt, ct, None, epoch, resumable_depth=depth)


# ---- same epoch (d_served_epoch) ------------------------------------------------

def test_pdflip_26_45_priced_short_on_its_predecessors_end_anchor(nf):
    f = _front(26)
    _served(f, "pdflip-26-44", 53748, 53748, 50688, 53760, 26)
    pending, credit, known, src = f.tspans.pending(_ids(55270), epoch=26)
    assert known and credit == 53696, "min(prompt 53748, depth 53760 -> own end anchor 53696)"
    assert pending == 1574 <= 3647, "SHORT (P prefilled 1510 on a 53760 hit)"
    assert src == "d_served_epoch"


def test_pdflip_24_40_priced_short_on_its_predecessors_end_anchor(nf):
    f = _front(24)
    _served(f, "pdflip-24-39", 47809, 47809, 45824, 47872, 24)
    pending, credit, _k, src = f.tspans.pending(_ids(50163), epoch=24)
    assert (credit, pending, src) == (47808, 2355, "d_served_epoch")
    assert pending <= 3602, "SHORT (P prefilled 2291 on a 47872 hit)"


def test_the_nextflash_row_runs_the_span():
    assert FM.PROFILES["nextflash"].agent_span is True


def test_explicit_span_off_is_the_old_price(nf):
    nf.setenv(SWITCH, "0")
    f = _front(26)
    _served(f, "pdflip-26-44", 53748, 53748, 50688, 53760, 26)
    assert f.tspans.pending(_ids(55270), epoch=26)[:2] == (4582, 50688)


# ---- across epochs (d_served_anchor) --------------------------------------------

@pytest.mark.parametrize("epoch", [None, 27, 28])
def test_the_end_anchor_survives_d_sleep_as_a_store_load(nf, epoch):
    # 26-44 served in epoch 26; priced in a P phase (None) or a later D phase.
    # Boot evidence: 14 of 14 follow-ups after a D sleep read >= the anchor
    # (26-45 P hit 53760, 27-46 55232 = 26-45's anchor, 64-104 on D 88336).
    f = _front(26)
    _served(f, "pdflip-26-44", 53748, 53748, 50688, 53760, 26)
    pending, credit, _k, src = f.tspans.pending(_ids(55270), epoch=epoch)
    assert (credit, pending, src) == (53696, 1574, "d_served_anchor")


def test_without_a_named_depth_the_epoch_guard_stands(nf):
    # no #59 depth: the held whole prompt is radix-only -> only in its epoch
    f = _front(26)
    _served(f, "x", 53748, 53748, 50688, None, 26)
    assert f.tspans.pending(_ids(55270), epoch=26)[1] == 53748
    assert f.tspans.pending(_ids(55270), epoch=27)[1] == 50688


def test_a_reading_d_did_not_serve_gets_no_anchor(nf):
    # W31 / reroute leg: held_epoch None -> the measured hit only (#1324)
    f = _front(26)
    _served(f, "x", 53748, 53748, 50688, 53760, None)
    assert f.tspans.pending(_ids(55270), epoch=None)[1:4:2] == (50688, "d_leg2_cached")


def test_px_a_divergence_before_the_anchor_falls_back_to_the_hit(nf):
    f = _front(26)
    _served(f, "pdflip-26-44", 53748, 53748, 50688, 53760, 26)
    other = _ids(55270)
    other[52000:] += 10 ** 6
    assert f.tspans.pending(other, epoch=None)[1] == 50688


def test_anchor_lost_retracts_by_ds_unclamped_depth(nf):
    f = _front(26)
    _served(f, "pdflip-26-44", 53748, 53748, 50688, 53760, 26)
    assert f.tspans.depth_caps and list(f.tspans.depth_caps.values()) == [53696]
    gone = F.retract_lost_anchors(f.tspans, [53760])
    assert len(gone) == 1
    assert f.tspans.pending(_ids(55270), epoch=None)[1] == 0
    assert not f.tspans.raw_depths
