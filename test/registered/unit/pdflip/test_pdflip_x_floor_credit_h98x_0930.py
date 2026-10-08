"""H98x (30.09., NF y4b D 03:50:21, rid pdflip-14-27): on a Form A group the X
gate priced ``total - min(max(head, store), floor)`` and credited the usable
floor only as a CAP. The Form A expert workers' own head walk is refused by
the mamba bytes they do not hold (``#904 match-census ... refusers=
MambaComponent:109440 why=MambaComponent:absent``), so the head MIN was 0,
the store arm 0 (the prefix was on TP0's device, not in the store), and W31
priced the whole prompt::

    X-GATE-TERMS rid=pdflip-14-27 total=110438 head=0 store=0 floor=109440
    uncached=110438 ... why=host_admission

although every rank admits the floor (``RU FORM-A FOLLOW tp0_depth=109440
worker_local=0 followed=109440``). W50, and P re-prefilled 110438 tokens of
a prefix resident on D; same shape for pdflip-22-45 and pdflip-30-69.

Now a Form A group credits the floor above the head/store terms. Classic
groups (floor from the head walk, never above it) and the switch off price
as before; a floor of 0 (the host itself admits nothing: pdflip-32-72
03:58:57, MambaComponent:absent on TP0) still prices the whole prompt."""

import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest

from flliper.srt.environ import envs
from flliper.srt.managers import scheduler as sched_mod
from flliper.srt.managers import tp_head_congruence as thc
from flliper.srt.managers import tp_match_floor as tmf

S = sched_mod.Scheduler
X = 12288


def _sched(rid, floor):
    tree = types.SimpleNamespace()
    setattr(tree, tmf.TREE_ATTR, {rid: floor})
    h = types.SimpleNamespace(tree_cache=tree)
    h.pdflip_uncached_extent = types.MethodType(S.pdflip_uncached_extent, h)
    return h


def _req(rid, total):
    return types.SimpleNamespace(
        rid=rid, full_untruncated_fill_ids=[0] * total, prefix_indices=[], host_hit_length=0
    )


def _inputs(rid, head, store):
    canon = thc.canonical_head_rids([rid])
    return thc.build_uniform_head_inputs(canon, [head], None, True, (), [store])


@pytest.fixture
def form_a(monkeypatch):
    monkeypatch.setattr(tmf, "form_a_follow_active", lambda: True)


def _price(rid, total, head, store, floor):
    h = _sched(rid, floor)
    return h.pdflip_uncached_extent(_req(rid, total), _inputs(rid, head, store)), h


def test_y4b_pdflip_14_27_form_a_credits_the_floor_every_rank_admits(form_a):
    got, h = _price("pdflip-14-27", 110438, 0, 0, 109440)
    assert got == 110438 - 109440            # base: 110438 -> W31 -> W50 -> P re-prefill
    assert got <= X
    assert h._pdflip_x_terms == (110438, 0, 0, 109440)


def test_y4b_pdflip_22_45_form_a(form_a):
    got, _ = _price("pdflip-22-45", 113708, 0, 0, 110400)
    assert got == 3308


def test_a_host_that_admits_nothing_still_prices_the_whole_prompt(form_a):
    """pdflip-32-72 03:58:57: TP0's own match refused (mamba absent), floor 0 --
    D cannot admit anything, the W31 stands."""
    got, _ = _price("pdflip-32-72", 95137, 0, 0, 0)
    assert got == 95137


def test_the_floor_still_caps_a_head_above_it(form_a):
    got, _ = _price("r", 50000, 49000, 49000, 30000)
    assert got == 20000


def test_a_classic_group_prices_as_before():
    got, _ = _price("pdflip-14-27", 110438, 0, 0, 109440)
    assert got == 110438


def test_the_switch_off_prices_as_before(form_a):
    with envs.FLLIPER_PDFLIP_ENABLE_X_FLOOR_CREDIT.override(False):
        got, _ = _price("pdflip-14-27", 110438, 0, 0, 109440)
    assert got == 110438


def test_the_store_arm_above_the_floor_is_still_capped(form_a):
    got, _ = _price("r", 50000, 0, 45000, 40000)
    assert got == 10000
