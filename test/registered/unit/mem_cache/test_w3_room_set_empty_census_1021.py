"""PARK-L3 task 3 (rc27 boot ..._893d09fa3f_1021_023225, P 02:37:09-02:37:25): an empty W3-ROOM-SET decision is
named, with its exclusion reasons.

THE METAL (P log). 02:37:09 ``ARENA-REF-CENSUS slots=6485 complete=6485 pinned=6485 refs=18172`` and the holders per
stage: PP0 ``tree=2532 tree_in_use=1347 prefetch=1931``, PP1 ``tree=2493 tree_in_use=3992``, PP2 ``tree=128
tree_in_use=5749`` (``park_kept=3098/8204`` is the #248 ORDER, not a reference). A slot frees only when all three
stages let go; PP2's running requests alone referenced 5749 of the 6485 slots. The group's first ``W3-ROOM-SET
PUBLISH`` of the boot came at 02:37:25 (gen=1 slots=1019). ``decide()`` printed nothing on an empty result, so why it
found no candidate at 02:37:12/15 stayed unproven.

THE RULE. PP0's empty decision is counted always and named at most every ``EMPTY_CENSUS_EVERY_S`` with every node
holding a host copy, by the FIRST rule that keeps it (reason:nodes/pages): running, host_locked, write_pending,
aux_host_only, staging, keep_payable, claimer_chain, told_head, host_only:<the host-leaf reason>, incomplete.

The three-stage harness of the F1 suite (one real C arena file, three pools)."""
import logging
import os
import shutil
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
import test_pdflip_f1_w3_room_set_1016 as F1  # noqa: E402
from test_pdflip_f1_w3_room_set_1016 import _env, pp3  # noqa: E402,F401

from flliper.srt.pdflip import w3_room_set as RS  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")
FULL = F1.FULL


@pytest.fixture(autouse=True)
def _fresh_throttle():
    RS._N.pop("empty_t", None)
    yield
    RS._N.pop("empty_t", None)


def test_an_empty_decision_names_running_requests_as_the_holder(pp3, caplog):
    """RED on a773b4ebc4 (nothing printed). Every PP0 node is held by a running request (the 02:37 shape): the
    spill releases nothing and the line names ``running`` with all nodes and pages."""
    pools, trees, nodes, store = pp3
    p0, a0 = pools[0]
    for n in nodes[0]:
        n.component_data[FULL].lock_ref = 1
    caplog.set_level(logging.INFO)
    n0 = RS._N.get("empty", 0)
    assert trees[0]._w3_arena_spill(p0, 1) == 0
    lines = [m for m in caplog.messages if m.startswith("W3-ROOM-SET EMPTY")]
    assert len(lines) == 1, caplog.messages
    assert "reasons=running:4/4" in lines[0], lines[0]
    assert RS._N["empty"] == n0 + 1


def test_the_census_splits_reasons_first_rule_wins(pp3):
    pools, trees, nodes, store = pp3
    p0, a0 = pools[0]
    ns = nodes[0]
    ns[0].component_data[FULL].lock_ref = 1                  # running
    ns[1].component_data[FULL].host_lock_ref = 1             # a load / a told window
    trees[0].ongoing_write_through[ns[2].id] = object()      # a write in flight
    ns[3].component_data[FULL].value = None                  # host-only leaf, host-locked too -> host-leaf rule
    ns[3].component_data[FULL].host_lock_ref = 1
    cen = RS.empty_census(trees[0], p0, RS._Ctx(trees[0], p0))
    assert cen.get("running") == [1, 1]
    assert cen.get("host_locked") == [1, 1]
    assert cen.get("write_pending") == [1, 1]
    assert cen.get("host_only:host_locked") == [1, 1], cen
    assert "candidate" not in cen


def test_the_census_is_throttled_but_the_count_always_moves(pp3, caplog):
    pools, trees, nodes, store = pp3
    p0, a0 = pools[0]
    for n in nodes[0]:
        n.component_data[FULL].lock_ref = 1
    caplog.set_level(logging.INFO)
    n0 = RS._N.get("empty", 0)
    for _ in range(3):
        trees[0]._w3_spill_quiet_until = 0.0
        trees[0]._w3_arena_spill(p0, 1)
    assert len([m for m in caplog.messages if m.startswith("W3-ROOM-SET EMPTY")]) == 1
    assert RS._N["empty"] == n0 + 3


def test_a_decision_with_slots_prints_no_empty_line(pp3, caplog):
    pools, trees, nodes, store = pp3
    p0, a0 = pools[0]
    caplog.set_level(logging.INFO)
    trees[0]._w3_arena_spill(p0, 1)
    assert not any(m.startswith("W3-ROOM-SET EMPTY") for m in caplog.messages)


def test_the_census_never_breaks_the_spill(pp3, monkeypatch, caplog):
    pools, trees, nodes, store = pp3
    p0, a0 = pools[0]
    for n in nodes[0]:
        n.component_data[FULL].lock_ref = 1
    monkeypatch.setattr(RS, "empty_census", lambda *a: (_ for _ in ()).throw(RuntimeError("x")))
    caplog.set_level(logging.INFO)
    assert trees[0]._w3_arena_spill(p0, 1) == 0
    assert any("census_failed:RuntimeError" in m for m in caplog.messages)
