"""S83 instrument on the W3 device round (rc18 ``..._1019_200330``: 13 of 14 FORMA-REQUESTs
``released_pages=0``, the device round ``candidates=0`` while the D tree held 3931 slots, and
the reasons it passed its nodes over were not counted: ``ongoing_write_through``,
``host_lock_ref``, the aux host-only state, staging rows). Every reason now lands in
``_s83_spill_why`` (reason -> [nodes, pages]) and an empty round emits ``S83 VICTIM-CENSUS``;
the spill's decisions are unchanged (the 0929 suite runs on the same tree).

Hermetic: the real C arena and the real ``_pdflip_direct_claim`` of the W3 suite."""
import importlib.util
import logging
import os
import shutil

import pytest

_spec = importlib.util.spec_from_file_location(
    "_s83_w3_0929", os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_w3_arena_spill_0929.py"))
_w3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_w3)
FULL, _claimer, _full_tree = _w3.FULL, _w3._claimer, _w3._full_tree

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")


def test_every_pass_over_reason_is_counted_with_pages(tmp_path):
    p, arena, root, t, nodes = _full_tree(tmp_path)
    t.ongoing_write_through[nodes[1].id] = object()
    nodes[2].component_data[FULL].host_lock_ref = 1
    nodes[3].component_data[FULL].value = None        # host-only: the HOST-LEAF round's
    m = _claimer(t)
    t._pdflip_direct_claim(m)
    why = t._s83_spill_why
    assert why.get("write_pending") == [1, 1], why
    assert why.get("host_locked") == [1, 1], why
    assert why.get("host_only") == [1, 1], why
    assert nodes[1].component_data[FULL].host_value is not None
    assert nodes[2].component_data[FULL].host_value is not None


def test_an_empty_round_says_who_holds_the_arena(tmp_path, caplog):
    """Every node passed over -> released 0 -> one VICTIM-CENSUS line naming the reasons."""
    p, arena, root, t, nodes = _full_tree(tmp_path)
    for n in nodes:
        n.component_data[FULL].host_lock_ref = 1
    m = _claimer(t)
    with caplog.at_level(logging.INFO):
        t._pdflip_direct_claim(m)
    lines = [r.getMessage() for r in caplog.records if "S83 VICTIM-CENSUS" in r.getMessage()]
    assert lines, "no census on an empty round"
    assert "site=spill" in lines[0] and "spill_why=host_locked:8/8" in lines[0], lines[0]
    assert "unpinned_complete=0 " in lines[0] and "tree_device=8 " in lines[0], lines[0]
