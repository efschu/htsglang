"""L3-COMPLETE (review K15b B1/A1, 16.10.): a page counts as L3-secured only when its arena slot is
COMPLETE (state 2) in the shared header. ``arena_ref_slots`` excludes only this rank's pending set, and
``complete_write`` drops a slot from it on status 0 too (merged, not full): under the token cut TP0
acks ~45 s before the workers, so TP0 saw half a page as finished. ``secure_rows_to_l3`` wrote those
slot bytes to L3 and reported ``pages=1 lost=0`` -- the K10 device round (W3-FORMA, live on h16s) and
the W3 host-leaf round then released the host copy: half a page silently in L3, never rewritten.

Hermetic: the real C arena and ArenaMHAHostPool of the dual15 harness (two ranks on one arena file).
RED on 1f4b3e73d4 (review probe RV4: ``sec={pages: 1, written: 1, lost: 0}``)."""
import os
import shutil
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
import test_w3_arena_spill_0929 as W  # noqa: E402
import test_w3_arena_spill_forma_dual15_0930 as F  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")
group = F.group
FULL = F.FULL


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv("FLLIPER_PDFLIP_L3_COMPLETE_ONLY", raising=False)


def _half_page(g, stem_hash="zz"):
    """Free one slot, claim it for a new page and complete only TP0's half (the token cut)."""
    p0, a0, a1 = g.p0, g.a0, g.a1
    old = [int(s) for s, _ in a0.find_slots(p0._stems(g.n0[-1].hash_value))]
    assert a0.ref_slots(old, -1) == 1 and a1.ref_slots(old, -1) == 1
    assert a0.drop_unreferenced(old) == 1
    stems = p0._stems([stem_hash])
    (slot, status, gen), = a0.claim_slots(stems, [W.PAGE])
    assert status == 0 and slot >= 0
    assert a0.complete_slots([slot], [gen], ((0, W.PAGE // 2),)) == [0]
    assert int(a0.find_slots(stems)[0][1]) == 1, "precondition: the slot is CLAIMED, not COMPLETE"
    return p0.arena_ids(torch.tensor([slot], dtype=torch.int64))


def test_rv4_half_page_is_not_secured_and_not_written(group):
    g = group
    rows = _half_page(g)
    sec = g.p0.secure_rows_to_l3(rows)
    assert sec["incomplete"] == 1 and sec["lost"] >= 1, sec
    assert not (g.store / "zz_sfx.bin").exists(), "half a page reached L3"


def test_half_page_with_an_old_file_on_disk_is_still_not_secured(group):
    """A file of the same stem is already on disk: the claimed slot is still no secured page."""
    g = group
    (g.store / "zz_sfx.bin").write_bytes(b"\x11" * W.PAGE)
    rows = _half_page(g)
    sec = g.p0.secure_rows_to_l3(rows)
    assert sec["incomplete"] == 1 and sec["lost"] >= 1 and sec["on_disk"] == 0, sec


def test_complete_pages_are_secured_as_before(group):
    g = group
    sec = g.p0.secure_rows_to_l3(g.n0[0].component_data[FULL].host_value)
    assert sec["incomplete"] == 0 and sec["lost"] == 0 and sec["pages"] == 1
    assert sec["on_disk"] + sec["written"] == 1


def test_k10_device_round_keeps_a_node_with_a_half_page(group):
    """The live K10 W3-FORMA device round (_w3_arena_spill) on TP0: the node whose page is half
    finished keeps its host copy and is not marked l3_present."""
    import unittest.mock as mock

    from flliper.srt.mem_cache import form_a_host_shadow as R

    g = group
    leaf = g.n0[-1]
    rows = _half_page(g)
    leaf.hash_value = ["zz"]
    leaf.component_data[FULL].host_value = rows
    for n in g.n0[:-1]:
        n.component_data[FULL].host_value = None        # only the half-page node is a candidate
    with mock.patch.object(R, "role", lambda: "host"):
        R.register_tree(g.t0)
        g.t0._w3_arena_spill(g.p0, 1)
    assert leaf.component_data[FULL].host_value is not None and not getattr(leaf, "l3_present_spilled", False)
    assert not (g.store / "zz_sfx.bin").exists()


def test_no_header_read_is_fail_closed(group, monkeypatch):
    """Review A4: no switch back onto the wrong-KV path; an arena without ``slot_states`` secures
    nothing (fail-closed, was fail-open)."""
    monkeypatch.setenv("FLLIPER_PDFLIP_L3_COMPLETE_ONLY", "0")          # the old switch is gone
    g = group
    rows = _half_page(g)
    assert g.p0.secure_rows_to_l3(rows)["incomplete"] == 1
    full = g.n0[0].component_data[FULL].host_value
    monkeypatch.setattr(type(g.a0), "slot_states", None, raising=False)
    sec = g.p0.secure_rows_to_l3(full)
    assert sec["incomplete"] == 1 and sec["lost"] >= 1 and sec["written"] == 0, sec


def test_review_p1_incomplete_node_does_not_eat_the_forma_cap(group):
    """Review A1 (probe P1): the half-page node carries only TP0's reference, so the group-equal order
    put it FIRST; the FORMA cap counted its page before secure_rows_to_l3 refused it and the complete
    nodes behind it were passed over (released 0, cap_used 1, cap_skipped 7). Now it is no candidate."""
    import unittest.mock as mock

    from flliper.srt.mem_cache import form_a_host_shadow as R

    g = group
    leaf = g.n0[-1]
    rows = _half_page(g)
    sl = [int(s) for s, _ in g.a0.find_slots(g.p0._stems(["zz"]))]
    assert g.a0.ref_slots(sl, +1) == 1               # TP0's complete_write took the node's reference
    leaf.hash_value = ["zz"]
    leaf.component_data[FULL].host_value = rows
    with mock.patch.object(R, "role", lambda: "host"):
        R.register_tree(g.t0)
        got = g.t0._w3_arena_spill(g.p0, 1, max_secure_pages=1)
    spilled = [n for n in g.n0[:-1] if n.component_data[FULL].host_value is None]
    assert got >= 1 and spilled, (got, g.t0._w3_spill_cap_used, g.t0._w3_spill_cap_skipped)
    assert leaf.component_data[FULL].host_value is not None


def test_review_p3_the_sink_refuses_a_claimed_slot_for_every_writer(group):
    """Review A2 (probe P3): the flush/park mamba spills call arena_secure_to_disk directly on
    'settled' slots (this rank's pending set only) -- the sink wrote the CLAIMED half slot."""
    g = group
    _half_page(g)
    sl = [int(s) for s, _ in g.a0.find_slots(g.p0._stems(["zz"]))]
    for writer in ("flush_spill", "park_first_spill", "claim_room"):
        sec = g.p0._backend.arena_secure_to_disk(g.a0, [(sl[0], 0, 0, W.PAGE)], writer=writer)
        assert sec["written"] == 0 and sec["on_disk"] == 0 and sec["incomplete"] == 1 and sec["lost"] == 1, sec
    assert not (g.store / "zz_sfx.bin").exists()


def test_the_sink_still_writes_a_complete_slot(group):
    g = group
    sl = [int(s) for s, _ in g.a0.find_slots(g.p0._stems(g.n0[0].hash_value))]
    sec = g.p0._backend.arena_secure_to_disk(g.a0, [(sl[0], 0, 0, W.PAGE)], writer="flush_spill")
    assert sec.get("incomplete", 0) == 0 and sec["lost"] == 0 and sec["on_disk"] + sec["written"] == 1, sec


def test_mixed_node_one_half_page_counts_lost(group):
    """A two-page node: page 0 COMPLETE, page 1 half -- the node is not secured (lost >= 1)."""
    g = group
    rows_half = _half_page(g)
    rows_full = g.n0[0].component_data[FULL].host_value
    sec = g.p0.secure_rows_to_l3(torch.cat([rows_full, rows_half]))
    assert sec["pages"] == 2 and sec["incomplete"] == 1 and sec["lost"] >= 1, sec


def test_host_leaf_round_counts_a_half_page_leaf_incomplete_not_unsecured(group):
    """Review A1/A5 for spill_host_only: a host-only leaf whose slot is not COMPLETE is no candidate;
    the line names it ``incomplete`` (no loss), the leaf keeps its host copy."""
    from flliper.srt.pdflip import host_only_spill as H

    g = group
    leaf = g.n0[-1]
    rows = _half_page(g)
    leaf.hash_value = ["zz"]
    cd = leaf.component_data[FULL]
    cd.host_value = rows
    cd.value = None
    for n in g.n0[:-1]:
        n.component_data[FULL].host_value = None
    assert H._reason(g.t0, leaf, g.p0, set()) is None, H._reason(g.t0, leaf, g.p0, set())
    got = H.spill_host_only(g.t0, g.p0, 1, W.PAGE_TOKENS if hasattr(W, "PAGE_TOKENS") else 1, set())
    assert got["incomplete"] == 1 and got["unsecured"] == 0 and got["released"] == 0, got
    assert cd.host_value is not None
    assert not (g.store / "zz_sfx.bin").exists()
