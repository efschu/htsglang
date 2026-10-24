# SPDX-License-Identifier: Apache-2.0
"""PROBE (Nachpruefung #231 END-ANCHOR-GUARD-1024, 24.10.; not for merge): two rank-agreement questions the review
round left open. Each test is GREEN when the named split EXISTS on the author head b636bdf7c3 (a probe documents a
finding; it is not a regression test).

P1 Form A (NF D: TP0 = host, TP1/TP2 = workers): the host holds a protected, unsecured leaf after its K15 save failed;
   a worker never holds (``_decide``: role worker -> False) and demotes it -> the two trees' device residency parts
   (K21: "ALL ranks keep the node on the device (a rank-local skip on TP0 alone parted the trees)").
P2 Shared-slot TP (27B D): ``TreeOps.agreed_unneeded`` marks a registry node secured on the rank whose ``_has_anchor``
   sees a rank-local mamba host copy (device state already tombstoned on every rank, KV still on the device); the
   other rank has no candidate for it and marks nothing. Once the device state is back on both ranks (not
   re-registered), ``protected`` / ``interior_hold`` answer differently per rank.
"""

from __future__ import annotations

import importlib.util
import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def G(monkeypatch):
    for k in ("FLLIPER_PDFLIP_END_ANCHOR_GUARD", "FLLIPER_PDFLIP_END_ANCHOR_GUARD_MAX_HELD",
              "FLLIPER_PDFLIP_DUAL_LAYOUT", "FLLIPER_PDFLIP_UD_LAST"):
        monkeypatch.delenv(k, raising=False)
    from flliper.srt.pdflip import end_anchor_guard as EAG

    monkeypatch.setattr(EAG, "_N", {k: 0 for k in EAG._N})
    monkeypatch.setattr(EAG, "_SAVE", {k: [0, 0, 0.0, 0.0] for k in EAG._SAVE})
    return _load("_probe_guard_helpers", "test_end_anchor_guard_1024.py")


def test_p1_form_a_host_holds_alone_and_the_worker_demotes(G, monkeypatch):
    from flliper.srt.pdflip import end_anchor_guard as EAG

    out = {}
    for role in ("host", "worker"):
        monkeypatch.setattr(EAG, "_role", lambda r=role: r)
        t, nodes = G._metal(tp=3, save_ok=False)        # TP0's own S108 deferral (Form A: the host's claim is the group's)
        assert not EAG.tp_slot_shared(t)
        G._peel(t, 5000)
        out[role] = (list(t.demoted), {k: n.component_data[G.M].value is not None for k, n in sorted(nodes.items())})
    print("P1", out)
    assert out["host"][0] == ["Y"] and out["worker"][0] == ["X"], out
    assert out["host"][1] != out["worker"][1], "device residency equal -- no split"


def test_p2_agreed_unneeded_marks_on_the_rank_with_a_local_host_copy_only(G, monkeypatch):
    from flliper.srt.pdflip import end_anchor_guard as EAG
    from flliper.srt.pdflip import park_anchor_rearm as par

    R = _load("_probe_rearm_world", "test_park_anchor_rearm_1024.py")
    monkeypatch.setattr(EAG, "_role", lambda: None)
    M = G.M
    trees, xs = [], []
    for r in (0, 1):
        t, root = G._fake_tree(tp=2)
        x = G._Node("X", root, 4096)                    # interior-shaped registry node, KV on the device
        EAG.register_end(t, x)
        x._pdflip_end_anchor = True
        x.component_data[M].value = None                # pass 2 tombstoned its mamba state on EVERY rank
        if r == 0:
            x.component_data[M].host_value = object()   # rank 0 alone kept a host copy (its own claim)
        t._pdflip_anchor_secured_holds = lambda n: False
        trees.append(t)
        xs.append(x)
    assert all(EAG.tp_slot_shared(t) for t in trees)
    ops = {r: par.TreeOps(trees[r], None, busy=False) for r in (0, 1)}
    res, g = R._run_ranks(2, lambda r, gm: par.run_pass(ops[r], gm, r, 2))
    marks = [bool(getattr(x, EAG.AGREED_ATTR, False)) for x in xs]
    print("P2 marks", marks, "agreed", {r: res[r].get("agreed") for r in res})
    assert marks == [True, False], marks
    # the device state comes back on both ranks (anchor read / re-insert), no new turn end at X
    for x in xs:
        x.component_data[M].value = object()
    prot = [EAG.protected(t, x) for t, x in zip(trees, xs)]
    holds = []
    for t, x in zip(trees, xs):
        prev = EAG.begin(t)
        try:
            holds.append(EAG.interior_hold(t, x, x.component_data[M]))
        finally:
            EAG.end(t, prev)
    print("P2 protected", prot, "interior_hold", holds)
    assert prot[0] is None and prot[1] == "end", prot
    assert holds == [False, True], holds
