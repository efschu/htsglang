"""#239 S4b part 5: tail adopt (E1/E2) under the Form A token cut, by owner.

Before part 5 every rank of a token-cut D group refused the tail adoption
(``dcp_active``: the tail rows were written by GLOBAL slot, the cut's KV pools
are COMPACT per owner), so a flip under the cut always re-extended the partial
page and never skipped. Part 5: each rank writes the rows of the partial page
it OWNS at their compact slot -- the same write rule the attention backends
use (``layers/dcp/owner.dcp_weighted_write_slots``) -- the host (share 0 in
0/48/16) writes none, a worker (no GDN layer) lands its rows at its first
attention step of the extend. What does not fit the owner install (plain
uneven DCP without the Form A cut; the QSA indexer's compressed or ring rows
on a worker) keeps the group-uniform refusal.

Hermetic, CPU tensors. RED on 72312793e4 (no owner path), GREEN with part 5.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from flliper.srt.pdflip import tail_adopt as ta

PAGE = 64
TP1 = (64, 0, 48, 48)   # the cut 0/48/16: TP1 owns page offsets [0, 48)
TP2 = (64, 48, 64, 16)  # TP2 owns [48, 64)
HOST = (64, 0, 0, 0)    # TP0: share 0


def _row(*shape, dtype="torch.bfloat16"):
    return ta.RowSpec(shape=list(shape), dtype=dtype)


def _held(owner, fa_rows=2, gdn=False, dcp=True):
    fa = {3: [_row(2, 8)] * fa_rows}
    g = {0: [_row(4, 4)]} if gdn else {}
    return ta.HeldShapes(fa=fa, gdn=g, qsa_ratio=0, dcp=dcp, owner=owner)


# ------------------------------------------------------------------ the gate
def test_the_gate_keeps_the_pre_cut_refusal_and_lifts_it_under_the_cut():
    assert ta.cut_gate(_held(None, dcp=False)) is None           # no DCP: untouched
    assert ta.cut_gate(_held(None)) == "dcp_active"              # plain uneven DCP: refused
    assert ta.cut_gate(_held(TP1)) == ""                         # cut worker, K/V only
    assert ta.cut_gate(_held(HOST, fa_rows=3, gdn=True)) == ""   # the host: GDN + QSA rows
    assert ta.cut_gate(_held(TP2, fa_rows=3)) == "cut_qsa_rows_on_worker"  # riegel


# ------------------------------------------------------------------ the write rule
def test_owner_rows_are_the_backends_write_rule():
    idx = torch.arange(10 * PAGE, 10 * PAGE + 49)  # partial page, 49 rows (c = 640+49)
    loc, keep = ta.owner_rows(idx, TP1)
    assert int(keep.sum()) == 48 and loc.tolist() == list(range(10 * 48, 10 * 48 + 48))
    loc2, keep2 = ta.owner_rows(idx, TP2)
    assert keep2.nonzero().flatten().tolist() == [48] and loc2.tolist() == [10 * 16]
    loc0, keep0 = ta.owner_rows(idx, HOST)
    assert int(keep0.sum()) == 0 and loc0.numel() == 0


def _install(owner, n_rows=49, gdn=False):
    k = torch.arange(n_rows * 2 * 8, dtype=torch.float32).view(n_rows, 2, 8)
    v = -k
    kbuf = torch.zeros(1024, 2, 8)
    vbuf = torch.zeros(1024, 2, 8)
    rows = torch.arange(10 * PAGE, 10 * PAGE + n_rows)
    inst = ta.Install(
        spec=SimpleNamespace(rid="pdflip-9-9", rows=n_rows), headers=[], fa={3: (k, v)},
        gdn=({0: (torch.zeros(1),)} if gdn else {}), fa_dst={3: (kbuf, vbuf)}, gdn_dst={},
        rows=rows, groups=None, slot=None, t0=0.0, owner=owner,
    )
    return inst, k, v, kbuf, vbuf


def test_a_cut_worker_writes_exactly_its_owned_rows_at_their_compact_slots():
    inst, k, v, kbuf, vbuf = _install(TP1)
    ta._put_fa(inst, inst.rows)
    assert torch.equal(kbuf[480:528], k[:48]) and torch.equal(vbuf[480:528], v[:48])
    assert int((kbuf[:480] != 0).sum()) == 0 and int((kbuf[528:] != 0).sum()) == 0
    inst2, k2, _v2, kbuf2, _ = _install(TP2)
    ta._put_fa(inst2, inst2.rows)
    assert torch.equal(kbuf2[160], k2[48])
    assert int((kbuf2 != 0).flatten(1).any(1).sum()) == 1


def test_the_host_of_share_zero_writes_no_kv_row():
    inst, _k, _v, kbuf, vbuf = _install(HOST, gdn=True)
    ta._put_fa(inst, inst.rows)
    assert int((kbuf != 0).sum()) == 0 and int((vbuf != 0).sum()) == 0


def test_without_the_cut_the_rows_go_by_global_slot_as_before():
    inst, k, _v, kbuf, _ = _install(None)
    ta._put_fa(inst, inst.rows)
    assert torch.equal(kbuf[640:689], k)


# ------------------------------------------------------------------ the worker's install point
def test_the_worker_install_lands_at_its_first_attention_step_and_leaves_the_host_install(monkeypatch):
    finished = []
    monkeypatch.setattr(ta, "_finish", lambda inst, verify: finished.append((inst, verify)))
    worker, _k, _v, kbuf, _ = _install(TP1)
    host, *_ = _install(HOST, gdn=True)
    monkeypatch.setattr(ta, "PENDING_INSTALLS", [worker, host])
    ta.install_worker_rows(3)
    assert ta.PENDING_INSTALLS == [host]   # the host's rides its GDN read
    assert finished == [(worker, False)] and worker.fa_done
    assert int((kbuf[480:528] != 0).sum()) > 0


def test_the_wiring_runs_the_install_before_the_worker_attention(monkeypatch):
    from flliper.srt import form_a_dcp_wiring as w

    order = []
    backend = SimpleNamespace(form_a_dcp=object(),
                              form_a_worker_attention=lambda fb, lid: order.append(("attn", lid)))
    pool = SimpleNamespace(full_attention_layer_id_mapping={3: 0})
    monkeypatch.setattr(ta, "PENDING_INSTALLS", [object()])
    monkeypatch.setattr(ta, "install_worker_rows", lambda lid: order.append(("install", lid)))
    attention, ids = w.form_a_worker_attention_step(backend, SimpleNamespace(), pool)
    attention(3)
    assert order == [("install", 3), ("attn", 3)] and ids == (3,)


def test_no_pending_install_no_extra_call(monkeypatch):
    from flliper.srt import form_a_dcp_wiring as w

    calls = []
    backend = SimpleNamespace(form_a_dcp=object(), form_a_worker_attention=lambda fb, lid: calls.append(lid))
    monkeypatch.setattr(ta, "PENDING_INSTALLS", [])
    monkeypatch.setattr(ta, "install_worker_rows", lambda lid: pytest.fail("called without an install"))
    attention, _ = w.form_a_worker_attention_step(
        backend, SimpleNamespace(), SimpleNamespace(full_attention_layer_id_mapping={3: 0}))
    attention(3)
    assert calls == [3]
