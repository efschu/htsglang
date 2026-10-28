# SPDX-License-Identifier: Apache-2.0
"""formhp-devdraft-vprep-1028: FORMHP-VPREP-EARLY -- the DFLASH verify prep of an early-draft round runs before the length wait.

CPU only: the order of prep / wait / fix / snapshot in staged_prep (old order, armed, self-test, eager redo, errors), the snapshot's
reach and cap, the bitwise compare, the fix after the wait, the verdict. Red on 6f5da68e5 (no dflash_vprep_early module)."""

import dataclasses
import types

import pytest
import torch

from flliper.srt.speculative import dflash_vprep_early as ve


@dataclasses.dataclass
class _Meta:
    idx: torch.Tensor
    n: int
    owner: object = None


@dataclasses.dataclass
class _FB:
    seq_lens_cpu: torch.Tensor = None
    seq_lens_sum: int = 0
    input_ids: torch.Tensor = None


def _runner(bs=2, cap=4):
    reg = types.SimpleNamespace(_slots={
        "seq_lens_cpu": types.SimpleNamespace(enabled=True, buffer=torch.full((cap,), 1, dtype=torch.int64)),
        "input_ids": types.SimpleNamespace(enabled=True, buffer=torch.zeros(cap * 8, dtype=torch.int64)),
    })
    be0 = types.SimpleNamespace(forward_metadata=_Meta(torch.zeros(3, dtype=torch.int32), 0))
    be1 = types.SimpleNamespace(forward_metadata=None)
    return types.SimpleNamespace(buffer_registry=reg, attn_backend=types.SimpleNamespace(attn_backend_list=[be0, be1]),
                                 raw_bs=bs, raw_num_token=bs * 8, bs=bs, _replay_graph_key="k")


class _World:
    """A prep that writes what the real one writes, from the host bound it is given (exact or reservation)."""

    def __init__(self, metadata_reads_bound=False, graph=True):
        self.runner = _runner()
        self.calls = []
        self.exact = None  # round N's lengths, known after the wait
        self.reservation = torch.tensor([30, 40])
        self.metadata_reads_bound = metadata_reads_bound
        self.graph = graph

    def prep(self):
        bound = self.exact + 8 if self.exact is not None else self.reservation
        self.calls.append(("prep", "exact" if self.exact is not None else "bound"))
        fb = _FB(seq_lens_cpu=bound.clone(), seq_lens_sum=int(bound.sum()), input_ids=torch.arange(16))
        reg = self.runner.buffer_registry._slots
        reg["seq_lens_cpu"].buffer[:2].copy_(bound)
        reg["input_ids"].buffer[:16].copy_(fb.input_ids)
        m = self.runner.attn_backend.attn_backend_list[0].forward_metadata
        m.idx.copy_(torch.tensor([1, 2, 3], dtype=torch.int32) + (int(bound[0]) if self.metadata_reads_bound else 0))
        return ("vi", fb, "pre", self.graph)

    def marks(self):
        self.calls.append(("marks",))

    def wait(self):
        self.calls.append(("wait",))
        self.exact = torch.tensor([20, 25])


def _run(state, w, eligible=True):
    return ve.staged_prep(state=state, eligible=eligible, prep=w.prep, marks=w.marks, wait=w.wait, exact_lens=lambda: w.exact,
                          runner=w.runner, block_size=8, bs=2)


def test_old_order_when_not_eligible_or_off(monkeypatch):
    monkeypatch.delenv(ve.ENV, raising=False)
    w = _World()
    _run(ve.VprepEarly(), w, eligible=False)
    assert w.calls == [("wait",), ("prep", "exact"), ("marks",)]
    monkeypatch.setenv(ve.ENV, "0")
    w = _World()
    _run(ve.VprepEarly(), w)
    assert w.calls == [("wait",), ("prep", "exact"), ("marks",)]


def test_selftest_rounds_use_the_old_prep_then_arm(monkeypatch):
    monkeypatch.delenv(ve.ENV, raising=False)
    st = ve.VprepEarly()
    for i in range(ve.SELFTEST_ROUNDS):
        w = _World()
        vp = _run(st, w)
        assert w.calls == [("prep", "bound"), ("marks",), ("wait",), ("prep", "exact")]
        assert torch.equal(vp[1].seq_lens_cpu, torch.tensor([28, 33]))  # the round's result is the old prep
    assert st.verdict is True
    w = _World()
    vp = _run(st, w)
    assert w.calls == [("prep", "bound"), ("marks",), ("wait",)]
    # the fix put the exact host values where the old order put them
    assert torch.equal(vp[1].seq_lens_cpu, torch.tensor([28, 33])) and vp[1].seq_lens_sum == 61
    slot = w.runner.buffer_registry._slots["seq_lens_cpu"].buffer
    assert slot.tolist() == [28, 33, 1, 1]


def test_selftest_refuses_when_the_metadata_reads_the_bound(monkeypatch):
    monkeypatch.delenv(ve.ENV, raising=False)
    st = ve.VprepEarly()
    w = _World(metadata_reads_bound=True)
    vp = _run(st, w)
    assert st.verdict is False and "attn0.forward_metadata.idx" in st.why
    assert torch.equal(vp[1].seq_lens_cpu, torch.tensor([28, 33]))
    w = _World()
    _run(st, w)
    assert w.calls == [("wait",), ("prep", "exact"), ("marks",)]


def test_eager_prep_prepares_again_after_the_wait(monkeypatch):
    monkeypatch.delenv(ve.ENV, raising=False)
    st = ve.VprepEarly()
    st.verdict = True
    w = _World(graph=False)
    vp = _run(st, w)
    assert w.calls == [("prep", "bound"), ("marks",), ("wait",), ("prep", "exact")]
    assert torch.equal(vp[1].seq_lens_cpu, torch.tensor([28, 33]))


def test_early_prep_error_falls_back_and_marks_once(monkeypatch):
    monkeypatch.delenv(ve.ENV, raising=False)
    st = ve.VprepEarly()
    w = _World()
    n = {"k": 0}
    real = w.prep

    def flaky():
        n["k"] += 1
        if n["k"] == 1:
            raise RuntimeError("x")
        return real()

    vp = ve.staged_prep(state=st, eligible=True, prep=flaky, marks=w.marks, wait=w.wait, exact_lens=lambda: w.exact,
                        runner=w.runner, block_size=8, bs=2)
    assert st.verdict is False and [c for c in w.calls if c == ("marks",)] == [("marks",)]
    assert vp[3] is True


def test_snapshot_reach_and_cap(monkeypatch):
    r = _runner()
    r.attn_backend.attn_backend_list[0].forward_metadata.owner = types.SimpleNamespace(big=torch.zeros(4))
    snap = ve.snapshot(r, _FB(seq_lens_cpu=torch.tensor([1]), seq_lens_sum=1))
    assert "attn0.forward_metadata.owner.big" in snap  # a SimpleNamespace is walked

    class Opaque:
        def __init__(self):
            self.t = torch.zeros(4)

    r.attn_backend.attn_backend_list[0].forward_metadata.owner = Opaque()
    snap = ve.snapshot(r, None)
    assert snap["attn0.forward_metadata.owner"].startswith("<") and "attn0.forward_metadata.owner.t" not in snap
    monkeypatch.setattr(ve, "_MAX_TOTAL_BYTES", 16)
    with pytest.raises(ve.SnapshotTooBig):
        ve.snapshot(_runner(), None)


def test_compare_bitwise():
    a = {"x": torch.tensor([1.0, -0.0]), "n": 3}
    assert ve.compare(a, {"x": torch.tensor([1.0, -0.0]), "n": 3})[0]
    assert not ve.compare(a, {"x": torch.tensor([1.0, 0.0]), "n": 3})[0]  # -0.0 vs 0.0: equal by value, not by bytes
    assert not ve.compare(a, {"x": torch.tensor([1.0, -0.0]), "n": 4})[0]
    assert not ve.compare(a, {"x": torch.tensor([1.0, -0.0])})[0]
