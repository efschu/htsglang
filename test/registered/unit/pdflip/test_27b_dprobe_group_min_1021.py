# SPDX-License-Identifier: Apache-2.0
"""27B NACHT 21.10. -- D-PROBE GROUP-MIN (SOLL R11; user order 19.10. "NICHTS SCHÄTZEN, EXAKTE ZAHLEN").

Metal class (27B INT8, done/27b-int8-abnahme-1021.md): ``D-PROBE-FAILED why=group_min_needed: asymmetric host tier
without Form A`` 22 of 22 (rc29 ..._041a871f4b_1021_001910) and on every arrival of rc29b
(..._c77da494f9_1021_011346, front.log 374: pdflip-0-1). Every arrival then went through the measuring barrier
(MEASURE_ON_D, front_x = halbX-1): pdflip-0-1 (u = 73772 - 44288 = 29484 >= X, a GROSS request) was seated on D as
"short", sat unstarted through the D->P flip (PARK-HANDBACK kept=committed), came back store-short after the flip
(``store_short_after_flip`` d_extent=29484) and cost a second flip pair (TTFT 236 s).

The 27B D group is TP3 over 3080/5090/3080 with an uneven DCP vector: its host tiers are asymmetric and it is not
Form A, so the admission's number is the RU MIN over the ranks. The probe now reads it the same way: every rank
peeks read-only, the group reduces by MIN on the TP CPU group. These tests drive the real scheduler methods with a
three-rank MIN emulated in-process (the other ranks' vectors are injected into the reduce).
"""
import types
from array import array

import pytest
import torch

from flliper.srt.managers import scheduler as S
from flliper.srt.managers import tp_match_floor as T
from flliper.srt.managers.io_struct import PdFlipMatchProbeReqInput, unwrap_from_pickle, wrap_as_pickle
from flliper.srt.pdflip import d_match_probe as dmp


class _AsymTree:
    """The 27B D tree: asymmetric host tier (``_hicache_prefetch_symmetric`` -- historical name -- is True)."""

    def _hicache_prefetch_symmetric(self):
        return True

    def match_prefix(self, params):  # pragma: no cover - the probe must never mutate the tree
        raise AssertionError("match_prefix must not be called")


def _fake(tree=None):
    fake = types.SimpleNamespace(
        tree_cache=_AsymTree() if tree is None else tree,
        tp_cpu_group=object(),
        model_config=types.SimpleNamespace(vocab_size=1000),
        pad_input_ids_func=None,
        _get_multimodal_inputs=lambda mm: None,
        _try_apply_padded_mm_input_ids=lambda recv, req, mm: False,
    )
    for m in ("_pdflip_match_probe_one", "_pdflip_match_probe_pad", "_pdflip_match_probe_local",
              "_pdflip_match_probe_group_min"):
        setattr(fake, m, (lambda n: lambda *a: getattr(S.Scheduler, n)(fake, *a))(m))
    return fake


@pytest.fixture
def three_ranks(monkeypatch):
    """A TP3 group: this rank's vector MIN-reduced with the other two ranks' vectors (queued per reduce call)."""
    others = []
    calls = []

    def _ws(grp):
        return 3

    def _all_reduce(t, op=None, group=None):
        assert op == torch.distributed.ReduceOp.MIN
        mine = [int(v) for v in t.tolist()]
        peer = others.pop(0) if others else [mine, mine]
        calls.append((mine, peer))
        red = [min(a, *(p[i] for p in peer)) for i, a in enumerate(mine)]
        t.copy_(torch.tensor(red, dtype=torch.int64))

    monkeypatch.setattr(torch.distributed, "get_world_size", _ws)
    monkeypatch.setattr(torch.distributed, "all_reduce", _all_reduce)
    monkeypatch.setattr(T, "form_a_follow_active", lambda: False)
    return types.SimpleNamespace(others=others, calls=calls)


def test_rc29b_pdflip_0_1_gets_the_group_number_not_group_min_needed(three_ranks, monkeypatch):
    """pdflip-0-1: n=73772. TP0 (5090 share) holds more on its host tier than TP1/TP2; the admission's MIN is the
    group's number. The probe answers it -- success=True, never ``group_min_needed``."""
    n = 73772
    monkeypatch.setattr(dmp, "peek_admission_match", lambda tc, ids, ek, limit: (30000, 20000, 50000))
    # TP1 / TP2: ok, dev+host, dev, key
    three_ranks.others.append([[1, 44288, 30000, 44288], [1, 47000, 31000, 47000]])
    out = S.Scheduler.handle_pdflip_match_probe(
        _fake(), PdFlipMatchProbeReqInput(rid="pdflip-0-1~probe", input_ids=array("q", range(n))))
    assert out.success is True, out.message
    assert out.device_tokens + out.host_tokens == 44288          # the MIN of dev+host (TP1)
    assert out.device_tokens == 30000 and out.host_tokens == 14288
    assert out.key_match_depth == 44288
    assert len(three_ranks.calls) == 1                            # exactly one reduce per probe
    # the front's price on this number: u = n - covered = 29484 >= X (12288) -> gross, R4 -> P (no D seat, no
    # store_short_after_flip after a flip)
    assert n - (out.device_tokens + out.host_tokens) == 29484


def test_a_rank_whose_walk_fails_still_enters_the_reduce_and_the_group_refuses_by_name(three_ranks, monkeypatch):
    """Rang-Einigkeit: a failing walk on ONE rank must not leave the others hanging in the reduce. This rank fails:
    it still reduces (ok=0), the group answers success=False by name."""
    def boom(*a):
        raise RuntimeError("walk broke")

    monkeypatch.setattr(dmp, "peek_admission_match", boom)
    three_ranks.others.append([[1, 500, 500, 500], [1, 500, 500, 500]])
    out = S.Scheduler.handle_pdflip_match_probe(
        _fake(), PdFlipMatchProbeReqInput(rid="r~probe", input_ids=array("q", range(1000))))
    assert len(three_ranks.calls) == 1 and three_ranks.calls[0][0][0] == 0
    assert out.success is False and out.message.startswith("group_probe_failed") and "walk broke" in out.message


def test_another_rank_failing_makes_this_rank_refuse_too(three_ranks, monkeypatch):
    monkeypatch.setattr(dmp, "peek_admission_match", lambda tc, ids, ek, limit: (100, 0, 100))
    three_ranks.others.append([[0, 0, 0, 0], [1, 100, 100, 100]])
    out = S.Scheduler.handle_pdflip_match_probe(
        _fake(), PdFlipMatchProbeReqInput(rid="r~probe", input_ids=array("q", range(200))))
    assert out.success is False and out.message.startswith("group_probe_failed")


def test_an_empty_prompt_still_reduces_once(three_ranks, monkeypatch):
    """n <= 1 walks nothing, but the reduce is entered like on every rank (the mode is launch-uniform)."""
    monkeypatch.setattr(dmp, "peek_admission_match", lambda *a: (_ for _ in ()).throw(AssertionError("no walk")))
    out = S.Scheduler.handle_pdflip_match_probe(_fake(), PdFlipMatchProbeReqInput(rid="r~probe", input_ids=None))
    assert out.success is True and out.device_tokens == 0 and out.host_tokens == 0
    assert len(three_ranks.calls) == 1


def test_a_coalesced_batch_reduces_once_per_probe_in_order(three_ranks, monkeypatch):
    monkeypatch.setattr(dmp, "peek_admission_match", lambda tc, ids, ek, limit: (len(ids) // 2, 0, len(ids)))
    three_ranks.others += [[[1, 3, 3, 3], [1, 9, 9, 9]], [[1, 50, 50, 50], [1, 7, 7, 7]], [[1, 99, 99, 99]] * 2]
    req = PdFlipMatchProbeReqInput(rid="a", input_ids=array("q", range(10)), batch=wrap_as_pickle([
        {"rid": "b", "input_ids": array("q", range(20)), "mm_inputs": None, "extra_key": None, "tokenize_ms": 1.0},
        {"rid": "c", "input_ids": array("q", range(40)), "mm_inputs": None, "extra_key": None, "tokenize_ms": 2.0}]))
    out = S.Scheduler.handle_pdflip_match_probe(_fake(), req)
    more = unwrap_from_pickle(out.batch)
    assert (out.rid, out.device_tokens) == ("a", 3)
    assert [(d["rid"], d["device_tokens"]) for d in more] == [("b", 7), ("c", 20)]
    assert len(three_ranks.calls) == 3


def test_form_a_and_symmetric_tiers_keep_the_local_answer_without_a_reduce(three_ranks, monkeypatch):
    """NF (Form A, TP0 is the authority) and a symmetric tier: TP0's number is the group's -- no new collective."""
    monkeypatch.setattr(dmp, "peek_admission_match", lambda tc, ids, ek, limit: (64, 0, 64))
    monkeypatch.setattr(T, "form_a_follow_active", lambda: True)
    monkeypatch.setattr(T, "this_rank_follows", lambda: False)
    out = S.Scheduler.handle_pdflip_match_probe(
        _fake(), PdFlipMatchProbeReqInput(rid="r~probe", input_ids=array("q", range(100))))
    assert out.success is True and out.device_tokens == 64 and three_ranks.calls == []

    class _Sym:
        def _hicache_prefetch_symmetric(self):
            return False

    monkeypatch.setattr(T, "form_a_follow_active", lambda: False)
    out = S.Scheduler.handle_pdflip_match_probe(
        _fake(_Sym()), PdFlipMatchProbeReqInput(rid="r~probe", input_ids=array("q", range(100))))
    assert out.success is True and out.device_tokens == 64 and three_ranks.calls == []


def test_the_refusal_text_is_gone_from_the_scheduler():
    src = open(S.__file__, encoding="utf-8").read()
    assert 'message="group_min_needed' not in src
