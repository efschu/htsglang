"""NF h18 fe799c4d6e death, 14.10. 16:23:24Z (D.log boot ..._1014_161406, line 48735):
``gloo::EnforceNotMet ... op.preamble.length <= op.nbytes. 8 vs 4 ... distributed collective mismatch``.

THE CHAIN (D.log):
* 16:22:28 weg2-10-28's read record differs by rank (Form A span bases): TP0 ``matched=0 completed_synced=0``,
  TP1/TP2 ``matched=19072 completed_synced=19072`` -> ``_1471_short`` (from this rank's own record) on TP0 only.
* 16:23:23 the #1471 settle tick: ``#1471w SETTLE-WRITER rid=weg2-10-28 writer=p-published prev=None
  action=reread`` on TP0 ONLY (the workers' action stays "poll"). The next tick TP0's action is "decide"
  (the ack was seen) -- and loop 1 SKIPPED ``_refetch`` (the progress check's collectives) for a "decide" member.
* 16:23:24 TP0's next collective was weg2-12-29's while TP1/TP2 checked weg2-10-28:
  TP0 ``#905 PREFETCH-COMPLETE req=weg2-12-29 completed=96256 min_synced=62848`` (62848 = the workers' 10-28),
  TP1/TP2 ``req=weg2-10-28 completed=62848 min_synced=62848``. TP0 deferred 12-29 alone (#1068) and re-issued it
  -> its prefetch thread ran an operation the workers did not -> the claim reduces met with 8 vs 4 bytes.

(2) SETTLE-UNIFORM-1014: every rank runs the progress check for every non-capwait settle member; the rank-local
    action shapes only the vote.  The rank-equality mutant below runs the tick on three "ranks" whose actions
    differ exactly like h18 and compares the per-rank sequences of collective-bearing calls.
(3) PREFETCH-RID-NET-1014: the termination collective carries the rid's fingerprint; ranks at different rids
    see it alike and answer "not done", named, instead of mixing two rids' numbers.
"""

import logging
import os
import sys
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_weg2_wake_read_seat_gate_1014 import _req, _sched  # noqa: E402

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.managers import scheduler as sched_mod  # noqa: E402
from sglang.srt.mem_cache import unified_radix_cache as URC  # noqa: E402


@pytest.fixture(autouse=True)
def _group_d(monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_GROUP", "D")
    monkeypatch.delenv("SGLANG_WEG2_SETTLE_UNIFORM_CHECK", raising=False)
    monkeypatch.delenv("SGLANG_WEG2_PREFETCH_RID_NET", raising=False)
    from sglang.srt.weg2 import park_l3
    with envs.SGLANG_WEG2_ENABLE_PARK_L3.override(True):
        yield


# ------------------------------------------------------------------ (2) the settle tick, rank by rank

def _rank_tick(monkeypatch, acts):
    """One #1471 settle tick on one 'rank': ``acts`` = this rank's writer action per rid.
    Returns the sequence of collective-bearing calls (``_refetch`` = the progress check)."""
    h = _sched()
    calls = []

    def _refetch(req, now, allow_reissue=True):
        calls.append(("refetch", req.rid, bool(allow_reissue)))
        return "reading"

    h._weg2_refetch_one = _refetch
    h._weg2_group_min_flags = lambda flags: list(flags)   # the vote's own collective: one per tick, any rank
    monkeypatch.setattr(sched_mod, "_weg2_settle_writer_action", lambda sched, req: acts[req.rid])
    settle = [_req(r, 900) for r in ("weg2-10-28", "weg2-12-29", "weg2-16-32")]
    from sglang.srt.weg2 import park_l3
    for r in settle:
        setattr(r, park_l3.DEFER_ATTR, False)
        setattr(r, park_l3.CAPWAIT_ATTR, False)
        r._1471_since = 0.0
    h.weg2_post_wake_settle = settle
    h._weg2_post_wake_settle_tick()
    return calls


H18_TP0 = {"weg2-10-28": "decide", "weg2-12-29": "poll", "weg2-16-32": "poll"}
H18_TP12 = {"weg2-10-28": "poll", "weg2-12-29": "poll", "weg2-16-32": "poll"}


def test_h18_every_rank_runs_the_same_progress_checks(monkeypatch):
    tp0 = _rank_tick(monkeypatch, H18_TP0)
    tp1 = _rank_tick(monkeypatch, H18_TP12)
    assert [c[1] for c in tp0 if not c[2]] == [c[1] for c in tp1 if not c[2]], (tp0, tp1)
    assert tp0[0][1] == "weg2-10-28", "TP0 checks 10-28 first, like its peers (h18: it went to 12-29)"


def test_rank_equality_mutant_the_old_skip_splits_the_ranks(monkeypatch):
    """The mutant: the switch off is the h18 code. The per-rank sequences differ -- this test is the
    detector that must stay red on the old form."""
    monkeypatch.setenv("SGLANG_WEG2_SETTLE_UNIFORM_CHECK", "0")
    tp0 = _rank_tick(monkeypatch, H18_TP0)
    tp1 = _rank_tick(monkeypatch, H18_TP12)
    assert [c[1] for c in tp0 if not c[2]] != [c[1] for c in tp1 if not c[2]]
    assert tp0[0][1] == "weg2-12-29"


def test_a_wait_member_is_checked_but_never_reissued(monkeypatch):
    acts = {"weg2-10-28": "wait", "weg2-12-29": "poll", "weg2-16-32": "poll"}
    calls = _rank_tick(monkeypatch, acts)
    assert ("refetch", "weg2-10-28", False) in calls
    assert ("refetch", "weg2-10-28", True) not in calls


def test_default_on():
    assert envs.SGLANG_WEG2_SETTLE_UNIFORM_CHECK.get()
    assert envs.SGLANG_WEG2_PREFETCH_RID_NET.get()


# ------------------------------------------------------------------ (3) the rid net in the termination collective

def _tree(peer_rids):
    """``can_terminate_prefetch`` of one rank whose all-reduce MAXes in the peers' states, built from the
    same operation shape for ``peer_rids`` (what the peers reached the collective with)."""
    t = object.__new__(URC.UnifiedRadixCache)
    t.prefetch_stop_policy = "timeout"
    t.page_size = 1
    t._prefetch_timeout_check_linear_func = lambda op: False

    def _reduce(tensor, op, label="hicache"):
        assert label == "can_terminate_prefetch"
        for rid in peer_rids:
            h = URC._prefetch_rid_fingerprint(rid)
            peer = torch.tensor([0, 0, 0] + ([h, -h] if tensor.numel() == 5 else []), dtype=torch.int)
            torch.maximum(tensor, peer, out=tensor)

    t._all_reduce_attn_groups = _reduce
    return t


def _op(rid):
    return types.SimpleNamespace(hash_value=[1, 2], completed_tokens=2, pool_transfers=None,
                                 is_terminated=lambda: False, request_id=rid, read_start_time=None)


def test_same_rid_on_every_rank_terminates():
    t = _tree(["weg2-10-28", "weg2-10-28"])
    assert t.can_terminate_prefetch(_op("weg2-10-28")) is True


def test_h18_different_rids_are_not_done_and_named(caplog):
    caplog.set_level(logging.ERROR)
    t = _tree(["weg2-10-28", "weg2-10-28"])          # TP1/TP2 at 10-28
    assert t.can_terminate_prefetch(_op("weg2-12-29")) is False   # TP0 at 12-29
    assert any("PREFETCH-RID-MISMATCH" in m and "rid=weg2-12-29" in m for m in caplog.messages)


def test_every_rank_sees_the_mismatch_alike():
    # the workers' side of the same collective: they hold 10-28 and meet TP0's 12-29
    t = _tree(["weg2-12-29"])
    assert t.can_terminate_prefetch(_op("weg2-10-28")) is False


def test_net_off_is_the_three_slot_form(monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_PREFETCH_RID_NET", "0")
    t = _tree(["weg2-10-28"])
    assert t.can_terminate_prefetch(_op("weg2-12-29")) is True   # the old form cannot see it


def test_fingerprint_is_stable_and_positive():
    a = URC._prefetch_rid_fingerprint("weg2-10-28")
    assert a == URC._prefetch_rid_fingerprint("weg2-10-28") and 0 < a < 2 ** 31
    assert a != URC._prefetch_rid_fingerprint("weg2-12-29")


# ------------------------------------------------------------------ STORE-HELD-DECIDE is group-uniform (suspect ruled out)
# The coordinator's suspect for both deaths: does STORE-HELD-DECIDE decide on completed_local? It decides on the
# drain's stamps ``_weg2_store_delivered`` (the reduced END, HFB) and ``_weg2_store_deliverable`` (absolute: the
# rank's span-relative deliverable + reduced END - its relative synced). Driven through the REAL drain chain on three
# stand-in ranks with the Form A span bases of the metal (TP0 base 12544/15104, workers 16384): the same verdict on
# every rank on every pass. The mutant decides on each rank's own completed_local and splits them.

import test_weg2_store_short_form_a_end_hfb as hfb  # noqa: E402


def test_store_held_decide_answers_alike_on_every_rank(monkeypatch):
    monkeypatch.delenv("SGLANG_WEG2_STORE_SHORT_HELD_DECIDE", raising=False)
    monkeypatch.setenv("SGLANG_WEG2_STORE_SHORT_TAIL", "0")
    ranks = hfb._ranks()
    for p in hfb.PASSES:
        outs = hfb._pass(ranks, *p)
        assert len(set(outs)) == 1, outs
    held = {round(getattr(r, "_weg2_store_deliverable", -1)) for _s, r in ranks}
    assert len(held) == 1, f"store_held differs by rank: {held}"


def test_store_held_decide_mutant_on_completed_local_splits_the_ranks(monkeypatch):
    monkeypatch.delenv("SGLANG_WEG2_STORE_SHORT_HELD_DECIDE", raising=False)
    monkeypatch.setenv("SGLANG_WEG2_STORE_SHORT_TAIL", "0")

    def _local_mutant(req):
        rec = None
        for s, r in ranks:
            if r is req:
                rec = s.tree_cache.prefetch_loaded_tokens_by_reqid[hfb.RID]
        return int(rec.materialized) < 10000   # a rank-local completed_local reading

    monkeypatch.setattr(sched_mod, "_weg2_store_held_short", _local_mutant)
    ranks = hfb._ranks()
    split = False
    for p in hfb.PASSES:
        outs = hfb._pass(ranks, *p)
        split = split or len(set(outs)) > 1
    assert split, "the detector must see a rank-local held decision"
