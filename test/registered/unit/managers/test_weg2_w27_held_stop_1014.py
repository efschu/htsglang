"""NF h16g (64991c27c4), P, 14.10. 17:49:53Z: ``#1233 #631 WIDTH GUARD (flip off):
16283 row(s) for 5723 token(s)`` on PP1 (P.log ``..._1014_174538`` 11701-12251).

THE METAL, line by line:

  17:49:36 PP0 SA SHORT-READ ANCHOR rid=weg2-0-10 landed=269 of 1876 anchor=267  -> told 3008+17088 = 20096
  17:49:43 PP1 SA SHORT-READ ANCHOR rid=weg2-0-10 landed=580 of 1876 anchor=559  -> state at 3008+35776 = 38784 ONLY
  17:49:43 PP1 #1423 INSERT-PLACED req=weg2-0-10 reg_depth=3008 inserted=35776 deepest=48
  17:49:50 PP0 #969N ADMIT slot=0 fwd_ct=15 bs=3 extend=16283 rids=[weg2-0-9,weg2-1-11,weg2-0-10]
  17:49:53 PP1 #1400 FOLLOWER-EARLY-SETTLE told=20096 own=38784 -> over
  17:49:53 PP1 W27-UNIFORM FOLLOWER TREE SHORT told=20096 own_record=20096 live=3008 site=satisfied
  17:49:53 PP1 SA SHORT-READ ANCHOR landed=60 of 267 pages anchor=0 read=none  (the re-read)
  17:49:53 PP1 W27-UNIFORM FOLLOWER HELD told=20096 live=3008 route=none
  17:49:53 PP1 #969N ADMIT slot=0 fwd_ct=15 bs=2 extend=5723 rids=[weg2-0-9,weg2-1-11]
  17:49:53 PP1 #1233 #631 WIDTH GUARD 16283 row(s) for 5723 token(s)

WHAT IT WAS -- NOT an eviction of the told span: node 48 (the inserted chain) appears in no
release line between 17:49:43 and the death (AUX-HOST-WITH-FULL named 49/44/36/40/28/31, all
weg2-0-9's chain; the #1417 span pin host-locks the chain, W3 spill / UD-H / AUX-HOST skip
host-locked nodes). ``live=3008`` is what ``rank_resumable(told=20096)`` answers on a tree whose
host run 3008..38784 carries its recurrent state at 38784 only: a match capped at 20096 finds
the anchor at 3008, the load-back stops there (``state_aligned_extent``) -- the #968
OVER-UNRESUMABLE class. Its guard is armed only with the PF ack stream, which was OFF on NF.

THE FIX: (1) PF on the nextflash row: the ack names the follower's resumable reach, PP0 answers
told=0 for EVERY rank before anyone seats it. (2) W27-HELD-STOP: a held told rid is a BARRIER
of the follower's admission loop (no overtake), and without a PF channel the unresolvable hold
stops the group by name at the hold instead of planning a different batch.
"""
from __future__ import annotations

import os
import re
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from sglang.srt.managers import weg2_store_told as m  # noqa: E402
from sglang.srt.managers import weg2_told_fallback as fb  # noqa: E402
from sglang.srt.managers import weg2_told_fidelity as tf  # noqa: E402
from sglang.srt.weg2 import form as FM  # noqa: E402
from sglang.srt.weg2 import p_intake  # noqa: E402
from sglang.srt.weg2 import p_twin_defer as twin  # noqa: E402

RID = "weg2-0-10"
HEAD = 3008                 # reg_depth=3008 on every stage (device head, state there)
TOLD = 20096                # PP0: 3008 + 17088 (anchor page 267)
OWN = 38784                 # PP1/PP2: 3008 + 35776 (anchor page 559)
PROMPT = 3008 + 120114
#: PP0's batch at fwd_ct=15 (#969 EXTENT): weg2-0-9 5698, weg2-1-11 25, weg2-0-10 10560
EXTENDS = {"weg2-0-9": 5698, "weg2-1-11": 25, RID: 10560}
PP0_ROWS = 16283
PP1_ROWS_METAL = 5723
#: literal, not ``m.ENV_HELD_STOP``: the file must run (and be red) at the base 853534a9ff too
HELD_STOP_ENV = "SGLANG_WEG2_W27_HELD_STOP"


def _node():
    mamba = SimpleNamespace(value="slot", host_value=None)
    kv = SimpleNamespace(value="kv", host_value=None)
    return SimpleNamespace(component_data=[kv, kv, mamba])


class _Tree:
    """One stage's tree for weg2-0-10: the device head, the host run beyond it and the
    depths that carry a recurrent state. A match capped at n reports the deepest state
    at or below its end -- the real tree's ``state_anchor_depth``."""

    is_eagle = False

    def __init__(self, dev, host, anchors, record):
        self.dev, self.host, self.anchors = dev, host, tuple(anchors)
        self.record = record
        self.prefetch_loaded_tokens_by_reqid = {RID: record}

    def check_prefetch_progress(self, rid):
        return True

    def completed_prefetch_tokens(self, rid):
        return self.record

    def pop_prefetch_loaded_tokens(self, rid):
        return self.prefetch_loaded_tokens_by_reqid.pop(rid, 0)

    def release_aborted_request(self, rid):
        # the PF fallback's upstream abort path: the read and its records go
        self.record = 0
        self.prefetch_loaded_tokens_by_reqid.pop(rid, None)

    def match_prefix(self, params):
        n = len(params.key)
        d = min(self.dev, n)
        h = max(0, min(self.host, n - d))
        reach = d + h
        anchor = max([a for a in self.anchors if a <= reach] or [0])
        return SimpleNamespace(
            device_indices=[0] * d, host_hit_length=h,
            last_device_node=_node(), last_host_node=_node(),
            state_anchor_depth=anchor if h > 0 else None, key_match_depth=reach,
        )


def _pp0_tree():
    return _Tree(dev=HEAD, host=TOLD - HEAD, anchors=(HEAD, TOLD), record=TOLD - HEAD)


def _follower_tree():
    # the early read: KV 3008..38784 host, the state at 38784 only (SA: deepest anchor read)
    return _Tree(dev=HEAD, host=OWN - HEAD, anchors=(HEAD, OWN), record=OWN - HEAD)


def _stage(rank, tree):
    s = SimpleNamespace()
    s.ps = SimpleNamespace(pp_rank=rank, pp_size=3, tp_size=1)
    s._weg2_store_held = {}
    s._weg2_store_told = {}
    s.tree_cache = tree
    s.waiting_queue = []
    s.calls = []

    def _prefetch_kvcache(req, limit_tokens=None):
        # 17:49:53 PP1: the told-limited re-read landed 60 of 267 pages, anchor=0 read=none
        # (#1028B FETCH CAP lost=60, arena full): the tree is unchanged
        s.calls.append((req.rid, limit_tokens))
        return "issued"

    s._prefetch_kvcache = _prefetch_kvcache
    return s


def _req(rid=RID):
    ids = list(range(PROMPT))
    return SimpleNamespace(rid=rid, origin_input_ids=ids, full_untruncated_fill_ids=ids,
                           extra_key=None, prefix_indices=None, host_hit_length=0,
                           _prefetch_registered_prefix_len=HEAD)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for k in ("SGLANG_WEG2_FOLLOWER_REACH_TOLD", "SGLANG_WEG2_DUAL_SHARE", "SGLANG_WEG2_DUAL_LAYOUT",
              tf.ENV, HELD_STOP_ENV):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv(m.ENV_FOLLOWER_EARLY_READ, "1")
    monkeypatch.setattr(m, "_absolute_armed", lambda: True)


def _absorb_admit(stage, req):
    """Weg2StoreAdmit(weg2-0-10, 20096) + twin absorbed, the early read done (DP-NACHLAUF)."""
    stage._weg2_store_told[RID] = TOLD
    twin.note_follower_twin(stage, RID)
    req._weg2_early_told = TOLD


def _plan(stage, queue, told_rids, budget=16384):
    """The scheduler's admission loop on the carrierless form (scheduler.py, the told branch):
    ``told_admission`` -> None = held; ``held_barrier`` -> break, else continue."""
    batch = []
    for req in queue:
        if req.rid in told_rids:
            credit = p_intake.told_admission(stage, req, lambda *a: None, m.admission)
            if credit is None:
                # the base scheduler had no barrier: ``continue`` (getattr = run red at the base)
                if getattr(m, "held_barrier", lambda _s, _r: False)(stage, req):
                    break
                continue
        take = min(EXTENDS[req.rid], budget)
        batch.append((req.rid, take))
        budget -= take
        if budget <= 0:
            break
    return batch


def _rows(batch):
    return sum(t for _r, t in batch)


def _queue():
    return [_req("weg2-0-9"), _req("weg2-1-11"), _req(RID)]


# -- the premise: live=3008 is the anchor, not an eviction ------------------------------------


def test_metal_live_3008_is_the_state_anchor_not_an_eviction():
    """W27 TREE SHORT live=3008 on a tree that still holds the WHOLE inserted span: the state
    sits at 38784, a match capped at told=20096 resumes at the head 3008."""
    pp1 = _stage(1, _follower_tree())
    assert tf.rank_resumable(pp1, _req(), TOLD) == HEAD == 3008
    assert tf.rank_resumable(pp1, _req(), OWN) == OWN
    assert tf.rank_resumable(_stage(0, _pp0_tree()), _req(), TOLD) == TOLD
    assert pp1.tree_cache.host == OWN - HEAD == 35776, "nothing of the span was released"


# -- (1) PF on: PP0 decides told=0 for every rank before anyone seats --------------------------


def test_pf_ack_names_the_reach_and_pp0_answers_told_zero_for_all(monkeypatch):
    monkeypatch.setenv(fb.ENV_FALLBACK, "1")
    acks = {}
    for rank in (1, 2):
        stage, req = _stage(rank, _follower_tree()), _req()
        twin.note_follower_twin(stage, RID)
        how = m.follower_early_settle(stage, req, TOLD, OWN, absolute=True)
        assert how == m.OVER_UNRESUMABLE, how     # was "over" -> SATISFIED with PF off
        assert RID not in (getattr(stage, "_weg2_store_told_satisfied", None) or {})
        acks[rank] = fb.own_prefix(stage, req, RID, TOLD)
        assert acks[rank] == OWN != TOLD
    pp0 = _stage(0, _pp0_tree())
    fb.pp0_open(pp0, RID, TOLD, published_at=37.0, own_read_s=6.42, window_s=8.02)
    fb._pp0_open_map(pp0)[RID].acks.update(acks)
    assert fb.pp0_decide(pp0, RID, now=43.5) == (0, fb.REASON_MISMATCH)


def test_pf_on_every_stage_plans_the_same_batch(monkeypatch):
    """RANK-EQUALITY, fixed side: told=0 on every rank (PP0's Admit(0, fallback)), every stage
    admits the same three rids with the same rows."""
    monkeypatch.setenv(fb.ENV_FALLBACK, "1")
    plans = {}
    for rank, tree in ((0, _pp0_tree()), (1, _follower_tree()), (2, _follower_tree())):
        stage = _stage(rank, tree)
        if rank == 0:
            fb.release_own_read(stage, RID)      # PP0 applies its own Admit(0) (pp0_retract_applied's half)
        else:
            twin.note_follower_twin(stage, RID)
            fb.follower_release(stage, RID)      # Admit(0, fallback) absorbed: the read is cut
        stage._weg2_store_told[RID] = 0          # the absorbed fallback Admit
        plans[rank] = _plan(stage, _queue(), {RID})
    assert plans[0] == plans[1] == plans[2]
    assert [r for r, _t in plans[0]] == ["weg2-0-9", "weg2-1-11", RID]


def test_nextflash_row_arms_pf():
    assert FM.PROFILES["nextflash"].told_group_fallback is True
    assert FM.PROFILE_SWITCH_DEFAULTS["nextflash"][fb.ENV_FALLBACK] is True
    assert FM.PROFILES["qwen27b"].told_group_fallback is False
    env = {FM.FORM_ENV: FM.Weg2Form(arch="moe", experts="offload", draft="mtp", p_draft="none",
                                    kv="qsa_forma", flip="family", vision="off", profile="nextflash",
                                    model="m").env_value()}
    assert FM.prefix_switch_state(fb.ENV_FALLBACK, env) == (True, "profile nextflash")


def _nf_form_env():
    return FM.Weg2Form(arch="moe", experts="offload", draft="mtp", p_draft="none", kv="qsa_forma",
                       flip="family", vision="off", profile="nextflash", model="m").env_value()


def _three_stage_plans(monkeypatch):
    """The 17:49 pass on all three stages with the switches the NEXTFLASH ROW gives (nothing
    set by hand): PF armed -> the followers' acks at read end, PP0's verdict from them; PF off
    -> PP0's window passes and it admits at told."""
    monkeypatch.delenv(fb.ENV_FALLBACK, raising=False)
    monkeypatch.setenv(FM.FORM_ENV, _nf_form_env())
    stages = {0: _stage(0, _pp0_tree()), 1: _stage(1, _follower_tree()), 2: _stage(2, _follower_tree())}
    queues = {r: _queue() for r in stages}
    pf = fb.env_on()
    told_final = TOLD
    if pf:
        acks = {}
        for r in (1, 2):
            twin.note_follower_twin(stages[r], RID)
            queues[r][2]._weg2_early_told = TOLD
            m.follower_early_settle_now(stages[r], queues[r][2], RID, TOLD)   # settled at the ack (N5p)
            acks[r] = fb.own_prefix(stages[r], queues[r][2], RID, TOLD)
        fb.pp0_open(stages[0], RID, TOLD, published_at=37.0, own_read_s=6.42, window_s=8.02)
        fb._pp0_open_map(stages[0])[RID].acks.update(acks)
        told_final, _why = fb.pp0_decide(stages[0], RID, now=45.0)
    plans = {}
    for r, st in stages.items():
        if told_final == 0:
            if r == 0:
                fb.release_own_read(st, RID)
            else:
                twin.note_follower_twin(st, RID)
                fb.follower_release(st, RID)
            st._weg2_store_told[RID] = 0
            plans[r] = _plan(st, queues[r], {RID})
        elif r == 0:
            st._weg2_store_told[RID] = TOLD
            plans[r] = _plan(st, queues[r], set())        # PP0's tree reaches its told (TF)
        else:
            _absorb_admit(st, queues[r][2])
            try:
                plans[r] = _plan(st, queues[r], {RID})
            except m.Weg2StoreToldMismatch as exc:         # a named stop is no batch
                plans[r] = exc
    return pf, told_final, plans


def test_rank_equality_three_stages_on_the_nextflash_row(monkeypatch):
    """RED at 853534a9ff: PF off on the row -> PP1/PP2 plan 5723 rows, PP0 16283 (the metal
    #1233). Fixed: PF on, PP0 answers told=0, every stage runs the same batch."""
    pf, told_final, plans = _three_stage_plans(monkeypatch)
    widths = {r: (_rows(p) if isinstance(p, list) else repr(p)) for r, p in plans.items()}
    assert pf is True, widths
    assert told_final == 0, widths
    assert plans[0] == plans[1] == plans[2], widths
    assert [r for r, _t in plans[0]] == ["weg2-0-9", "weg2-1-11", RID]


def test_rank_equality_mutant_row_without_pf_splits_or_stops(monkeypatch):
    """The same pass with the row's PF taken away (the metal boot): never one batch for all --
    with W27-HELD-STOP the follower stops by name instead of planning 5723 rows."""
    real = FM.prefix_switch_armed
    monkeypatch.setattr(FM, "prefix_switch_armed",
                        lambda name, environ=None: False if name == fb.ENV_FALLBACK else real(name, environ))
    pf, told_final, plans = _three_stage_plans(monkeypatch)
    assert pf is False and told_final == TOLD
    assert _rows(plans[0]) == PP0_ROWS
    for r in (1, 2):
        assert isinstance(plans[r], m.Weg2StoreToldMismatch), plans[r]
        assert "W27-HELD-STOP" in str(plans[r])
    monkeypatch.setenv(HELD_STOP_ENV, "0")
    pf, told_final, plans = _three_stage_plans(monkeypatch)
    assert {_rows(plans[r]) for r in (1, 2)} == {PP1_ROWS_METAL}, "the metal split, reproduced"


# -- (2) W27-HELD-STOP, with PF off (the metal boot) -------------------------------------------


def test_mutant_old_hold_splits_the_group(monkeypatch):
    """RANK-EQUALITY MUTANT: PF off + the hold as before (continue). PP1 plans 5723 rows
    against PP0's 16283 -- exactly the metal #1233 WIDTH GUARD."""
    monkeypatch.setenv(fb.ENV_FALLBACK, "0")
    monkeypatch.setenv(HELD_STOP_ENV, "0")
    pp0 = _stage(0, _pp0_tree())
    pp0._weg2_store_told[RID] = TOLD
    pp0_batch = _plan(pp0, _queue(), set())       # PP0's own tree reaches told (TF checked)
    pp1 = _stage(1, _follower_tree())
    q = _queue()
    _absorb_admit(pp1, q[2])
    pp1_batch = _plan(pp1, q, {RID})
    assert _rows(pp0_batch) == PP0_ROWS == 16283
    assert _rows(pp1_batch) == PP1_ROWS_METAL == 5723
    assert pp1_batch != pp0_batch, "the split the width guard named"
    assert pp1.calls == [(RID, TOLD)], "the bounded re-read ran first"
    assert m.reach_unresolved(pp1, RID)[:2] == (TOLD, HEAD)


def test_without_pf_the_unresolvable_hold_stops_by_name_before_planning(monkeypatch, caplog):
    monkeypatch.setenv(fb.ENV_FALLBACK, "0")
    pp1 = _stage(1, _follower_tree())
    q = _queue()
    _absorb_admit(pp1, q[2])
    with caplog.at_level("WARNING", logger=m.logger.name):
        with pytest.raises(m.Weg2FollowerToldUnreachable) as ei:
            _plan(pp1, q, {RID})
    text = str(ei.value)
    assert "W27-HELD-STOP" in text and "rid=weg2-0-10" in text
    assert "told=20096" in text and "live=3008" in text and "route=none" in text
    assert isinstance(ei.value, m.Weg2StoreToldMismatch), "the group-stop exception family"
    assert any("FOLLOWER HELD" in x for x in caplog.messages)


def test_a_reread_still_in_flight_holds_with_the_barrier(monkeypatch):
    """Not yet unresolvable: the re-read is still landing -> no stop, but nothing behind the
    held rid is admitted (the barrier); its next visit asks again."""
    monkeypatch.setenv(fb.ENV_FALLBACK, "0")
    tree = _follower_tree()
    tree.prefetch_is_ongoing = lambda rid: True
    pp1 = _stage(1, tree)
    q = [_req("weg2-0-9"), _req(RID), _req("weg2-1-11")]   # a rid BEHIND the held one
    _absorb_admit(pp1, q[1])
    batch = _plan(pp1, q, {RID})
    assert [r for r, _t in batch] == ["weg2-0-9"], "weg2-1-11 overtook the held rid"
    tree.prefetch_is_ongoing = lambda rid: False            # the read ended, still short
    with pytest.raises(m.Weg2FollowerToldUnreachable):
        _plan(pp1, q, {RID})


def test_with_a_pf_channel_the_hold_is_a_barrier_not_a_stop(monkeypatch):
    """route=pf-ack: PP0 can still answer told=0 for every rank -> no stop; the barrier keeps
    the follower's loop where PP0's ends for every reason other than seating the rid."""
    monkeypatch.setenv(fb.ENV_FALLBACK, "1")
    pp1 = _stage(1, _follower_tree())
    fb.follower_expect(pp1, RID, TOLD)                     # the paced read-ahead carried ack
    q = [_req("weg2-0-9"), _req(RID), _req("weg2-1-11")]
    # the settle happened at the ack; this rank was satisfied by another route (local prefix)
    pp1._weg2_store_told[RID] = TOLD
    twin.note_follower_twin(pp1, RID)
    pp1._weg2_store_told_satisfied = {RID: TOLD}
    batch = _plan(pp1, q, {RID})
    assert [r for r, _t in batch] == ["weg2-0-9"]
    assert m.reach_unresolved(pp1, RID)[2] == fb.ROUTE_PF
    assert any(e[0] == RID for e in fb._fstate(pp1).outbox), "PP0 is re-asked"


def test_the_switch_off_restores_the_old_hold(monkeypatch):
    monkeypatch.setenv(fb.ENV_FALLBACK, "0")
    monkeypatch.setenv(HELD_STOP_ENV, "0")
    pp1 = _stage(1, _follower_tree())
    q = [_req("weg2-0-9"), _req(RID), _req("weg2-1-11")]
    _absorb_admit(pp1, q[1])
    assert [r for r, _t in _plan(pp1, q, {RID})] == ["weg2-0-9", "weg2-1-11"]
    assert m.held_barrier(pp1, q[1]) is False


def test_pp0_is_never_a_barrier():
    pp0 = _stage(0, _pp0_tree())
    pp0._w27u_unreached = {RID: (TOLD, HEAD, fb.ROUTE_NONE)}
    assert m.held_barrier(pp0, _req()) is False


def test_kept_visit_lets_the_named_stop_through(monkeypatch):
    """p_intake._reach_kept's catch-all keeps a failing PROBE from breaking the admission; the
    named stop must pass it (else the kept credit seats a short tree)."""
    monkeypatch.setenv(fb.ENV_FALLBACK, "0")
    pp1 = _stage(1, _follower_tree())
    entry = p_intake._Kept(req=_req(), told=TOLD, credit=TOLD, reach=True)
    with pytest.raises(m.Weg2FollowerToldUnreachable):
        p_intake._reach_kept(pp1, entry.req, entry)


# -- the scheduler seam ---------------------------------------------------------------------------


def test_scheduler_breaks_on_the_held_barrier():
    import inspect

    from sglang.srt.managers import scheduler as S

    src = inspect.getsource(S)
    i = src.index("_told_loaded = _p_intake.told_admission(")
    window = src[i:i + 700]
    assert re.search(r"if _told_loaded is None:\s+if weg2_store_told\.held_barrier\(self, req\):", window), window
    j = window.index("held_barrier")
    assert window.index("break", j) < window.index("continue", j)
