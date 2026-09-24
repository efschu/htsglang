"""fnFL2 H45: D's tail vote waits for P's COMPLETE manifest (E2 was a race).

Hermetic (no CUDA). Metal fnFL2x150/x151: P's three PP ranks publish their
tail parts from background threads; D's first prefetch check came before the
last of them was written, the staging read the part list ONCE (parts=1-2 of
3), TP0 -- the rank holding every attention/GDN layer -- refused E1 with
'fa_layer_missing:<gid of the missing part>', the group MIN fell to 0 and D
ran a real extend (x151 weg2-0-4: flip 3.57 s; x150 small flips 3.9 s). The
one flip whose manifest was complete at the first check adopted the END
state (x151 weg2-2-6: 1.95 s). TP0 printed first_token=-1 only because its
END header read sat behind the refused E1 staging; TP1/TP2 read the same END
headers and printed P's token.

What these cases pin (derived properties / bookkeeping a later diff can
silently break):
* the manifest is counted: P's header names its part count; 'partial' is
  never staged, the part list is re-read at every check;
* a rank whose staging is unfinished HOLDS the prefetch termination (the
  vote's collective) through a slot of the existing MAX -- the group waits
  uniformly, bounded (SGLANG_WEG2_TAIL_WAIT_MS; 300 ms with no part at all);
* the replayed x151 race ends in adopt=done with P's END state, TP0 naming
  first_token from the parts' END headers (token_src=publish) and the hold
  (waited_ms); a lapsed bound is a NAMED uniform refusal
  (parts_partial:2/3), never a split and never a second extend;
* MUTANTS: staging a partial manifest or a hold that never holds puts the
  x151 race back (the vote falls before P's token and all parts are there);
  a termination that ignores the hold slot fails the MAX-slot case.
"""

import contextlib
import logging
import threading
import time
from types import SimpleNamespace

import msgspec
import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.weg2 import tail_adopt as ta
from sglang.srt.weg2 import tail_handoff as th

RID = "weg2-0-4"
PAGE, RATIO = 64, 4
N = 241  # c = 240, page prefix 192; N % 4 == 1: one open-group member (240)
PREFIX, C = 192, 240
FIRST = 21483  # x151 weg2-0-4: P's sampled token
FA_GIDS, GDN_GIDS = [3, 7, 11], [0, 1, 2, 4, 5, 6, 8, 9, 10]
#: P's partition (PP0 -> PP1 -> PP2), global layer ids; GDN precedes FA
PARTS = {"pp0-1": ([3], [0, 1, 2]), "pp1-2": ([7], [4, 5, 6]), "pp2-3": ([11], [8, 9, 10])}
SLOTS, SLOT = 5, 2
REQ_SLOTS, D_RPI = 6, 1
KV_ROWS = 6 * PAGE
D_PAGE = 3
FP8 = torch.float8_e4m3fn


# ------------------------------------------------------------------ P side
def _payloads(seed=45):
    g = torch.Generator().manual_seed(seed)
    rows1, rows = C - PREFIX, N - PREFIX
    fa1 = {gid: (torch.randn(rows1, 2, 8, generator=g).to(FP8), torch.randn(rows1, 2, 8, generator=g).to(FP8),
                 torch.randn(rows1 // RATIO, 1, 4, generator=g).to(torch.bfloat16)) for gid in FA_GIDS}
    gdn1 = {gid: (torch.randn(1, 2, 4, 4, generator=g), torch.randn(1, 6, 3, generator=g).to(torch.bfloat16))
            for gid in GDN_GIDS}
    fa = {gid: (torch.randn(rows, 2, 8, generator=g).to(FP8), torch.randn(rows, 2, 8, generator=g).to(FP8),
                torch.randn(12, 1, 4, generator=g).to(torch.bfloat16)) for gid in FA_GIDS}
    gdn = {gid: (torch.randn(1, 2, 4, 4, generator=g), torch.randn(1, 6, 3, generator=g).to(torch.bfloat16))
           for gid in GDN_GIDS}
    ring = {gid: (torch.randn(1, 1, 4, generator=g).to(torch.bfloat16),) for gid in FA_GIDS}
    rope = torch.full((1, 3), 240, dtype=torch.int64)
    return fa1, gdn1, fa, gdn, ring, rope


PAY = _payloads()


def _publish(parts=tuple(PARTS), n_parts=3, first=FIRST):
    """P's publish threads, one call per part that has landed so far."""
    ids = list(range(N))
    spec = th.spec_for(RID, ids, None, PAGE, RATIO)
    fa1, gdn1, fa, gdn, ring, rope = PAY
    for part in parts:
        fl, gl = PARTS[part]
        end = th.EndPayload(first_token=first, key=th.tail_key(ids, N, None), rows=N - PREFIX, groups=12,
                            ring_rows=1, fa={g: fa[g] for g in fl}, gdn={g: gdn[g] for g in gl},
                            ring={g: ring[g] for g in fl}, rope=rope)
        th.write_part(spec, part, {g: fa1[g] for g in fl}, {g: gdn1[g] for g in gl}, end=end, n_parts=n_parts)
    return spec


# ------------------------------------------------------------------ D side
def _d_pools(worker: bool):
    from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
    from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool

    heads = 0 if worker else 2
    kv = object.__new__(QSATokenToKVPool)
    kv.qsa_compress_ratio = RATIO
    kv.full_attention_layer_id_mapping = {11: 0, 3: 1, 7: 2}
    kv.full_kv_pool = SimpleNamespace(
        k_buffer=[torch.zeros(KV_ROWS, heads, 8, dtype=FP8) for _ in FA_GIDS],
        v_buffer=[torch.zeros(KV_ROWS, heads, 8, dtype=FP8) for _ in FA_GIDS],
    )
    kv.qsa_compressed_k_buffer_pool = [torch.zeros(KV_ROWS // RATIO, 1, 4, dtype=torch.bfloat16) for _ in FA_GIDS]
    kv.qsa_key_state_buffer_pool = [torch.zeros(REQ_SLOTS * RATIO, 1, 4, dtype=torch.bfloat16) for _ in FA_GIDS]
    kv.qsa_rope_position_buffer = torch.zeros(REQ_SLOTS * RATIO, 3, dtype=torch.int64)
    rp = object.__new__(HybridReqToTokenPool)
    rp.mamba_map = {gid: len(GDN_GIDS) - 1 - i for i, gid in enumerate(GDN_GIDS)}
    rp.mamba_pool = SimpleNamespace(mamba_cache=SimpleNamespace(
        temporal=torch.zeros(len(GDN_GIDS), SLOTS, 0 if worker else 2, 4, 4),
        conv=[torch.zeros(len(GDN_GIDS), SLOTS, 0 if worker else 6, 3, dtype=torch.bfloat16)],
    ))
    rp.req_to_token = torch.zeros(REQ_SLOTS, 512, dtype=torch.int32)
    return kv, rp


class Rank:
    def __init__(self, worker: bool):
        self.kv, self.rp = _d_pools(worker)
        self.tree = SimpleNamespace(token_to_kv_pool_allocator=SimpleNamespace(get_kvcache=lambda: self.kv),
                                    req_to_token_pool=self.rp)
        self.jobs, self.agreed, self.pending, self.skips = {}, {}, [], {}
        self.req = None

    @contextlib.contextmanager
    def active(self):
        saved = (ta._JOBS, ta._AGREED, ta.PENDING_INSTALLS, ta.SKIP_PLANS)
        ta._JOBS, ta._AGREED, ta.PENDING_INSTALLS, ta.SKIP_PLANS = self.jobs, self.agreed, self.pending, self.skips
        try:
            yield
        finally:
            ta._JOBS, ta._AGREED, ta.PENDING_INSTALLS, ta.SKIP_PLANS = saved


def _req():
    ids = list(range(N))
    sp = SimpleNamespace(frequency_penalty=0.0, presence_penalty=0.0, repetition_penalty=1.0, min_new_tokens=0)
    return SimpleNamespace(rid=RID, origin_input_ids=ids, full_untruncated_fill_ids=list(ids), extra_key=None,
                           prefix_indices=torch.arange(PAGE, PAGE + PREFIX, dtype=torch.int64),
                           mamba_pool_idx=torch.tensor(SLOT), req_pool_idx=D_RPI, return_logprob=False,
                           return_hidden_states=False, grammar=None, sampling_params=sp)


def _join(name):
    for t in threading.enumerate():
        if t.name == name:
            t.join(10)


@pytest.fixture
def d_group(tmp_path, monkeypatch):
    import sglang.srt.managers.schedule_policy as sp
    import sglang.srt.mem_cache.common as common

    monkeypatch.setenv("SGLANG_HICACHE_ARENA_DIR", str(tmp_path))
    monkeypatch.setattr(sp, "_WEG2_END_ANCHOR", False)  # group D
    monkeypatch.setattr(ta, "_SKIP_SERVER", [True])
    monkeypatch.setattr(common, "alloc_token_slots",
                        lambda tree_cache, n: torch.arange(D_PAGE * PAGE, D_PAGE * PAGE + n, dtype=torch.int64))
    with envs.SGLANG_WEG2_TAIL_HANDOFF.override(True), envs.SGLANG_WEG2_TAIL_ADOPT.override(True), \
            envs.SGLANG_WEG2_TAIL_VERIFY.override(True), envs.SGLANG_WEG2_TAIL_SKIP_EXTEND.override(True), \
            envs.SGLANG_WEG2_TAIL_WAIT_MS.override(1500):
        yield [Rank(worker=False), Rank(worker=True), Rank(worker=True)]


# ------------------------------------------------------------------ the group's progress check
def _terminates(ranks, read_done=True, honour_hold=True):
    """One ``check_prefetch_progress`` round of the TP group up to the
    termination verdict: every rank stages (re-reads the manifest) and
    proposes its hold, the MAX is the group's answer (as in
    ``UnifiedRadixCache.can_terminate_prefetch``)."""
    holds = []
    for r in ranks:
        with r.active():
            ta.stage(RID, r.tree)
            holds.append(ta.vote_hold(RID))
    held = max(holds) > 0 and honour_hold
    if read_done and held:
        for r in ranks:
            with r.active():
                ta.note_held(RID)
    return read_done and not held


def _vote_and_admit(ranks):
    """The packed MIN (level vote) + agree + admission on every rank."""
    votes = []
    for r in ranks:
        with r.active():
            votes.append(ta.local_vote(RID))
    group = min(votes)
    plans = []
    for r in ranks:
        r.req = _req()
        with r.active():
            ta.agree(RID, group)
            plan = ta.plan_adopt(r.req, len(r.req.prefix_indices), batch_empty=True)
            if plan is not None:
                ta.commit_adopt(r.req, plan, tree_cache=r.tree, page_size=PAGE)
            plans.append(plan)
    return votes, plans


def _x151_race(ranks, honour_hold=True):
    """x151 weg2-0-4 replayed: PP1 and PP2 have written, PP0 (the biggest
    part) is still being written when D's read finishes; it lands two
    progress rounds later. Returns (votes, plans, rounds_held)."""
    _publish(parts=("pp1-2", "pp2-3"))
    rounds = 0
    while not _terminates(ranks, honour_hold=honour_hold):
        rounds += 1
        if rounds == 2:
            _publish(parts=("pp0-1",))  # PP0's publish thread finished
        _join("weg2-tail-stage")
        assert rounds < 50, "the hold never released"
    _join("weg2-tail-stage")
    votes, plans = _vote_and_admit(ranks)
    return votes, plans, rounds


# ------------------------------------------------------------------ manifest
def test_manifest_state_counts_the_parts(d_group):
    assert th.manifest_state([]) == ("none", 0, 0)
    _publish(parts=("pp1-2",))
    assert th.manifest_state(th.headers_for(RID)) == ("partial", 1, 3)
    _publish(parts=("pp2-3",))
    assert th.manifest_state(th.headers_for(RID)) == ("partial", 2, 3)
    _publish(parts=("pp0-1",))
    hs = th.headers_for(RID)
    assert th.manifest_state(hs) == ("complete", 3, 3) and {h.n_parts for h in hs} == {3}
    # a stale part of the same PP rank from another process
    spec = hs[0].spec
    th.write_part(spec, "pp0-999", {}, {}, n_parts=3)
    assert th.manifest_state(th.headers_for(RID)) == ("excess", 4, 3)


def test_pre_h45_headers_are_legacy_and_staged_as_they_are(d_group):
    _publish(n_parts=0)
    hs = th.headers_for(RID)
    assert th.manifest_state(hs) == ("legacy", 3, 0)
    # a header JSON written before the field existed decodes with n_parts=0
    raw = msgspec.json.decode(msgspec.json.encode(hs[0]))
    raw.pop("n_parts")
    assert msgspec.json.decode(msgspec.json.encode(raw), type=th.TailHeader).n_parts == 0


# ------------------------------------------------------------------ the race, fixed
def test_x151_race_waits_for_all_parts_and_adopts_the_end_state(d_group, caplog):
    with caplog.at_level(logging.INFO, logger="sglang.srt.weg2.tail_adopt"):
        votes, plans, rounds = _x151_race(d_group)
    assert rounds >= 2  # the finished read was held until PP0's part landed
    assert votes == [2, 2, 2] and all(p is not None and p.skip for p in plans)
    assert all(len(r.req.prefix_indices) == C for r in d_group)  # E1's shape, no forward (E2)
    assert all(RID in r.skips for r in d_group) and not d_group[0].pending  # no E1 install queued
    text = caplog.text
    ready = [ln for ln in text.splitlines() if "WEG2-TAIL-READY" in ln]
    assert len(ready) == 3
    for ln in ready:
        assert f"state_at={N} extend=0 parts=3/3" in ln
        assert f"first_token={FIRST} token_src=publish waited_ms=" in ln
        assert "adopt=done" in ln
    assert sum("verdict=ready adopt=done end=ready" in ln for ln in ready) == 1  # TP0
    assert sum("verdict=not_mine adopt=done end=not_mine" in ln for ln in ready) == 2
    waited = [float(ln.split("waited_ms=")[1].split()[0]) for ln in ready]
    assert all(w >= 0 for w in waited)
    assert "fa_layer_missing" not in text


def test_no_hold_votes_on_the_partial_manifest_and_names_it(d_group, caplog):
    """SGLANG_WEG2_TAIL_WAIT_MS=0: the read terminates at once, TP0 has only
    2 of 3 parts -> a named, uniform refusal: every rank on today's extend,
    TP0 still names P's token from the same END headers as its peers."""
    with envs.SGLANG_WEG2_TAIL_WAIT_MS.override(0), \
            caplog.at_level(logging.INFO, logger="sglang.srt.weg2.tail_adopt"):
        _publish(parts=("pp1-2", "pp2-3"))
        assert _terminates(d_group)
        votes, plans = _vote_and_admit(d_group)
    assert votes == [0, 0, 0] and plans == [None] * 3
    assert all(len(r.req.prefix_indices) == PREFIX for r in d_group)
    assert all(not r.pending and not r.skips for r in d_group)
    text = caplog.text
    assert text.count("parts=2/3") == 3
    assert text.count("verdict=parts_partial:2/3 adopt=skipped:group_vote") == 3
    assert text.count(f"first_token={FIRST} token_src=publish") == 3  # TP0 included


def test_the_hold_is_bounded(d_group, caplog):
    _publish(parts=("pp1-2", "pp2-3"))
    assert not _terminates(d_group)  # held: 2 of 3
    for r in d_group:
        r.jobs[RID].t_first -= 1.0
    assert not _terminates(d_group)  # 1.0 s < 1.5 s
    for r in d_group:
        r.jobs[RID].t_first -= 0.6
    assert _terminates(d_group)  # past SGLANG_WEG2_TAIL_WAIT_MS on every rank
    with caplog.at_level(logging.INFO, logger="sglang.srt.weg2.tail_adopt"):
        votes, plans = _vote_and_admit(d_group)
    assert votes == [0, 0, 0] and plans == [None] * 3
    assert caplog.text.count("verdict=parts_partial:2/3") == 3
    waited = [float(ln.split("waited_ms=")[1].split()[0]) for ln in caplog.text.splitlines() if "waited_ms=" in ln]
    assert len(waited) == 3 and all(w >= 0 for w in waited)


def test_no_part_at_all_holds_only_briefly(d_group, caplog):
    """P may publish nothing (no partial page below the cut): the hold is
    NO_PARTS_WAIT_S, then today's extend, one READY line per rank."""
    assert not _terminates(d_group)
    for r in d_group:
        r.jobs[RID].t_first -= ta.NO_PARTS_WAIT_S + 0.01
    assert _terminates(d_group)
    with caplog.at_level(logging.INFO, logger="sglang.srt.weg2.tail_adopt"):
        votes, plans = _vote_and_admit(d_group)
    assert votes == [0, 0, 0] and plans == [None] * 3
    assert caplog.text.count(f"WEG2-TAIL-READY rid={RID} parts=0/? verdict=no_parts") == 3


def test_one_slow_rank_holds_the_whole_group(d_group):
    """The hold is a MAX: TP1's own staging done, TP0's not -> nobody
    terminates; the ranks never disagree on WHEN the vote falls."""
    _publish()
    for r in d_group[1:]:
        with r.active():
            ta.stage(RID, r.tree)
    _join("weg2-tail-stage")
    for r in d_group[1:]:
        with r.active():
            assert ta.vote_hold(RID) == 0
    with d_group[0].active():
        assert ta.vote_hold(RID) == 0  # no job yet on TP0 ...
    tp0_job = ta._Job(box=[], t_first=time.perf_counter(), state="partial", have=2, want=3)
    d_group[0].jobs[RID] = tp0_job
    with d_group[0].active():
        assert ta.vote_hold(RID) == 1  # ... TP0 still sees 2 of 3
    holds = []
    for r in d_group:
        with r.active():
            holds.append(ta.vote_hold(RID))
    assert holds == [1, 0, 0] and max(holds) == 1


def test_excess_parts_refuse_at_once(d_group, caplog):
    spec = _publish()
    th.write_part(spec, "pp0-999", {}, {}, n_parts=3)  # a stale PP0 part
    assert _terminates(d_group)  # no hold: the manifest can never complete
    with caplog.at_level(logging.INFO, logger="sglang.srt.weg2.tail_adopt"):
        votes, plans = _vote_and_admit(d_group)
    assert votes == [0, 0, 0] and plans == [None] * 3
    assert caplog.text.count("verdict=parts_excess:4/3") == 3


def test_tp0_names_the_token_even_when_its_e1_staging_refuses(d_group, caplog):
    """x151: TP0 'fa_layer_missing:43 ... end=absent first_token=-1' beside
    TP1/TP2 'first_token=21483'. The token is read from the END headers on
    every rank, whatever the E1 verdict."""
    _publish()
    _j, ppath = th.part_paths(RID, "pp1-2")
    bundle = torch.load(ppath)
    bundle["fa"][7] = (bundle["fa"][7][0], bundle["fa"][7][1], bundle["fa"][7][2] + 1)
    torch.save(bundle, ppath)
    with caplog.at_level(logging.INFO, logger="sglang.srt.weg2.tail_adopt"):
        while not _terminates(d_group):
            _join("weg2-tail-stage")
        votes, plans = _vote_and_admit(d_group)
    assert votes == [0, 2, 2] and plans == [None] * 3
    tp0 = [ln for ln in caplog.text.splitlines() if "verdict=digest_MISMATCH:pp1-2" in ln]
    assert len(tp0) == 1 and f"end=absent first_token={FIRST} token_src=publish" in tp0[0]


def test_end_token_source_names_why_it_is_missing():
    assert ta.end_token([]) == (-1, "none:no_parts")


# ------------------------------------------------------------------ the tree's MAX slot
class _Op(SimpleNamespace):
    def is_terminated(self):
        return False


def _tree(peer_hold):
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    tree = object.__new__(UnifiedRadixCache)
    tree.prefetch_stop_policy = "wait_complete"
    tree.page_size = PAGE
    seen = {}

    def _peer_max(states, op, label=""):
        seen["len"], seen["label"] = int(states.numel()), label
        states[2] = max(int(states[2].item()), peer_hold)

    tree._all_reduce_attn_groups = _peer_max
    return tree, seen


def test_the_termination_max_carries_the_hold(d_group):
    op = _Op(request_id=RID, hash_value=["h"], completed_tokens=PAGE, pool_transfers=None)
    _publish(parts=("pp1-2",))
    with d_group[0].active():
        ta.stage(RID, d_group[0].tree)
        tree, seen = _tree(peer_hold=0)
        assert tree.can_terminate_prefetch(op, tail_hold=0)  # nobody holds: today's answer
        assert seen == {"len": 3, "label": "can_terminate_prefetch"}
        assert d_group[0].jobs[RID].held_since < 0
        tree, _ = _tree(peer_hold=1)  # a PEER's staging is unfinished
        assert not tree.can_terminate_prefetch(op, tail_hold=0)
        assert d_group[0].jobs[RID].held_since > 0  # the wait is stamped
        tree, _ = _tree(peer_hold=0)
        assert not tree.can_terminate_prefetch(op, tail_hold=1)  # this rank's own hold
        # an unfinished read is not a hold: the stamp only marks a finished one
        op.completed_tokens = 0
        d_group[0].jobs[RID].held_since = -1.0
        tree, _ = _tree(peer_hold=1)
        assert not tree.can_terminate_prefetch(op, tail_hold=1)
        assert d_group[0].jobs[RID].held_since < 0


def test_the_progress_check_proposes_the_hold(d_group, monkeypatch):
    """check_prefetch_progress passes vote_hold(rid) into the termination."""
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    tree = object.__new__(UnifiedRadixCache)
    tree.token_to_kv_pool_allocator = d_group[0].tree.token_to_kv_pool_allocator
    tree.req_to_token_pool = d_group[0].rp
    op = _Op(request_id=RID, host_indices=torch.zeros(1), hash_value=["h"], completed_tokens=PAGE)
    tree.ongoing_prefetch = {RID: (None, None, None, op, None, {})}
    seen = []

    def _can_terminate(operation, tail_hold=0):
        seen.append(tail_hold)
        return False

    tree.can_terminate_prefetch = _can_terminate
    _publish(parts=("pp1-2", "pp2-3"))
    with d_group[0].active():
        assert tree.check_prefetch_progress(RID) is False
        _publish(parts=("pp0-1",))
        assert tree.check_prefetch_progress(RID) is False
        _join("weg2-tail-stage")
        assert tree.check_prefetch_progress(RID) is False
    # partial -> (staging: 1 while the read runs) -> staged
    assert len(seen) == 3 and seen[0] == 1 and seen[2] == 0


# ------------------------------------------------------------------ mutants
def test_mutant_staging_a_partial_manifest_loses_the_race(d_group, monkeypatch, caplog):
    """Mutant: the part count is ignored (every non-empty list is staged as
    it is -- the pre-H45 snapshot) -> TP0 stages 2 of 3 parts and the group
    votes before PP0's part is there."""
    monkeypatch.setattr(th, "manifest_state", lambda hs: ("legacy", len(hs), 0) if hs else ("none", 0, 0))
    with caplog.at_level(logging.INFO, logger="sglang.srt.weg2.tail_adopt"):
        votes, plans, _rounds = _x151_race(d_group)
    assert votes[0] == 0 and plans == [None] * 3
    assert "verdict=fa_layer_missing:3" in caplog.text


def test_mutant_a_hold_that_never_holds_loses_the_race(d_group, monkeypatch, caplog):
    monkeypatch.setattr(ta, "vote_hold", lambda rid: 0)
    with caplog.at_level(logging.INFO, logger="sglang.srt.weg2.tail_adopt"):
        votes, plans, rounds = _x151_race(d_group)
    assert rounds == 0 and votes == [0, 0, 0] and plans == [None] * 3
    assert "verdict=parts_partial:2/3" in caplog.text

