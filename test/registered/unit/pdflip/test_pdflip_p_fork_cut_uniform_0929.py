"""P-FORK-CUT rang-einig (29.09.): the cut is PP0's verdict and rides its told.

THE DEATH OF THE FIRST FORM. NF rc12z30j (bb82fbcb68), P log
...dauer09290109_bb82fbcb68_0929_011011, 01:16:17-20:
  PP0  'PDFLIP P-FORK-CUT CUT rid=pdflip-4-6 ... prefix=0 extend=16384 prompt=16919
        fork=4608 src=store cut=4608'  ->  '#969N ADMIT ... extend=4608'
  PP1  '#969N ADMIT ... extend=16384'  ->  W27 PPWidthDivergenceRefused
The first form read the fork from THIS rank's store probe (`_STORE_UNCAPPED`,
noted by the prefetch thread) -- rank-local: on the carrierless P form a
follower registers only the told span (here told=0, 'declined:pdflip_told_zero'),
so it never probed and never cut. Its premise "followers execute PP0's
forwarded extent (#791)" does not hold on this form ('#1245 ARMED ... the
no-flip PP form carries no #631 row carrier').

THE FORM. PP0 decides the fork depth when it publishes the told (#1400/#1416:
the object every follower already waits for before admitting) and the told
carries it; `admission()` hands it to the request on every rank; the adder
cuts at the told fork only. No told fork -> no cut, on every rank.

The group is driven through the SHIPPED protocol (intake -> pp0_publish ->
follower_absorb -> admission) on three stage doubles, and the cut through the
adder hook itself (`p_fork_cut.apply`) -- with the store probe held per rank,
as the per-process module state is on the metal.
"""

import inspect
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from flliper.srt.managers import pdflip_store_told as m
from flliper.srt.pdflip import p_fork_cut as pfc
from flliper.srt.pdflip import p_twin_defer as tw

PAGE = 64
CHUNK = 16384
PROMPT = 16919  # pdflip-4-6
FORK_PAGES = 72  # 4608 tokens: the metal fork


class _Tree:
    def __init__(self):
        self.progress_left = {}
        self.prefetch_loaded_tokens_by_reqid = {}
        self._prefetch_completed_tokens = {}
        self._pending = {}

    def register(self, rid, completed, flips=0):
        self.progress_left[rid] = flips
        self._pending[rid] = completed

    def check_prefetch_progress(self, rid):
        if rid not in self.progress_left:
            return True
        if self.progress_left[rid] > 0:
            self.progress_left[rid] -= 1
            return False
        del self.progress_left[rid]
        done = self._pending.pop(rid)
        self.prefetch_loaded_tokens_by_reqid[rid] = done
        self._prefetch_completed_tokens[rid] = done
        return True

    def completed_prefetch_tokens(self, rid):
        return self._prefetch_completed_tokens.get(rid)

    def pop_prefetch_loaded_tokens(self, rid):
        return self.prefetch_loaded_tokens_by_reqid.pop(rid, 0)


class _Sched:
    def __init__(self, pp_rank, pp_size=3):
        self.ps = SimpleNamespace(pp_rank=pp_rank, pp_size=pp_size, tp_size=1)
        self.enable_hicache_storage = True
        self.tree_cache = _Tree()
        self.waiting_queue = []
        self.chunked_req = None
        self.running_batch = None
        self.pp_flip_counters = None
        self.page_size = PAGE
        self.chunked_prefill_size = CHUNK
        #: THIS rank's store probe (the per-process `_STORE_UNCAPPED`).
        self.store_probe = {}

    def _prefetch_kvcache(self, req, rematch=True, limit_tokens=None):
        req._prefetch_registered_prefix_len = 0
        req.prefix_indices = []
        req.host_hit_length = 0
        self.tree_cache.register(req.rid, completed=0)
        return "issued"


class _Req:
    def __init__(self, rid, n):
        self.rid = rid
        self.origin_input_ids = list(range(n))
        self.full_untruncated_fill_ids = list(range(n))
        self.extra_key = None
        self.prefetch_deferred = None
        self.key_match_depth = None
        self.done = False

    def finished(self):
        return self.done


@contextmanager
def _as_rank(s):
    """Run on stage `s`: the module's store probe is this rank's own."""
    saved = dict(pfc._STORE_UNCAPPED)
    pfc._STORE_UNCAPPED.clear()
    pfc._STORE_UNCAPPED.update(s.store_probe)
    try:
        yield
    finally:
        s.store_probe = dict(pfc._STORE_UNCAPPED)
        pfc._STORE_UNCAPPED.clear()
        pfc._STORE_UNCAPPED.update(saved)


@pytest.fixture
def group_p(monkeypatch):
    import flliper.srt.runtime_context as rc

    monkeypatch.setattr(
        rc, "get_server_args",
        lambda: SimpleNamespace(chunked_prefill_size=CHUNK, pp_size=3, tp_size=1),
    )
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "P")
    monkeypatch.delenv("FLLIPER_PDFLIP_MAMBA_ANCHOR_INTERVAL", raising=False)
    monkeypatch.delenv("FLLIPER_PDFLIP_FORM", raising=False)
    monkeypatch.delenv(m.ENV_ARMED, raising=False)
    monkeypatch.delenv(tw.ENV, raising=False)
    monkeypatch.delenv("FLLIPER_PDFLIP_TOLD_PACED", raising=False)
    pfc._STORE_UNCAPPED.clear()
    yield
    pfc._STORE_UNCAPPED.clear()


def _group():
    ranks = [_Sched(0), _Sched(1), _Sched(2)]
    for s in ranks:
        assert m.armed(s)
    return ranks


def _drive(ranks, rid, n):
    """intake on every rank, PP0 publishes, PP1/PP2 absorb (in pipe order),
    every rank admits; returns each rank's request after admission."""
    reqs = []
    for s in ranks:
        r = _Req(rid, n)
        with _as_rank(s):
            m.intake(s, r, lambda g: None)
        s.waiting_queue.append(r)
        reqs.append(r)
    with _as_rank(ranks[0]):
        wire = m.pp0_publish(ranks[0], [])
    assert any(isinstance(w, m.PdFlipStoreTold) for w in wire)
    for s in ranks[1:]:
        with _as_rank(s):
            m.follower_absorb(s, list(wire))
    for s, r in zip(ranks, reqs):
        with _as_rank(s):
            assert m.admission(s, r, lambda k, v: None) is not None
    return reqs


def _cut(s, r):
    adder = SimpleNamespace(page_size=PAGE, token_to_kv_pool_allocator=None)
    with _as_rank(s):
        return pfc.apply(adder, r, 0, CHUNK, "add_one_req")


@pytest.mark.parametrize("paced", ["0", "1"])
def test_pdflip_4_6_every_rank_cuts_at_pp0s_fork(group_p, monkeypatch, paced):
    """The z30j death: only PP0's probe saw the uncapped store KV. Every stage
    must execute the same chunk -- 4608, PP0's verdict (single-phase and
    #1416e paced told alike; told=0 publishes single-phase in both)."""
    monkeypatch.setenv("FLLIPER_PDFLIP_TOLD_PACED", paced)
    ranks = _group()
    assert ranks[0]._pdflip_told_paced_on is (paced == "1")
    ranks[0].store_probe = {"pdflip-4-6": (FORK_PAGES, 0)}
    reqs = _drive(ranks, "pdflip-4-6", PROMPT)
    widths = [_cut(s, r) for s, r in zip(ranks, reqs)]
    assert widths[0] == widths[1] == widths[2], (
        f"W27 width divergence: PP0/PP1/PP2 extend {widths}")
    assert widths[0] == FORK_PAGES * PAGE


def test_a_follower_probe_alone_never_cuts(group_p):
    """Rank-local knowledge is never a cut: a follower that happens to hold a
    probe note while PP0 has none cuts nothing, like PP0."""
    ranks = _group()
    ranks[1].store_probe = {"pdflip-9-9": (FORK_PAGES, 0)}
    reqs = _drive(ranks, "pdflip-9-9", PROMPT)
    assert [_cut(s, r) for s, r in zip(ranks, reqs)] == [CHUNK, CHUNK, CHUNK]


def test_a_follower_tree_match_alone_never_cuts(group_p):
    ranks = _group()
    reqs = _drive(ranks, "pdflip-9-10", PROMPT)
    reqs[2].key_match_depth = FORK_PAGES * PAGE
    assert [_cut(s, r) for s, r in zip(ranks, reqs)] == [CHUNK, CHUNK, CHUNK]


def test_the_told_carries_the_fork(group_p):
    ranks = _group()
    ranks[0].store_probe = {"pdflip-4-7": (FORK_PAGES, 0)}
    r = _Req("pdflip-4-7", PROMPT)
    with _as_rank(ranks[0]):
        m.intake(ranks[0], r, lambda g: None)
        ranks[0].waiting_queue.append(r)
        wire = m.pp0_publish(ranks[0], [])
    told = [w for w in wire if isinstance(w, m.PdFlipStoreTold)][0]
    assert getattr(told, m.WIRE_FORK) == FORK_PAGES * PAGE


def test_a_told_without_fork_carries_no_wire_attribute(group_p):
    """No cut -> the told is the pre-fork object (byte-identical wire)."""
    ranks = _group()
    r = _Req("pdflip-4-8", PROMPT)
    with _as_rank(ranks[0]):
        m.intake(ranks[0], r, lambda g: None)
        ranks[0].waiting_queue.append(r)
        wire = m.pp0_publish(ranks[0], [])
    told = [w for w in wire if isinstance(w, m.PdFlipStoreTold)][0]
    assert m.WIRE_FORK not in vars(told)


def test_paid_cut_is_skipped_on_every_rank_alike(group_p):
    """40000 tokens, fork 4608: the cut would cost a forward -- no rank cuts."""
    ranks = _group()
    ranks[0].store_probe = {"pdflip-5-5": (FORK_PAGES, 0)}
    reqs = _drive(ranks, "pdflip-5-5", 40000)
    assert [_cut(s, r) for s, r in zip(ranks, reqs)] == [CHUNK, CHUNK, CHUNK]


def test_wired_publish_absorb_admission_and_both_adder_sites():
    from flliper.srt.managers import schedule_policy as sp

    assert "_pdflip_p_fork_cut.apply(" in inspect.getsource(sp.PrefillAdder.add_chunked_req)
    assert "_pdflip_p_fork_cut.apply(" in inspect.getsource(sp.PrefillAdder.add_one_req)
    src = inspect.getsource(m)
    assert "_pp0_fork(" in inspect.getsource(m.pp0_publish)
    assert "_pp0_fork(" in inspect.getsource(m._pp0_publish_paced)
    assert "fork" in inspect.getsource(m._follower_absorb_impl)
    assert "_pdflip_fork_told" in inspect.getsource(m.admission)
    assert "_pdflip_fork_told" in inspect.getsource(pfc.apply)
    assert "_STORE_UNCAPPED" not in inspect.getsource(pfc.apply), src[:0]
