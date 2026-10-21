"""D-FORK-ANCHOR review (21.10., independent reviewer; feature 33cf26403c).

Two gaps the feature's own suite (test_nf_rc30_d_fork_anchor_1021.py) does not pin, both on the REAL
UnifiedRadixCache of test_pdflip_turn_anchor_0929, hermetic on the CPU:

(R1) THE FOLLOW WALK IS LOAD-BEARING FOR KV, NOT ONLY FOR THE PROTECTED LENGTH. ``_pdflip_turn_insert_one`` inserts
     ``key[:t]`` with ``prev_prefix_len = cache_protected_len``; where the path to t already exists in the tree (a
     sibling inserted it), ``insert`` FREES the request's duplicate KV slots of [protected, t) (``dup_start``) and only
     the re-match rewrites the request's rows to the tree's. On a rank whose node at t carries no state (a Form A
     worker, or TP0 that planned split-only), a mamba-validated re-match stops below t: protected stays short, the rows
     [short, t) keep pointing at the freed slots, and the step's own insert frees them a SECOND time. Mutant "no
     follow walk in the D re-match" survives the feature suite (13 passed); it is caught here.

(R2) THE FORK SLOT IS TP0-ONLY, THE DONATION SLOT IS NOT (strict xfail = a named open finding). On D the step's own
     chunk insert draws a fresh mamba slot (``MambaComponent.prepare_for_caching_req`` -> ``_alloc_mamba_slot``;
     ``None`` -> ``return 0`` -> "cache nothing this step"). TP0 now holds one more slot than the workers (the fork
     anchor, pinned by the request's lock that the fork insert moved onto it); with the pool otherwise full and locked
     TP0 alone skips the step's node while every worker inserts it -> the trees differ (R12 node keys) and
     ``cache_protected_len`` differs. With the feature off both ranks insert alike (the control below).
"""
from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
from array import array
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

_HERE = pathlib.Path(__file__).resolve().parent


def _load(name, path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


TA0 = _load("_turn_anchor_0929_for_d_fork", _HERE / "test_pdflip_turn_anchor_0929.py")
DF = _load("_d_fork_anchor_1021_for_review", _HERE / "test_nf_rc30_d_fork_anchor_1021.py")
IM, PROMPT, N, I_LAST = TA0.IM, TA0.PROMPT, TA0.N, TA0.I_LAST

group_d = DF.group_d          # the feature suite's fixture (group D, token, bigram-exact, END anchor)
_structure = DF._structure
_as_worker = DF._as_worker


def _free_list(fx):
    alloc = fx.allocator
    for name in ("free_pages", "free_slots"):
        v = getattr(alloc, name, None)
        if v is not None:
            return [int(x) for x in v.view(-1).tolist()]
    raise AssertionError("allocator exposes no free list")


def _new_req(fx, ids, rid):
    req = TA0._req(fx, ids)
    req.rid = rid
    return req


def _chunk_step(fx, req, end, *, marks=("gdn",)):
    """One UNFINISHED D step [protected, end) through the scheduler's plan (note_step on the real pool), the forward's
    marks, and ``cache_unfinished_req`` -- the chunk path whose own insert draws a donation slot."""
    from flliper.srt.pdflip import turn_anchor as ta

    TA0._write_kv(fx, req, end)
    batch = SimpleNamespace(reqs=[req], req_to_token_pool=fx.pool)
    prefix = len(req.prefix_indices)
    desc = ta.note_step(batch=batch, desc=None, req=req, row=0, prefix=prefix, end=end,
                        track_mask=True, main_track=end, chunk=1, page=1, tok=IM)
    plans = [getattr(req, a) for a in (ta.PENDING_ATTR, ta.CLOSE_PENDING_ATTR) if getattr(req, a, None)]
    for p in plans:
        if p[1] is not None:
            TA0._state(fx, p[1], TA0.S_TURN)
    if desc is not None:
        desc.done.update(marks)
    TA0._state(fx, req.mamba_pool_idx, TA0.S_EARLY)
    fx.cache.cache_unfinished_req(req, chunked=True)
    return plans


# -- (R1) the follow walk keeps the request's rows on the tree's KV ---------------------------------------------------

@pytest.mark.parametrize("bigram", [False, True], ids=["unigram", "bigram"])
@pytest.mark.parametrize("rank", ["worker", "host_without_free_slot"])
def test_a_split_only_rank_rewrites_its_rows_to_the_tree_where_a_sibling_owns_the_path(group_d, rank, bigram):
    _as_worker(group_d, rank == "worker")
    fx = TA0._fixture(bigram)
    held = []
    if rank == "host_without_free_slot":
        while fx.pool.mamba_allocator.available_size() > 0:
            held.append(fx.pool.mamba_allocator.alloc(1))
        fx.pool.mamba_allocator.free(held.pop())     # the sibling's own active slot, nothing for its fork plans
    # the sibling: PROMPT prefilled on D -> the path [0, N) and a node at the fork I_LAST (split-only here: no state)
    DF._d_prefill(fx)
    if held:
        fx.pool.mamba_allocator.free(held.pop())     # the request's own active slot; its fork plan again finds none
    # the request: the same session's sibling sharing ids[:I_LAST + 1], then its own last message (a side request)
    ids = PROMPT[: I_LAST + 1] + [TA0.ASSISTANT, TA0.NL] + list(range(8000, 8030)) + [TA0.IM_END, TA0.NL] + TA0.GEN
    req = _new_req(fx, ids, "pdflip-2-64")
    assert req.cache_protected_len < I_LAST, "the node at the fork has no state on this rank: the match stops below"
    end = I_LAST + 16
    plans = _chunk_step(fx, req, end)
    assert any(int(p[0]) == I_LAST for p in plans), plans
    free = _free_list(fx)
    assert len(free) == len(set(free)), "a KV slot is on the free list twice (the duplicate span freed twice)"
    rows = [int(x) for x in fx.pool.req_to_token[req.req_pool_idx, :end].tolist()]
    assert not set(rows) & set(free), "the request's rows point at freed KV slots"
    assert req.cache_protected_len >= I_LAST


# -- (R3) D runs NEXTN: bigram anchor keys (is_eagle) -- the feature suite drives unigram keys only ------------------

def test_bigram_keys_host_and_worker_hold_the_same_nodes_and_the_host_resumes_at_the_fork(group_d):
    _as_worker(group_d, False)
    host = TA0._fixture(True)
    DF._d_prefill(host)
    _as_worker(group_d, True)
    worker = TA0._fixture(True)
    DF._d_prefill(worker)
    assert _structure(host) == _structure(worker), "the same nodes on every rank under bigram keys"
    assert TA0._claim(host, TA0.NEXT) == (I_LAST, pytest.approx(TA0.S_TURN))


# -- (R2) the donation slot under a full pool: TP0 alone skips the step's node -----------------------------------------

def _starved_step(monkeypatch, worker: bool, feature_on: bool = True):
    from flliper.srt.pdflip import turn_anchor as ta

    if not feature_on:
        monkeypatch.setenv("FLLIPER_PDFLIP_D_FORK_ANCHOR", "0")
    _as_worker(monkeypatch, worker)
    fx = TA0._fixture(False)
    req = _new_req(fx, PROMPT, "pdflip-2-63")
    # every other slot held by running requests (outside the tree, unevictable): ONE free slot left, the same on
    # every rank before this step (the workers' bookkeeping pools mirror TP0's)
    held = []
    while fx.pool.mamba_allocator.available_size() > 1:
        held.append(fx.pool.mamba_allocator.alloc(1))
    end = I_LAST + 16                                   # a long last message: the turn fork in an INNER chunk
    if feature_on:
        plans = _chunk_step(fx, req, end)
    else:
        # switch 0: armed() is None on D, prepare_for_extend never calls note_step -- the stock chunk insert
        TA0._write_kv(fx, req, end)
        TA0._state(fx, req.mamba_pool_idx, TA0.S_EARLY)
        fx.cache.cache_unfinished_req(req, chunked=True)
        plans = []
    monkeypatch.delenv("FLLIPER_PDFLIP_D_FORK_ANCHOR", raising=False)
    return fx, req, plans


def test_control_with_the_feature_off_every_rank_inserts_the_step_alike(group_d):
    host, hreq, _ = _starved_step(group_d, worker=False, feature_on=False)
    wk, wreq, _ = _starved_step(group_d, worker=True, feature_on=False)
    assert _structure(host) == _structure(wk)
    assert hreq.cache_protected_len == wreq.cache_protected_len == I_LAST + 16


# REVIEW FINDING F1 fixed (A1, UnifiedRadixCache._d_fork_donation_room): the strict-xfail marker is removed.
def test_a_full_pool_does_not_make_tp0_skip_a_step_node_the_workers_insert(group_d):
    host, hreq, hplans = _starved_step(group_d, worker=False)
    wk, wreq, wplans = _starved_step(group_d, worker=True)
    assert [int(p[0]) for p in hplans] == [int(p[0]) for p in wplans], "the position plans agree"
    assert _structure(host) == _structure(wk), (sorted(_structure(host)), sorted(_structure(wk)))
    assert hreq.cache_protected_len == wreq.cache_protected_len


# -- (R4) Nachprüfung A1 (9a98793acf): a fork plan IN FLIGHT starves ANOTHER request's donation on TP0 only ----------

def _inflight_pair(monkeypatch, worker: bool):
    """R1's final step planned a fork (note_step at prepare N; consumed only in process_batch_result(N), i.e. after the
    next scheduling). Meanwhile R2's chunk is cached (get_next_batch_to_run N+1 -> cache_unfinished_req, or an earlier
    entry of the same process_batch_result loop): R2's donation draws from a pool TP0 alone has shrunk by R1's slot.
    R2 has no fork plan of its own, so the A1 guard (evaluated only inside R1's fork insert) never runs for it."""
    from flliper.srt.pdflip import turn_anchor as ta

    _as_worker(monkeypatch, worker)
    fx = TA0._fixture(False)
    r1 = _new_req(fx, PROMPT, "pdflip-2-63")
    r2_ids = list(range(9000, 9060))                     # no turn token: R2 plans nothing
    r2 = _new_req(fx, r2_ids, "pdflip-2-70")
    held = []
    while fx.pool.mamba_allocator.available_size() > 1:  # one free slot, the same on every rank before prepare N
        held.append(fx.pool.mamba_allocator.alloc(1))
    TA0._write_kv(fx, r1, N)
    batch = SimpleNamespace(reqs=[r1], req_to_token_pool=fx.pool)
    desc = ta.note_step(batch=batch, desc=None, req=r1, row=0, prefix=0, end=N,
                        track_mask=True, main_track=N - 1, chunk=1, page=1, tok=IM)
    if desc is not None:
        desc.done.add("gdn")
    # R2's chunk insert while R1's plan is still unconsumed
    TA0._write_kv(fx, r2, 32)
    TA0._state(fx, r2.mamba_pool_idx, TA0.S_EARLY)
    fx.cache.cache_unfinished_req(r2, chunked=True)
    return fx, r1, r2


# REVIEW NACHPRUEFUNG G1 fixed (turn_anchor.revoke_for_donation at the step donation): the strict-xfail marker is removed.
def test_an_inflight_fork_plan_does_not_make_tp0_skip_another_requests_step_node(group_d):
    host, _h1, h2 = _inflight_pair(group_d, worker=False)
    wk, _w1, w2 = _inflight_pair(group_d, worker=True)
    assert h2.cache_protected_len == w2.cache_protected_len, (h2.cache_protected_len, w2.cache_protected_len)
    assert _structure(host) == _structure(wk)


# -- (R5) Nachprüfung G1 (b197da0c60): the in-flight registry must not outlive a pool reset -------------------------

@pytest.mark.xfail(strict=True, reason=(
    "REVIEW NACHPRUEFUNG N1 (b197da0c60): turn_anchor._INFLIGHT validates an entry only by 'the request still carries "
    "that plan tuple'. A pool reset (flush/sleep leg Scheduler reset -> req_to_token_pool.clear(), vram_dial shrink, "
    "wake restore) puts the plan's slot back on the free list while a surviving request object still carries the "
    "unconsumed plan; the next donation revoke then hands out a slot another owner already holds (#924 aliasing)."))
def test_a_revoke_never_hands_out_a_slot_a_pool_reset_already_gave_away(group_d):
    from flliper.srt.pdflip import turn_anchor as ta

    group_d.setattr(ta, "_INFLIGHT", [])
    _as_worker(group_d, False)
    fx = TA0._fixture(False)
    r1 = _new_req(fx, PROMPT, "pdflip-2-63")
    TA0._write_kv(fx, r1, N)
    batch = SimpleNamespace(reqs=[r1], req_to_token_pool=fx.pool)
    ta.note_step(batch=batch, desc=None, req=r1, row=0, prefix=0, end=N,
                 track_mask=True, main_track=N - 1, chunk=1, page=1, tok=IM)
    plan = getattr(r1, ta.PENDING_ATTR)
    assert plan[1] is not None and ta.inflight_count() >= 1
    # the reset of a flush / sleep leg / vram_dial shrink: every slot free again, r1's object (and plan) survive
    fx.pool.mamba_allocator.clear()
    owned = set()
    while True:                                          # new owners take every slot, the planned one included
        s = fx.pool.mamba_allocator.alloc(1)
        if s is None:
            break
        owned.add(int(s.view(-1)[0]))
    got = ta.revoke_for_donation()
    assert got is None or int(got.view(-1)[0]) not in owned, (
        f"revoke handed out slot {int(got.view(-1)[0])} that a new owner already holds")
