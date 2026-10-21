"""D-FORK-ANCHOR (21.10.): the TURN / CLOSE-FORK / GEN-FORK tracks on group D.

THE MEASURED GAP (NF rc30 1d822bd150, boot ..._1021_063252, D TP0): 17 ``S109R ... store_miss`` lines, every one at
``MATCH-CENSUS-DEEP reached == branching`` -- a request forking at the start of a sibling's LAST message (pdflip-2-64 vs
pdflip-2-63: both on the anchor 70208, prompts 70394 / 70495, branch 70336; front SESSION-PREFIX pdflip-2-103 common=44620
-> branch page 44608). Those siblings were prefilled ON D (front ``verdict=short``, ``rule=R2``), and
``turn_anchor.armed()`` armed the fork tracks on group P only (``FLLIPER_PDFLIP_GROUP == "P"`` and no overlap schedule),
so D never wrote the state at the fork: P counted ``CLOSE-FORK-ANCHOR SHARED`` 27 / ``TURN-ANCHOR INSERT`` 24, D zero.

Pinned here (hermetic, CPU, the REAL UnifiedRadixCache / GDN backend helpers of test_pdflip_turn_anchor_0929):
  (1) the switch: group D arms WITH the overlap schedule, its own switch turns it off, P unchanged;
  (2) rank agreement: on D the plan is a POSITION plan -- a state-holding rank draws a slot (or plans split-only when
      none is free), a Form A worker always plans split-only; the insert puts key[:t] into EVERY rank's tree (state on
      the host, none on the worker), the protected length is t everywhere; MUTANT "only TP0 splits" (the worker's
      split-only plan dropped like a P plan without a slot) -> the trees differ;
  (3) ordering: a plan consumed without its marks is inserted split-only and its slot freed (no anchor, no error); the
      fork node goes in BEFORE the step's own node in the same insert call; the fork rows join the step's own track
      gathers (one kernel) at D's geometry (page 64, prefix > 0, several rows);
  (4) the next turn resumes at the fork with the fork state; the live witnesses ``D-FORK-ANCHOR SET`` / ``READ``.
"""
from __future__ import annotations

import importlib.util
import logging
import os
import pathlib
import sys
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

from flliper.srt.mem_cache.base_prefix_cache import MatchPrefixParams  # noqa: E402
from flliper.srt.mem_cache.unified_cache_components.tree_component import ComponentType  # noqa: E402

_HERE = pathlib.Path(__file__).resolve().parent


def _load(name, path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


TA0 = _load("_turn_anchor_0929_for_d_fork", _HERE / "test_pdflip_turn_anchor_0929.py")
IM, PROMPT, NEXT, N, I_LAST = TA0.IM, TA0.PROMPT, TA0.NEXT, TA0.N, TA0.I_LAST
TA_LOG = "flliper.srt.pdflip.turn_anchor"


@pytest.fixture
def group_d(monkeypatch):
    from flliper.srt.mem_cache import unified_radix_cache as urc

    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "D")
    monkeypatch.setenv(TA0.TOKEN_ENV, str(IM))
    monkeypatch.setenv("FLLIPER_PDFLIP_BIGRAM_ANCHOR_EXACT", "1")
    monkeypatch.setenv("FLLIPER_PDFLIP_GEN_FORK_ANCHOR", "0")   # the turn track alone (page 1: gen joins otherwise)
    monkeypatch.delenv("FLLIPER_PDFLIP_D_FORK_ANCHOR", raising=False)
    monkeypatch.setattr(urc, "_PDFLIP_END_ANCHOR", True)
    from flliper.srt.pdflip import turn_anchor as ta

    # review G1: the in-flight registry is per process -- every test starts with an empty one (a plan another test's
    # fixture left unconsumed must not lend this test's donation a slot of a different pool)
    monkeypatch.setattr(ta, "_INFLIGHT", [])
    return monkeypatch


def _as_worker(monkeypatch, worker: bool):
    from flliper.srt.pdflip import turn_anchor as ta

    monkeypatch.setattr(ta, "d_split_only_rank", lambda: worker)


# -- (1) the switch --------------------------------------------------------------------------------------------------

def test_group_d_arms_with_the_overlap_schedule_and_has_its_own_switch(monkeypatch):
    from flliper.srt.pdflip import turn_anchor as ta

    sa = SimpleNamespace(enable_mamba_extra_buffer=lambda: True, mamba_checkpoint_interval=None,
                         disable_overlap_schedule=False)              # D: overlap ON
    monkeypatch.setenv(TA0.TOKEN_ENV, str(IM))
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "D")
    assert ta.armed(sa) == IM and ta.d_mode()
    monkeypatch.setenv("FLLIPER_PDFLIP_D_FORK_ANCHOR", "0")
    assert ta.armed(sa) is None and not ta.d_mode()
    monkeypatch.delenv("FLLIPER_PDFLIP_D_FORK_ANCHOR")
    assert ta.armed(SimpleNamespace(**{**sa.__dict__, "mamba_checkpoint_interval": 4096})) is None
    assert ta.armed(SimpleNamespace(**{**sa.__dict__, "enable_mamba_extra_buffer": lambda: False})) is None
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "P")
    assert ta.armed(sa) is None and not ta.d_mode(), "group P keeps its overlap condition"
    monkeypatch.delenv(TA0.TOKEN_ENV)
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "D")
    assert ta.armed(sa) is None and not ta.d_mode(), "no token: off"


def test_the_launcher_hands_the_token_to_group_d_too():
    from flliper.srt.pdflip import launcher as L
    import inspect

    src = inspect.getsource(L)
    assert src.count('env_d.update(turn_anchor_env(getattr(ns, "turn_anchor_token", None)))') == 2
    assert L.turn_anchor_env(None) == {}


# -- (2) the plan: a position plan on every rank ---------------------------------------------------------------------

def test_host_with_a_free_slot_plans_the_state(group_d):
    from flliper.srt.pdflip import turn_anchor as ta

    _as_worker(group_d, False)
    alloc = TA0._Alloc([11, 12])
    req = TA0._sreq(PROMPT)
    desc = TA0._plan(None, req, alloc, 0, N, N - 1)
    t, slot, d, s0, s1 = getattr(req, ta.PENDING_ATTR)
    assert (t, int(slot), d is desc) == (I_LAST, 11, True) and desc.targets[0] == I_LAST
    close = getattr(req, ta.CLOSE_PENDING_ATTR)
    assert close[0] == I_LAST - 3 and int(close[1]) == 12, "CLOSE-FORK joins on D"


def test_host_without_a_free_slot_still_plans_the_position_split_only(group_d):
    """rank agreement: the plan never depends on a rank-local free-slot count (P skips here)."""
    from flliper.srt.pdflip import turn_anchor as ta

    _as_worker(group_d, False)
    alloc = TA0._Alloc([])
    req = TA0._sreq(PROMPT)
    desc = TA0._plan(None, req, alloc, 0, N, N - 1)
    assert desc is None, "no slot, no gather row"
    assert getattr(req, ta.PENDING_ATTR) == (I_LAST, None, None, 0, N)
    assert getattr(req, ta.CLOSE_PENDING_ATTR) == (I_LAST - 3, None, None, 0, N)


def test_a_form_a_worker_plans_split_only_and_draws_no_slot(group_d):
    from flliper.srt.pdflip import turn_anchor as ta

    _as_worker(group_d, True)
    alloc = TA0._Alloc([11, 12])
    req = TA0._sreq(PROMPT)
    assert TA0._plan(None, req, alloc, 0, N, N - 1) is None
    assert getattr(req, ta.PENDING_ATTR) == (I_LAST, None, None, 0, N)
    assert alloc.free_ids == [11, 12], "a worker's anchors are bookkeeping: nothing drawn"


def test_group_p_without_a_slot_is_unchanged(monkeypatch):
    from flliper.srt.pdflip import turn_anchor as ta

    monkeypatch.setenv("FLLIPER_PDFLIP_GEN_FORK_ANCHOR", "0")
    monkeypatch.setenv("FLLIPER_PDFLIP_SYS_GRID_ANCHOR", "0")
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "P")
    req = TA0._sreq(PROMPT)
    assert TA0._plan(None, req, TA0._Alloc([]), 0, N, N - 1) is None
    assert getattr(req, ta.PENDING_ATTR, None) is None


# -- (2)/(3)/(4) the real tree -----------------------------------------------------------------------------------------

def _structure(fx):
    """Every node as (end depth, key length) -- the rank-invariant shape R12 keys on (last hash, end depth)."""
    cache = getattr(fx, "cache", fx)
    out, stack = set(), [(cache.root_node, 0)]
    while stack:
        node, d = stack.pop()
        for c in node.children.values():
            dd = d + len(c.key)
            out.add((dd, len(c.key)))
            stack.append((c, dd))
    return out


def _d_prefill(fx, *, marks=("gdn",), insert_order=None):
    """One D prefill of PROMPT through the scheduler's plan (note_step on the real pool), the forward's marks, the
    finish insert -- the fork state S_TURN where a slot was drawn."""
    from flliper.srt.pdflip import turn_anchor as ta

    req = TA0._req(fx, PROMPT)
    TA0._write_kv(fx, req, N)
    batch = SimpleNamespace(reqs=[req], req_to_token_pool=fx.pool)
    desc = ta.note_step(batch=batch, desc=None, req=req, row=0, prefix=0, end=N,
                        track_mask=True, main_track=N - 1, chunk=1, page=1, tok=IM)
    plan = getattr(req, ta.PENDING_ATTR)
    if plan[1] is not None:
        TA0._state(fx, plan[1], TA0.S_TURN)
    if desc is not None:
        desc.done.update(marks)
    TA0._state(fx, req.mamba_pool_idx, TA0.S_END)
    if insert_order is not None:
        real = fx.cache.insert

        def _rec(params):
            insert_order.append(len(params.key))
            return real(params)

        fx.cache.insert = _rec
    TA0._finish(fx, req)
    return req, plan


def test_host_and_worker_trees_hold_the_same_nodes_and_the_next_turn_resumes_at_the_fork(group_d, caplog):
    caplog.set_level(logging.INFO, TA_LOG)
    _as_worker(group_d, False)
    host = TA0._fixture(False)
    _d_prefill(host)
    _as_worker(group_d, True)
    worker = TA0._fixture(False)
    _d_prefill(worker)
    assert _structure(host) == _structure(worker), "the same nodes on every rank (R12 node keys)"
    assert (I_LAST, I_LAST) in _structure(host) or any(d == I_LAST for d, _ in _structure(host))
    # the host resumes the next turn AT the fork with the fork's state (red on rc30: below it)
    assert TA0._claim(host, NEXT) == (I_LAST, pytest.approx(TA0.S_TURN))
    # the worker's node at the fork carries no state (bookkeeping rank)
    mr = worker.cache.match_prefix(MatchPrefixParams(key=TA0._key(worker, PROMPT[:I_LAST])))
    from flliper.srt.managers.tp_match_floor import follow_walk

    with follow_walk(worker.cache):
        mr = worker.cache.match_prefix(MatchPrefixParams(key=TA0._key(worker, PROMPT[:I_LAST])))
    assert len(mr.device_indices) == I_LAST, "the worker's KV path reaches the fork (H98 follow walk)"
    assert mr.last_device_node.component_data[ComponentType.MAMBA].value is None
    msgs = [m for m in caplog.messages if "PDFLIP D-FORK-ANCHOR SET" in m]
    assert any("state=taken" in m and "role=host" in m for m in msgs), msgs
    assert any("state=split_only" in m and "role=worker" in m for m in msgs), msgs


def test_mutant_only_tp0_splits_leaves_the_trees_different(group_d):
    """The danger direction: a worker that drops its split-only plan (the P rule: no slot, no node) -- TP0's tree has
    a node at the fork the workers do not (R12 'unreconcilable')."""
    from flliper.srt.pdflip import turn_anchor as ta

    _as_worker(group_d, False)
    host = TA0._fixture(False)
    _d_prefill(host)
    _as_worker(group_d, True)
    worker = TA0._fixture(False)
    real_insert_one = worker.cache._pdflip_turn_insert_one

    def _mutant(req, pend, *, kind, is_insert):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(ta, "d_mode", lambda: False)        # the worker inserts like group P: no state, no node
            return real_insert_one(req, pend, kind=kind, is_insert=is_insert)

    worker.cache._pdflip_turn_insert_one = _mutant
    _d_prefill(worker)
    assert _structure(host) != _structure(worker)


def test_a_plan_consumed_without_its_marks_is_inserted_split_only_and_frees_its_slot(group_d, caplog):
    """(3) overlap: a plan whose forward never ran its rows (a batch that did not run, a geometry change, a captured
    graph) -- the node is still there on every rank, the slot goes back, nothing raises."""
    caplog.set_level(logging.INFO, TA_LOG)
    _as_worker(group_d, False)
    fx = TA0._fixture(False)
    free0 = fx.pool.mamba_allocator.available_size()
    _req, plan = _d_prefill(fx, marks=())
    assert plan[1] is not None
    assert fx.pool.mamba_allocator.available_size() == free0 - 1, "only the END anchor's slot stays in the tree"
    _as_worker(group_d, True)
    wk = TA0._fixture(False)
    _d_prefill(wk)
    assert _structure(fx) == _structure(wk)
    depth, state = TA0._claim(fx, NEXT)
    assert state != pytest.approx(TA0.S_TURN)
    assert any("state=split_only" in m and "why=unmarked:gdn" in m for m in caplog.messages), caplog.messages


def test_the_fork_node_goes_in_before_the_step_node_in_the_same_insert_call(group_d):
    _as_worker(group_d, False)
    fx = TA0._fixture(False)
    order = []
    _d_prefill(fx, insert_order=order)
    assert order and order[0] == I_LAST - 3 and I_LAST in order and order[-1] >= N - 1, order
    assert order.index(I_LAST) < len(order) - 1


def test_the_read_witness_names_a_resume_on_a_d_fork_anchor(group_d, caplog):
    caplog.set_level(logging.INFO, TA_LOG)
    _as_worker(group_d, False)
    fx = TA0._fixture(False)
    _d_prefill(fx)
    from array import array

    req = TA0.Req(rid="pdflip-2-64", origin_input_text="", origin_input_ids=array("q", NEXT),
                  sampling_params=TA0.SamplingParams(temperature=0, max_new_tokens=1))
    fx.cache.match_prefix(MatchPrefixParams(key=TA0._key(fx, NEXT[:I_LAST + 1]), req=req))
    assert any("PDFLIP D-FORK-ANCHOR READ" in m and "rid=pdflip-2-64" in m and "kind=turn" in m
               for m in caplog.messages), caplog.messages


# -- (3) the D backend: the fork rows join the step's own track gathers ----------------------------------------------

def test_the_fork_rows_join_the_step_track_gathers_at_d_geometry(group_d):
    """Form A x DCP token cut: TP0 runs every GDN head over the whole extend (the cut is the full-attention KV only),
    so the forward's geometry is the planned one. Two D rows on the page-64 grid, prefix > 0, the fork on row 0 only
    (row 1's plan was split-only: no gather row); the fork row is the h row / conv window the upstream track math
    picks for a track at t, appended to the SAME tensors as the step tracks (one kernel launch)."""
    from flliper.srt.pdflip import turn_anchor as ta

    sa = TA0.ServerArgs(model_path="dummy", page_size=64)
    sa._mamba_cache_chunk_size = 64
    TA0.set_global_server_args_for_scheduler(sa)
    prefixes, ext = [70208, 62528], [187, 1152]          # pdflip-2-63-like and pdflip-2-15-like steps
    t = 70336
    turn = ta.TurnTracks(2)
    turn.add(0, torch.tensor([9]), t, 70208, 187)
    fb = TA0._fb(prefixes, ext, [70208 + 128 + 1, 63680 + 1], [40, 41], turn)
    be = TA0._backend()
    be.forward_metadata = TA0._metadata(be, fb)
    n_main = len(be.forward_metadata.track_ssm_h_dst)
    be._pdflip_turn_rows(fb)
    md = be.forward_metadata
    assert "gdn" in turn.done
    assert len(md.track_ssm_h_dst) == n_main + 1 and md.track_ssm_h_dst[-1].item() == 9
    ref = TA0._metadata(TA0._backend(), TA0._fb(prefixes, ext, [t + 1, 63680 + 1], [9, 41]))
    assert md.track_ssm_h_src[-1].item() == ref.track_ssm_h_src[0].item()
    assert md.track_conv_indices[-1].tolist() == ref.track_conv_indices[0].tolist()
    # a forward whose geometry is NOT the planned one (a sliced batch) adds nothing: the plan goes split-only
    turn2 = ta.TurnTracks(2)
    turn2.add(0, torch.tensor([9]), t, 70208, 187)
    fb2 = TA0._fb([70208], [187], [70337], [40], turn2)
    be2 = TA0._backend()
    be2.forward_metadata = TA0._metadata(be2, fb2)
    before = be2.forward_metadata.track_ssm_h_dst.tolist()
    be2._pdflip_turn_rows(fb2)
    assert be2.forward_metadata.track_ssm_h_dst.tolist() == before and not turn2.complete()


def test_a_classic_rank_holds_its_state_and_the_rank_check_never_swallows_an_error(monkeypatch):
    """Classic boot (no Form A plan, no weightless lane): the rank plans slots. An undecidable rank form is NOT
    swallowed into 'holds state' (H99 bare-except trap): the named stop propagates."""
    from flliper.srt.managers import tp_match_floor as m
    from flliper.srt.pdflip import turn_anchor as ta

    assert ta.d_split_only_rank() is False

    def _undecidable():
        raise m.FormAFollowUndecidable("plan unreadable")

    monkeypatch.setattr(m, "this_rank_follows", _undecidable)
    with pytest.raises(m.FormAFollowUndecidable):
        ta.d_split_only_rank()


# -- review 21.10. (A1 / A4 / F6 / F7) --------------------------------------------------------------------------------

REV = _load("_d_fork_review_for_feature", _HERE / "test_nf_rc30_d_fork_anchor_review_1021.py")


def test_a1_mutant_without_the_donation_room_guard_splits_the_step_node(group_d):
    """The danger direction of F1: without the guard TP0 keeps the fork state pinned on its last free slot, the step's
    donation finds nothing and TP0 alone skips the chunk node the workers insert."""
    from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    group_d.setattr(UnifiedRadixCache, "_d_fork_donation_room", lambda self: True)
    host, hreq, _ = REV._starved_step(group_d, worker=False)
    wk, wreq, _ = REV._starved_step(group_d, worker=True)
    assert _structure(host) != _structure(wk) or hreq.cache_protected_len != wreq.cache_protected_len


def test_a1_the_starved_fork_yields_its_state_and_is_named(group_d, caplog):
    caplog.set_level(logging.INFO, TA_LOG)
    host, hreq, plans = REV._starved_step(group_d, worker=False)
    assert any("state=split_only" in m and "why=no_donation_room" in m and "role=host" in m
               for m in caplog.messages), caplog.messages
    assert hreq.cache_protected_len == I_LAST + 16, "the step's node is in TP0's tree"


def test_a1_with_room_the_fork_keeps_its_state(group_d):
    _as_worker(group_d, False)
    fx = TA0._fixture(False)
    _d_prefill(fx)
    assert TA0._claim(fx, NEXT) == (I_LAST, pytest.approx(TA0.S_TURN))


def test_f6_an_unreadable_switch_stops_by_name(monkeypatch):
    from flliper.srt.environ import envs
    from flliper.srt.pdflip import turn_anchor as ta

    def _boom():
        raise RuntimeError("env unreadable")

    monkeypatch.setattr(envs.FLLIPER_PDFLIP_D_FORK_ANCHOR, "get", _boom)
    with pytest.raises(RuntimeError):
        ta.d_fork_anchor_on()


def test_f7_the_read_witness_names_only_the_forks_own_depth(group_d, caplog):
    from flliper.srt.mem_cache.unified_cache_components import mamba_component as mc
    from flliper.srt.pdflip import turn_anchor as ta

    caplog.set_level(logging.INFO, TA_LOG)
    node = SimpleNamespace(component_data={ComponentType.MAMBA: SimpleNamespace(value=torch.tensor([3]),
                                                                                host_value=None)})
    setattr(node, ta.D_FORK_KIND_ATTR, ("turn", 128))
    mc._obs_d_fork_read(SimpleNamespace(rid="r-other"), node, 64)
    assert not any("D-FORK-ANCHOR READ" in m for m in caplog.messages)
    mc._obs_d_fork_read(SimpleNamespace(rid="r-own"), node, 128)
    assert any("D-FORK-ANCHOR READ" in m and "rid=r-own" in m and "kind=turn" in m for m in caplog.messages)


def test_a4_an_anchor_short_on_a_fork_node_is_counted_apart(monkeypatch):
    from flliper.srt.mem_cache import form_a_host_shadow as r12
    from flliper.srt.pdflip import turn_anchor as ta

    monkeypatch.setattr(r12, "_host_flags", lambda node: (True, False))     # KV host yes, anchor host no
    root = SimpleNamespace(parent=None, key=())
    fork = SimpleNamespace(parent=root, key=tuple(range(64)))
    setattr(fork, ta.D_FORK_KIND_ATTR, ("turn", 64))
    deeper = SimpleNamespace(parent=root, key=tuple(range(96)))     # a split moved the mark onto a deeper piece
    setattr(deeper, ta.D_FORK_KIND_ATTR, ("turn", 64))
    assert r12._d_fork_anchor_short(deeper, 1, 1, 1) is False, "not at the fork's depth: a real short"
    assert r12._d_fork_anchor_short(fork, 1, 1, 1) is True
    assert r12._d_fork_anchor_short(SimpleNamespace(), 1, 1, 1) is False, "not a fork node: a real short"
    monkeypatch.setattr(r12, "_host_flags", lambda node: (False, False))    # the KV host copy missing too
    assert r12._d_fork_anchor_short(fork, 1, 1, 1) is False, "a missing KV host copy stays short"


# -- re-review G1: an in-flight fork plan never starves another request's donation on TP0 alone ---------------------

def test_g1_the_donation_takes_back_an_inflight_fork_slot_and_names_it(group_d, caplog):
    from flliper.srt.pdflip import turn_anchor as ta

    caplog.set_level(logging.INFO, TA_LOG)
    host, r1, r2 = REV._inflight_pair(group_d, worker=False)
    assert any("PDFLIP D-FORK-ANCHOR REVOKED" in m and "rid=pdflip-2-63" in m for m in caplog.messages), caplog.messages
    plan = getattr(r1, ta.PENDING_ATTR)
    assert plan[1] is None and plan[2] == "revoked_for_donation"
    assert ta.insert_verdict(plan) == (False, "revoked_for_donation")
    assert r2.cache_protected_len == 32, "R2's step node is in TP0's tree"


def test_g1_mutant_without_the_revoke_splits_the_trees(group_d):
    from flliper.srt.pdflip import turn_anchor as ta

    group_d.setattr(ta, "revoke_for_donation", lambda: None)
    host, _h1, h2 = REV._inflight_pair(group_d, worker=False)
    wk, _w1, w2 = REV._inflight_pair(group_d, worker=True)
    assert h2.cache_protected_len != w2.cache_protected_len or _structure(host) != _structure(wk)


def test_g1_a_worker_and_group_p_never_revoke(group_d, monkeypatch):
    from flliper.srt.pdflip import turn_anchor as ta

    _as_worker(group_d, True)
    assert ta.revoke_for_donation() is None
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "P")
    assert ta.revoke_for_donation() is None


def test_g1_only_the_step_donation_may_take_a_fork_slot():
    """A non-donation allocation (S90b / S109R device read: an H2D copy on the CURRENT stream, not ordered behind the
    fork batch's late gather on the forward stream) must never take an in-flight fork slot: exactly the three step
    donation sites pass donation=True (two ping-pong branches, first write on the forward stream; the copy branch
    fences before its current-stream copy)."""
    import inspect

    from flliper.srt.mem_cache.unified_cache_components import mamba_component as mc

    body = inspect.getsource(mc.MambaComponent._prepare_for_caching_req_impl)
    src = inspect.getsource(mc.MambaComponent)
    assert body.count("_alloc_mamba_slot(donation=True)") == 3
    assert src.count("_alloc_mamba_slot(donation=True)") == 3, "no other caller may revoke"
    i = body.index("mamba_value_donated = self._alloc_mamba_slot(donation=True)")
    assert body.index("torch.cuda.synchronize()", i) < body.index("copy_from(", i), "the fence precedes the copy"


def test_g1_a_revoked_plan_is_inserted_split_only_on_the_host(group_d, caplog):
    caplog.set_level(logging.INFO, TA_LOG)
    host, r1, r2 = REV._inflight_pair(group_d, worker=False)
    TA0._state(host, r1.mamba_pool_idx, TA0.S_END)
    TA0._finish(host, r1)
    wk, w1, w2 = REV._inflight_pair(group_d, worker=True)
    TA0._state(wk, w1.mamba_pool_idx, TA0.S_END)
    TA0._finish(wk, w1)
    assert _structure(host) == _structure(wk)
    assert any("state=split_only" in m and "why=revoked_for_donation" in m for m in caplog.messages), caplog.messages


# -- re-review 2 N1: a pool reset neutralises the in-flight plans ---------------------------------------------------

def test_n1_a_pool_clear_turns_the_inflight_plans_split_only_and_nothing_frees_them_again(group_d, caplog):
    from flliper.srt.pdflip import turn_anchor as ta

    caplog.set_level(logging.INFO, TA_LOG)
    _as_worker(group_d, False)
    fx = TA0._fixture(False)
    r1 = TA0._req(fx, PROMPT)
    TA0._write_kv(fx, r1, N)
    batch = SimpleNamespace(reqs=[r1], req_to_token_pool=fx.pool)
    ta.note_step(batch=batch, desc=None, req=r1, row=0, prefix=0, end=N,
                 track_mask=True, main_track=N - 1, chunk=1, page=1, tok=IM)
    assert getattr(r1, ta.PENDING_ATTR)[1] is not None and ta.inflight_count() >= 1
    fx.pool.mamba_allocator.clear()
    plan = getattr(r1, ta.PENDING_ATTR)
    assert plan[1] is None and plan[2] == "pool_reset" and ta.inflight_count() == 0
    assert ta.insert_verdict(plan) == (False, "pool_reset")
    assert any("D-FORK-ANCHOR POOL-RESET" in m for m in caplog.messages)
    # a later stale-plan walk (the next note_step of r1) frees nothing: the plan holds no slot any more
    free = fx.pool.mamba_allocator.available_size()
    ta.note_step(batch=batch, desc=None, req=r1, row=0, prefix=N, end=N, track_mask=False, main_track=None,
                 chunk=1, page=1, tok=IM)
    assert fx.pool.mamba_allocator.available_size() == free


def test_n1_mutant_without_the_clear_hook_hands_out_an_owned_slot(group_d):
    from flliper.srt.pdflip import turn_anchor as ta

    group_d.setattr(ta, "on_pool_clear", lambda alloc: 0)
    group_d.setattr(ta, "_epoch", lambda alloc: 0)
    _as_worker(group_d, False)
    fx = TA0._fixture(False)
    r1 = TA0._req(fx, PROMPT)
    TA0._write_kv(fx, r1, N)
    batch = SimpleNamespace(reqs=[r1], req_to_token_pool=fx.pool)
    ta.note_step(batch=batch, desc=None, req=r1, row=0, prefix=0, end=N,
                 track_mask=True, main_track=N - 1, chunk=1, page=1, tok=IM)
    fx.pool.mamba_allocator.clear()
    owned = set()
    while True:
        s = fx.pool.mamba_allocator.alloc(1)
        if s is None:
            break
        owned.add(int(s.view(-1)[0]))
    got = ta.revoke_for_donation()
    assert got is not None and int(got.view(-1)[0]) in owned, "the mutant aliases an owned slot"


# -- V1 (merge with IMAGE-END TRACK): the image track is a position plan on D too -----------------------------------

def test_v1_the_image_track_on_d_is_a_registered_position_plan(group_d):
    from flliper.srt.pdflip import turn_anchor as ta

    if not hasattr(ta, "IMAGE_PENDING_ATTR"):
        pytest.skip("IMAGE-END TRACK not on this base")
    ids = list(PROMPT)
    ids[10:20] = [1_000_007] * 10
    _as_worker(group_d, True)
    alloc = TA0._Alloc([11, 12, 13])
    req = TA0._sreq(ids)
    TA0._plan(None, req, alloc, 0, N, N - 1)
    wplan = getattr(req, ta.IMAGE_PENDING_ATTR)
    assert wplan is not None and wplan[1] is None and alloc.free_ids == [11, 12, 13], "worker: split-only, no draw"
    _as_worker(group_d, False)
    req2 = TA0._sreq(ids)
    TA0._plan(None, req2, TA0._Alloc([]), 0, N, N - 1)
    assert getattr(req2, ta.IMAGE_PENDING_ATTR)[0] == wplan[0] and getattr(req2, ta.IMAGE_PENDING_ATTR)[1] is None, \
        "host without a free slot: the same position, split-only"
    req3 = TA0._sreq(ids)
    TA0._plan(None, req3, TA0._Alloc([21, 22, 23]), 0, N, N - 1)
    assert getattr(req3, ta.IMAGE_PENDING_ATTR)[1] is not None
    assert any(e[1] == ta.IMAGE_PENDING_ATTR and e[0] is req3 for e in ta._INFLIGHT), "in the G1 registry"
    assert ta._ATTR_KIND[ta.IMAGE_PENDING_ATTR] == "image"
