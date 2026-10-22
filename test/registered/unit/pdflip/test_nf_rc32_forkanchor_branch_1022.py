"""NF rc32 (22.10.): the branching point no longer replaces group D's step track (review A2 of nf-rc31-forkanchor).

THE MEASURED GAP (rc32 ded00bd5a7, boot ..._1022_030836, D log):

  TP0 MATCH-CENSUS-DEEP rid=pdflip-4-121 reached=57088 accepted=56832 branching=57088 prefix=56832
  TP0/TP1/TP2 #969 EXTENT ('pdflip-4-121', 56832, 58201, 56832, 1369)
  TP1/TP2 PDFLIP CLOSE-FORK-ANCHOR SHARED rid=pdflip-4-121 at=57984 step=[56832,58201)
  TP1/TP2 PDFLIP D-FORK-ANCHOR SET rid=pdflip-4-121 kind=turn at=57984 state=split_only why=no_plan role=worker
  TP0     (no PLAN / TRACK / SET line for pdflip-4-121)
  TP0 #59 FINISH-TRACK rid=pdflip-4-121 ... branching=57088   TP1/TP2 ... branching=-

Upstream's extend-track setup let the branching point REPLACE the step's own track. Only the state-holding rank has a
branching point (a Form A worker's follow walk drops it), so TP0's main track was rank-local: ``step_target`` refused
the turn position 57984 >= 57088 on TP0 alone (7 of 7 branching readers of the boot: worker fork keys TP0 never had),
and TP0 lost the step-end anchor (pdflip-4-143 resumed on pdflip-4-141's branching anchor 101760 instead of its
step-end anchor 102784: 1024 tokens recomputed).

Pinned here (hermetic, CPU; the REAL ``_mamba_radix_cache_v2_req_prepare_for_extend``, ``turn_anchor.note_step`` and
UnifiedRadixCache of test_pdflip_turn_anchor_0929):
  (1) metal repro 4-121 / 4-141: on group D the step keeps its step-end track on every rank, the branching point goes
      to a second-track plan; the fork plan set (turn) is the same on TP0 and the workers;
  (2) the real tree: host and worker trees hold the same nodes, the next turn resumes at the turn anchor, a reader
      diverging at the branching point resumes on the branch state;
  (3) MUTANT "Ränge uneins" (the danger direction): with upstream's replacement the TP0 plan set and tree differ;
  (4) edges: group P / output ids / page not a multiple of the chunk keep upstream's replacement; a stale branch plan
      frees its slot; the pending helpers know the new kind;
  (5) the divergence instrument names the ids where a request leaves the tree.
"""
from __future__ import annotations

import importlib.util
import logging
import os
import pathlib
import sys
from array import array
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


TA0 = _load("_turn_anchor_0929_for_rc32_branch", _HERE / "test_pdflip_turn_anchor_0929.py")
IM, IM_END, NL, USER, ASSISTANT = TA0.IM, TA0.IM_END, TA0.NL, TA0.USER, TA0.ASSISTANT
GEN = TA0.GEN
TA_LOG = "flliper.srt.pdflip.turn_anchor"
S_BR, S_CLOSE = 1.75, 2.5


@pytest.fixture
def group_d(monkeypatch):
    from flliper.srt.mem_cache import unified_radix_cache as urc
    from flliper.srt.pdflip import turn_anchor as ta

    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "D")
    monkeypatch.setenv(TA0.TOKEN_ENV, str(IM))
    monkeypatch.setenv("FLLIPER_PDFLIP_BIGRAM_ANCHOR_EXACT", "1")
    monkeypatch.setenv("FLLIPER_PDFLIP_GEN_FORK_ANCHOR", "0")
    monkeypatch.delenv("FLLIPER_PDFLIP_D_FORK_ANCHOR", raising=False)
    monkeypatch.setattr(urc, "_PDFLIP_END_ANCHOR", True)
    monkeypatch.setattr(ta, "_INFLIGHT", [])
    return monkeypatch


def _as_worker(monkeypatch, worker: bool):
    from flliper.srt.pdflip import turn_anchor as ta

    monkeypatch.setattr(ta, "d_split_only_rank", lambda: worker)


# -- (1) metal geometry: the track setup and the plan ------------------------------------------------------------------

# pdflip-4-121: prompt 58201, D prefix 56832 (pdflip-4-119's step-end anchor), branching 57088 (inside 4-119's answer),
# last message's <|im_start|> at 57990 -> turn floor_64 = 57984 (= the close-fork position, SHARED on the workers)
N121, PRE121, BR121, I121 = 58201, 56832, 57088, 57990
E121 = PRE121 + (N121 - PRE121) // 64 * 64          # 58176: the step-end grid point every rank tracks


def _ids_121():
    ids = [7] * N121
    ids[I121 - 2:I121 + 3] = [IM_END, NL, IM, USER, NL]
    ids[N121 - 7:] = [IM_END, NL] + GEN
    return ids


@pytest.fixture
def d_setup(group_d):
    import flliper.srt.managers.schedule_batch as sb

    group_d.setattr(sb, "get_server_args", lambda: SimpleNamespace(
        mamba_cache_chunk_size=64, mamba_checkpoint_interval=None,
        enable_mamba_extra_buffer=lambda: True, enable_mamba_extra_buffer_lazy=lambda: False))
    return group_d, sb


def _step(sb, ids, prefix, branching, *, page=64, free=(21, 22, 23), rid="pdflip-4-121", output=()):
    """The track setup of one step (the real ScheduleBatch helper) and the D fork plan note_step makes right after it,
    with the main track the setup left on the request -- what prepare_for_extend does per request."""
    from flliper.srt.pdflip import turn_anchor as ta

    alloc = TA0._Alloc(list(free))
    batch = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(get_mamba_ping_pong_other_idx=lambda i: 1 - i, mamba_allocator=alloc),
        token_to_kv_pool_allocator=SimpleNamespace(page_size=page))
    req = SimpleNamespace(rid=rid, origin_input_ids=array("q", ids), output_ids=list(output),
                          return_logprob=False, input_embeds=None, session_id=None, multimodal_inputs=None,
                          prefix_indices=range(prefix), mamba_ping_pong_track_buffer=torch.tensor([0, 1]),
                          mamba_next_track_idx=0, mamba_branching_seqlen=branching,
                          mamba_last_track_seqlen=None, cache_protected_len=prefix)
    req.extend_range = SimpleNamespace(start=prefix, end=len(ids), length=len(ids) - prefix)
    batch.reqs = [req]
    entry = sb.ScheduleBatch._mamba_radix_cache_v2_req_prepare_for_extend(batch, req)
    desc = ta.note_step(batch=batch, desc=None, req=req, row=0, prefix=prefix, end=len(ids),
                        track_mask=bool(entry.track_mask), main_track=req.mamba_last_track_seqlen,
                        chunk=64, page=page, tok=IM)
    return req, desc, entry, alloc


def _plans(req):
    """The fork plans the step left: kind -> (position, has a slot)."""
    from flliper.srt.pdflip import turn_anchor as ta

    out = {}
    for kind, attr in (("turn", ta.PENDING_ATTR), ("close", ta.CLOSE_PENDING_ATTR), ("gen", ta.GEN_PENDING_ATTR),
                       ("image", ta.IMAGE_PENDING_ATTR), ("branch", ta.BRANCH_PENDING_ATTR)):
        p = getattr(req, attr, None)
        if p is not None:
            out[kind] = (int(p[0]), p[1] is not None)
    return out


def _fork_keys(req):
    """The rank-agreed part of the plan set: every kind but the host's branch state plan."""
    return {(k, v[0]) for k, v in _plans(req).items() if k != "branch"}


def test_metal_4_121_the_host_keeps_the_step_end_track_and_plans_the_turn_like_the_workers(d_setup):
    """Red on ded00bd5a7: the host's main track was 57088, its plan set empty, the workers planned turn 57984."""
    mp, sb = d_setup
    _as_worker(mp, False)
    host, hdesc, hentry, _ = _step(sb, _ids_121(), PRE121, BR121)
    _as_worker(mp, True)
    worker, _wdesc, wentry, walloc = _step(sb, _ids_121(), PRE121, None)
    assert host.mamba_last_track_seqlen == worker.mamba_last_track_seqlen == E121, \
        "the step-end track on every rank (base: TP0 57088)"
    assert int(hentry.track_seqlen) == int(wentry.track_seqlen), "the same kernel track position on every rank"
    assert _fork_keys(host) == _fork_keys(worker) == {("turn", 57984)}, (_plans(host), _plans(worker))
    assert _plans(host)["turn"] == (57984, True) and _plans(worker)["turn"] == (57984, False)
    assert _plans(host)["branch"] == (BR121, True), "the branching point rides a second track with its own slot"
    assert "branch" not in _plans(worker) and walloc.free_ids == [21, 22, 23], "a worker draws nothing"
    assert sorted(hdesc.targets) == [BR121, 57984] and set(hdesc.prefixes) == {PRE121}, "two gather rows on TP0"


def test_metal_4_141_the_step_end_anchor_4_143_needed_is_tracked_on_tp0(d_setup):
    """pdflip-4-141 [101312, 102841) branching 101760: TP0 tracked 101760 and lost 102784, so pdflip-4-143 (fork at
    102976) resumed at 101760 -- 1216 tokens recomputed, 1024 of them below the step-end anchor."""
    mp, sb = d_setup
    _as_worker(mp, False)
    ids = [7] * 102841
    ids[-7:] = [IM_END, NL] + GEN
    host, _d, _e, _a = _step(sb, ids, 101312, 101760, rid="pdflip-4-141")
    assert host.mamba_last_track_seqlen == 102784
    assert _plans(host)["branch"] == (101760, True)


def test_without_a_free_slot_the_branch_plan_is_split_only_and_the_turn_still_planned(d_setup):
    mp, sb = d_setup
    _as_worker(mp, False)
    host, desc, _e, _a = _step(sb, _ids_121(), PRE121, BR121, free=())
    assert host.mamba_last_track_seqlen == E121 and desc is None
    assert _plans(host) == {"turn": (57984, False), "branch": (BR121, False)}


def test_the_branch_point_handed_over_is_consumed_by_the_plan(d_setup):
    from flliper.srt.pdflip import turn_anchor as ta

    mp, sb = d_setup
    _as_worker(mp, False)
    host, _d, _e, _a = _step(sb, _ids_121(), PRE121, BR121)
    assert getattr(host, ta.BRANCH_POS_ATTR, None) is None, "never left standing for a later step"


# -- (3) the mutant: upstream's replacement on group D -----------------------------------------------------------------

def test_mutant_upstream_replacement_splits_the_rank_plan_sets(d_setup):
    """The danger direction 'ranks disagree': the branching point replacing TP0's track again (``d_branch_rides`` ->
    False) -- TP0's main track is its branching point, the turn plan exists on the workers only."""
    from flliper.srt.pdflip import turn_anchor as ta

    mp, sb = d_setup
    mp.setattr(ta, "d_branch_rides", lambda *a, **k: False)
    _as_worker(mp, False)
    host, _d, _e, _a = _step(sb, _ids_121(), PRE121, BR121)
    _as_worker(mp, True)
    worker, _wd, _we, _wa = _step(sb, _ids_121(), PRE121, None)
    assert host.mamba_last_track_seqlen == BR121 != worker.mamba_last_track_seqlen
    assert _fork_keys(host) != _fork_keys(worker), "the metal split: worker turn 57984, none on TP0"


# -- (4) edges ---------------------------------------------------------------------------------------------------------

def test_group_p_keeps_the_upstream_replacement(d_setup):
    mp, sb = d_setup
    mp.setenv("FLLIPER_PDFLIP_GROUP", "P")
    req, _d, _e, _a = _step(sb, _ids_121(), PRE121, BR121)
    assert req.mamba_last_track_seqlen == BR121
    assert "branch" not in _plans(req)


def test_the_d_switch_off_keeps_the_upstream_replacement(d_setup):
    mp, sb = d_setup
    mp.setenv("FLLIPER_PDFLIP_D_FORK_ANCHOR", "0")
    req, _d, _e, _a = _step(sb, _ids_121(), PRE121, BR121)
    assert req.mamba_last_track_seqlen == BR121


def test_a_page_not_a_multiple_of_the_chunk_keeps_the_upstream_replacement(d_setup):
    """page 16 < chunk 64: floor_chunk(reached) may lie below the KV match end -- no node boundary on the workers."""
    from flliper.srt.pdflip import turn_anchor as ta

    mp, sb = d_setup
    _as_worker(mp, False)
    req, _d, _e, _a = _step(sb, _ids_121(), PRE121, BR121, page=16)
    assert req.mamba_last_track_seqlen == BR121
    assert ta.d_branch_rides(sb.get_server_args(), 128, 64, False)
    assert not ta.d_branch_rides(sb.get_server_args(), 16, 64, False)
    assert not ta.d_branch_rides(sb.get_server_args(), None, 64, False)


def test_a_request_with_output_keeps_the_upstream_replacement(d_setup):
    """a retracted re-prefill: note_step plans nothing for it, so the branching point keeps the main track"""
    mp, sb = d_setup
    _as_worker(mp, False)
    req, _d, _e, _a = _step(sb, _ids_121(), PRE121, BR121, output=(5,))
    assert req.mamba_last_track_seqlen == BR121


def test_a_branch_point_at_the_step_end_track_needs_no_plan(d_setup):
    """branching == the step-end grid point: the step's own track writes it."""
    mp, sb = d_setup
    _as_worker(mp, False)
    req, _d, _e, _a = _step(sb, _ids_121(), PRE121, E121)
    assert req.mamba_last_track_seqlen == E121 and "branch" not in _plans(req)


def test_a_stale_branch_plan_frees_its_slot_and_the_pending_helpers_know_it(group_d):
    from flliper.srt.pdflip import turn_anchor as ta

    _as_worker(group_d, False)
    alloc = TA0._Alloc([])
    req = TA0._sreq(TA0.PROMPT)
    setattr(req, ta.BRANCH_PENDING_ATTR, (9, torch.tensor([33]), None, 0, 10))
    assert ta.has_pending(req)
    TA0._plan(None, req, alloc, 0, TA0.N, TA0.N - 1)
    assert 33 in alloc.freed and getattr(req, ta.BRANCH_PENDING_ATTR, None) is None
    setattr(req, ta.BRANCH_PENDING_ATTR, (9, None, None, 0, 10))
    kinds = [k for k, _p in ta.pop_all_pending(req)]
    assert "branch" in kinds and not ta.has_pending(req)
    assert ta._ATTR_KIND[ta.BRANCH_PENDING_ATTR] == "branch" and ta._label("branch") == "BRANCH-ANCHOR"


# -- (2) the real tree ---------------------------------------------------------------------------------------------------

# a sibling (pdflip-4-119's role) whose answer the reader (4-121's role) leaves at B = 128; the shared anchor at 64
COMMON = [IM, 8948, NL] + list(range(3000, 3059)) + [IM_END, NL]            # 64 ids, its state at 64
ANS = list(range(5000, 5100))                                            # the sibling's answer
SIB = COMMON + ANS + [IM_END, NL] + [IM, USER, NL] + list(range(6100, 6112)) + [IM_END, NL] + GEN
B = 128
R = COMMON + ANS[:B - len(COMMON)] + list(range(8000, 8020)) + [IM_END, NL] \
    + [IM, USER, NL] + list(range(6000, 6012)) + [IM_END, NL] + GEN
NR = len(R)
I_R = B + 20 + 2                                                            # the reader's last <|im_start|>
NEXT_R = R[:I_R + 1] + [ASSISTANT, NL] + list(range(7000, 7020)) + [IM_END, NL] + GEN
SIDE_B = R[:B] + list(range(9000, 9030)) + [IM_END, NL] + GEN              # leaves at the branching point too


def _finish(fx, req, n):
    req.kv_committed_len = n
    req.kv_allocated_len = n
    fx.cache.cache_finished_req(req, is_insert=True)


def _seed(fx):
    """The tree before the reader: COMMON with its state at 64, then the sibling's whole path with its own end state
    (its answer holds no anchor below its last track -- the decode-output shape)."""
    for ids, st in ((COMMON, TA0.S_EARLY), (SIB, 9.0)):
        req = TA0._req(fx, ids)
        TA0._write_kv(fx, req, len(ids))
        TA0._state(fx, req.mamba_pool_idx, st)
        _finish(fx, req, len(ids))


def _reader(fx, branch_at, main):
    """The reader's D prefill: match (reached B, accepted 64, branching B on the host), the plan note_step makes with
    the main track the setup left, the forward's marks, the finish insert."""
    from flliper.srt.pdflip import turn_anchor as ta

    req = TA0._req(fx, R)
    mr = fx.cache.match_prefix(MatchPrefixParams(key=TA0._key(fx, R)))
    TA0._write_kv(fx, req, NR)
    if branch_at is not None:
        setattr(req, ta.BRANCH_POS_ATTR, branch_at)
    batch = SimpleNamespace(reqs=[req], req_to_token_pool=fx.pool)
    desc = ta.note_step(batch=batch, desc=None, req=req, row=0, prefix=len(req.prefix_indices), end=NR,
                        track_mask=True, main_track=main, chunk=1, page=1, tok=IM)
    for attr, st in ((ta.PENDING_ATTR, TA0.S_TURN), (ta.BRANCH_PENDING_ATTR, S_BR),
                     (ta.CLOSE_PENDING_ATTR, S_CLOSE)):
        p = getattr(req, attr, None)
        if p is not None and p[1] is not None:
            TA0._state(fx, p[1], st)
    if desc is not None:
        desc.done.update(("gdn",))
    TA0._state(fx, req.mamba_pool_idx, TA0.S_END)
    _finish(fx, req, NR)
    return req, mr


def _structure(fx):
    out, stack = set(), [(fx.cache.root_node, 0)]
    while stack:
        node, d = stack.pop()
        for c in node.children.values():
            dd = d + len(c.key)
            out.add((dd, len(c.key)))
            stack.append((c, dd))
    return out


def _host_and_worker(mp, host_main):
    _as_worker(mp, False)
    host = TA0._fixture(False)
    _seed(host)
    _hreq, mr = _reader(host, B, host_main)
    _as_worker(mp, True)
    worker = TA0._fixture(False)
    _seed(worker)
    _reader(worker, None, NR - 1)
    return host, worker, mr


def test_the_tree_scenario_is_the_metal_shape(group_d):
    _as_worker(group_d, False)
    fx = TA0._fixture(False)
    _seed(fx)
    mr = fx.cache.match_prefix(MatchPrefixParams(key=TA0._key(fx, R)))
    assert len(mr.device_indices) == len(COMMON) and mr.mamba_branching_seqlen == B, \
        "reached B inside the sibling's answer, accepted the anchor below it"


def test_host_and_worker_trees_hold_the_same_nodes_and_the_host_resumes_on_both_fork_states(group_d, caplog):
    caplog.set_level(logging.INFO, TA_LOG)
    host, worker, _mr = _host_and_worker(group_d, NR - 1)
    assert _structure(host) == _structure(worker), "the branch node is the divergence split on every rank"
    assert TA0._claim(host, NEXT_R) == (I_R, pytest.approx(TA0.S_TURN)), "the next turn resumes at the turn anchor"
    assert TA0._claim(host, SIDE_B) == (B, pytest.approx(S_BR)), "a reader leaving at B resumes on the branch state"
    msgs = [m for m in caplog.messages if "PDFLIP D-FORK-ANCHOR SET" in m]
    assert any("kind=branch" in m and "state=taken" in m and "role=host" in m for m in msgs), msgs
    assert any("kind=turn" in m and "state=taken" in m and "role=host" in m for m in msgs), msgs
    assert any("kind=turn" in m and "state=split_only" in m and "role=worker" in m for m in msgs), msgs
    assert not any("kind=branch" in m and "role=worker" in m for m in msgs), msgs


def test_mutant_upstream_main_track_on_the_host_leaves_the_trees_different(group_d):
    """Danger direction 'ranks disagree' on the tree: TP0 planning against its own branching point (main = B) puts
    no turn / close node in -- the workers' tree has them."""
    host, worker, _mr = _host_and_worker(group_d, B)
    assert _structure(host) != _structure(worker)


# -- (5) the divergence instrument -------------------------------------------------------------------------------------

def test_the_divergence_instrument_names_the_ids_where_the_request_leaves_the_tree(caplog):
    from flliper.srt.mem_cache.radix_cache import RadixKey
    from flliper.srt.mem_cache.unified_cache_components import mamba_component as mc

    caplog.set_level(logging.INFO, mc.logger.name)
    mc._BRANCH_DIVERGE_N[0] = 0
    reached = 2048
    ids = list(range(10000, 10000 + reached)) + [1, 2, 3, 99, 5]
    child = SimpleNamespace(key=RadixKey(array("q", [1, 2, 3, 4, 5, 6])))
    node = SimpleNamespace(children={1: child})
    req = SimpleNamespace(rid="pdflip-4-121")
    mc._obs_branch_diverge(req, RadixKey(array("q", ids)), node, reached, 1984, reached)
    line = [m for m in caplog.messages if "MATCH-BRANCH-DIVERGE" in m]
    assert line and "rid=pdflip-4-121" in line[0] and "first_diff=3" in line[0], line
    assert "[1, 2, 3, 4, 5, 6]" in line[0] and "req_ids[2040:]" in line[0]
    caplog.clear()
    mc._obs_branch_diverge(req, RadixKey(array("q", ids)), node, reached, reached, None)
    assert not [m for m in caplog.messages if "MATCH-BRANCH-DIVERGE" in m], "no branching: no line"
