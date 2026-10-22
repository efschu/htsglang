"""Review of NF rc32 FORKANCHOR2 (af2a6b9cf6, 22.10.): the branch track on group D.

Pinned here (hermetic, CPU; the author's harness of test_nf_rc32_forkanchor_branch_1022 and the REAL
``_mamba_radix_cache_v2_req_prepare_for_extend``, ``turn_anchor.note_step``, ``GDNAttnBackend`` track rows,
``UnifiedRadixCache`` match/insert):

  R1 position, end to end: the 4-121 step's branch row lands on exactly the h row / conv window upstream's own
     replacement track (``_force_track_h(B)``) picked, and the state after B tokens lands in the branch slot; the
     step-end row is the worker's row.
  R2 a branching point at a position another fork track writes is SHARED: one plan, one slot.
  R3 slot ownership (K19 class): a revoked branch plan is never freed again; a stale-freed branch plan is never
     revoked afterwards; a pool clear neutralises it.
  R4 a match the key limit cut (the prompt lies wholly on a longer path) gives TP0 a branching point that is NOT a
     divergence -- still a node on every rank, because every rank's admission match splits at its end.
  R5 the MATCH-BRANCH-DIVERGE instrument on the REAL match: ``tree_children`` must be the sibling's continuation AT
     ``reached`` (the node the match ended on), not the children of the accepted-depth node.
  R6 an ARMED group P (overlap off) keeps upstream's replacement (the author's P edge runs unarmed).
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

_HERE = pathlib.Path(__file__).resolve().parent


def _load(name, path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


AU = _load("_rc32_branch_author_for_review", _HERE / "test_nf_rc32_forkanchor_branch_1022.py")
TA0 = AU.TA0
group_d = AU.group_d          # fixtures, re-exported
d_setup = AU.d_setup
server_args_64 = TA0.server_args_64
IM, IM_END, NL, GEN = TA0.IM, TA0.IM_END, TA0.NL, TA0.GEN
PRE121, BR121, N121, E121 = AU.PRE121, AU.BR121, AU.N121, AU.E121


# -- R1: the branch row's position, from the track setup to the GDN gather ------------------------------------------

def test_r1_branch_row_is_upstreams_replacement_row_and_the_main_row_the_workers(d_setup, server_args_64):
    mp, sb = d_setup
    AU._as_worker(mp, False)
    host, desc, hentry, _ = AU._step(sb, AU._ids_121(), PRE121, BR121)
    AU._as_worker(mp, True)
    _w, _wd, wentry, _wa = AU._step(sb, AU._ids_121(), PRE121, None)
    ext = N121 - PRE121
    br_slot = next(s for s, t in zip(desc.slots, desc.targets) if t == BR121)

    fb = TA0._fb([PRE121], [ext], [int(hentry.track_seqlen)], [40], desc)
    be = TA0._backend()
    be.forward_metadata = TA0._metadata(be, fb)
    base_h = be.forward_metadata.track_ssm_h_src.clone()
    be._pdflip_turn_rows(fb)
    md = be.forward_metadata
    assert "gdn" in desc.done

    # the main row: the same as a worker's (no branching there)
    wfb = TA0._fb([PRE121], [ext], [int(wentry.track_seqlen)], [40])
    wmd = TA0._metadata(TA0._backend(), wfb)
    assert base_h.tolist() == wmd.track_ssm_h_src.tolist(), "the step-end row is rank-agreed"

    # the reference: upstream's own replacement at B (base ded00bd5a7 path: track_seqlen = _force_track_h(B) = B + 1)
    ref = TA0._metadata(TA0._backend(), TA0._fb([PRE121], [ext], [BR121 + 1], [int(br_slot)]))
    i = [int(x) for x in md.track_ssm_h_dst.tolist()].index(int(br_slot))
    assert md.track_ssm_h_src[i].item() == ref.track_ssm_h_src[-1].item(), "branch h row == upstream's row at B"
    j = len(md.track_conv_indices) - len(desc) + list(desc.slots).index(br_slot)
    assert md.track_conv_indices[j].tolist() == ref.track_conv_indices[-1].tolist(), "conv window == upstream's"

    # the copy the forward runs: the state after (B - prefix) / 64 chunks of this step lands in the branch slot
    n_h = (ext - 1) // 64 + 1
    h = torch.arange(n_h, dtype=torch.float32).view(1, n_h, 1, 1, 1).expand(1, n_h, 1, 2, 2)
    ssm = torch.zeros(64, 1, 2, 2)
    be._track_mamba_state_extend(fb, h, ssm, md)
    assert ssm[int(br_slot)].flatten().tolist() == [float((BR121 - PRE121) // 64)] * 4


# -- R2: SHARED -------------------------------------------------------------------------------------------------------

def test_r2_a_branching_point_on_the_turn_position_is_shared_one_plan_one_slot(d_setup):
    mp, sb = d_setup
    AU._as_worker(mp, False)
    host, desc, _e, alloc = AU._step(sb, AU._ids_121(), PRE121, 57984)
    assert AU._plans(host) == {"turn": (57984, True)}, AU._plans(host)
    assert len(desc) == 1 and alloc.free_ids == [22, 23], "one gather row, one slot drawn"
    assert host.mamba_last_track_seqlen == E121


# -- R3: slot ownership --------------------------------------------------------------------------------------------

def _branch_only(sb, free=(21, 22)):
    ids = [7] * 102841
    ids[-7:] = [IM_END, NL] + GEN
    return AU._step(sb, ids, 101312, 101760, rid="pdflip-4-141", free=free)


def test_r3_a_revoked_branch_plan_is_never_freed_again(d_setup):
    from flliper.srt.pdflip import turn_anchor as ta

    mp, sb = d_setup
    AU._as_worker(mp, False)
    req, _d, _e, alloc = _branch_only(sb)
    assert AU._plans(req) == {"branch": (101760, True)} and ta.inflight_count() == 1
    slot = ta.revoke_for_donation()
    assert slot is not None and int(slot.view(-1)[0]) == 21
    assert getattr(req, ta.BRANCH_PENDING_ATTR)[1] is None, "the plan turned split-only"
    # the next step's stale sweep must not free the slot the donation now owns
    batch = SimpleNamespace(reqs=[req], req_to_token_pool=SimpleNamespace(mamba_allocator=alloc))
    ta.note_step(batch=batch, desc=None, req=req, row=0, prefix=102784, end=102841, track_mask=False,
                 main_track=None, chunk=64, page=64, tok=IM)
    assert alloc.freed == [] and getattr(req, ta.BRANCH_PENDING_ATTR, None) is None


def test_r3_a_stale_freed_branch_plan_is_never_revoked_afterwards(d_setup):
    from flliper.srt.pdflip import turn_anchor as ta

    mp, sb = d_setup
    AU._as_worker(mp, False)
    req, _d, _e, alloc = _branch_only(sb)
    batch = SimpleNamespace(reqs=[req], req_to_token_pool=SimpleNamespace(mamba_allocator=alloc))
    ta.note_step(batch=batch, desc=None, req=req, row=0, prefix=102784, end=102841, track_mask=False,
                 main_track=None, chunk=64, page=64, tok=IM)
    assert alloc.freed == [21]
    assert ta.revoke_for_donation() is None, "a freed slot handed to a donation would have two owners"


def test_r3_a_pool_clear_neutralises_the_branch_plan(d_setup):
    from flliper.srt.pdflip import turn_anchor as ta

    mp, sb = d_setup
    AU._as_worker(mp, False)
    req, _d, _e, alloc = _branch_only(sb)
    assert ta.on_pool_clear(alloc) == 1
    p = getattr(req, ta.BRANCH_PENDING_ATTR)
    assert p[1] is None and p[2] == "pool_reset"


# -- R4: residual -- a branching point that is not a divergence (key limit) -----------------------------------------

COMMON = AU.COMMON                                     # 64 ids, its state at 64
LONG = COMMON + list(range(5000, 5180)) + [IM_END, NL] + GEN   # 251 ids (the fixture's context is 256)
L4 = 192                                               # the reader's prompt: LONG[:192], wholly on LONG's path
R4 = LONG[:L4]


def _seed_long(fx):
    for ids, st in ((COMMON, TA0.S_EARLY), (LONG, 9.0)):
        req = TA0._req(fx, ids)
        TA0._write_kv(fx, req, len(ids))
        TA0._state(fx, req.mamba_pool_idx, st)
        AU._finish(fx, req, len(ids))


def _capped_branching(fx):
    from flliper.srt.managers.schedule_batch import Req

    claim = Req._compute_max_prefix_len(SimpleNamespace(return_logprob=False, logprob_start_len=0), L4)
    return fx.cache.match_prefix(MatchPrefixParams(key=TA0._key(fx, R4[:claim])))


def _reader4(fx, branch_at, admission_match=True):
    from flliper.srt.pdflip import turn_anchor as ta

    if admission_match:
        # the admission match every rank runs with the key limit; on NF's page 64 the walk matches whole pages only,
        # floor_64(len - 1) -- this page-1 tree is handed that page-floored key (the same split position)
        from flliper.srt.managers.schedule_batch import Req

        claim = Req._compute_max_prefix_len(SimpleNamespace(return_logprob=False, logprob_start_len=0), L4)
        mr = fx.cache.match_prefix(MatchPrefixParams(key=TA0._key(fx, R4[:claim // 64 * 64])))
        assert mr.mamba_branching_seqlen in (None, 128)
    req = TA0._req(fx, R4)
    TA0._write_kv(fx, req, L4)
    if branch_at is not None:
        setattr(req, ta.BRANCH_POS_ATTR, branch_at)
    batch = SimpleNamespace(reqs=[req], req_to_token_pool=fx.pool)
    desc = ta.note_step(batch=batch, desc=None, req=req, row=0, prefix=len(req.prefix_indices), end=L4,
                        track_mask=True, main_track=L4, chunk=64, page=64, tok=IM)
    p = getattr(req, ta.BRANCH_PENDING_ATTR, None)
    if p is not None and p[1] is not None:
        TA0._state(fx, p[1], AU.S_BR)
    if desc is not None:
        desc.done.update(("gdn",))
    TA0._state(fx, req.mamba_pool_idx, TA0.S_END)
    AU._finish(fx, req, L4)
    return req


def test_r4_geometry_a_key_limited_match_gives_the_host_a_branching_point_inside_a_path(group_d):
    AU._as_worker(group_d, False)
    fx = TA0._fixture(False)
    _seed_long(fx)
    mr = _capped_branching(fx)
    assert len(mr.device_indices) == 64 and mr.mamba_branching_seqlen == 128, \
        "reached floor_64(191) = 128 on LONG's path (no divergence there), accepted COMMON's 64"


def _host_worker4(group_d, admission_match):
    AU._as_worker(group_d, False)
    host = TA0._fixture(False)
    _seed_long(host)
    hreq = _reader4(host, 128, admission_match)
    AU._as_worker(group_d, True)
    worker = TA0._fixture(False)
    _seed_long(worker)
    _reader4(worker, None, admission_match)
    from flliper.srt.pdflip import turn_anchor as ta

    assert getattr(hreq, ta.BRANCH_PENDING_ATTR, None) is None, "the branch plan was consumed by the insert"
    return host, worker


def test_r4_a_key_limit_branching_point_is_a_node_on_every_rank_because_the_admission_match_splits_there(group_d):
    """The branching point need not be a divergence: a prompt wholly on a longer path is cut by the key limit
    (``_compute_max_prefix_len``: len - 1) and branches at floor_64(len - 1). Every rank's admission match runs with
    that same key limit (page 64: it matches floor_64(len - 1), here handed page-floored to the page-1 test tree),
    and the match walk SPLITS a partly matched node at its end
    (``UnifiedRadixCache._match_prefix_helper``), so the node at B is on every rank's tree before any insert -- TP0's
    branch insert adds no node a worker lacks."""
    host, worker = _host_worker4(group_d, admission_match=True)
    assert AU._structure(host) == AU._structure(worker), (sorted(AU._structure(host)), sorted(AU._structure(worker)))
    assert (128, 64) in AU._structure(worker), "the worker's split at B came from its own admission match"


def test_r4_control_without_the_admission_match_the_host_alone_has_the_split(group_d):
    """Control for the statement above (what the tree would look like if the split came from the insert alone)."""
    host, worker = _host_worker4(group_d, admission_match=False)
    assert AU._structure(host) != AU._structure(worker)


# -- R5: the divergence instrument on the real match -----------------------------------------------------------------

@pytest.mark.parametrize("bigram", [False, True], ids=["unigram", "bigram"])
def test_r5_the_divergence_line_names_the_siblings_ids_at_reached_on_the_real_match(group_d, caplog, bigram):
    from flliper.srt.mem_cache.unified_cache_components import mamba_component as mc

    AU._as_worker(group_d, False)
    group_d.setattr(mc, "_DEEP_CENSUS_MIN_DEPTH", 1)
    mc._BRANCH_DIVERGE_N[0] = 0
    caplog.set_level(logging.INFO, mc.logger.name)
    fx = TA0._fixture(bigram)
    AU._seed(fx)
    req = TA0.Req(rid="pdflip-4-121", origin_input_text="", origin_input_ids=array("q", AU.R),
                  sampling_params=TA0.SamplingParams(temperature=0, max_new_tokens=1))
    mr = fx.cache.match_prefix(MatchPrefixParams(key=TA0._key(fx, AU.R), req=req))
    assert mr.mamba_branching_seqlen is not None
    lines = [m for m in caplog.messages if "MATCH-BRANCH-DIVERGE" in m]
    assert lines, caplog.messages
    import re

    reached = int(re.search(r"reached=(\d+)", lines[0])[1])
    assert reached == (AU.B - 1 if bigram else AU.B), lines[0]   # bigram keys count N-1 units for N ids
    sibling_at_reached = AU.SIB[reached:reached + 40]          # pdflip-4-119's answer from the divergence on
    assert f"tree_children=[{sibling_at_reached}]" in lines[0], lines[0]
    assert f"first_diff={1 if bigram else 0}" in lines[0], lines[0]   # a bigram child repeats the boundary id


# -- R6: group P ARMED (overlap off, the turn anchor's P form) keeps upstream's replacement ----------------------------

def test_r6_armed_group_p_keeps_the_upstream_replacement(d_setup):
    """The author's P edge runs with ``armed() is None`` on P (no ``disable_overlap_schedule`` in its server args), so
    it never reaches the group check in ``d_branch_rides``. Here P is armed exactly as it boots (token, extra_buffer,
    no checkpoint interval, overlap off): the branching point must still REPLACE the track there -- P has no branch
    plan (``_note_branch_step`` is D-only), a P step that kept its step-end track would lose its branching anchor."""
    from flliper.srt.pdflip import turn_anchor as ta

    mp, sb = d_setup
    mp.setenv("FLLIPER_PDFLIP_GROUP", "P")
    mp.setattr(sb, "get_server_args", lambda: SimpleNamespace(
        mamba_cache_chunk_size=64, mamba_checkpoint_interval=None, disable_overlap_schedule=True,
        enable_mamba_extra_buffer=lambda: True, enable_mamba_extra_buffer_lazy=lambda: False))
    assert ta.armed(sb.get_server_args()) == IM, "the P turn anchor is armed in this setup"
    assert not ta.d_branch_rides(sb.get_server_args(), 64, 64, False)
    req, _d, entry, _a = AU._step(sb, AU._ids_121(), PRE121, BR121)
    assert req.mamba_last_track_seqlen == BR121 and int(entry.track_seqlen) == BR121 + 1
    assert "branch" not in AU._plans(req)
