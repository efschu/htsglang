"""TAIL_FOLD_SHORT (0929): the H63 END-only hand-off also for a cut ON the page boundary.

Hermetic (no CUDA). Metal y3r (image c1c012dd5e, ...dauer09292330, P+D logs):
P published no tail part for 23 of 53 finished prompts -- every one of them
N % 64 in {1, 2, 4} (weg2-34-45/46 N=65602, weg2-0-4 N=2113, weg2-48-67..70
N=16449, ...). ``spec_for`` returns None whenever c = floor_4(N-1) sits on a
page boundary (N % page in 1..grain): the E1 rows [floor_page(c), c) are empty.
Under the fold (E1 absent, END-only parts) that None threw the END away as
well; D logged 'TAIL-READY verdict=no_parts' and ran a real extend of 1-65
tokens -- 2 tokens 0.59-0.70 s (a cold expert pass: 'Prefill rank batch,
#new-token: 2 ... gpu-ms: 567.2'), 65 tokens 1.5-2.4 s, split 64 + 1 by the
post-wake corridor chunk (ep38/ep50: four passes). Over the 17 wakes with
wake_to_last_decode >= 2 s that class held 19.9 s of EXTEND run_ms.

What these cases pin:

* P: ``fold_spec`` gives an END-only spec for N % page in 1..grain (rows 0),
  and for N % page == 1 under the bigram claim its page_prefix is the
  reader's claim floor_page(N-2) = N-65 (where P's recurrent anchor sits and
  D's read re-enters), everywhere else the ``spec_for`` geometry unchanged;
* P: ``arm_fold`` registers the capture and ``publish_rows`` writes the
  END-only part (rows [page_prefix, N), GDN after N, P's token);
* D: the rows-0 END-only part is the E2 skip on every rank (no forward, the
  prefix is not grown, END rows land on the extend's slots); the claim-form
  part (rows 64) likewise;
* the switch SGLANG_WEG2_TAIL_FOLD_SHORT=0 restores the old form (no part).

Red on c1c012dd5e (no ``fold_spec``; D 'skipped:geometry'), green with the fix.
"""

import logging
import os
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.weg2 import tail_adopt as ta  # noqa: E402
from sglang.srt.weg2 import tail_handoff as th  # noqa: E402
from test_weg2_tail_fold_h63 import (  # noqa: E402,F401  (fixtures by name)
    D_PAGE,
    D_RPI,
    FA_GIDS,
    FIRST,
    FP8,
    GDN_GIDS,
    P_RPI,
    PAGE,
    PARTS,
    RATIO,
    RID,
    SLOT,
    _e2,
    _join,
    _p_pools,
    _p_req,
    _req,
    _run_group,
    _skip_on,
    d_group,
    p_arena,
)

N_SHORT, PREFIX_SHORT = 194, 192  # N % 64 == 2: c = floor_4(193) = 192 = floor_page(c)
N_CLAIM, CLAIM, CUT_CLAIM = 193, 128, 192  # N % 64 == 1: the bigram reader claims floor_page(191) = 128


# ------------------------------------------------------------------ P: the geometry
def test_spec_for_has_no_spec_on_the_page_boundary():
    """The base form, pinned: E1 has nothing to hand over here."""
    for n in (N_SHORT, N_CLAIM, 196, 65602, 2113, 16449):
        assert th.spec_for(RID, list(range(n)), None, PAGE, RATIO) is None


def test_fold_spec_rows_zero_on_the_page_boundary():
    with envs.SGLANG_WEG2_TAIL_FOLD_SHORT.override(True):
        s = th.fold_spec(RID, list(range(N_SHORT)), None, PAGE, RATIO)
        assert (s.page_prefix, s.cut, s.n_tokens, s.rows, s.extend) == (PREFIX_SHORT, PREFIX_SHORT, N_SHORT, 0, 2)
        assert s.key == th.tail_key(list(range(N_SHORT)), PREFIX_SHORT, None)
        # y3r weg2-34-45: N=65602, D re-entered at 65600
        s = th.fold_spec(RID, list(range(65602)), None, PAGE, RATIO, claim=th.reader_claim_end(65602, PAGE, True))
        assert (s.page_prefix, s.cut, s.rows) == (65600, 65600, 0)


def test_fold_spec_mod_one_starts_at_the_readers_claim():
    """y3r weg2-0-4 N=2113: P's anchor and D's read at 2048 (the claim), not
    at c = 2112 -- the END section is [2048, 2113)."""
    with envs.SGLANG_WEG2_TAIL_FOLD_SHORT.override(True):
        s = th.fold_spec(RID, list(range(N_CLAIM)), None, PAGE, RATIO, claim=CLAIM)
        assert (s.page_prefix, s.cut, s.rows, s.extend) == (CLAIM, CUT_CLAIM, 64, 1)
        s = th.fold_spec(RID, list(range(2113)), None, PAGE, RATIO, claim=th.reader_claim_end(2113, PAGE, True))
        assert (s.page_prefix, s.cut) == (2048, 2112)


@pytest.mark.parametrize("bigram", [True, False])
def test_fold_spec_keeps_every_existing_geometry(bigram):
    """Wherever spec_for has a spec, fold_spec is the same spec (the claim
    agrees with floor_page(c) unless N-1 is a page multiple)."""
    with envs.SGLANG_WEG2_TAIL_FOLD_SHORT.override(True):
        for n in range(130, 700):
            ids = list(range(n))
            old = th.spec_for(RID, ids, None, PAGE, RATIO)
            new = th.fold_spec(RID, ids, None, PAGE, RATIO, claim=th.reader_claim_end(n, PAGE, bigram))
            if old is not None:
                assert new == old, n
            else:
                assert new is not None and new.extend >= 1 and new.page_prefix % PAGE == 0, n


def test_fold_spec_switch_off_is_spec_for():
    with envs.SGLANG_WEG2_TAIL_FOLD_SHORT.override(False):
        assert th.fold_spec(RID, list(range(N_SHORT)), None, PAGE, RATIO) is None
        assert th.fold_spec(RID, list(range(N_CLAIM)), None, PAGE, RATIO, claim=CLAIM) is None
        assert th.fold_spec(RID, list(range(241)), None, PAGE, RATIO) == th.spec_for(
            RID, list(range(241)), None, PAGE, RATIO)


# ------------------------------------------------------------------ P: arm + publish
def _claim_tree(monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_GROUP", "P")
    return SimpleNamespace(page_size=PAGE, bigram_anchor_exact=True)


def test_arm_fold_registers_the_short_tail(p_arena):
    _kv, _rp, alloc = _p_pools()
    req = _p_req(list(range(N_SHORT)), 128, N_SHORT)
    with _e2(fold=True), envs.SGLANG_WEG2_TAIL_FOLD_SHORT.override(False):
        assert th.arm_fold([req], alloc, PAGE, None) == 0  # the old form: no capture, no part
    with _e2(fold=True), envs.SGLANG_WEG2_TAIL_FOLD_SHORT.override(True):
        assert th.arm_fold([req], alloc, PAGE, None) == 1
    cap = th._CAPTURES[RID]
    assert cap.e1 is False and (cap.spec.page_prefix, cap.spec.cut) == (PREFIX_SHORT, PREFIX_SHORT)


def test_arm_fold_takes_the_claim_from_the_tree(p_arena, monkeypatch):
    _kv, _rp, alloc = _p_pools()
    req = _p_req(list(range(N_CLAIM)), 128, N_CLAIM)
    with _e2(fold=True), envs.SGLANG_WEG2_TAIL_FOLD_SHORT.override(True):
        assert th.arm_fold([req], alloc, PAGE, None, tree_cache=_claim_tree(monkeypatch)) == 1
    spec = th._CAPTURES[RID].spec
    assert (spec.page_prefix, spec.cut, spec.n_tokens) == (CLAIM, CUT_CLAIM, N_CLAIM)


def test_short_tail_publishes_an_end_only_part(p_arena, caplog):
    kv, rp, alloc = _p_pools()
    ids = list(range(N_SHORT))
    req = _p_req(ids, 0, N_SHORT)
    with _e2(fold=True), envs.SGLANG_WEG2_TAIL_FOLD_SHORT.override(True), \
            caplog.at_level(logging.INFO, logger="sglang.srt.weg2.tail_handoff"):
        assert th.arm_fold([req], alloc, PAGE, None) == 1
        rp.mamba_pool.mamba_cache.temporal[:, SLOT] += 1.0  # the last chunk runs, tail included
        kv_indices = torch.arange(N_SHORT) + PAGE
        th.publish_rows(req, kv_indices, alloc, "pp0-1", req_to_token_pool=rp, n_parts=3)
        _join("weg2-tail-publish")
    (h,) = th.headers_for(RID)
    assert h.e1 is False and h.spec.rows == 0
    e = h.end
    assert (e.first_token, e.rows, e.groups, e.ring_rows) == (FIRST, 2, 0, 2)
    assert e.key == th.tail_key(ids, N_SHORT, None)
    bundle = th.verify_part(h)
    assert bundle is not None and th.end_digest_refusal(h, bundle) == ""
    rows = kv_indices[PREFIX_SHORT:N_SHORT]
    for gid, local in kv.full_attention_layer_id_mapping.items():
        assert torch.equal(bundle["end"]["fa"][gid][0], kv.full_kv_pool.k_buffer[local][rows])
    for gid, local in rp.mamba_map.items():
        assert torch.equal(bundle["end"]["gdn"][gid][0][0], rp.mamba_pool.mamba_cache.temporal[local, SLOT])
    assert f"WEG2-TAIL-PUBLISH rid={RID} page_prefix={PREFIX_SHORT} tail_rows=2 state_at={N_SHORT}" in caplog.text


# ------------------------------------------------------------------ D side
def _payload(rows, groups, ring_rows, seed=29):
    g = torch.Generator().manual_seed(seed)
    fa = {gid: (torch.randn(rows, 2, 8, generator=g).to(FP8), torch.randn(rows, 2, 8, generator=g).to(FP8),
                torch.randn(groups, 1, 4, generator=g).to(torch.bfloat16)) for gid in FA_GIDS}
    gdn = {gid: (torch.randn(1, 2, 4, 4, generator=g), torch.randn(1, 6, 3, generator=g).to(torch.bfloat16))
           for gid in GDN_GIDS}
    ring = {gid: (torch.randn(ring_rows, 1, 4, generator=g).to(torch.bfloat16),) for gid in FA_GIDS}
    rope = torch.full((ring_rows, 3), 7, dtype=torch.int64)
    return fa, gdn, ring, rope


def _publish(n, claim=None):
    ids = list(range(n))
    with envs.SGLANG_WEG2_TAIL_FOLD_SHORT.override(True):
        spec = th.fold_spec(RID, ids, None, PAGE, RATIO, claim=claim)
    rows, groups, ring_rows = th.end_geometry(spec, RATIO)
    fa, gdn, ring, rope = _payload(rows, groups, ring_rows)
    for part, (fl, gl) in PARTS.items():
        end = th.EndPayload(first_token=FIRST, key=th.tail_key(ids, n, None), rows=rows, groups=groups,
                            ring_rows=ring_rows, fa={g: fa[g] for g in fl}, gdn={g: gdn[g] for g in gl},
                            ring={g: ring[g] for g in fl}, rope=rope)
        th.write_part(spec, part, {}, {}, end=end, n_parts=len(PARTS), e1=False)
    return spec, fa, gdn


def _d_req(n, prefix):
    ids = list(range(n))
    return _req(ids, full_untruncated_fill_ids=list(ids),
                prefix_indices=torch.arange(PAGE, PAGE + prefix, dtype=torch.int64))


def _prepare(rank, n):
    req = rank.req
    k = len(req.prefix_indices)
    rank.rp.req_to_token[D_RPI, :k] = req.prefix_indices.to(torch.int32)
    last = int(req.prefix_indices[-1])
    rank.rp.req_to_token[D_RPI, k:n] = torch.arange(last + 1, last + 1 + n - k, dtype=torch.int32)


@pytest.mark.parametrize("n, claim, prefix, cut", [
    (N_SHORT, None, PREFIX_SHORT, PREFIX_SHORT),  # rows 0: nothing below the cut
    (N_CLAIM, CLAIM, CLAIM, CUT_CLAIM),  # N % 64 == 1: the claim form, rows 64
])
def test_d_skips_the_short_tail_on_every_rank(d_group, caplog, n, claim, prefix, cut):
    spec, fa, gdn = _publish(n, claim)
    assert (spec.page_prefix, spec.cut) == (prefix, cut)
    reqs = [_d_req(n, prefix) for _ in d_group]
    with caplog.at_level(logging.INFO, logger="sglang.srt.weg2.tail_adopt"):
        votes, plans = _run_group(d_group, reqs=reqs)
        assert "skipped:geometry" not in caplog.text
        assert votes == [2, 2, 2] and all(p is not None and p.skip for p in plans)
        for r in d_group:
            assert len(r.req.prefix_indices) == cut  # rows 0: the prefix is not grown
            _prepare(r, n)
        tokens = [_skip_on(r) for r in d_group]
    assert tokens == [[FIRST]] * 3
    tp0 = d_group[0]
    slots = tp0.rp.req_to_token[D_RPI, prefix:n].to(torch.int64)
    full = tp0.kv.full_kv_pool
    for gid, local in tp0.kv.full_attention_layer_id_mapping.items():
        assert torch.equal(full.k_buffer[local][slots].view(torch.uint8), fa[gid][0].view(torch.uint8))
        assert torch.equal(full.v_buffer[local][slots].view(torch.uint8), fa[gid][1].view(torch.uint8))
    cache = tp0.rp.mamba_pool.mamba_cache
    for gid, local in tp0.rp.mamba_map.items():
        assert torch.equal(cache.temporal[local, SLOT], gdn[gid][0][0])
    assert f"WEG2-TAIL-ADOPT rid={RID} page_prefix={prefix} tail_rows={n - prefix} state_at={n} extend=0 " \
           f"fa_rows_written={n - prefix} fa_layers=3 gdn_layers=9 digest=match" in caplog.text


def test_d_refuses_rows_zero_for_an_e1_part():
    """E1 with no row below the cut stays refused: only END-only parts may
    carry rows 0."""
    with envs.SGLANG_WEG2_TAIL_FOLD_SHORT.override(True):
        spec = th.fold_spec(RID, list(range(N_SHORT)), None, PAGE, RATIO)
    ids = list(range(N_SHORT))
    assert ta.uniform_refusal(spec, ids, N_SHORT, None, PREFIX_SHORT) == "geometry"
    assert ta.uniform_refusal(spec, ids, N_SHORT, None, PREFIX_SHORT, end_only=True) == ""
