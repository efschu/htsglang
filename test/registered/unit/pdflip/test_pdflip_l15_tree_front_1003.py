# SPDX-License-Identifier: Apache-2.0
"""L15-TREE-FRONT: a follow-up turn of a FINISHED request takes a held tree tip.

The tree-candidate holds (L15-TREE-CAND) are published under rids
``tree:<digest>``, not client rids, so the front's rid-keyed hot hint
(``_sess_prev`` -> prev rid live on D) never named them. The hint now carries
the whole prompt (``tree`` flag) and every P stage picks the held tip the
prompt extends from D's published descriptor: same descriptor, same ids, same
answer on every stage (no reserve, no collective).
"""

from __future__ import annotations

import inspect
from array import array
from types import SimpleNamespace

from flliper.srt.pdflip import l15_bind, l15_share_take
from flliper.srt.pdflip import l15_share_admit as sa
from flliper.srt.pdflip import l15_tree_cand as tc

TIP_A = [1, 2, 3, 4]
TIP_B = [1, 2, 9, 9, 9, 9]


def _span(toks, rid=None, **kw):
    return {"rid": rid or tc.RID_PREFIX + tc.digest_of(toks), "depth": len(toks), **kw}


def _d0(*spans):
    return {"cap0": [], "prefix": [0, 2, 3], "spans": list(spans)}


# -- the pure match -------------------------------------------------------------

def test_match_tip_picks_the_tip_the_prompt_extends():
    spans = [_span(TIP_A), _span(TIP_B), {"rid": "live-1", "depth": 4}]
    got, tips = tc.match_tip(spans, TIP_A + [7, 7, 7])
    # plain tree: raw tokens == KV slots (depth)
    assert got == (tc.RID_PREFIX + tc.digest_of(TIP_A), 4, 4) and len(tips) == 2
    got, _ = tc.match_tip(spans, TIP_B + [5])
    assert got[1] == 6 and got[2] == 6


def test_match_tip_longest_wins_and_diverging_or_short_prompts_miss():
    spans = [_span(TIP_A[:2]), _span(TIP_A)]
    assert tc.match_tip(spans, TIP_A + [8])[0][1] == 4
    # the prompt leaves the tip before its end: no whole-tip prefix, no match
    got, tips = tc.match_tip([_span(TIP_A)], [1, 2, 3, 99, 5])
    assert got is None and len(tips) == 1
    # the prompt is shorter than the tip
    assert tc.match_tip([_span(TIP_A)], [1, 2, 3])[0] is None
    # no tree spans at all (live rids only)
    assert tc.match_tip([{"rid": "r1", "depth": 4}], TIP_A)[0] is None
    assert tc.match_tip([], TIP_A) == (None, [])


def test_match_tip_is_deterministic_for_every_stage():
    spans = [_span(TIP_B), _span(TIP_A)]
    ids = TIP_A + [3, 3]
    assert len({tc.match_tip(spans, ids)[0] for _ in range(5)}) == 1
    # span order in the descriptor does not change the answer
    assert tc.match_tip(list(reversed(spans)), ids)[0] == tc.match_tip(spans, ids)[0]


# -- the P-stage resolution -----------------------------------------------------

def test_resolve_passes_a_plain_hint_through_and_resolves_a_tree_hint():
    logs = []
    plain = {"prev_rid": "r1", "n": 30}
    assert sa.resolve_tree_hint(plain, _d0(), [1], "r2", logs.append) is plain
    hint = {"prev_rid": sa.TREE_PREV, "n": 7, "tree": True}
    out = sa.resolve_tree_hint(hint, _d0(_span(TIP_A)), TIP_A + [5, 6, 7], "r3", logs.append)
    assert out["prev_rid"] == tc.RID_PREFIX + tc.digest_of(TIP_A) and out["n"] == 4
    assert out["raw_extra"] == 0
    assert any("rid=r3 tree-match tip=tree:" in x and "depth=4 raw=4" in x for x in logs)


def test_resolve_names_the_miss():
    logs = []
    hint = {"prev_rid": sa.TREE_PREV, "n": 5, "tree": True}
    assert sa.resolve_tree_hint(hint, _d0(_span(TIP_A)), [1, 2, 3, 9, 9], "r4", logs.append) is None
    assert any("rid=r4 tree-miss tips=1 depths=[4] prompt=5" in x for x in logs)
    assert sa.resolve_tree_hint(hint, None, [1], "r5", logs.append) is None


def test_hint_file_carries_the_tree_flag_and_ids(tmp_path):
    d = str(tmp_path)
    sa.write_hot_hint(d, "r6", sa.TREE_PREV, 3, ids=[11, 12, 13], tree=True)
    h = sa.hot_hint(d, "r6")
    assert h["tree"] is True and h["n"] == 3 and sa.hint_ids(h) == [11, 12, 13]
    sa.write_hot_hint(d, "r7", "r4", 3, ids=[1, 2, 3])
    assert "tree" not in sa.hot_hint(d, "r7")


def _wake_fixture(monkeypatch, d0):
    geom = SimpleNamespace(dev=0, n_d=2, kv_alloc=None, mamba_alloc=None,
                           tree_cache=None, p_buffers={1: None}, spec=None,
                           stage_linear=(0, 2), p_temporal=None, p_conv=None,
                           req_to_token_pool=None)
    monkeypatch.setattr(sa, "_first_share", lambda dd, f: d0)
    monkeypatch.setattr(sa, "stage_geometry", lambda sched, d: geom)
    monkeypatch.setattr(l15_bind, "live_host_pools", lambda t: (None, None))
    seen = []

    def fake_admit(**kw):
        seen.append((kw["rid"], dict(kw["hint"]), kw["token_ids"]))
        kw["verdict"](True)
        kw["stats"].update(card_bytes=1, l2_bytes=0, anchor_card_bytes=0)
        return None

    monkeypatch.setattr(sa, "admit", fake_admit)
    return SimpleNamespace(pp_rank=0, pp_size=1, tp_worker=None), seen


def test_take_all_at_wake_adopts_a_follow_up_of_a_finished_request(monkeypatch, tmp_path):
    d = str(tmp_path)
    sched, seen = _wake_fixture(monkeypatch, _d0(_span(TIP_A)))
    prompt = TIP_A + [20, 21]
    sa.write_hot_hint(d, "hit", sa.TREE_PREV, len(prompt), ids=prompt, tree=True)
    sa.write_hot_hint(d, "miss", sa.TREE_PREV, 4, ids=[1, 2, 3, 99], tree=True)
    logs = []
    n = sa.take_all_at_wake(sched, {"FLLIPER_PDFLIP_L15_SHARE_DIR": d}, logs.append)
    assert n == 1 and [s[0] for s in seen] == ["hit"]
    # admit saw the RESOLVED tip (rid + depth), with the whole prompt's ids
    assert seen[0][1]["prev_rid"] == tc.RID_PREFIX + tc.digest_of(TIP_A)
    assert seen[0][1]["n"] == 4 and seen[0][2] == prompt
    assert any("rid=hit tree-match tip=tree:" in x for x in logs)
    assert any("rid=hit at=wake result=adopted n=4" in x for x in logs)
    assert any("rid=miss tree-miss" in x for x in logs)
    assert any("rid=miss at=wake result=fallback" in x and "fallback=tree:" in x for x in logs)


def test_a_rid_hint_and_a_tree_hint_coexist_in_one_wake(monkeypatch, tmp_path):
    d = str(tmp_path)
    sched, seen = _wake_fixture(monkeypatch, _d0(_span(TIP_A)))
    sa.write_hot_hint(d, "live", "r1", 2, ids=[5, 6])
    sa.write_hot_hint(d, "tree", sa.TREE_PREV, 5, ids=TIP_A + [0], tree=True)
    n = sa.take_all_at_wake(sched, {"FLLIPER_PDFLIP_L15_SHARE_DIR": d}, lambda m: None)
    assert n == 2
    byrid = {s[0]: s[1] for s in seen}
    assert byrid["live"]["prev_rid"] == "r1" and byrid["tree"]["prev_rid"].startswith("tree:")


def test_admission_mode_resolves_from_the_requests_own_ids(monkeypatch, tmp_path):
    d = str(tmp_path)
    sa.write_hot_hint(d, "r9", sa.TREE_PREV, 6, tree=True)
    d0 = _d0(_span(TIP_A))
    monkeypatch.setattr(sa, "_first_share", lambda dd, f: d0)
    votes = []
    monkeypatch.setattr(sa, "stage_verdict",
                        lambda directory, rid, stage, n_stages, ok, tmo: votes.append(ok) or "fallback")
    geom = SimpleNamespace(dev=0)
    monkeypatch.setattr(sa, "stage_geometry", lambda sched, dd: geom)
    monkeypatch.setattr(l15_bind, "live_host_pools", lambda t: (None, None))
    seen = []

    def fake_admit(**kw):
        seen.append(dict(kw["hint"]))
        return None

    monkeypatch.setattr(sa, "admit", fake_admit)
    env = {"FLLIPER_PDFLIP_L15_SHARE_DIR": d, "FLLIPER_PDFLIP_L15_HOT_AT_WAKE": "0"}
    geom.__dict__.update(n_d=2, kv_alloc=None, mamba_alloc=None, tree_cache=None,
                         p_buffers={}, spec=None, stage_linear=(0, 2), p_temporal=None,
                         p_conv=None, req_to_token_pool=None)
    sched = SimpleNamespace(pp_rank=0, pp_size=1, tp_worker=None)
    logs = []
    # a prompt that diverges inside the tip: counter-marker, a "fail" vote, no take
    miss = SimpleNamespace(rid="r9", origin_input_ids=[1, 2, 3, 77, 5, 6])
    why = sa.admit_for_sched(sched, miss, env, logs.append)
    assert why and "tree" in why and votes == [False] and not seen
    assert any("rid=r9 tree-miss" in x for x in logs)
    # a prompt that extends the tip: the stage goes on with the resolved tip
    hit = SimpleNamespace(rid="r9", origin_input_ids=TIP_A + [5, 6])
    sa.admit_for_sched(sched, hit, env, logs.append)
    assert seen and seen[0]["prev_rid"] == tc.RID_PREFIX + tc.digest_of(TIP_A)
    assert any("rid=r9 tree-match" in x for x in logs)


# -- the front: where the hint is written ---------------------------------------

def test_front_wires_the_tree_hint_at_the_flip_and_at_leg1():
    from flliper.srt.pdflip import front

    src = inspect.getsource(front)
    # ids of a follow-up are kept, bounded, behind the tree switch family
    i = src.index("L15-TREE-FRONT: a follow-up of a session's earlier turn keeps")
    blk = src[i:i + 900]
    assert "_l15_tc.env_on(os.environ)" in blk
    assert "_tree_ids" in blk and "len(tids) > 16" in blk
    # flip begin: only requests whose previous rid is NOT live on D
    j = src.index("L15-TREE-FRONT: a queued follow-up whose session's previous")
    blk = src[j:j + 2600]
    assert "str(_pv[0]) in _dl" in blk and "tree=True" in blk
    assert "HOT-HANDOVER-HINT at=wake tree n=%d" in blk
    assert "self._l15_wake_hints.append(_r)" in blk      # reaped with the others
    # leg 1 (admission mode)
    k = src.index("HOT-HANDOVER-HINT rid=%s tree n=%d")
    assert "tree=True" in src[k - 600:k]


# -- bigram keys (DFLASH + FLLIPER_HICACHE_BIGRAM_KEYS=1, the 27B boot form) -----
# A bigram tree files N raw tokens as N-1 units: the hold's span depth counts
# KV slots (N-1) while the tip digest runs over the N raw tokens = the
# request's own token sequence (L15-UNHOLDABLE: chain_tokens keeps the
# boundary token once).

def _bigram_tip(ids):
    """D side: a finished request in a REAL bigram UnifiedRadixCache; returns
    (fixture, tip pseudo req, the published-span view of the tip)."""
    import test_pdflip_end_anchor_exact_probe_0928 as ea

    fx = ea._fixture(exact=True, bigram=True)
    ea._finish(fx, ea._req(fx, ids, "d1"))
    reqs = tc.build(fx.cache, lambda v: [v], 0, {}, None)
    assert len(reqs) == 1
    slots, _node, _anchor = l15_bind.match_parked(reqs[0], fx.cache)
    return fx, reqs[0], {"rid": reqs[0].rid, "depth": len(slots)}


def test_bigram_tip_digest_is_the_requests_token_sequence_and_depth_is_one_less():
    ids = list(range(1000, 1024))
    _fx, req, span = _bigram_tip(ids)
    assert span["depth"] == len(ids) - 1                  # KV slots (units)
    assert req.rid == tc.RID_PREFIX + tc.digest_of(ids)   # raw tokens, boundary once
    got, _ = tc.match_tip([span], ids + [7, 8, 9])        # the follow-up turn
    assert got == (req.rid, len(ids) - 1, len(ids))
    # a prompt that leaves the finished request's tokens early does not match
    assert tc.match_tip([span], ids[:10] + [5] * 20)[0] is None


def test_bigram_tip_is_adopted_by_p_and_the_extra_raw_token_row_is_given_back(monkeypatch):
    import test_pdflip_end_anchor_exact_probe_0928 as ea

    from flliper.srt.mem_cache.base_prefix_cache import MatchPrefixParams
    from flliper.srt.mem_cache.radix_cache import RadixKey

    ids = list(range(1000, 1024))
    _dfx, req, span = _bigram_tip(ids)
    prompt = ids + [7, 8, 9]
    logs = []
    hint = sa.resolve_tree_hint({"prev_rid": sa.TREE_PREV, "n": len(prompt), "tree": True},
                                _d0(span), prompt, "f1", logs.append)
    assert hint["n"] == len(ids) - 1 and hint["raw_extra"] == 1
    pfx = ea._fixture(exact=True, bigram=True)            # P's empty bigram tree
    kv0 = pfx.allocator.available_size()
    mb0 = pfx.pool.mamba_allocator.available_size()
    taken = {}

    def fake_kv(shares, *, rid, n, stage_layers, p_buffers, p_rows, map_extent,
                skip_ranks=()):
        taken["rid"], taken["n"], taken["rows"] = rid, n, list(p_rows)
        return n

    monkeypatch.setattr(l15_share_take, "take_kv", fake_kv)
    monkeypatch.setattr(l15_share_take, "take_anchor", lambda *a, **k: 64)
    why = sa.admit(rid="f1", token_ids=prompt, hint=hint,
                   fetch=lambda r: ({"prefix": [0, 2, 3], "spans": [span]}, []),
                   n_d_ranks=2, kv_alloc=pfx.allocator, mamba_alloc=pfx.pool.mamba_allocator,
                   tree_cache=pfx.cache, stage_att_layers=[1], p_buffers={}, spec=None,
                   stage_linear=(0, 2), p_temporal=None, p_conv=None, map_extent=None,
                   log=logs.append)
    assert why is None, why
    # the copy fills exactly the KV slots; D's span is named by the tip rid
    assert taken["rid"] == req.rid and taken["n"] == len(ids) - 1
    assert len(taken["rows"]) == len(ids) - 1
    # the follow-up's match is a device hit over the whole tip
    res = pfx.cache.match_prefix(MatchPrefixParams(
        key=RadixKey(array("q", prompt), is_bigram=True), cow_mamba=False))
    assert len(res.device_indices) == len(ids) - 1
    # no leak: the extra raw-token row went back, only the slots + 1 anchor are held
    assert pfx.allocator.available_size() == kv0 - (len(ids) - 1)
    assert pfx.pool.mamba_allocator.available_size() == mb0 - 1
    assert any(x.startswith("HOT-HANDOVER rid=f1 from=tree:") for x in logs)


def test_bigram_tip_on_a_non_bigram_tree_is_refused_not_adopted():
    import torch

    kv = SimpleNamespace(free_pages=torch.arange(1, 60))
    mb = SimpleNamespace(free_slots=torch.arange(1, 5))
    why = sa.admit(rid="f2", token_ids=list(range(40)),
                   hint={"prev_rid": "tree:x", "n": 20, "raw_extra": 1},
                   fetch=lambda r: ({"prefix": [0, 2, 3], "spans": []}, []), n_d_ranks=2,
                   kv_alloc=kv, mamba_alloc=mb, tree_cache=SimpleNamespace(),
                   stage_att_layers=[1], p_buffers={}, spec=None, stage_linear=(0, 2),
                   p_temporal=None, p_conv=None, map_extent=None, log=lambda m: None)
    assert why and "non-bigram" in why
