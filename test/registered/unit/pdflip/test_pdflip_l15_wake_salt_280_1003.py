# SPDX-License-Identifier: Apache-2.0
"""L15-WAKE-SALT (desk 280): the wake-mode hot hint carries the request's
extra_key; a tip held under cache_salt A is never adopted into P's tree for a
request of salt B / no salt.

Before: the wake hint carried token ids only, so ``resolve_tree_hint`` ->
``match_tip`` ran with ANY_EXTRA_KEY and matched a salted tip by tokens alone,
and ``l15_p_adopt.adopt`` was called without extra_key (the prefix landed in
P's tree UNSALTED) -- cross-salt prefix sharing, a breach of the cache_salt
contract (tenant isolation).

Now: the hint's ``extra_key`` field (front: write_hot_hint) is pinned like in
admission mode; an old hint without the field means "unsalted only"; adopt
inserts under the tip's extra_key (the salt namespace survives on P).

Multimodal: the pad tokens carry the mm hash (schedule_batch.py
MultimodalDataItem.set_pad_value -> _compute_pad_value, pad = 1_000_000 +
hash % 2^30), so two images never share a token sequence; the tip digest runs
over those tokens.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import tempfile
from array import array
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from flliper.srt.pdflip import l15_share_admit as sa  # noqa: E402
from flliper.srt.pdflip import l15_share_publish as sp  # noqa: E402
from flliper.srt.pdflip import l15_share_take  # noqa: E402
from flliper.srt.pdflip import l15_tree_cand as tc  # noqa: E402

TIP = [1, 2, 3, 4]
EK = "tenant-A"


def _cand(toks, ek=None):
    kw = {}
    if "match_key" in tc.TreeCand.__dataclass_fields__:
        kw["match_key"] = tc.digest_of(toks, None)
    return tc.TreeCand(digest=tc.digest_of(toks, ek), n_tokens=len(toks), last_access=1.0,
                       tokens=tuple(toks), extra_key=ek, **kw)


def _span(toks, ek=None):
    req = tc.pseudo_req(_cand(toks, ek), 0, 1)
    hs = SimpleNamespace(rid=req.rid, depth=len(toks), slots=(), anchor_slot=1, l2_slots=(),
                         l2_gens=(), anchor_l2_slot=-1, anchor_l2_gen=-1)
    return sp.build_descriptor(epoch=1, rank=0, prefix=[0, 1], bases=[], spans=[hs])["spans"][0]


def _d0(*spans):
    return {"cap0": [], "prefix": [0, 2, 3], "spans": list(spans)}


def _wake_hint(**kw):
    """The hint file a front writes at the D->P flip, read back as P reads it."""
    with tempfile.TemporaryDirectory() as d:
        sa.write_hot_hint(d, "q1", sa.TREE_PREV, 6, ids=TIP + [9, 9], tree=True, **kw)
        return sa.hot_hint(d, "q1")


def _old_hint():
    """A pre-280 hint file: tree + ids, no extra_key field at all."""
    with tempfile.TemporaryDirectory() as d:
        sa.write_hot_hint(d, "q1", sa.TREE_PREV, 6, ids=TIP + [9, 9], tree=True)
        body = json.load(open(os.path.join(d, "hot.q1.json")))
        assert "extra_key" not in body
        return sa.hot_hint(d, "q1")


def _resolve(hint, span):
    logs = []
    out = sa.resolve_tree_hint(hint, _d0(span), TIP + [9, 9], "q1", logs.append)
    return out, logs


# ------------------------------------------------ (a) salted tip, unsalted hint

def test_a_salted_tip_unsalted_wake_hint_does_not_match():
    out, logs = _resolve(_old_hint(), _span(TIP, EK))
    assert out is None
    assert any("tree-miss" in x for x in logs)


def test_a_salted_tip_other_salt_wake_hint_does_not_match():
    out, _ = _resolve(_wake_hint(extra_key="tenant-B"), _span(TIP, EK))
    assert out is None


def test_a_match_tip_default_is_unsalted_only_never_any():
    assert tc.match_tip([_span(TIP, EK)], TIP + [9])[0] is None
    assert tc.match_tip([_span(TIP, None)], TIP + [9])[0] is not None
    # the diagnostics opt-in still exists
    assert tc.match_tip([_span(TIP, EK)], TIP + [9], extra_key=tc.ANY_EXTRA_KEY)[0] is not None


# ------------------------------------------------ (b) same salt: match + adopt

def test_b_same_salt_wake_hint_matches_and_carries_the_key_on():
    hint = _wake_hint(extra_key=EK)
    assert sa.hint_extra_key(hint) == EK
    out, logs = _resolve(hint, _span(TIP, EK))
    assert out is not None and out["n"] == 4
    assert sa.hint_extra_key(out) == EK
    assert any("tree-match" in x for x in logs)


def _bigram_tip(ids, ek):
    import test_pdflip_end_anchor_exact_probe_0928 as ea
    from flliper.srt.pdflip import l15_bind

    fx = ea._fixture(exact=True, bigram=True)
    req = ea._req(fx, ids, "d1")
    req.extra_key = ek
    ea._finish(fx, req)
    (pr,) = tc.build(fx.cache, lambda v: [v], 0, {}, None)
    slots, _n, _a = l15_bind.match_parked(pr, fx.cache)
    hs = SimpleNamespace(rid=pr.rid, depth=len(slots), slots=(), anchor_slot=1, l2_slots=(),
                         l2_gens=(), anchor_l2_slot=-1, anchor_l2_gen=-1)
    span = sp.build_descriptor(epoch=1, rank=0, prefix=[0, 1], bases=[], spans=[hs])["spans"][0]
    return pr, span


def _adopt_on_p(monkeypatch, hint, span, prompt, rid="w1"):
    import test_pdflip_end_anchor_exact_probe_0928 as ea

    pfx = ea._fixture(exact=True, bigram=True)
    monkeypatch.setattr(l15_share_take, "take_kv", lambda shares, *, rid, n, **k: n)
    monkeypatch.setattr(l15_share_take, "take_anchor", lambda *a, **k: 64)
    logs = []
    why = sa.admit(rid=rid, token_ids=prompt, hint=hint,
                   fetch=lambda r: ({"prefix": [0, 2, 3], "spans": [span]}, []),
                   n_d_ranks=2, kv_alloc=pfx.allocator, mamba_alloc=pfx.pool.mamba_allocator,
                   tree_cache=pfx.cache, stage_att_layers=[1], p_buffers={}, spec=None,
                   stage_linear=(0, 2), p_temporal=None, p_conv=None, map_extent=None,
                   log=logs.append)
    return pfx, why


def _p_hit(pfx, prompt, ek):
    from flliper.srt.mem_cache.base_prefix_cache import MatchPrefixParams
    from flliper.srt.mem_cache.radix_cache import RadixKey

    res = pfx.cache.match_prefix(MatchPrefixParams(
        key=RadixKey(array("q", prompt), extra_key=ek, is_bigram=True), cow_mamba=False))
    return len(res.device_indices)


def test_b_same_salt_tip_is_adopted_under_that_salt_only(monkeypatch):
    ids = list(range(3000, 3024))
    prompt = ids + [7, 8, 9]
    _pr, span = _bigram_tip(ids, EK)
    hint0 = {"prev_rid": sa.TREE_PREV, "n": len(prompt), "tree": True, "extra_key": EK}
    hint = sa.resolve_tree_hint(hint0, _d0(span), prompt, "w1", lambda m: None)
    assert hint is not None and sa.hint_extra_key(hint) == EK
    pfx, why = _adopt_on_p(monkeypatch, hint, span, prompt)
    assert why is None, why
    assert _p_hit(pfx, prompt, EK) == len(ids) - 1      # the salt hits
    assert _p_hit(pfx, prompt, None) == 0               # no salt misses
    assert _p_hit(pfx, prompt, "tenant-B") == 0         # another salt misses


def test_b_unsalted_tip_is_adopted_unsalted(monkeypatch):
    ids = list(range(3100, 3124))
    prompt = ids + [7]
    _pr, span = _bigram_tip(ids, None)
    hint = sa.resolve_tree_hint({"prev_rid": sa.TREE_PREV, "n": len(prompt), "tree": True},
                                _d0(span), prompt, "w2", lambda m: None)
    assert hint is not None
    pfx, why = _adopt_on_p(monkeypatch, hint, span, prompt, rid="w2")
    assert why is None, why
    assert _p_hit(pfx, prompt, None) == len(ids) - 1
    assert _p_hit(pfx, prompt, EK) == 0


# ------------------------------------------------ (c) old hint: unsalted only

def test_c_old_hint_matches_an_unsalted_tip_only():
    hint = _old_hint()
    assert sa.hint_extra_key(hint) is None
    out, _ = _resolve(hint, _span(TIP, None))
    assert out is not None and out["n"] == 4
    assert sa.hint_extra_key(out) is None
    out, _ = _resolve(hint, _span(TIP, EK))
    assert out is None


def test_c_two_tips_same_tokens_old_hint_picks_the_unsalted():
    spans = [_span(TIP, EK), _span(TIP, None)]
    out = sa.resolve_tree_hint(_old_hint(), _d0(*spans), TIP + [9, 9], "q1", lambda m: None)
    assert out is not None and "@" not in out["prev_rid"]


# ------------------------------------------------ admission mode / non-tree

def test_admission_mode_request_key_is_authoritative_over_the_file():
    hint = _wake_hint(extra_key="forged")
    span = _span(TIP, EK)
    out = sa.resolve_tree_hint(hint, _d0(span), TIP + [9, 9], "q", lambda m: None, extra_key=EK)
    assert out is not None and sa.hint_extra_key(out) == EK
    assert sa.resolve_tree_hint(hint, _d0(span), TIP + [9, 9], "q", lambda m: None,
                                extra_key=None) is None


def test_non_tree_hint_gets_the_requests_key_pinned_in_admission_mode():
    plain = {"prev_rid": "r0", "n": 4}
    out = sa.resolve_tree_hint(plain, None, TIP, "q", lambda m: None, extra_key=EK)
    assert sa.hint_extra_key(out) == EK
    out = sa.resolve_tree_hint(plain, None, TIP, "q", lambda m: None, extra_key=None)
    assert sa.hint_extra_key(out) is None


# ------------------------------------------------ front side

def test_payload_extra_key_mirrors_the_serving_layer():
    assert sa.payload_extra_key({"messages": []}) == (True, None)
    assert sa.payload_extra_key({"cache_salt": "s"}) == (True, "s")
    assert sa.payload_extra_key({"cache_salt": "s", "extra_key": "e"}) == (True, "se")
    assert sa.payload_extra_key({"lora_path": "x"})[0] is False     # fail closed
    assert sa.payload_extra_key({"model": "base:adapter"})[0] is False
    assert sa.payload_extra_key({"cache_salt": 5})[0] is False


def test_hint_file_roundtrips_the_key_and_omits_it_when_none():
    with tempfile.TemporaryDirectory() as d:
        sa.write_hot_hint(d, "a", sa.TREE_PREV, 3, ids=[1, 2, 3], tree=True, extra_key=EK)
        sa.write_hot_hint(d, "b", sa.TREE_PREV, 3, ids=[1, 2, 3], tree=True)
        assert sa.hint_extra_key(sa.hot_hint(d, "a")) == EK
        assert "extra_key" not in json.load(open(os.path.join(d, "hot.b.json")))


def test_front_hints_carry_the_key_and_gate_on_the_held_key():
    import inspect

    from flliper.srt.pdflip import front

    src = inspect.getsource(front)
    assert src.count("extra_key=_ek") == 4          # the four hint writers
    assert "_l15_hint_ek" in src and "_l15_ek_note(rid, payload)" in src
    ek = front.Front._l15_hint_ek
    fake = SimpleNamespace(_l15_ek={"a": (True, "X"), "b": (True, "X"),
                                    "c": (True, "Y"), "l": (False, None)})
    assert ek(fake, "a") == (True, "X")
    assert ek(fake, "a", "b") == (True, "X")        # same salt as the held request
    assert ek(fake, "a", "c") == (False, None)      # another salt: no hint
    assert ek(fake, "l") == (False, None)           # LoRA: unknowable
    assert ek(fake, "zz") == (False, None)          # never noted
