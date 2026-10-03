# SPDX-License-Identifier: Apache-2.0
"""L15-EXTRAKEY (desk 240): a held tree tip whose key carries an extra_key
(cache salt, multimodal hash) must be matchable by the front.

qwen review of 130 on the 27B standard boot (03.10.): ``local_candidates``
stamps each tip digest WITH the node's extra_key (``digest_of(toks, ek)``), so
the rid is ``tree:<digest>``; the only front lookup, ``match_tip``, recomputed
``digest_of(arr[:raw], None)`` (l15_share_admit.resolve_tree_hint passed no
extra_key), so a held tip with a non-None extra_key could never match.

Fix: every span publishes a ``match_key`` = the digest of the tip's raw tokens
WITHOUT the extra_key (build_descriptor), ``match_tip`` compares against it;
the rid / agreement digest and the pseudo_req's extra_key (-> the bind) are
unchanged. Rank-uniform (a pure function of the tokens), no collective, no
reserve.

Second point (proved here, not fixed): ``match_tip`` tries raw = depth and
depth + 1 only, and no other raw length exists (see
``test_raw_length_is_units_or_units_plus_one``).
"""

from __future__ import annotations

import pathlib
import sys
from array import array
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import pytest  # noqa: E402

from sglang.srt.weg2 import l15_share_admit as sa  # noqa: E402
from sglang.srt.weg2 import l15_share_publish as sp  # noqa: E402
from sglang.srt.weg2 import l15_tree_cand as tc  # noqa: E402

TIP = [1, 2, 3, 4]
EK = "tenant-salt"


def _cand(toks, ek=None):
    kw = {}
    if "match_key" in tc.TreeCand.__dataclass_fields__:
        kw["match_key"] = tc.digest_of(toks, None)
    return tc.TreeCand(
        digest=tc.digest_of(toks, ek), n_tokens=len(toks), last_access=1.0,
        tokens=tuple(toks), extra_key=ek, **kw)


def _hold_span(rid, depth):
    return SimpleNamespace(rid=rid, depth=depth, slots=(), anchor_slot=1, l2_slots=(),
                           l2_gens=(), anchor_l2_slot=-1, anchor_l2_gen=-1)


def _published(c, depth=None):
    """The span a P stage reads: pseudo_req rid -> HoldSpan -> descriptor."""
    req = tc.pseudo_req(c, 0, 1)
    hs = _hold_span(req.rid, depth or c.n_tokens)
    return sp.build_descriptor(epoch=1, rank=0, prefix=[0, 1], bases=[], spans=[hs])["spans"][0]


# ------------------------------------------------------------ the bug itself

def test_salted_tip_matches_the_prompt_that_extends_it():
    span = _published(_cand(TIP, EK))
    got, tips = tc.match_tip([span], TIP + [9, 9])
    assert got is not None and got[1] == 4 and got[2] == 4 and len(tips) == 1


def test_salted_tip_matches_through_the_p_stage_resolution():
    span = _published(_cand(TIP, EK))
    d0 = {"spans": [span]}
    logs = []
    out = sa.resolve_tree_hint({"tree": True, "prev_rid": "tree:", "n": 6}, d0,
                               TIP + [9, 9], "q1", logs.append)
    assert out is not None and out["prev_rid"] == span["rid"] and out["n"] == 4
    assert any("tree-match" in x for x in logs)


def test_admission_mode_pins_the_tips_own_extra_key():
    span = _published(_cand(TIP, EK))
    ids = TIP + [9]
    assert tc.match_tip([span], ids, extra_key=EK)[0] is not None
    # a prompt of another salt / no salt is not the tip's
    assert tc.match_tip([span], ids, extra_key="other")[0] is None
    assert tc.match_tip([span], ids, extra_key=None)[0] is None
    # an unsalted tip is not matched by a salted prompt either
    plain = _published(_cand(TIP, None))
    assert tc.match_tip([plain], ids, extra_key=EK)[0] is None
    assert tc.match_tip([plain], ids, extra_key=None)[0] is not None
    # the P-stage entry takes the live request's extra_key
    d0 = {"spans": [span]}
    hint = {"tree": True, "prev_rid": "tree:", "n": 5}
    assert sa.resolve_tree_hint(hint, d0, ids, "q", lambda _m: None, extra_key=EK)
    assert sa.resolve_tree_hint(hint, d0, ids, "q", lambda _m: None, extra_key="x") is None


def test_admit_for_sched_passes_the_requests_extra_key():
    import inspect

    src = inspect.getsource(sa.admit_for_sched)
    assert 'extra_key=getattr(req, "extra_key", None)' in src


# -------------------------------------------------- identity is not disturbed

def test_unsalted_rid_and_span_are_unchanged():
    c = _cand(TIP, None)
    assert tc.pseudo_req(c, 0, 1).rid == tc.RID_PREFIX + tc.digest_of(TIP)
    span = _published(c)
    assert span["match_key"] == tc.digest_of(TIP)


def test_salted_rid_keeps_the_extra_key_digest_and_names_the_match_key():
    c = _cand(TIP, EK)
    rid = tc.pseudo_req(c, 0, 1).rid
    assert rid == "tree:%s@%s" % (tc.digest_of(TIP, EK), tc.digest_of(TIP, None))
    assert tc.digest_of(TIP, EK) != tc.digest_of(TIP, None)
    assert tc.split_rid(rid) == (tc.digest_of(TIP, EK), tc.digest_of(TIP, None))
    assert _published(c)["match_key"] == tc.digest_of(TIP, None)
    # the pseudo req still carries the extra_key to the bind
    assert tc.pseudo_req(c, 0, 1).extra_key == EK


def test_same_tokens_two_salts_are_two_tips_with_one_match_key():
    a, b = _cand(TIP, "A"), _cand(TIP, "B")
    ra, rb = tc.pseudo_req(a, 0, 2).rid, tc.pseudo_req(b, 1, 2).rid
    assert ra != rb
    sa_, sb_ = _published(a), _published(b)
    assert sa_["match_key"] == sb_["match_key"]
    # admission mode picks the one of the prompt's own salt
    assert tc.match_tip([sa_, sb_], TIP + [5], extra_key="B")[0][0] == rb
    assert tc.match_tip([sa_, sb_], TIP + [5], extra_key="A")[0][0] == ra


def test_old_descriptor_without_match_key_still_matches_unsalted_tips():
    span = {"rid": tc.RID_PREFIX + tc.digest_of(TIP), "depth": 4}      # pre-240
    got, _ = tc.match_tip([span], TIP + [7])
    assert got == (span["rid"], 4, 4)


def test_ranks_derive_the_same_match_key():
    """No collective: the match key is a function of the tokens, so two ranks
    with different recency clocks publish the same rid / match key, and the
    gather still agrees on the extra_key digest."""
    c0 = _cand(TIP, EK)
    c1 = tc.TreeCand(digest=c0.digest, n_tokens=4, last_access=99.0, tokens=tuple(TIP),
                     extra_key=EK, match_key=c0.match_key)
    assert tc.pseudo_req(c0, 0, 1).rid == tc.pseudo_req(c1, 0, 1).rid
    agreed = tc.agree([c0], lambda mine: [mine, [(c1.digest, c1.n_tokens)]], 4)
    assert [x.digest for x in agreed] == [c0.digest]


# -------------------------------------------------------- the tree walk side

def test_local_candidates_carry_the_match_key_lazy_and_eager():
    import torch

    from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType

    FULL, MAMBA = int(ComponentType.FULL), int(ComponentType.MAMBA)
    NS = SimpleNamespace

    def node(parent, toks, ek):
        n = NS(parent=parent, children={}, key=NS(token_ids=list(toks), extra_key=ek),
               last_access_time=1.0)
        cds = [NS(value=None, host_value=None) for _ in range(3)]
        cds[FULL].value = torch.arange(len(toks), dtype=torch.int64) + 1
        cds[MAMBA].value = torch.tensor([5], dtype=torch.int64)
        n.component_data = cds
        parent.children[tuple(toks[:1])] = n
        return n

    root = NS(parent=None, children={}, key=NS(token_ids=[], extra_key=None),
              last_access_time=0, component_data=[NS(value=None, host_value=None)] * 3)
    node(root, [1, 2, 3, 4], EK)
    tree = NS(root_node=root)
    for lazy in (False, True):
        (c,) = tc.local_candidates(tree, None, lazy_tokens=lazy)
        assert c.digest == tc.digest_of(TIP, EK)
        assert c.match_key == tc.digest_of(TIP, None)
        assert tc.pseudo_req(c, 0, 1).rid.endswith("@" + tc.digest_of(TIP, None))
    # build() logs and returns the published rid
    logs = []
    reqs = tc.build(tree, lambda v: [v], 0, {}, logs.append)
    assert [r.rid for r in reqs] == [tc.rid_of(c)]
    assert any(tc.rid_of(c) in x for x in logs)


def test_real_bigram_tree_salted_tip_is_matched():
    """The 27B form: a REAL bigram UnifiedRadixCache, a finished request with a
    cache salt; D publishes the tip, the follow-up prompt extends it."""
    import test_weg2_end_anchor_exact_probe_0928 as ea
    from sglang.srt.weg2 import l15_bind

    fx = ea._fixture(exact=True, bigram=True)
    ids = list(range(2000, 2024))
    req = ea._req(fx, ids, "d1")
    req.extra_key = EK
    ea._finish(fx, req)
    (pr,) = tc.build(fx.cache, lambda v: [v], 0, {}, None)
    assert pr.extra_key == EK and "@" in pr.rid
    slots, _node, _anchor = l15_bind.match_parked(pr, fx.cache)   # the bind still resolves it
    span = sp.build_descriptor(epoch=1, rank=0, prefix=[0, 1], bases=[],
                               spans=[_hold_span(pr.rid, len(slots))])["spans"][0]
    got, _ = tc.match_tip([span], ids + [7, 8, 9])
    assert got == (pr.rid, len(ids) - 1, len(ids))
    assert tc.match_tip([span], ids + [7], extra_key=EK)[0] is not None
    assert tc.match_tip([span], ids + [7], extra_key=None)[0] is None


# ------------------------------------------- point 2: no third raw length

@pytest.mark.parametrize("page", [1, 16, 64])
@pytest.mark.parametrize("bigram", [False, True])
def test_raw_length_is_units_or_units_plus_one(page, bigram):
    """Every insert / match goes through ``key.page_aligned(page_size)``
    (unified_radix_cache.insert / match_prefix); a tip's chain is a
    concatenation of such keys, so its raw length is units (plain) or
    units + 1 (bigram: boundary token kept once). The span's ``depth`` is the
    matched unit count (l15_bind.match_parked), and an off-grid mamba anchor
    only drops the node's mamba VALUE (mamba_component.py: mamba_value=None,
    a tombstone, KV kept at full length), never shortens the key."""
    from sglang.srt.mem_cache.radix_cache import RadixKey

    for n in range(1, 3 * page + 3):
        key = RadixKey(array("q", range(n)), None, is_bigram=bigram).page_aligned(page)
        units, raw = len(key), len(key.raw_token_ids())
        if units == 0:
            continue
        assert raw == (units + 1 if bigram else units)
        # and the chain a tip digests is exactly those raw tokens
        root = SimpleNamespace()
        node = SimpleNamespace(parent=root, key=key)
        assert len(tc.chain_tokens(node, root)) == raw
