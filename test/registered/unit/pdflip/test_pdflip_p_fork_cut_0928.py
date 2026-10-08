"""P-FORK-CUT (28.09.2026): group P ends a chunk at the shared-prefix fork.

Numbers from NF rc12z30e (ca2a9706ec), P log ...09282117_ca2a9706ec_0928_211748,
PP0 21:27:17: pdflip-14-37, 16869 tokens, '#1028B FETCH CAP n=7: kv=251
claimed=0 lost=251 ... anchors_in_range mamba (0,-1)', chunk 16384, page 64.
"""

import inspect
from types import SimpleNamespace

import pytest

from flliper.srt.pdflip import p_fork_cut as pfc

PAGE = 64
CHUNK = 16384
PROMPT = 16869
KV_PAGES = 251  # 16064 tokens of shared KV in the store


def test_1437_cut_is_free_and_lands_on_the_fork():
    new_len, why = pfc.fork_cut(0, CHUNK, PROMPT, KV_PAGES * PAGE, CHUNK, PAGE)
    assert (new_len, why) == (16064, "cut")
    # two forwards either way: 16384 + 485 before, 16064 + 805 after
    assert pfc.forwards(PROMPT - 16064, CHUNK) == pfc.forwards(PROMPT - CHUNK, CHUNK) == 1


def test_a_cut_that_costs_a_forward_is_not_taken():
    # 40000 tokens, fork at 4000: 16384+16384+7232 (3) vs 4000+16384+16384+3232 (4)
    assert pfc.fork_cut(0, CHUNK, 40000, 4000, CHUNK, PAGE) == (None, "paid")


def test_fork_outside_the_extend_changes_nothing():
    assert pfc.fork_cut(16384, 485, PROMPT, 16064, CHUNK, PAGE) == (None, "outside")
    assert pfc.fork_cut(0, CHUNK, 40000, 20000, CHUNK, PAGE) == (None, "outside")


def test_interval_rule_only_cuts_where_an_anchor_is_donated():
    assert pfc.fork_cut(0, CHUNK, PROMPT, 2048, CHUNK, PAGE, interval=4096) == (None, "interval")
    assert pfc.fork_cut(0, CHUNK, PROMPT, 16064, CHUNK, PAGE, interval=4096)[1] == "cut"


def test_cut_is_floored_to_the_grain():
    new_len, why = pfc.fork_cut(0, CHUNK, PROMPT, 16090, CHUNK, PAGE)
    assert why == "cut" and new_len == 16064


def test_store_note_keeps_only_capped_answers():
    pfc._STORE_UNCAPPED.clear()
    pfc.note_store_uncapped("full-claim", 264, 264)
    pfc.note_store_uncapped("pdflip-14-37", KV_PAGES, 0)
    assert "full-claim" not in pfc._STORE_UNCAPPED
    assert pfc._STORE_UNCAPPED["pdflip-14-37"] == (KV_PAGES, 0)


def _server_args(monkeypatch):
    import flliper.srt.runtime_context as rc

    monkeypatch.setattr(
        rc, "get_server_args", lambda: SimpleNamespace(chunked_prefill_size=CHUNK, pp_size=3, tp_size=1)
    )
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "P")
    monkeypatch.delenv("FLLIPER_PDFLIP_MAMBA_ANCHOR_INTERVAL", raising=False)
    monkeypatch.delenv("FLLIPER_PDFLIP_FORM", raising=False)


def test_apply_1437_from_the_store_probe(monkeypatch):
    """The metal form end to end: PP0's probe noted 251 uncapped pages at
    registration depth 0, PP0's told verdict carries 16064, the adder
    truncates to 16384 -- the hook ends the chunk at 16064 instead. (29.09.:
    the adder reads the TOLD fork, never the probe -- test_pdflip_p_fork_cut_
    uniform_0929 drives the group.)"""
    _server_args(monkeypatch)
    assert pfc.armed()
    pfc._STORE_UNCAPPED.clear()
    pfc.note_store_uncapped("pdflip-14-37", KV_PAGES, 0)
    adder = SimpleNamespace(page_size=PAGE, token_to_kv_pool_allocator=None)
    req = SimpleNamespace(
        rid="pdflip-14-37",
        full_untruncated_fill_ids=list(range(PROMPT)),
        _prefetch_registered_prefix_len=0,
        key_match_depth=None,
    )
    # the probe alone cuts nothing -- it is rank-local
    assert pfc.apply(adder, req, 0, CHUNK, "add_one_req") == CHUNK
    req._pdflip_fork_told = pfc.pp0_fork_verdict(req, PAGE)
    assert req._pdflip_fork_told == KV_PAGES * PAGE
    assert pfc.apply(adder, req, 0, CHUNK, "add_one_req") == 16064
    # the continuation from the fork is left alone
    assert pfc.apply(adder, req, 16064, PROMPT - 16064, "add_chunked_req") == PROMPT - 16064


def test_apply_tree_fork(monkeypatch):
    _server_args(monkeypatch)
    pfc._STORE_UNCAPPED.clear()
    adder = SimpleNamespace(page_size=PAGE, token_to_kv_pool_allocator=None)
    req = SimpleNamespace(rid="t", full_untruncated_fill_ids=list(range(PROMPT)), key_match_depth=16090)
    assert pfc.apply(adder, req, 0, CHUNK, "add_one_req") == CHUNK
    req._pdflip_fork_told = pfc.pp0_fork_verdict(req, PAGE)
    assert pfc.apply(adder, req, 0, CHUNK, "add_one_req") == 16064


@pytest.mark.parametrize(
    "env,sa",
    [
        ({"FLLIPER_PDFLIP_GROUP": "D"}, dict(pp_size=1, tp_size=3)),  # D: TP ranks decide alone
        ({"FLLIPER_PDFLIP_GROUP": "P"}, dict(pp_size=1, tp_size=1)),  # no forwarded schedule
    ],
)
def test_disarmed_off_group_p_pp(monkeypatch, env, sa):
    import flliper.srt.runtime_context as rc

    monkeypatch.setattr(rc, "get_server_args", lambda: SimpleNamespace(chunked_prefill_size=CHUNK, **sa))
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    pfc._STORE_UNCAPPED.clear()
    pfc.note_store_uncapped("x", KV_PAGES, 0)
    adder = SimpleNamespace(page_size=PAGE, token_to_kv_pool_allocator=None)
    req = SimpleNamespace(rid="x", full_untruncated_fill_ids=list(range(PROMPT)),
                          _prefetch_registered_prefix_len=0, key_match_depth=None)
    assert pfc.pp0_fork_verdict(req, PAGE) == 0
    req._pdflip_fork_told = KV_PAGES * PAGE
    assert pfc.apply(adder, req, 0, CHUNK, "add_one_req") == CHUNK


def test_wired_into_both_truncating_sites_and_the_probe():
    from flliper.srt.managers import schedule_policy as sp
    from flliper.srt.mem_cache.hybrid_cache import hybrid_cache_controller as hcc

    chunked = inspect.getsource(sp.PrefillAdder.add_chunked_req)
    one = inspect.getsource(sp.PrefillAdder.add_one_req)
    assert "_pdflip_p_fork_cut.apply(" in chunked
    # before the end-anchor split, which then sees a non-final chunk
    assert chunked.index("_pdflip_p_fork_cut.apply(") < chunked.index("_pdflip_end_anchor_split(")
    assert "_pdflip_p_fork_cut.apply(" in one
    assert "p_fork_cut.note_store_uncapped(" in inspect.getsource(hcc)
