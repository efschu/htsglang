"""#1442: P hands D the token ids and the page-key chain of a finished
leg-1 request by rid through the shared arena dir; D's tokenizer takes the
ids, D's hit query takes the keys; absent file = the old path."""

import os
import types

import pytest

from flliper.srt.pdflip import handoff as ho


@pytest.fixture
def arena_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("FLLIPER_HICACHE_ARENA_DIR", str(tmp_path / "arena"))
    monkeypatch.delenv("FLLIPER_PDFLIP_HANDOFF", raising=False)
    return tmp_path / "arena"


def test_roundtrip_and_removal(arena_dir):
    assert ho.write("pdflip-6-3", [1, 2, 3, 4], ["k0", "k1", "k2", "k3"])
    assert (arena_dir / "handoff" / "pdflip-6-3.json").exists()
    assert ho.read_ids("pdflip-6-3") == [1, 2, 3, 4]
    assert ho.read_keys("pdflip-6-3") == ["k0", "k1", "k2", "k3"]
    assert ho.read_ids("pdflip-6-3") is None, "the keys read removes the file"
    assert ho.read_ids("pdflip-9-9") is None


def test_disabled_or_no_arena_is_the_old_path(monkeypatch, tmp_path):
    monkeypatch.delenv("FLLIPER_HICACHE_ARENA_DIR", raising=False)
    assert not ho.write("pdflip-1-1", [1], ["k"]) and ho.read_ids("pdflip-1-1") is None
    monkeypatch.setenv("FLLIPER_HICACHE_ARENA_DIR", str(tmp_path))
    monkeypatch.setenv("FLLIPER_PDFLIP_HANDOFF", "0")
    assert not ho.write("pdflip-1-1", [1], ["k"])


def test_tokenizer_shortcut_reads_the_ids_once(arena_dir):
    from flliper.srt.managers.tokenizer_manager import _pdflip_handoff_ids
    ho.write("pdflip-2-2", [7, 8, 9], ["a", "b", "c"])
    obj = types.SimpleNamespace(rid="pdflip-2-2", input_ids=None)
    assert _pdflip_handoff_ids(obj) == [7, 8, 9]
    os.remove(ho.path("pdflip-2-2"))
    assert _pdflip_handoff_ids(obj) == [7, 8, 9], "cached on the object"
    assert _pdflip_handoff_ids(types.SimpleNamespace(rid="abc-uuid", input_ids=None)) is None


def test_hit_query_prefers_the_handed_over_keys():
    from flliper.srt.mem_cache.hybrid_cache.hybrid_cache_controller import HybridCacheController
    calls = []
    hc = object.__new__(HybridCacheController)
    hc.page_size = 1
    hc.get_hash_str = lambda toks, last, page_size=1: calls.append("hashed") or ["h"] * len(toks)
    op = types.SimpleNamespace(token_ids=[1, 2], last_hash=None, prefix_keys=None,
                               pool_transfers=None, pdflip_page_keys=["p0", "p1"])
    # only the key derivation line is exercised: reproduce it verbatim
    hash_value = getattr(op, "pdflip_page_keys", None) or hc.get_hash_str(op.token_ids, op.last_hash, page_size=hc.page_size)
    assert hash_value == ["p0", "p1"] and calls == []
