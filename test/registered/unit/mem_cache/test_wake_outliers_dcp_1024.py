"""WAKE-OUTLIERS-1024, the dcp owner form (coordinator 24.10.: "Zweirang-Test unter dcp_owner_mode, page_size 1, ohne
Owner-Rows"). Two HiCacheFile ranks of one D group on one store, through the REAL backend read and write paths.

What it shows, and what it does not:
* READ path: both ranks name the SAME file for a KV page (the page-1 owner form drops the tp term from the KV key) and
  the backend serves every rank's read of the group's key list -- nothing is filtered by owner (the #239 owner-row form
  is different: there the host rank abstains). A peer's read-path T line is therefore the owner's duplicate WHEN the
  owner reads the same key list. That premise is a group fact the backend cannot show; on metal it holds: int8
  ``_1024_080355`` PDFLIP-READ-STAGES 8776008 pages on TP0, TP1 and TP2; duo ``_1024_070103`` the +3360 pages of TP1
  are overlap of reads reaped at different depths (pdflip-0-1: TP0 4786 / 68556 / 19520, TP1 3864 / 86784 / 19520 --
  the group cuts to the MIN anchor and re-issues from there on every rank, so both unions are [0, 86784)).
* WRITE path: a page is written only by its owner rank (cache_controller._page_backup, kv_page_owner_mask), so the
  ``_touch_present`` T line of a peer's own page is the page's ONLY one -- the peer keeps it (danger direction: a lost
  T line lets the owner evict the page first and the next read finds a shorter prefix).
Every test is hard-bounded (20 s SIGALRM).
"""

import os
import signal
import tempfile
from unittest import mock

import pytest
import torch

from flliper.srt.mem_cache.hicache_storage import HiCacheFile, HiCacheStorageConfig
from flliper.srt.mem_cache.storage.file import store_journal as SJ

HARD_LIMIT_S = 20
ENV_PEERS = "FLLIPER_HICACHE_L3_JOURNAL_TOUCH_TP_PEERS"
PAGE_BYTES = 256


@pytest.fixture(autouse=True)
def _hard_limit():
    def _boom(signum, frame):
        raise TimeoutError(f"hard test limit {HARD_LIMIT_S} s")

    old = signal.signal(signal.SIGALRM, _boom)
    signal.alarm(HARD_LIMIT_S)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


@pytest.fixture
def root(monkeypatch):
    for k in (SJ.ENV, "FLLIPER_HICACHE_L3_JOURNAL_TOUCH", "FLLIPER_HICACHE_L3_JOURNAL_TOUCH_S", ENV_PEERS):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("FLLIPER_PDFLIP_L3_EPOCH", "100")
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "D")
    with tempfile.TemporaryDirectory() as d:
        yield d


def _rank(root, tp_rank, owner_rows=None):
    cfg = HiCacheStorageConfig(
        tp_rank=tp_rank, tp_size=2, pp_rank=0, pp_size=1, attn_cp_rank=0, attn_cp_size=1,
        is_mla_model=False, enable_storage_metrics=False, is_page_first_layout=False,
        model_name="Qwen3.8-27B-NVFP4", model_identity_hash="feedfacecafebeef",
        dcp_owner_mode=True, canonical_kv_page=None, canonical_kv_owner_rows=owner_rows,
    )
    return HiCacheFile(cfg, file_path=root)


def _t_lines(backend):
    with open(backend._evictor._journal.path, "rb") as f:
        recs, _ = SJ.parse_lines(f.read())
    return [r[3] for r in recs if r[1] == "T"]


def _page(v):
    return torch.full((PAGE_BYTES,), v, dtype=torch.uint8)


def test_both_ranks_name_one_file_and_the_backend_serves_every_rank_the_whole_key_list(root, monkeypatch):
    t0, t1 = _rank(root, 0), _rank(root, 1)
    keys = ["a1", "a2", "a3"]
    assert {t0._get_suffixed_key(k) for k in keys} == {t1._get_suffixed_key(k) for k in keys}, "one file per page"
    assert t0._evictor._is_storage_owner and not t1._evictor._is_storage_owner
    for k in keys:                                   # the page owner writes; the bytes are complete (full kv-heads)
        assert t0.set(k, _page(7))
    monkeypatch.setenv(ENV_PEERS, "1")               # observe the undeduplicated lines first
    outs0 = [torch.zeros(PAGE_BYTES, dtype=torch.uint8) for _ in keys]
    outs1 = [torch.zeros(PAGE_BYTES, dtype=torch.uint8) for _ in keys]
    assert all(r is not None for r in t0.batch_get(keys, outs0))
    assert all(r is not None for r in t1.batch_get(keys, outs1))
    assert all(bool((o == 7).all()) for o in outs0 + outs1), "every rank reads every page of the key list"
    assert _t_lines(t1) == _t_lines(t0)[-3:] == [t0._get_suffixed_key(k) for k in keys], (
        "the peer's read-path lines are exactly the owner's")


def test_read_path_lines_of_the_peer_are_deduplicated(root):
    t0, t1 = _rank(root, 0), _rank(root, 1)
    for k in ("b1", "b2"):
        assert t0.set(k, _page(3))
    before0 = len(_t_lines(t0))
    out = torch.zeros(PAGE_BYTES, dtype=torch.uint8)
    for b in (t0, t1):
        assert b.get("b1", out) is not None
        assert all(r is not None for r in b.batch_get(["b2"], [torch.zeros(PAGE_BYTES, dtype=torch.uint8)]))
    assert _t_lines(t0)[before0:] == [t0._get_suffixed_key("b1"), t0._get_suffixed_key("b2")]
    assert _t_lines(t1) == [], "the owner carries the read recency of the group-wide page"


def test_write_path_line_of_the_peers_own_page_is_kept(root):
    """The page-1 owner form writes a page only on its owner rank: TP1's re-write of ITS page ('already on disk ->
    refresh recency') is that page's only T line -- dropping it would let the owner evict the page first."""
    t0, t1 = _rank(root, 0), _rank(root, 1)
    assert t1.set("c1", _page(5))                    # TP1 owns c1 (written only here)
    assert t1.set("c1", _page(5))                    # backup again: found on disk -> _touch_present
    assert _t_lines(t1) == [t1._get_suffixed_key("c1")]
    assert _t_lines(t0) == [], "the owner never wrote nor read c1: no other line exists"


def test_owner_row_form_keeps_every_peer_line(root):
    """#239 owner rows: the host rank abstains from reads its peers do -- no key is the owner's duplicate (backend
    suffix answer and evictor flag both)."""
    import types

    from flliper.srt.mem_cache.storage.file.lru_file_evictor import LRUFileEvictor

    fake = types.SimpleNamespace(dcp_owner_mode=True, _kv_owner_rows=(64, 64, 0, 46),
                                 _kv_config_suffix_is_group_wide=True, kv_config_suffix="_m_h")
    assert HiCacheFile._touch_dedupe_suffix(fake) is None
    fake._kv_owner_rows = None
    assert HiCacheFile._touch_dedupe_suffix(fake) == "_m_h", "page-1 owner form: the group-wide suffix"
    ev = LRUFileEvictor(root, "_m_h", tp_rank=1, writes_shared_keys=True, dcp_owner_mode=True, dcp_owner_rows=True,
                        extra_config={"max_size": str(10 ** 9), "max_size_scope": "shared"})
    assert not ev._journal_touch_tp_peer


def test_evictor_flags_follow_the_form(root):
    t1 = _rank(root, 1)
    ev = t1._evictor
    assert ev._journal_touch_tp_peer and ev._journal_touch_read_only, "page-1 dcp owner form: read path only"
    with mock.patch.dict(os.environ, {"FLLIPER_PDFLIP_GROUP": ""}):
        assert not _rank(root, 1)._evictor._journal_touch_tp_peer, "outside a Weg-2 group: no dedupe"


def test_t_source_instrument_names_the_path(root):
    t0 = _rank(root, 0)
    assert t0.set("d1", _page(1))
    assert t0.get("d1", torch.zeros(PAGE_BYTES, dtype=torch.uint8)) is not None
    assert t0.set("d1", _page(1))                   # already on disk -> _touch_present<set
    src = t0._evictor._t_source
    assert src.get("read:get") == 1 and sum(v for k, v in src.items() if k.startswith("_touch_present<")) == 1, src
