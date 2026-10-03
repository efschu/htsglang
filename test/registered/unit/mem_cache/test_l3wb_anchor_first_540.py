"""540 L3WB-ANCHOR-FIRST: the L3 write-behind reaches the anchor arena on every pass.

27B y8r (55c95a89c7, 03.10. 09:06), D TP0: the write-behind visited the arenas in open order --
the KV arena (720896 x 32 KiB, opened first) and then the mamba anchor arena (112 x 78 MiB). The KV
arena never ended its cycle under load (``new`` 50-150k stems, every pass ``budget=hit cont=1`` for
minutes), so the pass always stopped inside it and the anchor arena was never reached. Every anchor
claim of the D>P flush's FLUSH-PUBLISH then evicted a slot WITHOUT an L3 copy and wrote it right
there, synchronously (#257 d: CRC + fsync of 78 MiB): 09:14:00 TP1 five such ``ARENA-DROP ...
written:1`` inside a sweep of 1039 ms, while its last pass with four ``on_disk:1`` drops took 89 ms.

Pinned here on the real C arena and the real L3 index/evictor (the 0930 slice scaffolding): a KV
arena with a backlog that keeps every pass sliced, an anchor arena opened AFTER it, and a new anchor
page arriving in the middle of the KV cycle. The anchor page reaches L3 on the very next pass; with
the switch at 0 (the open order) it does not reach it at all while the KV cycle continues.
"""
from __future__ import annotations

import os
import shutil
import time as _real_time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

from sglang.srt.mem_cache import hicache_storage as HS  # noqa: E402
from sglang.srt.mem_cache.canonical_page_store import CanonicalExtentWindow  # noqa: E402
from sglang.srt.mem_cache.hicache_storage import HiCacheFile  # noqa: E402
from sglang.srt.mem_cache.storage.file.hicache_arena import ShmArena  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")

TOTAL = 64                  # a KV page (KiB-class)
A_TOTAL = 1 << 20           # an anchor page (MiB-class; the mamba blob is 78 MiB on the 27B)
SFX = "_NF_540anch"
SLICE_S = 10 / 1024
STEM_S = 1 / 1024


class _KVPage:
    pass


@pytest.fixture(autouse=True)
def _awake_gate():
    from sglang.srt.mem_cache import l3_write_behind as gate

    gate._reset_for_tests()
    yield
    gate._reset_for_tests()


class _Clock:
    def __init__(self):
        self.t = 2048.0

    def perf_counter(self):
        return self.t

    def thread_time(self):
        return self.t

    def __getattr__(self, name):
        return getattr(_real_time, name)


def _backend(tmp_path, monkeypatch):
    from sglang.srt.mem_cache.storage.file.l3_index import L3Index
    from sglang.srt.mem_cache.storage.file.lru_file_evictor import LRUFileEvictor

    monkeypatch.setenv("SGLANG_WEG2_L3_PERSIST", "1")
    root = tmp_path / "store"
    root.mkdir(parents=True, exist_ok=True)
    (root / "L3_IDENTITY.json").write_text("{}")
    be = object.__new__(HiCacheFile)
    be.file_path = str(root)
    be._known_shards = set()
    be._legacy_flat = False
    be._key_geom = {"is_mla_model": False}
    be.metadata_cache = None
    be.dcp_owner_mode = False
    be.canonical_kv_page = _KVPage()
    be._canonical_kv_extents = CanonicalExtentWindow(TOTAL, ((0, TOTAL),))
    be.canonical_qsa_page = None
    be.canonical_mamba_blob = None
    be.canonical_draft_page = None
    be.kv_config_suffix = SFX
    be._kv_config_suffix_is_group_wide = True
    be.config_suffix = SFX + "_0_1"
    be._config_suffix_is_group_wide = False
    be._evictor = LRUFileEvictor(
        str(root), SFX, tp_rank=0, writes_shared_keys=False,
        path_for_stem=be._existing_path, iter_existing=be._iter_existing_files,
    )
    idx = L3Index(str(tmp_path / "l3idx.bin"), cap=1 << 14)
    be._l3idx = idx
    be._l3idx_tried = True
    be._evictor.l3_index = idx
    adir = tmp_path / "shm"
    adir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("SGLANG_HICACHE_ARENA_DIR", str(adir))
    kv = ShmArena(str(adir / f"arena-{TOTAL}.bin"), TOTAL, 2048)
    an = ShmArena(str(adir / f"arena-{A_TOTAL}.bin"), A_TOTAL, 4)
    # the y8r open order: the KV arena first, the anchor arena behind it
    be._arenas = {TOTAL: kv, A_TOTAL: an}
    return be, kv, an


def _put(arena, stem, total, fill=0):
    pay = torch.full((total,), fill & 0xFF, dtype=torch.uint8)
    assert arena.write([stem], [total], [((0, total),)], [pay.data_ptr()]) == [1]


def _slow_stat(be, clock, monkeypatch):
    real = be._stat_stems

    def slow(stems):
        clock.t += STEM_S * len(stems)
        return real(stems)

    monkeypatch.setattr(be, "_stat_stems", slow)


def _no_quiet():
    return None


def _run(tmp_path, monkeypatch):
    """A KV backlog that keeps the cycle running (every pass sliced), then a new anchor page in
    the middle of that cycle, then three more continuation passes. Returns (be, anchor stem, passes)."""
    be, kv, an = _backend(tmp_path, monkeypatch)
    clock = _Clock()
    monkeypatch.setattr(HS, "time", clock)
    _slow_stat(be, clock, monkeypatch)
    for i in range(1500):
        _put(kv, be._get_suffixed_key(f"{i:05d}" + "ab" * 30), TOTAL, i)
    passes = []
    cont = False
    for _ in range(3):
        tot = be.l3_write_behind_pass(budget_bytes=1 << 40, quiet=_no_quiet, cont=cont,
                                      slice_s=SLICE_S)
        passes.append(tot)
        cont = bool(tot.get("sliced"))
    assert cont, "the KV cycle must still be running (the y8r shape)"
    a_stem = be._get_suffixed_key("f" * 64 + ".mamba")
    _put(an, a_stem, A_TOTAL, 7)
    for _ in range(3):
        tot = be.l3_write_behind_pass(budget_bytes=1 << 40, quiet=_no_quiet, cont=cont,
                                      slice_s=SLICE_S)
        passes.append(tot)
        cont = bool(tot.get("sliced"))
    assert cont, "still the same KV cycle"
    return be, a_stem, passes


def test_the_anchor_page_reaches_l3_on_the_next_pass_while_the_kv_cycle_runs(tmp_path, monkeypatch):
    monkeypatch.delenv("SGLANG_WEG2_L3_WRITE_BEHIND_ANCHOR_MIN_MIB", raising=False)
    be, a_stem, passes = _run(tmp_path, monkeypatch)
    assert be._stem_exists(a_stem)                       # RED before 540: never reached
    assert passes[3]["anchor_written"] == 1              # the first pass after it arrived
    assert sum(p["anchor_written"] for p in passes) == 1  # written once, not every pass
    # the KV arena keeps its progress: every continuation still verified KV stems
    assert all(p["on_disk"] + p["written"] - p["anchor_written"] > 0 for p in passes)


def test_switch_zero_is_the_open_order_and_starves_the_anchor_arena(tmp_path, monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_L3_WRITE_BEHIND_ANCHOR_MIN_MIB", "0")
    be, a_stem, passes = _run(tmp_path, monkeypatch)
    assert not be._stem_exists(a_stem)                   # the y8r defect, reproduced
    assert sum(p["anchor_written"] for p in passes) == 0


def test_an_already_secured_anchor_page_costs_no_write_on_later_passes(tmp_path, monkeypatch):
    monkeypatch.delenv("SGLANG_WEG2_L3_WRITE_BEHIND_ANCHOR_MIN_MIB", raising=False)
    be, kv, an = _backend(tmp_path, monkeypatch)
    a_stem = be._get_suffixed_key("e" * 64 + ".mamba")
    _put(an, a_stem, A_TOTAL, 3)
    first = be.l3_write_behind_pass(budget_bytes=1 << 40, quiet=_no_quiet, slice_s=0)
    assert first["anchor_written"] == 1 and be._stem_exists(a_stem)
    for cont in (True, True, False):
        again = be.l3_write_behind_pass(budget_bytes=1 << 40, quiet=_no_quiet, cont=cont, slice_s=0)
        assert again["anchor_written"] == 0 and again["new"] == 0


def test_the_default_threshold_separates_anchor_from_kv_pages():
    from sglang.srt.environ import envs

    assert envs.SGLANG_WEG2_L3_WRITE_BEHIND_ANCHOR_MIN_MIB.get() == 1
    assert HiCacheFile._l3wb_anchor_min_bytes() == 1 << 20
    # 27B: KV page 32 KiB, mamba blob 78446592 B; NF: KV 786 KiB, mamba 56 MiB
    assert 32768 < HiCacheFile._l3wb_anchor_min_bytes() <= 78446592
    assert 786 * 1024 < HiCacheFile._l3wb_anchor_min_bytes() <= 56 << 20


def test_nf_the_anchor_arena_goes_before_a_qsa_backlog(tmp_path, monkeypatch):
    """NF: the QSA index arena goes before KV (pair rule) and its backlog follows KV 1:1, so an
    anchor arena behind QSA would starve the same way. Anchors go first, even before QSA."""
    monkeypatch.delenv("SGLANG_WEG2_L3_WRITE_BEHIND_ANCHOR_MIN_MIB", raising=False)
    q_total = 32
    be, kv, an = _backend(tmp_path, monkeypatch)
    be.canonical_qsa_page = CanonicalExtentWindow(q_total, ((0, q_total),), label="qsa")
    be._canonical_probe_mismatch = lambda: None
    q = ShmArena(str(tmp_path / "shm" / f"arena-q-{q_total}.bin"), q_total, 2048)
    be._arenas = {TOTAL: kv, q_total: q, A_TOTAL: an}
    clock = _Clock()
    monkeypatch.setattr(HS, "time", clock)
    _slow_stat(be, clock, monkeypatch)
    for i in range(1500):
        _put(q, be._get_suffixed_key(f"{i:05d}" + "cd" * 30 + ".qsa"), q_total, i)
    a_stem = be._get_suffixed_key("d" * 64 + ".mamba")
    _put(an, a_stem, A_TOTAL, 9)
    first = be.l3_write_behind_pass(budget_bytes=1 << 40, quiet=_no_quiet, slice_s=SLICE_S)
    assert first["sliced"]                                # the QSA backlog keeps the pass busy
    assert first["anchor_written"] == 1 and be._stem_exists(a_stem)
