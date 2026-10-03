"""#257 (ii): an inner mamba anchor displaced from the arena keeps an L3 copy.

27B 62e7b80aed (NF P logs, 9 boots with the tree-key probe, 20 anchor clamps):
8 clamps were forks below a held anchor -- NF's anchor interval is 0, the
inner chunk anchors live only in the 32-slot mamba arena, and the H19
displacement dropped them with NO disk write (``arena_drop_unreferenced``).
Every fork below such an anchor fell back to the previous surviving one and
re-prefilled the difference.

Now the rank whose release leaves a displaced slot unreferenced pins it,
writes it to L3 through the claim's own disk half
(``HiCacheFile.arena_secure_to_disk``), unpins and drops -- once per anchor,
not once per rank.

Hermetic: the H19 harness (one real shared arena, three P rank pools, real
``UnifiedRadixCache._weg2_mamba_claim``) on the L3P store of #257 (d) (real
``LRUFileEvictor``, real #1459 index, ``_l3p_seed_index``); every rank writes
its own extent of each anchor blob."""
from __future__ import annotations

import importlib.util
import os
import shutil

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from sglang.srt.mem_cache.hicache_storage import HiCacheFile  # noqa: E402
from sglang.srt.mem_cache.pool_host.arena_pool import ArenaMHAHostPool  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")

_HERE = os.path.dirname(__file__)


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


h19 = _load("_t_h19", "../weg2/test_weg2_mamba_arena_displace_h19.py")
d257 = _load("_t_257d", "test_probe_hold_257.py")

CAP = 2
CHUNKS = 6


def _fill(h):
    return sum(map(ord, h)) % 250 + 1


class _Rank(h19._Rank):
    """The H19 rank whose sweep also writes its extent of each anchor blob
    before the write completes (what the device->arena copy does)."""

    def __init__(self, arena, rank, cfg, backend):
        super().__init__(arena, rank, cfg)
        self.rank = rank
        self.arena = arena
        self.mp._backend = backend

    def sweep(self):
        while self.unbacked:
            n, h = self.unbacked[0]
            rows = self.cache._weg2_mamba_claim(n, self.mp, h)
            if rows is None:
                return False
            slot = int(rows[0]) - self.mp.staging_rows
            view = self.arena.slot_view(slot, h19.SB)
            lo = self.rank * h19.EXT
            view[lo:lo + h19.EXT] = bytes([_fill(h)]) * h19.EXT
            n.component_data[h19.M].host_value = rows
            self.mp.complete_write(rows)
            self.unbacked.pop(0)
        return True


def _store(tmp_path, monkeypatch):
    be, idx = d257._l3p_backend(tmp_path, monkeypatch)
    be._get_suffixed_key = lambda k: f"{k}.sfx"
    be._log_key = lambda pool, k: f"{k}.mamba"
    return be, idx


def _run(tmp_path, monkeypatch, pre_on_disk=()):
    be, idx = _store(tmp_path, monkeypatch)
    for h in pre_on_disk:
        d257._previous_boot_left(be, h19._stem(h), 0x7E)
    be._l3p_seed_index(idx)
    arena = h19._arena(tmp_path, 32)
    ranks = [_Rank(arena, r, cfg=CAP, backend=be) for r in range(h19.RANKS)]
    w0 = getattr(ArenaMHAHostPool, "_257_written_to_l3", 0)
    l0 = getattr(ArenaMHAHostPool, "_257_dropped_without_l3", 0)
    hs = h19._prefill(ranks, "A", CHUNKS)
    written = getattr(ArenaMHAHostPool, "_257_written_to_l3", 0) - w0
    lost = getattr(ArenaMHAHostPool, "_257_dropped_without_l3", 0) - l0
    return be, idx, arena, hs, written, lost


def test_a_displaced_inner_anchor_is_on_disk_for_the_fork_below_it(tmp_path, monkeypatch):
    """RED on e5399c8471: the four inner anchors the share displaced are in
    neither the arena nor L3 -- a fork below any of them re-prefills. GREEN:
    each is in L3 (file + index) with the bytes of all three rank extents, and
    the fork's read fills it back from disk."""
    be, idx, arena, hs, written, lost = _run(tmp_path, monkeypatch)
    inner = hs[:-CAP]
    assert all(h19._state(arena, h) != 2 for h in inner), "H19 displaced the inner anchors"
    assert [h19._state(arena, h) for h in hs[-CAP:]] == [2] * CAP
    stems = [h19._stem(h) for h in inner]
    missing = [s for s in stems if not os.path.exists(be._sharded_path(s))]
    assert not missing, f"inner anchors lost from L2 and L3 at once: {missing}"
    assert all(idx.has(stems)), "an anchor on disk the next probe cannot see"
    assert lost == 0
    back = HiCacheFile.arena_fill_from_disk(be, arena, stems, h19.SB)
    for h, slot in zip(inner, back):
        assert slot is not None
        assert bytes(arena.slot_view(slot, h19.SB)) == bytes([_fill(h)]) * h19.SB, \
            "the copy holds every rank's extent"


def test_each_displaced_anchor_is_written_once_not_once_per_rank(tmp_path, monkeypatch):
    """Three P ranks release the same anchor; only the last releaser finds it
    unreferenced -- one write per anchor."""
    _be, _idx, _arena, hs, written, lost = _run(tmp_path, monkeypatch)
    assert (written, lost) == (CHUNKS - CAP, 0)


def test_an_anchor_already_in_l3_is_not_written_again(tmp_path, monkeypatch):
    be, _idx, arena, hs, written, lost = _run(tmp_path, monkeypatch, pre_on_disk=("A-h0",))
    with open(be._sharded_path(h19._stem("A-h0")), "rb") as f:
        assert f.read() == bytes([0x7E]) * 64
    assert (written, lost) == (CHUNKS - CAP - 1, 0)
