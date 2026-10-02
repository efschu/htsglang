"""DP-NACHLAUF 02.10. (N5p b6a6a5c08d): two mamba transfers in ONE
start_loading must both take the layer-0 all-layers state load and skip the
per-layer path on every later layer.

Every 27B START-LOADING printed two ``WEG2-ARENA-STATE-LOAD`` lines and
``mamba.pre_n=66`` (2 x 33 layers); the single ``_state_loaded_key`` kept only
the second transfer's key, so on each layer > 0 the first missed it and ran the
per-layer path (``device_indices.cpu()``: a host wait behind the KV H2D on the
load stream). PP0 ``mamba=206`` ms against 9 ms of timed sub-stages, linear in
the KV tokens (~7 GB/s).

Pinned (red before): interleaved per layer like the controller calls them
(A0 B0 A1 B1 ...), no per-layer path runs with the key set, the device rows are
exactly those of two separate loads; the switch off restores the single key
(the per-layer path, counted on the line as mamba.perlayer_n).
"""
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest  # noqa: E402
import torch  # noqa: E402

import test_arena_mamba_direct_1427 as H  # noqa: E402
from sglang.srt.mem_cache.pool_host import arena_mamba_pool as amp  # noqa: E402
from sglang.srt.mem_cache.storage.file.hicache_arena import ShmArena  # noqa: E402

pytestmark = H.pytestmark


def _two_blobs(tmp_path):
    arena = ShmArena(str(tmp_path / "mamba.bin"), H.TOTAL, 8)
    p = H._pool(arena)
    dev = H._device_pool()
    ra = p.alloc_write(["a"]); p.backup_from_device_all_layer(dev, ra, torch.tensor([3]), "direct"); p.complete_write(ra)
    rb = p.alloc_write(["b"]); p.backup_from_device_all_layer(dev, rb, torch.tensor([2]), "direct"); p.complete_write(rb)
    return p, dev, ra, rb


def _interleaved(p, ra, rb):
    back = H._device_pool(fill=False)
    da, db = torch.tensor([1]), torch.tensor([0])
    for l in range(H.SPEC.num_layers):
        p.load_to_device_per_layer(back, ra, da, l, "direct")
        p.load_to_device_per_layer(back, rb, db, l, "direct")
    return back


@pytest.mark.parametrize("mode", ["cpu", "dma"])
def test_two_transfers_interleaved_take_no_per_layer_path(tmp_path, monkeypatch, mode):
    monkeypatch.setenv("SGLANG_WEG2_ARENA_PAGE_LOAD_MODE", mode)
    monkeypatch.delenv(amp.ENV_MAMBA_STATE_KEYS, raising=False)
    p, dev, ra, rb = _two_blobs(tmp_path)
    p._weg2_load_sub = {}
    back = _interleaved(p, ra, rb)
    assert p._weg2_load_sub.get("perlayer_n", 0) == 0, p._weg2_load_sub
    for l in range(H.SPEC.num_layers):
        assert torch.equal(back.mamba_cache.temporal[l][1], dev.mamba_cache.temporal[l][3])
        assert torch.equal(back.mamba_cache.temporal[l][0], dev.mamba_cache.temporal[l][2])
        assert torch.equal(back.mamba_cache.conv[0][l][1], dev.mamba_cache.conv[0][l][3])
        assert torch.equal(back.mamba_cache.conv[0][l][0], dev.mamba_cache.conv[0][l][2])


def test_switch_off_is_the_single_key_and_the_line_names_it(tmp_path, monkeypatch):
    monkeypatch.setenv(amp.ENV_MAMBA_STATE_KEYS, "0")
    p, dev, ra, rb = _two_blobs(tmp_path)
    p._weg2_load_sub = {}
    back = _interleaved(p, ra, rb)
    assert p._weg2_load_sub.get("perlayer_n", 0) >= 1      # the miss is visible
    for l in range(H.SPEC.num_layers):                       # and still correct
        assert torch.equal(back.mamba_cache.temporal[l][1], dev.mamba_cache.temporal[l][3])
        assert torch.equal(back.mamba_cache.temporal[l][0], dev.mamba_cache.temporal[l][2])


def test_key_memory_is_bounded_and_dropped_on_failure():
    p = object.__new__(amp.ArenaMambaPoolHost)
    for k in range(20):
        amp._state_key_note(p, ("k", k))
    assert len(p._state_loaded_keys) == 8 and p._state_loaded_key == ("k", 19)
    assert amp._state_key_loaded(p, ("k", 12)) and not amp._state_key_loaded(p, ("k", 3))
    amp._state_key_drop(p, ("k", 12))
    assert not amp._state_key_loaded(p, ("k", 12)) and p._state_loaded_key is None
