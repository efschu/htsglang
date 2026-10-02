"""DP-NACHLAUF 02.10. (N5p b6a6a5c08d): the host load-back's two mamba
transfers (the node anchor and the request's CoW slot) read the SAME host
slot -- two WEG2-ARENA-STATE-LOAD of 54 MB per load on PP0. Loaded once, the
second device row is copied on the card per layer before the layer's event.

Pinned (red before): the split keeps the first of each (name, host slots) and
pairs the later ones with it; distinct slots, card-side host ids, unequal
counts, the switch off and the PLE form stay two loads; one arena load plus
the per-layer card copy lands exactly the rows two loads land; start_loading
loads only the kept transfers and copies before producer_event.complete(i).
"""
import inspect
import os
import sys
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from sglang.srt.mem_cache.hybrid_cache import mamba_load_dedup as d  # noqa: E402


def _t(name, host, dev):
    return SimpleNamespace(name=name, host_indices=torch.tensor(host), device_indices=torch.tensor(dev))


def test_split_pairs_the_same_host_slot(monkeypatch):
    monkeypatch.delenv(d.ENV, raising=False)
    monkeypatch.setattr(d, "_ple_on", lambda: False)
    node, cow, other = _t("mamba", [7], [2]), _t("mamba", [7], [5]), _t("mamba", [9], [1])
    kept, dups = d.split_duplicates([node, cow, other])
    assert kept == [node, other] and dups == [(cow, node)]


def test_what_stays_two_loads(monkeypatch):
    monkeypatch.delenv(d.ENV, raising=False)
    monkeypatch.setattr(d, "_ple_on", lambda: False)
    a, b = _t("mamba", [7], [2]), _t("mamba", [8], [5])
    assert d.split_duplicates([a, b]) == ([a, b], [])
    c = _t("swa", [7], [5])
    assert d.split_duplicates([a, c])[1] == []                      # another pool
    e = SimpleNamespace(name="mamba", host_indices=torch.tensor([7]), device_indices=None)
    assert d.split_duplicates([a, e])[1] == []                      # no device rows
    monkeypatch.setenv(d.ENV, "0")
    assert d.split_duplicates([a, _t("mamba", [7], [5])])[1] == []  # switch off
    monkeypatch.delenv(d.ENV)
    monkeypatch.setattr(d, "_ple_on", lambda: True)
    assert d.split_duplicates([a, _t("mamba", [7], [5])])[1] == []  # PLE side states


def test_one_arena_load_plus_card_copy_equals_two_loads(tmp_path, monkeypatch):
    import test_arena_mamba_direct_1427 as H
    from sglang.srt.mem_cache.storage.file.hicache_arena import ShmArena

    if H.shutil.which("gcc") is None:
        import pytest
        pytest.skip("needs gcc")
    arena = ShmArena(str(tmp_path / "mamba.bin"), H.TOTAL, 8)
    p = H._pool(arena)
    dev = H._device_pool()
    r = p.alloc_write(["a"]); p.backup_from_device_all_layer(dev, r, torch.tensor([3]), "direct"); p.complete_write(r)
    two = H._device_pool(fill=False)
    for l in range(H.SPEC.num_layers):
        p.load_to_device_per_layer(two, r, torch.tensor([1]), l, "direct")
        p.load_to_device_per_layer(two, r, torch.tensor([0]), l, "direct")
    one = H._device_pool(fill=False)
    for l in range(H.SPEC.num_layers):
        p.load_to_device_per_layer(one, r, torch.tensor([1]), l, "direct")
        d.copy_layer(one, l, torch.tensor([1]), torch.tensor([0]))
    assert torch.equal(one.mamba_cache.temporal, two.mamba_cache.temporal)
    assert torch.equal(one.mamba_cache.conv[0], two.mamba_cache.conv[0])
    assert torch.equal(one.mamba_cache.temporal[:, 0], dev.mamba_cache.temporal[:, 3])


def test_start_loading_loads_the_kept_and_copies_before_the_layer_event():
    from sglang.srt.mem_cache.hybrid_cache import hybrid_cache_controller as hcc

    src = inspect.getsource(hcc.HybridCacheController.start_loading)
    assert "pool_transfers=load_transfers," in src
    i_load = src.index("pool_transfers=load_transfers,")
    i_copy = src.index("_mld.copy_dups(")
    i_ev = src.index("producer_event.complete(i)")
    assert i_load < i_copy < i_ev
    # the index record still covers every transfer (dups included)
    assert "resolved_pool_transfers," in src[i_ev:]


def test_copy_dups_maps_global_to_local_layers():
    calls = []
    ent = SimpleNamespace(local_layer=lambda g: {3: 0, 5: 1}.get(g), device_pool="dp")
    host = SimpleNamespace(entry_map={"mamba": ent})
    orig = d.copy_layer
    try:
        d.copy_layer = lambda dp, ll, s, t: calls.append((dp, ll))
        dup, srcT = _t("mamba", [7], [5]), _t("mamba", [7], [2])
        assert [d.copy_dups(host, [(dup, srcT)], g) for g in (2, 3, 4, 5)] == [0, 1, 0, 1]
    finally:
        d.copy_layer = orig
    assert calls == [("dp", 0), ("dp", 1)]
    assert d.split_for(SimpleNamespace(), [1, 2]) == ([1, 2], [])
