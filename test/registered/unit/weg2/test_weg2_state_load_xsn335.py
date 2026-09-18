"""xsn332/335: the arena's Mamba states load as one blob per slot; the
release drain is capped per pass."""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402


def test_release_drain_cap_env(monkeypatch):
    from sglang.srt.mem_cache import unified_radix_cache as u
    monkeypatch.delenv("SGLANG_WEG2_RELEASE_DRAIN_CAP", raising=False)
    assert u._weg2_release_drain_cap() == 0          # xsn335: off by default
    monkeypatch.setenv("SGLANG_WEG2_RELEASE_DRAIN_CAP", "64")
    assert u._weg2_release_drain_cap() == 64
    monkeypatch.setenv("SGLANG_WEG2_RELEASE_DRAIN_CAP", "x")
    assert u._weg2_release_drain_cap() == 0
    src = open(u.__file__).read()
    i = src.index("def drain_storage_control_queues")
    assert "_cap = _weg2_release_drain_cap()" in src[i:i + 3000]


def test_state_loader_splits_layers_like_the_per_layer_path(monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_ARENA_STATE_LOAD_BLOCK_BYTES", str(1 << 20))  # the floor: forces blocks of 1 MiB // slot_bytes
    """A synthetic slot blob: L layers of temporal rows and 3 conv extents per
    layer; the all-layers loader must land the same bytes as per-layer copies."""
    from sglang.srt.mem_cache.pool_host import arena_mamba_pool as amp
    import types
    L, A, n = 3, 8, 4
    t_shape = (2, 4); width = 3; conv_shape = (6, width)
    e = 2  # bf16
    t_row = 8 * e
    c_row = 6 * width * e
    slot_bytes = L * (t_row + c_row)
    blob = torch.randint(0, 255, (A, slot_bytes), dtype=torch.uint8)
    t_ext, c_ext = [], []
    off = 0
    for l in range(L):
        t_ext.append((off, t_row)); off += t_row
        segs = []
        for j, n_j in enumerate((2, 3, 1)):
            segs.append((off, n_j * width * e)); off += n_j * width * e
        c_ext.append(segs)
    pool = amp.ArenaMambaPoolHost.__new__(amp.ArenaMambaPoolHost)
    pool._slot_view = blob
    pool._page_bytes = slot_bytes
    pool._state_stage = None
    pool.temporal_dtype = torch.bfloat16
    pool.conv_dtype = torch.bfloat16
    pool._layout = {"L": L, "t_shape": t_shape, "conv_shape": conv_shape, "width": width, "t_ext": t_ext, "c_ext": c_ext}
    temporal = [torch.zeros((16,) + t_shape, dtype=torch.bfloat16) for _ in range(L)]
    conv = [torch.zeros((16,) + conv_shape, dtype=torch.bfloat16) for _ in range(L)]
    dp = types.SimpleNamespace(mamba_cache=types.SimpleNamespace(temporal=temporal, conv=[conv]))
    slots = torch.tensor([5, 1, 7, 2]); didx = torch.tensor([3, 9, 0, 12])
    pool._load_states_all_layers(dp, slots, didx)
    for l in range(L):
        for s, d in zip(slots.tolist(), didx.tolist()):
            off, ln = t_ext[l]
            ref_t = blob[s, off:off + ln].contiguous().view(torch.bfloat16).view(t_shape)
            assert torch.equal(temporal[l][d].contiguous().view(torch.uint8), ref_t.contiguous().view(torch.uint8))  # bytes: random bf16 holds NaN
            ch0 = 0
            for (off_j, ln_j) in c_ext[l]:
                n_j = ln_j // (width * e)
                ref_c = blob[s, off_j:off_j + ln_j].contiguous().view(torch.bfloat16).view(n_j, width)
                assert torch.equal(conv[l][d, ch0:ch0 + n_j].contiguous().view(torch.uint8), ref_c.contiguous().view(torch.uint8))
                ch0 += n_j
