"""H2D phase 1 (a), 28.09.: the load's and the read's sub-stage timers.

NF rc12z26: ``PDFLIP-START-LOADING components_ms mamba=`` median 46 / p90 276 /
max 334 ms of scheduler-thread CPU for ONE 58.8-MB state slot, and the aux
read (``PDFLIP-LOAD-DEVICE read_ms``) 126-560 ms of zero-copy addressing; neither
line said where. These pins keep the instrument: the mamba load's sub-stages
fold into the components line as ``<pool>.<stage>``, the read's stages land on
the operation and print as one ``PDFLIP-READ-STAGES`` line; nothing else changes.
"""
from __future__ import annotations

import logging
import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402
import torch  # noqa: E402


def _state_pool(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_ARENA_STATE_LOAD_BLOCK_BYTES", str(1 << 20))
    from flliper.srt.mem_cache.pool_host import arena_mamba_pool as amp

    L, A = 3, 8
    t_shape = (2, 4); width = 3; conv_shape = (6, width); e = 2
    t_row = 8 * e; c_row = 6 * width * e
    slot_bytes = L * (t_row + c_row)
    blob = torch.randint(0, 255, (A, slot_bytes), dtype=torch.uint8)
    t_ext, c_ext, off = [], [], 0
    for l in range(L):
        t_ext.append((off, t_row)); off += t_row
        segs = []
        for n_j in (2, 3, 1):
            segs.append((off, n_j * width * e)); off += n_j * width * e
        c_ext.append(segs)
    pool = amp.ArenaMambaPoolHost.__new__(amp.ArenaMambaPoolHost)
    pool._slot_view = blob
    pool._page_bytes = slot_bytes
    pool._state_stage = None
    pool.temporal_dtype = torch.bfloat16
    pool.conv_dtype = torch.bfloat16
    pool._layout = {"L": L, "t_shape": t_shape, "conv_shape": conv_shape, "width": width,
                    "t_ext": t_ext, "c_ext": c_ext}
    temporal = [torch.zeros((16,) + t_shape, dtype=torch.bfloat16) for _ in range(L)]
    conv = [torch.zeros((16,) + conv_shape, dtype=torch.bfloat16) for _ in range(L)]
    dp = types.SimpleNamespace(mamba_cache=types.SimpleNamespace(temporal=temporal, conv=[conv]))
    return pool, dp


def test_the_state_load_records_its_sub_stages(monkeypatch):
    pool, dp = _state_pool(monkeypatch)
    pool._load_states_all_layers(dp, torch.tensor([5, 1, 7, 2]), torch.tensor([3, 9, 0, 12]))
    sub = pool._pdflip_load_sub
    for k in ("idx", "issue", "split"):
        assert k in sub and sub[k] >= 0.0


def test_the_timers_change_no_byte(monkeypatch):
    # same bytes land as without the instrument (compared to the untimed copy of the
    # loader's arithmetic: the per-layer reference of test_pdflip_state_load_xsn335)
    pool, dp = _state_pool(monkeypatch)
    slots = torch.tensor([5, 1, 7, 2]); didx = torch.tensor([3, 9, 0, 12])
    pool._load_states_all_layers(dp, slots, didx)
    lay = pool._layout
    for l in range(lay["L"]):
        off, ln = lay["t_ext"][l]
        for s, d in zip(slots.tolist(), didx.tolist()):
            want = pool._slot_view[s, off:off + ln].view(torch.bfloat16).view(lay["t_shape"])
            # bytes, not values: random uint8 bytes read as bf16 hold NaN patterns (NaN != NaN)
            assert torch.equal(dp.mamba_cache.temporal[l][d].view(torch.int16), want.view(torch.int16))


def test_the_hybrid_pool_folds_the_sub_stages_into_the_components_line():
    from flliper.srt.mem_cache import memory_pool_host as mph

    calls = []

    class _Host:
        _pdflip_load_sub = {}

        def load_to_device_per_layer(self, *a):
            calls.append(a)
            self._pdflip_load_sub.update(idx=2.0, issue=3.0, split=5.0)

    host = _Host()
    entry = types.SimpleNamespace(host_pool=host, device_pool=object(), local_layer=lambda i: i)
    hp = mph.HostPoolGroup.__new__(mph.HostPoolGroup)
    hp.anchor_entry = types.SimpleNamespace(local_layer=lambda i: None)
    hp._entry_for_transfer = lambda t, d: entry
    tr = types.SimpleNamespace(name="mamba", host_indices=torch.tensor([1]), device_indices=torch.tensor([2]))
    type(hp).load_to_device_per_layer(hp, None, torch.tensor([], dtype=torch.long), None, 0, "direct",
                                      pool_transfers=[tr])
    acc = hp._pdflip_load_ms
    assert acc["mamba.idx"] == 2.0 and acc["mamba.issue"] == 3.0 and acc["mamba.split"] == 5.0
    assert "mamba" in acc
    assert host._pdflip_load_sub == {}      # cleared for the next start_loading


def _arena_ctl(found):
    from flliper.srt.managers import cache_controller as cc

    class _A:
        def find_slots_np(self, stems):
            return (np.asarray([s for s, _ in found], dtype=np.int64),
                    np.asarray([t for _, t in found], dtype=np.int8))

        def ref_slots_np(self, a, d):
            return len(a)

        def ref_slots(self, s, d):
            return 1

    backend = types.SimpleNamespace(arena_fill_from_disk=lambda ar, st, nb, prefix=False: [900 + i for i in range(len(st))])
    pool = types.SimpleNamespace(arena_read=True, arena=_A(), _page_bytes=64,
                                 ensure_bound=lambda be, role: True, resolve_rows=lambda hi, s: None)
    ctl = types.SimpleNamespace(mem_pool_host=pool, storage_backend=backend, page_size=1)
    op = types.SimpleNamespace(probe_pins=None, completed_tokens=0, n=0, request_id="r1")
    op.increment = lambda k: setattr(op, "n", op.n + k)
    return cc, ctl, op


def test_the_arena_read_records_find_adopt_ref_l3fill_resolve():
    cc, ctl, op = _arena_ctl([(10, 2), (11, 2), (-1, 0), (-1, 0)])
    orig = cc.pdflip_suffixed_stems
    cc.pdflip_suffixed_stems = lambda be, hv: ["s%d" % i for i in range(len(hv))]
    try:
        got = cc.HiCacheController._arena_page_get(ctl, op, [0, 1, 2, 3], None)
    finally:
        cc.pdflip_suffixed_stems = orig
    assert got == 4
    rs = op._pdflip_rs
    for k in ("find", "adopt", "ref", "l3fill", "resolve"):
        assert rs[k] >= 0.0
    assert rs["pages"] == 4 and rs["l3fill_pages"] == 2


def test_one_read_stages_line(caplog):
    from flliper.srt.managers import cache_controller as cc

    op = types.SimpleNamespace(request_id="pdflip-3-8",
                               _pdflip_rs={"find": 1.0, "adopt": 0.5, "ref": 2.0, "l3fill": 40.0,
                                         "resolve": 3.0, "kv": 47.0, "extra": 90.0, "pages": 200,
                                         "l3fill_pages": 1})
    with caplog.at_level(logging.INFO, logger=cc.logger.name):
        cc._read_stages_line(op, 430.0)
    lines = [r.getMessage() for r in caplog.records if "PDFLIP-READ-STAGES" in r.getMessage()]
    assert len(lines) == 1
    for f in ("req=pdflip-3-8", "pages=200", "total_ms=430", "kv_ms=47", "extra_ms=90", "l3fill_ms=40"):
        assert f in lines[0]


def test_the_aux_loop_prints_the_line_after_the_read():
    import inspect
    from flliper.srt.managers import cache_controller as cc

    src = inspect.getsource(cc)
    i = src.index("self._page_transfer(operation)\n                operation.read_end_time")
    assert "_read_stages_line(operation" in src[i:i + 400]
