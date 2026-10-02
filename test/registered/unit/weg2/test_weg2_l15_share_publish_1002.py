# SPDX-License-Identifier: Apache-2.0
"""L15-10 S4n-a: D publishes its held KV (descriptor + fds) for the waking P;
P fetches it; D's wake closes everything."""

from __future__ import annotations

import json
import os
import tempfile
from types import SimpleNamespace

from sglang.srt.weg2 import l15_share_publish as sp
from sglang.srt.weg2.l15_hold_share import L15ShareError


def _span():
    return SimpleNamespace(rid="r1", depth=5, slots=(1, 2, 3, 4, 5), anchor_slot=1)


def test_publish_fetch_close_roundtrip(tmp_path):
    fds = []
    for i in range(3):
        f, _p = tempfile.mkstemp()
        os.write(f, b"%d" % i)
        fds.append(f)
    bases = [{"role": "k", "layer": 0, "view_off": 0, "unit": 64,
              "extents": [[0, 2097152]]},
             {"role": "v", "layer": 0, "view_off": 0, "unit": 64,
              "extents": [[0, 2097152], [4194304, 2097152]]}]
    desc = sp.build_descriptor(epoch=4, rank=1, prefix=[0, 7, 11, 16],
                               bases=bases, spans=[_span()])
    assert desc["n_fds"] == 3
    pub = sp.SharePublisher(str(tmp_path), 1, desc, fds)
    pub.start()
    try:
        on_disk = json.load(open(tmp_path / "D.1.json"))
        assert on_disk["spans"][0]["slots"] == [1, 2, 3, 4, 5]
        header, got = sp.fetch_share(str(tmp_path), 1)
        assert header["epoch"] == 4 and len(got) == 3
        for i, f in enumerate(got):
            os.lseek(f, 0, 0)
            assert os.read(f, 1) == b"%d" % i
            os.close(f)
    finally:
        pub.close()
    assert not (tmp_path / "D.1.json").exists()
    assert not (tmp_path / "D.1.sock").exists()
    for f in fds:
        try:
            os.fstat(f)
        except OSError:
            continue
        raise AssertionError("publisher left fd %d open after close" % f)


def test_fetch_without_a_publisher_is_named(tmp_path):
    try:
        sp.fetch_share(str(tmp_path), 2)
    except L15ShareError as exc:
        assert "D rank 2" in str(exc)
    else:
        raise AssertionError("missing share must raise")


def test_publish_for_sched_exports_each_base_once(tmp_path, monkeypatch):
    import torch

    from sglang.srt.weg2 import l15_hold_share, l15_keep_split

    l15_keep_split.forget_all()
    k = [torch.zeros(8, 4), torch.zeros(8, 4)]
    v = [torch.zeros(8, 4), torch.zeros(8, 4)]
    temporal = torch.zeros(3, 5, 2)     # 3 layers share ONE base
    pool = SimpleNamespace(k_buffer=k, v_buffer=v)
    mc = SimpleNamespace(temporal=temporal, conv=[])
    sched = SimpleNamespace(
        tp_worker=SimpleNamespace(model_runner=SimpleNamespace(token_to_kv_pool=pool)),
        req_to_token_pool=SimpleNamespace(mamba_pool=SimpleNamespace(mamba_cache=mc)),
        ps=SimpleNamespace(tp_rank=1))
    for t in k + v:
        l15_keep_split._HOLD[int(t.data_ptr())] = ((0, 64),)
    tb = int(temporal.data_ptr())
    lay = temporal.stride(0) * temporal.element_size()
    l15_keep_split._HOLD[tb] = ((0, 8), (lay, lay + 8), (2 * lay, 2 * lay + 8))
    calls = []

    def fake_export(ptr, holds, export=None):
        calls.append(ptr)
        out = []
        for o, h in holds:
            f, _p = tempfile.mkstemp()
            out.append((o, h - o, f))
        return out

    monkeypatch.setattr(l15_hold_share, "export_hold_extents", fake_export)
    monkeypatch.setattr(
        "sglang.srt.distributed.utils.get_cp_token_ratios", lambda: [7, 4, 5])
    man = SimpleNamespace(epoch=3, spans=[_span()])
    pub = sp.publish_for_sched(sched, man, {"SGLANG_WEG2_L15_SHARE_DIR": str(tmp_path)},
                               lambda m: None)
    try:
        assert pub is not None
        d = pub.descriptor
        assert len(calls) == 5, "each base exported once (4 kv + 1 temporal)"
        assert d["n_fds"] == 4 + 3 and len(pub.fds) == 7
        tviews = [b for b in d["bases"] if b["role"] == "mamba_temporal"]
        assert [len(b["extents"]) for b in tviews] == [1, 1, 1]
        assert len({b["extents"][0][2] for b in tviews}) == 3
        assert d["prefix"] == [0, 7, 11, 16]
    finally:
        pub.close()
        l15_keep_split.forget_all()
