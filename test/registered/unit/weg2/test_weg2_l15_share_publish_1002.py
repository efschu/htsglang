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
