"""02.10. NF y7l: the front's hint names P's first chunk; PP0's told confirms it.

Boot ...dauer10021206_99d1977a63_1002_120637, every D->P wake (4/4): the hint
read the prompt's tail (``started source=hint dormant=1 start=63100``), PP0's
told named the chunk behind the store prefix (``start=6208``), the hint dropped
``start_moved`` and P's first forward after the flip waited on PP0's own read
(``PLE-PREFETCH chunk=0 ready=no wait_ms=540.8``; 406 / 241 at later wakes).
The front had the start (``DP-WAIT ... presence_span=6208 span_known=True``):
the hint now carries it (``start_hint``), P floors it to its page, and the told
at that start is ``confirmed`` -- the forward finds its rows (``ready=yes``).
RED on 99d1977a63 (``admit_ple_hint`` takes no ``start_hint``).
Desk: the real admitting gather and worker processes of the H43 test.
"""

import logging
import os
import sys
import time
from unittest import mock

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_qwen4_exp_ple_admit_h43 as h43  # noqa: E402

from flliper.srt.models import qwen4_exp_ple_admit as adm  # noqa: E402
from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


@pytest.fixture
def rig(tmp_path):
    clock = h43._Clock()
    with mock.patch.object(adm, "_clock", clock):
        r = h43._Rig(h43._table(str(tmp_path)), delay_s=0.3)
        r.clock = clock
        try:
            yield r
        finally:
            r.close()
    assert adm._SINKS == []


def _req(rid, n, seed):
    return h43._Req(rid, torch.randint(0, 300, (n,), generator=torch.Generator().manual_seed(seed)))


def test_y7l_shape_the_told_confirms_the_spanned_hint(rig, caplog):
    """pdflip-2-7 (79484 tokens, store span 6208, chunk 16384, page 64) scaled to
    the rig's chunk of 200: 800 tokens, store span 70 -> page 8 -> told 64. The
    tail window would be 600..800 and the told at 64 would drop it."""
    rig.chunk_of(h43._Req("warm", torch.randint(0, 256, (100,))), 0)
    w = _req("pdflip-2-7", 800, 7)
    with caplog.at_level(logging.INFO):
        assert adm.admit_ple_hint(w.rid, w.origin_input_ids, rig.chunk, dormant=True,
                                  start_hint=70, page_size=8) == "started"
        assert rig.g._adm.start == 64
        time.sleep(0.5)  # the flip runs: the hint's read finishes
        assert adm.admit_ple_request(w, rig.chunk, start=64, source="told") == "confirmed"
        t = time.monotonic()
        ids, out = rig.batch([(w, 64, 264)])  # P's first chunk after the flip
        fwd_s = time.monotonic() - t
    assert h43._same(out, h43._serial(rig.table, ids))
    line = h43._chunks(caplog.records)[-1]
    assert (line["ready"], line["read"]) == ("yes", 0)
    assert line["wait_ms"] < 60.0 and fwd_s < 0.2  # not the 0.3 s read
    lines = h43._admit_lines(caplog.records)
    assert not any("dropped" in m for m in lines), lines
    assert any("rid=pdflip-2-7 confirmed source=told (admitted by hint)" in m for m in lines), lines
    assert rig.g.stats["admit_started"] == 1 and rig.g.stats["admit_used"] == 1


def test_without_the_span_the_same_told_drops_the_tail_hint(rig, caplog):
    """The base form, for contrast: the tail window, start_moved, a cold read."""
    rig.chunk_of(h43._Req("warm", torch.randint(0, 256, (100,))), 0)
    w = _req("pdflip-2-7", 800, 7)
    adm.admit_ple_hint(w.rid, w.origin_input_ids, rig.chunk, dormant=True)
    assert rig.g._adm.start == 600
    time.sleep(0.5)
    with caplog.at_level(logging.INFO):
        assert adm.admit_ple_request(w, rig.chunk, start=64, source="told") in ("started", "queued")
        time.sleep(0.45)
        ids, out = rig.batch([(w, 64, 264)])
    assert h43._same(out, h43._serial(rig.table, ids))
    assert any("rid=pdflip-2-7 dropped reason=start_moved" in m for m in h43._admit_lines(caplog.records))
