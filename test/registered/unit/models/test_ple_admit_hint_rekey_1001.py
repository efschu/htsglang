"""1001: the front's PLE hint reads the prompt's tail; the told re-keys it.

NF bfpgwv (boot ...dauer10011823_a11cc7cbc6), PP0: 27 hint admissions, 0
used. Every hint was read from token 0 (``admit rid=pdflip-12-87 rows=262144
started source=hint dormant=1 start=0``) and dropped at PP0's told as
``tokens_differ`` -- the told named the first chunk behind the store prefix
(``rows=155040 started source=told ... start=84544``), and that read started
~150 ms before the forward (``ready=no``). Same in 10011949 / 10012029 / y6o
(14 / 17 / 14 drops, every one a different start, never different tokens).
Now: the hint reads the last chunk of the prompt; an exact admission whose
chunk lies inside that window takes its rows where they are (``rekeyed``);
one outside drops as ``start_moved``; exact admissions wait ahead of hints.
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

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


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


def _warm(r):
    r.chunk_of(h43._Req("warm", torch.randint(0, 256, (100,))), 0)


def _req(rid, n, seed):
    return h43._Req(rid, torch.randint(0, 300, (n,), generator=torch.Generator().manual_seed(seed)))


def test_the_hint_reads_the_tail_and_the_told_rekeys_it(rig, caplog):
    """pdflip-12-87's shape (94234 tokens, told 84544, chunk 16384) scaled to the
    rig's chunk of 200: 500 tokens, store prefix 420. The told arrives right
    before the forward -- the hint's read, done long before, serves it."""
    _warm(rig)
    w = _req("pdflip-12-87", 500, 1)
    with caplog.at_level(logging.INFO):
        assert adm.admit_ple_hint(w.rid, w.origin_input_ids, rig.chunk, dormant=True) == "started"
        assert rig.g._adm.start == 300  # the last chunk of the prompt
        time.sleep(0.5)  # D decodes, the flip runs: the hint's read finishes
        assert adm.admit_ple_request(w, rig.chunk, start=420, source="told") == "rekeyed"
        t = time.monotonic()
        ids, out = rig.batch([(w, 420, 500)])  # the first chunk, right after the told
        fwd_s = time.monotonic() - t
    assert h43._same(out, h43._serial(rig.table, ids))
    line = h43._chunks(caplog.records)[-1]
    assert (line["ready"], line["hit"], line["read"]) == ("yes", 80, 0)
    assert line["wait_ms"] < 60.0 and fwd_s < 0.2  # not the 0.3 s read
    lines = h43._admit_lines(caplog.records)
    assert not any("dropped" in m for m in lines), lines
    assert any("rid=pdflip-12-87 rekeyed source=told (admitted by hint) start=300->420 rows=80 row_off=120"
               in m for m in lines), lines
    assert rig.g.stats["admit_started"] == 1 and rig.g.stats["admit_used"] == 1


def test_an_intake_inside_the_window_rekeys_too(rig, caplog):
    _warm(rig)
    w = _req("w", 450, 2)
    adm.admit_ple_hint("w", w.origin_input_ids, rig.chunk)
    time.sleep(0.45)
    w._prefetch_registered_prefix_len = 256  # no store read: the registered head
    assert adm.admit_ple_request(w, rig.chunk) == "rekeyed"
    with caplog.at_level(logging.INFO):
        ids, out = rig.batch([(w, 256, 450)])
    assert h43._same(out, h43._serial(rig.table, ids))
    assert h43._chunks(caplog.records)[-1]["read"] == 0


def test_a_told_before_the_window_drops_as_start_moved(rig, caplog):
    """The rest behind the store prefix is longer than one chunk (pdflip-42-307:
    told 22400, many chunks): no re-key, named, the chunk gathered right."""
    _warm(rig)
    w = _req("w", 500, 3)
    adm.admit_ple_hint("w", w.origin_input_ids, rig.chunk)
    time.sleep(0.45)
    with caplog.at_level(logging.INFO):
        v = adm.admit_ple_request(w, rig.chunk, start=40, source="told")
        assert v in ("started", "queued")
        time.sleep(0.45)
        ids, out = rig.batch([(w, 40, 240)])
    assert h43._same(out, h43._serial(rig.table, ids))
    lines = h43._admit_lines(caplog.records)
    assert any("rid=w dropped reason=start_moved" in m for m in lines), lines
    assert not any("tokens_differ" in m for m in lines), lines


def test_different_tokens_in_the_window_drop_as_tokens_differ(rig, caplog):
    _warm(rig)
    w = _req("w", 500, 4)
    other = list(w.origin_input_ids)
    other[450] = (other[450] + 1) % 300  # a re-rendered tail
    adm.admit_ple_hint("w", other, rig.chunk)
    time.sleep(0.45)
    with caplog.at_level(logging.INFO):
        assert adm.admit_ple_request(w, rig.chunk, start=300, source="told") in ("started", "queued")
        time.sleep(0.45)
        ids, out = rig.batch([(w, 300, 500)])
    assert h43._same(out, h43._serial(rig.table, ids))
    assert any("rid=w dropped reason=tokens_differ" in m for m in h43._admit_lines(caplog.records))


def test_exact_admissions_wait_ahead_of_hints(rig, caplog):
    _warm(rig)
    hold = _req("hold", 150, 5)
    assert adm.admit_ple_request(hold, rig.chunk) == "started"  # holds the slot
    for i in range(3):
        h = _req(f"h{i}", 150, 10 + i)
        assert adm.admit_ple_hint(h.rid, h.origin_input_ids, rig.chunk) == "queued"
    x = _req("x", 150, 20)
    assert adm.admit_ple_request(x, rig.chunk, start=0, source="told") == "queued"
    assert list(rig.g._adm_queue) == ["x", "h0", "h1", "h2"]
    # a full queue: the next exact admission displaces the newest hint
    y = _req("y", 150, 21)
    with caplog.at_level(logging.INFO):
        assert adm.admit_ple_request(y, rig.chunk, start=0, source="told") == "queued"
    assert list(rig.g._adm_queue) == ["x", "y", "h0", "h1"]
    assert any("rid=h2 dropped reason=displaced_by_exact" in m for m in h43._admit_lines(caplog.records))
    time.sleep(0.45)
    rig.chunk_of(hold, 0)  # hold served: the slot frees, x reads first
    assert rig.g._adm is not None and rig.g._adm.rid == "x"


def test_a_short_prompt_hint_reads_it_whole_as_before():
    assert adm.ple_hint_start(150, 200) == 0
    assert adm.ple_hint_start(94234, 16384) == 77850
    assert adm.ple_hint_start(500, None) == 0
