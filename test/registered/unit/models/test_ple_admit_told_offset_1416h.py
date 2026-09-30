"""#1416h: the PLE first-chunk admission starts at PP0's told offset.

NF z30e (ca2a9706ec, boot ...stvsyncbar1dauer09282117), PP0: 33 admissions
``PLE-PREFETCH admit ... rows=262144 started`` (token 0, at intake), 31 of them
``dropped reason=cached_prefix`` -- the first chunk started behind a store
prefix -- and that chunk was then gathered cold inside the forward
(``PLE-PREFETCH chunk=1 ... ready=none wait_ms`` mean 186, max 511 ms).
Now: at intake a request with a store read in flight is not admitted; PP0's
told admits it at the offset its first chunk will start; without a store
read the first chunk starts at the registered head. Desk: the real admitting
gather and worker processes of the H43 test, the real told publisher.
"""

import os
import sys
import time
import logging
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "managers"))

import test_qwen4_exp_ple_admit_h43 as h43  # noqa: E402

from flliper.srt.models import qwen4_exp_ple_admit as adm  # noqa: E402
from flliper.srt.models import qwen4_exp_ple_prefetch as pf  # noqa: E402
from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


@pytest.fixture
def rig(tmp_path):
    clock = h43._Clock()
    with mock.patch.object(adm, "_clock", clock):
        r = h43._Rig(h43._table(str(tmp_path)), delay_s=0.2)
        r.clock = clock
        try:
            yield r
        finally:
            r.close()
    assert adm._SINKS == []


def _req(rid, n, gen, **attrs):
    q = h43._Req(rid, torch.randint(0, 300, (n,), generator=gen))
    for k, v in attrs.items():
        setattr(q, k, v)
    return q


def _warm(r):
    w0 = h43._Req("warm", torch.randint(0, 256, (100,)))
    r.chunk_of(w0, 0)


def test_the_told_offset_admission_is_used_by_the_first_chunk(rig, caplog):
    """pdflip-8-29's shape: store prefix 40 of 190 tokens; the admission at the
    told reads rows 40..190 ahead, the chunk joins it ready and reads nothing."""
    _warm(rig)
    g = torch.Generator().manual_seed(3)
    w = _req("pdflip-8-29", 190, g)
    assert adm.admit_ple_request(w, rig.chunk, start=40, source="told") == "started"
    a = rig.g._adm
    assert (a.start, a.lead) == (40, pf.PLE_PREFETCH_LEAD_TOKENS)
    assert torch.equal(a.tokens, w.ids[40 - a.lead:190])
    time.sleep(0.35)
    with caplog.at_level(logging.INFO):
        ids, out = rig.batch([(w, 40, 190)])
    assert h43._same(out, h43._serial(rig.table, ids))
    line = h43._chunks(caplog.records)[-1]
    assert (line["ready"], line["hit"], line["read"]) == ("yes", 150, 0)
    assert rig.g.stats["admit_used"] == 1
    assert not any("dropped reason=cached_prefix" in m for m in h43._admit_lines(caplog.records))


def test_intake_with_a_store_read_in_flight_waits_for_the_told(rig):
    _warm(rig)
    g = torch.Generator().manual_seed(4)
    w = _req("w", 190, g, _969c_verdict="issued", _prefetch_registered_prefix_len=0)
    assert adm.admit_ple_request(w, rig.chunk) == "skipped:told_pending"
    assert rig.g._adm is None and not rig.g._adm_queue
    t = _req("t", 190, g, _969c_verdict="declined:pdflip_twin_deferred")
    assert adm.admit_ple_request(t, rig.chunk) == "skipped:told_pending"


def test_intake_without_a_store_read_starts_at_the_registered_head(rig, caplog):
    _warm(rig)
    g = torch.Generator().manual_seed(5)
    w = _req("w", 190, g, _969c_verdict="declined:store_absent", _prefetch_registered_prefix_len=64)
    assert adm.admit_ple_request(w, rig.chunk) == "started"
    assert rig.g._adm.start == 64
    time.sleep(0.35)
    with caplog.at_level(logging.INFO):
        ids, out = rig.batch([(w, 64, 190)])
    assert h43._same(out, h43._serial(rig.table, ids))
    assert h43._chunks(caplog.records)[-1]["read"] == 0


def test_a_moved_start_drops_and_the_chunk_is_gathered_as_before(rig, caplog):
    _warm(rig)
    g = torch.Generator().manual_seed(6)
    w = _req("w", 190, g)
    adm.admit_ple_request(w, rig.chunk, start=64, source="told")
    time.sleep(0.3)
    with caplog.at_level(logging.INFO):
        ids, out = rig.batch([(w, 40, 190)])  # the extend started earlier than told
    assert h43._same(out, h43._serial(rig.table, ids))
    assert any("rid=w dropped reason=start_moved" in m for m in h43._admit_lines(caplog.records))


# --------------------------------------------------------------------------
# The told publisher admits at the offset (PP0, both publish forms).
# --------------------------------------------------------------------------

from flliper.srt.managers import pdflip_store_told as told_mod  # noqa: E402


@pytest.mark.parametrize("told,absolute,head,want", [
    (16384, True, 0, 16384),      # TK absolute told (the z30e form)
    (16384, False, 2560, 18944),  # span-relative told on a registered head
    (0, False, 2560, 2560),       # nothing from the store: the head
])
def test_the_hook_computes_the_first_chunk_start(told, absolute, head, want):
    calls = []
    with mock.patch.object(adm, "ple_admission_armed", lambda: True), \
            mock.patch.object(adm, "admit_ple_request",
                              lambda req, cs, **kw: calls.append((cs, kw))):
        sched = SimpleNamespace(chunked_prefill_size=16384, pdflip_dormant=False)
        req = SimpleNamespace(rid="r", _prefetch_registered_prefix_len=head)
        told_mod._ple_admit_at_told(sched, req, told, absolute)
    assert calls == [(16384, {"dormant": False, "start": want, "source": "told"})]


@pytest.mark.parametrize("paced", [True, False])
def test_the_publisher_calls_the_hook_with_pp0s_told(monkeypatch, paced):
    import test_pdflip_store_told_paced_1416e as ring_mod  # the #1416e ring (managers dir)

    seen = []
    monkeypatch.setattr(told_mod, "_ple_admit_at_told",
                        lambda s, req, told, absolute: seen.append((req.rid, told)), raising=False)
    ring = ring_mod._Ring(monkeypatch, {"aaaa-told": 16384}, {r: {"aaaa-told": 0.1} for r in range(3)},
                          paced=paced)
    ring.arrive("aaaa-told")
    ring.run(20)
    assert seen == [("aaaa-told", 16384)]
