# SPDX-License-Identifier: Apache-2.0
"""D-NORECOMPUTE / SEQ-HASH (NF 02ddfc4906) on the 27B line -- the two 27B cases.

* D-NACHRECHNEN, b9a960c2b4 (D log ..._1002_065831.D.log) pdflip-0-3: the wake read
  delivered 90112 of 92866 ('HiCache prefetch INCOMPLETE req=pdflip-0-3 ...
  deliverable=92866 shortfall=2754'), '#1471 SETTLE-RELEASE rid=pdflip-0-3
  state=reissued' at 07:02:07, then 'PREFETCH-DEFER-FALLBACK rid=pdflip-0-3
  delivered=90112 tail=2759 ... reason=settled_no_writer' -- D extended 2759
  tokens after the flip. The settle release now asks d_would_compute and sends
  it through RESUME-VIA-P (P prefills from the store, D resumes under E2).
* the ~4k class, N3y 6dd8d7e68c (front log ..._1002_083747): pdflip-12-32 (prompt
  62604, 4045 decoded) finished 08:46:08.622 with 'PRESENCE-OWN-TEXT-CLAMP ...
  cached_d=62592 prompt_d=62604 depth_d=66560 credited=62592'; pdflip-14-37 (66942,
  reused=62604) priced 08:46:10.185 'pending=4350 ... credit=62592' -> LONG at
  X=4096 in a D phase, P read exactly 66560 and computed 382. With D's sequence
  mark at the finish it is credited 66560 (src=d_seq_anchor) -> SHORT.
"""
from __future__ import annotations

import collections
import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402

from flliper.srt.managers import pdflip_seq_hash as SH  # noqa: E402
from flliper.srt.pdflip import d_norecompute as DN  # noqa: E402
from flliper.srt.pdflip import front as F  # noqa: E402
from flliper.srt.pdflip.front_tokens import TokenSpans  # noqa: E402


def _ids(n, salt):
    rng = np.random.default_rng(4242 + salt)
    return rng.integers(0, 150000, size=n, dtype=np.int64).astype(np.int32)


# ---- b9a960c2b4 pdflip-0-3 ----------------------------------------------------------------

def _req_0_3(stream=True):
    n = 92871  # its prompt; the fork-cut read span ended at 92867 (D extends the tail itself)
    return types.SimpleNamespace(rid="pdflip-0-3", full_untruncated_fill_ids=list(range(n)),
                                 _pdflip_store_delivered=90112, stream=stream,
                                 multimodal_inputs=None, origin_input_ids=list(range(n)),
                                 output_ids=[])


def test_b9a_pdflip_0_3_d_would_compute_the_tail():
    assert DN.d_would_compute(_req_0_3()) == 92871 - 90112 == 2759, "the log's tail=2759"


def test_b9a_pdflip_0_3_goes_resume_via_p_not_to_ds_extend(monkeypatch, tmp_path):
    from flliper.srt.pdflip import d_park_runtime as dpr
    from flliper.srt.pdflip import resume_via_p as rvp

    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "D")
    monkeypatch.setenv("FLLIPER_HICACHE_ARENA_DIR", str(tmp_path))
    monkeypatch.delenv("FLLIPER_PDFLIP_D_NORECOMPUTE", raising=False)
    sched = types.SimpleNamespace(server_args=types.SimpleNamespace(tp_prefill_max_tokens=4096),
                                  ps=types.SimpleNamespace(tp_rank=0), pdflip_d_parked=[])
    req = _req_0_3()
    assert DN.enabled()
    assert DN.divert(sched, req, DN.d_would_compute(req)) is True
    assert any(r is req for r in dpr.parked_list(sched)), "kept parked on D for P"
    recs = rvp.take_requests(rvp.needs_p_dir())
    assert [r["rid"] for r in recs] == ["pdflip-0-3"] and recs[0]["reason"] == DN.REASON


def test_the_switch_off_releases_as_before(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_D_NORECOMPUTE", "0")
    assert not DN.enabled()


# ---- N3y pdflip-12-32 -> pdflip-14-37 -------------------------------------------------------------

def _front():
    f = object.__new__(F.Front)
    f.counters = collections.Counter()
    f.x_exact = True
    f.epoch = 14
    f.awake = "D"
    f.state = "serving"
    f.tp_prefill_max_tokens = 4096
    f.queue = collections.deque()
    f.tspans = TokenSpans(agent_span=True)
    f.ftok = types.SimpleNamespace(m={}, ids_for=lambda t, _m=None: None)
    return f


def test_n3y_pdflip_14_37_extends_12_32s_generated_tokens_and_is_short():
    prompt = _ids(62604, 1)                       # pdflip-12-32's prompt
    out = _ids(4045, 2)                           # its 4045 decoded tokens
    seq = np.concatenate([prompt, out])
    mark = SH.mark(seq, 66560)                    # D: depth_d=66560 at the finish
    f = _front()
    ts = f.tspans
    # the finish reading as the log records it (K2 own-text clamp -> 62592)
    ts.record_presence(prompt, 62592, prompt_tokens=62604, held_epoch=None, resumable_depth=66560)
    nxt = np.concatenate([seq[:66560], _ids(66942 - 66560, 3)])  # pdflip-14-37: 66942 tokens
    before = ts.pending(nxt)
    assert before[:2] == (4350, 62592), "the log's price: LONG at X=4096"
    assert f._seq_record("pdflip-12-32", prompt, mark, "finish") is True
    pending, credit, known, src = ts.pending(nxt)
    assert (credit, src, known) == (66560, "d_seq_anchor", True)
    assert pending == 382 <= 4096, "SHORT -- P computed exactly 382 on its 66560 store hit"


def test_a_follow_up_that_does_not_carry_the_output_gets_no_seq_credit():
    prompt = _ids(62604, 1)
    seq = np.concatenate([prompt, _ids(4045, 2)])
    f = _front()
    f._seq_record("pdflip-12-32", prompt, SH.mark(seq, 66560), "finish")
    other = np.concatenate([prompt, _ids(4338, 9)])   # same prompt, a different continuation
    assert f.tspans.pending(other)[3] != "d_seq_anchor"


def test_a_flush_lost_seq_depth_is_dropped():
    prompt = _ids(62604, 1)
    seq = np.concatenate([prompt, _ids(4045, 2)])
    f = _front()
    f._seq_record("pdflip-12-32", prompt, SH.mark(seq, 66560), "finish")
    F.retract_lost_anchors(f.tspans, [66560])
    nxt = np.concatenate([seq[:66560], _ids(382, 3)])
    assert f.tspans.pending(nxt)[3] != "d_seq_anchor"


def test_an_sk_x_fresh_confirmation_read_ignores_seq_marks():
    # 27B port: pending(since_seq=...) asks for D evidence recorded after an SK-X void;
    # a seq mark is not such evidence
    prompt = _ids(62604, 1)
    seq = np.concatenate([prompt, _ids(4045, 2)])
    f = _front()
    f._seq_record("pdflip-12-32", prompt, SH.mark(seq, 66560), "finish")
    nxt = np.concatenate([seq[:66560], _ids(382, 3)])
    assert f.tspans.pending(nxt, since_seq=10 ** 9)[1] == 0
