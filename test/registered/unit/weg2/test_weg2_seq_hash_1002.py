"""SEQ-HASH (D-NORECOMPUTE (d), 02.10.): a follow-up that extends the previous
turn's GENERATED tokens is credited with D's anchor on that sequence.

y6y (0e1967fd36): weg2-18-117 finished 08:06:50.48 'PRESENCE-OWN-TEXT-CLAMP ...
prompt_d=85001 depth_d=89344 credited=84992'; weg2-18-130 (89398) priced 4406 >
X=4305 at 08:06:52.7 -> LONG, P read exactly 89344 and computed 54.
weg2-16-92 (79960, 08:01:53.5) priced 11544 > 4373 -> LONG, P read 78912 and
computed 1048: it extends weg2-12-44's decode (prompt 68476; '#59b
PARK-RESUMABLE weg2-12-44=78080' 08:00:02, '=79104' 08:01:19, finish 79872).
"""
from __future__ import annotations

import asyncio
import collections
import inspect
import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402

from sglang.srt.managers import weg2_seq_hash as SH  # noqa: E402
from sglang.srt.weg2 import front as F  # noqa: E402
from sglang.srt.weg2.front_tokens import TokenSpans  # noqa: E402


def _seq(n, salt=0):
    rng = np.random.default_rng(1234 + salt)
    return rng.integers(0, 150000, size=n, dtype=np.int64).astype(np.int32)


class _Tok:
    def __init__(self):
        self.m = {}

    def ids_for(self, text):
        return self.m.get(text)


def _front(epoch=18):
    f = object.__new__(F.Front)
    f.counters = collections.Counter()
    f.x_exact = True
    f.epoch = epoch
    f.awake = "D"
    f.state = "serving"
    f.tp_prefill_max_tokens = 4305
    f.queue = collections.deque()
    f.tspans = TokenSpans(agent_span=True)
    f.ftok = _Tok()
    f._x_exact_rid = collections.OrderedDict()
    return f


def test_the_mark_is_deterministic_and_parses():
    a = _seq(1000)
    assert SH.mark(a, 640) == SH.mark(list(int(x) for x in a), 640)
    d, h = SH.parse(SH.mark(a, 640))
    assert d == 640 and len(h) == 32
    assert SH.mark(a, 2000) is None and SH.parse("x") is None and SH.parse("12:short") is None
    b = a.copy()
    b[639] += 1
    assert SH.digest(b, 640) != SH.digest(a, 640) and SH.digest(b, 639) == SH.digest(a, 639)


def test_weg2_18_130_priced_short_on_18_117s_decode_anchor():
    f = _front()
    seq = _seq(85001 + 4379)                    # 18-117: prompt + its 4379 decoded tokens
    prompt = seq[:85001]
    f.ftok.m["18-117"] = prompt
    f._x_exact_record("weg2-18-117", "18-117", 85001, 84736, None, 18,
                      resumable_depth=89344, seq_mark=SH.mark(seq, 89344))
    b = np.concatenate([seq[:89344], _seq(54, salt=9)])    # 18-130: 89398 tokens
    pending, credit, known, src = f.tspans.pending(b, epoch=18)
    assert (credit, src) == (89344, "d_seq_anchor")
    assert pending == 54 <= 4305, "SHORT (P computed 54 on an 89344 hit)"
    assert f.counters["seq_marks"] == 1


def test_weg2_16_92_priced_short_on_12_44s_park_anchor():
    f = _front(epoch=16)
    seq = _seq(79892, salt=3)                   # 12-44's sequence (prompt 68476 + 11416)
    f.ftok.m["12-44"] = seq[:68476]
    f._leg2_text = {"weg2-12-44": "12-44"}
    f._x_exact_reprice_queue = lambda why: 0
    marks = {"weg2-12-44": SH.mark(seq, 78080)}
    assert f._seq_park(marks) == 1
    f._seq_park({"weg2-12-44": SH.mark(seq, 79104)})
    b = np.concatenate([seq[:78950], _seq(79960 - 78950, salt=4)])  # diverges at 78950
    pending, credit, _k, src = f.tspans.pending(b, epoch=None)
    assert (credit, src) == (78080, "d_seq_anchor"), "79104 lies past the divergence"
    assert pending == 1880 <= 4373, "SHORT (P computed 1048 on a 78912 hit)"


def test_a_prompt_that_does_not_extend_the_sequence_gets_nothing():
    ts = TokenSpans(agent_span=True)
    seq = _seq(5000)
    ts.record_seq(seq[:3000], 4480, SH.digest(seq, 4480))
    other = seq.copy()
    other[2000] += 1                            # leaves A's prompt
    assert ts.seq_credit(other[:4600]) == (0, None)
    assert ts.seq_credit(seq[:4600]) == (4480, 3000)
    assert ts.record_seq(seq[:3000], 2000, SH.digest(seq, 2000)) is False, "inside the prompt: LCP prices it"


def test_anchor_lost_drops_the_seq_marks():
    f = _front()
    seq = _seq(5000)
    f.tspans.record_seq(seq[:3000], 4480, SH.digest(seq, 4480))
    F.retract_lost_anchors(f.tspans, [4480])
    assert f.tspans.seq_credit(seq[:4600]) == (0, None)


def test_the_park_answer_carries_marks_from_the_tokenizer_manager():
    from sglang.srt.managers import tokenizer_control_mixin as TCM

    seq = list(_seq(2000))
    out = types.SimpleNamespace(weg2_resumable_depth={"r1": 1920, "gone": 640}, weg2_seq_hash={})

    class _Self:
        rid_to_state = {"r1": types.SimpleNamespace(prompt_token_ids=None, weg2_prompt_ids=seq[:1500],
                                                     output_ids=seq[1500:])}

        def auto_create_handle_loop(self):
            pass

        async def weg2_park_running_communicator(self, obj):
            return [out]

    got = asyncio.run(TCM.TokenizerControlMixin.weg2_park_running(_Self(), None))
    assert got.weg2_seq_hash == {"r1": SH.mark(seq, 1920)}


def test_the_wire_carries_the_mark():
    from sglang.srt.entrypoints.anthropic import serving as AS
    from sglang.srt.entrypoints.openai.protocol import SglExt

    ext = SglExt(weg2_resumable_depth=89344, weg2_seq_hash="89344:" + "a" * 32)
    obj = types.SimpleNamespace(sglext=ext)
    assert AS._sglext_of(obj).weg2_seq_hash == "89344:" + "a" * 32
    assert F.d_seq_mark({"sglext": {"weg2_seq_hash": "1:" + "b" * 32}}) == "1:" + "b" * 32
    tail = b'data: {"type":"message_delta","sglext":{"weg2_resumable_depth":5,"weg2_seq_hash":"5:' + \
        b"c" * 32 + b'"}}\n\n'
    assert F.d_seq_mark_stream_tail(tail) == "5:" + "c" * 32
    assert F.park_seq_marks('{"parked":[],"weg2_seq_hash":{"r":"7:' + "d" * 32 + '"}}') == {"r": "7:" + "d" * 32}
    from sglang.srt.managers import tokenizer_manager as TM

    src = inspect.getsource(TM)
    i = src.index("meta_info[weg2_resumable_depth.FIELD] = resumable")
    assert "meta_info[_weg2_seq_hash.FIELD] = _sm" in src[i:i + 800]
