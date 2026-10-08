"""DP-NACHLAUF 02.10.: the PP pass names its segments.

N5d (0c996cf05c 1002_124821, D->P epoch 25): PP0 `#1466 PASS-STALL ...
input_ms=303` with vote/send/proc_input ~0, PP1/PP2 `other_ms=378` -- whole
passes between P's first store read and its first prefill forward that no
term named. The pass gets a segment clock: `_pp_seg(self, name)` marks the end
of a named segment (loop body: recv, input, pre_admission, admission_ring,
void_check, launch_proxy, commit_proxy; intake: in.vote_hook, in.trace,
in.commit_send, in.told_publish, in.send, in.vote_after, in.absorb,
in.process), and PASS-STALL prints `segs[name=ms,...] seg_t0=<epoch s>`.
Instrument only (red before: no _pp_seg / _pp_segs_text)."""
from __future__ import annotations

import inspect
import os
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.managers import scheduler_pp_mixin as m  # noqa: E402


def test_segments_largest_first_with_rest():
    segs = [("start", 10.0), ("recv", 10.01), ("input", 10.31), ("void_check", 10.33)]
    assert m._pp_segs_text(segs, 10.73) == "rest=400,input=300,void_check=20,recv=10"
    assert m._pp_segs_text(segs, 10.73, top=2) == "rest=400,input=300"
    assert m._pp_segs_text([], 1.0) == ""


def test_seg_marks_append_and_never_raise():
    o = SimpleNamespace()
    m._pp_seg(o, "a")
    m._pp_seg(o, "b")
    assert [n for n, _ in o._pp_segs] == ["a", "b"]
    m._pp_seg(object(), "x")   # no __dict__: silently nothing


def test_the_pass_and_the_intake_are_marked_and_printed():
    src = inspect.getsource(m)
    for name in ("recv", "input", "pre_admission", "admission_ring", "void_check", "launch_proxy",
                 "commit_proxy", "in.vote_hook", "in.trace", "in.commit_send", "in.told_publish",
                 "in.send", "in.vote_after", "in.absorb", "in.process"):
        assert '_pp_seg(self, "%s")' % name in src, name
    assert "segs[%s] seg_t0=%.3f" in src
    assert "_pp_segs_text(_pp_segs_prev, _1466_now)" in src
