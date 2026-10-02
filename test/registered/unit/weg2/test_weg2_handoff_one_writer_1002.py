"""DP-NACHLAUF 02.10.: the #1442 hand-off file is written by PP0 alone, in one
shot.

N5t epoch 11 (P>D quiesce 443 ms): PP0 served at 50.972, PP1 finished its
last chunk at 51.076, PP2 at 51.319 (PASS-TAIL process_ms 226), the idle
200 at 51.424. Every stage wrote the SAME /dev/shm hand-off file at its
finish (json.dump, 7.5 MB, pure-Python streaming encoder) -- PP2's write is on
the path to P's idle. Pinned (red before): only stage 0 writes with the switch
on; every stage with it off; the file bytes of the one-shot writer equal
json.dump's; a later stage still marks the hand-off pending (#243) with the
exact key count; D's readers get the same ids and keys.
"""
from __future__ import annotations

import inspect
import json
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import handoff as ho  # noqa: E402


def test_one_writer_rule(monkeypatch):
    monkeypatch.delenv(ho.ONE_WRITER_ENV, raising=False)
    assert ho.writes_on_stage(0) and not ho.writes_on_stage(1) and not ho.writes_on_stage(2)
    assert ho.writes_on_stage(None)
    monkeypatch.setenv(ho.ONE_WRITER_ENV, "0")
    assert ho.writes_on_stage(1) and ho.writes_on_stage(2)


def test_one_shot_file_equals_streamed_and_reads_back(tmp_path, monkeypatch):
    monkeypatch.setenv("SGLANG_HICACHE_ARENA_DIR", str(tmp_path))
    monkeypatch.delenv(ho.ENV, raising=False)
    ids = list(range(1000, 1500))
    keys = [f"{i:064x}" for i in range(499)]
    monkeypatch.delenv(ho.ONE_WRITER_ENV, raising=False)
    assert ho.write("weg2-1-1", ids, keys)
    one = open(ho.path("weg2-1-1"), "rb").read()
    monkeypatch.setenv(ho.ONE_WRITER_ENV, "0")
    assert ho.write("weg2-1-1", ids, keys)
    streamed = open(ho.path("weg2-1-1"), "rb").read()
    assert one == streamed
    assert json.loads(one) == {"input_ids": ids, "page_keys": keys}
    assert ho.read_ids("weg2-1-1") == ids and ho.read_keys("weg2-1-1") == keys


def test_tree_skips_the_write_on_later_stages_but_keeps_the_mark():
    from sglang.srt.mem_cache import unified_radix_cache as urc

    src = inspect.getsource(urc.UnifiedRadixCache._weg2_handoff_write)
    i_rule = src.index("_ho.writes_on_stage(")
    i_write = src.index("(not _writer) or _ho.write(rid, ids, keys)")
    i_mark = src.index("_hp.mark(rid, len(keys)")
    assert i_rule < i_write < i_mark
