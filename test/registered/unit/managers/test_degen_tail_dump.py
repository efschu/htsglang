"""TAIL DUMP at the DEGEN-SUSPECT hook (managers/degen_detect.py, EG 28.09.).

27B boot ...dkr27breleasedraftbar1w109281340: weg2-0-8 / weg2-1-13 produced
35789 / 39729 completion tokens and NOTHING kept their text, so whether they
were a loop or a legitimate long answer could not be decided afterwards.
Pinned here: a DEGEN-SUSPECT and a finished request past DUMP_LONG hand the
part's last <= 2048 output ids to the dump hook; TailDumper writes ids +
decoded text as one JSON, from its own thread, at most max_files, and an
instrument failure never raises into the detokenizer.
"""

import json
import os
import random

from sglang.srt.managers import degen_detect as dd


class _Tok:
    def decode(self, ids):
        return " ".join(f"t{i}" for i in ids)


def _loop_ids(n, period=37, seed=3):
    rng = random.Random(seed)
    pat = [rng.randrange(1000, 50000) for _ in range(period)]
    return [pat[i % period] for i in range(n)]


def _feed(det, rid, ids, finished=False, batch=16):
    for k in range(0, len(ids), batch):
        last = finished and k + batch >= len(ids)
        det.observe_chunk(rid, ids[k:k + batch], 0, finished=last)


def test_a_suspect_hands_the_window_to_the_dump_hook():
    got = []
    det = dd.DegenDetector(None, on_dump=lambda *a: got.append(a))
    _feed(det, "r1", _loop_ids(3000))
    assert got, "no dump at the suspect"
    rid, reason, part, ids, meta = got[0]
    assert (rid, reason, part) == ("r1", "suspect", "content")
    assert 0 < len(ids) <= dd.WINDOW and meta["period"] == 37 and meta["out_len"] >= 512


def test_a_long_finished_request_is_dumped_once_and_a_short_one_not():
    got = []
    rng = random.Random(5)
    det = dd.DegenDetector(None, on_dump=lambda *a: got.append(a), dump_long=4096)
    _feed(det, "long", [rng.randrange(1000, 50000) for _ in range(5000)], finished=True)
    _feed(det, "short", [rng.randrange(1000, 50000) for _ in range(1000)], finished=True)
    reasons = [(g[0], g[1]) for g in got]
    assert reasons == [("long", "long")], reasons
    assert got[0][4]["out_len"] == 5000 and len(got[0][3]) == dd.WINDOW


def test_dump_long_zero_is_off():
    got = []
    rng = random.Random(6)
    det = dd.DegenDetector(None, on_dump=lambda *a: got.append(a), dump_long=0)
    _feed(det, "x", [rng.randrange(1000, 50000) for _ in range(5000)], finished=True)
    assert got == []


def test_tail_dumper_writes_ids_and_text_and_respects_the_cap(tmp_path):
    d = dd.TailDumper(_Tok(), str(tmp_path), max_files=2, start=False)
    p = d.write("weg2-1-13", "long", "reasoning", [1, 2, 3], {"out_len": 39729})
    body = json.load(open(p))
    assert body["rid"] == "weg2-1-13" and body["text"] == "t1 t2 t3" and body["ids"] == [1, 2, 3]
    assert body["out_len"] == 39729 and body["part"] == "reasoning"
    d(rid="a", reason="long", part="c", ids=[1], meta={})   # queued (1 written + 1 queued = cap)
    d(rid="b", reason="long", part="c", ids=[1], meta={})   # over the cap -> dropped, counted
    assert d.dropped == 1 and d.q.qsize() == 1


def test_tail_dumper_thread_writes_off_the_caller(tmp_path):
    d = dd.TailDumper(_Tok(), str(tmp_path), max_files=4)
    d("r", "suspect", "content", [7, 8], {"period": 2})
    d.q.join if False else None
    for _ in range(200):
        if os.listdir(tmp_path):
            break
        import time
        time.sleep(0.01)
    files = os.listdir(tmp_path)
    assert len(files) == 1 and files[0].startswith("degen_") and "_r_suspect_" in files[0]


def test_a_failing_dump_hook_never_raises_into_the_detokenizer():
    def boom(*a):
        raise RuntimeError("disk full")
    det = dd.DegenDetector(None, on_dump=boom)
    _feed(det, "r", _loop_ids(3000))          # must not raise
    assert det.on_dump is None                # hook switched off after its first failure


def test_env_defaults():
    from sglang.srt.environ import envs
    assert envs.SGLANG_WEG2_DEGEN_DUMP_MAX.get() == 16
    assert envs.SGLANG_WEG2_DEGEN_DUMP_LONG.get() == 16384
