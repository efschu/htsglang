"""H81 (fnNV4f2, P PP0 05:08:43): a tail part under write is never removed,
and two writers never share a temp name.

THE METAL: the second leg 1 of the burst (requeue after W50) re-published
weg2-8-12, -14 and -15 on PP0 within one second, one publish thread each
(write_part, then _prune). The thread of weg2-8-12 pruned by the count rule
(capture_keep() = 4 newest rids, by HEADER mtime): weg2-8-14's only header was
its FIRST leg's (05:08:22), so it was the oldest rid -- and `remove` globbed
`weg2-8-14.tail.*`, the in-flight `weg2-8-14.tail.pp0-267295.pt.267295.tmp`
included. The weg2-8-14 thread died at `os.replace(tmp, ppath)`:

  WEG2-TAIL-PUBLISH write failed rid=weg2-8-14
  FileNotFoundError: ... 'weg2-8-14.tail.pp0-267295.pt.267295.tmp' -> '...pt'

(Not the cause of the 413: D never reached the tail of these rids, their
mamba anchor was missing -- see test_weg2_arena_reset_release_h81.py. A lost
part costs D the extend of [page_prefix, N) it would have skipped.)

The interleavings are forced deterministically: the payload save of one
writer runs the other thread's whole publish before it returns, exactly the
window between `torch.save(bundle, tmp)` and `os.replace(tmp, ppath)`.
Hermetic, CPU, a temp dir as the arena dir.
"""
from __future__ import annotations

import os
import threading

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.weg2 import tail_handoff as th

PAGE, RATIO = 64, 4
PART = "pp0-267295"


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("SGLANG_HICACHE_ARENA_DIR", str(tmp_path))
    monkeypatch.setattr(th, "capture_keep", lambda: 4)  # P --max-running-requests 4
    with envs.SGLANG_WEG2_TAIL_HANDOFF.override(True), envs.SGLANG_WEG2_TAIL_KEEP_MIB.override(0):
        yield tmp_path / "handoff"


def _spec(rid, n=4209):
    return th.spec_for(rid, list(range(n)), None, PAGE, RATIO)


def _publish(rid, fill=0.0, t=None, part=PART, rows=5):
    """One rank's publish thread: write_part, then the prune."""
    spec = _spec(rid)
    th.write_part(spec, part, {3: (torch.full((rows, 8), fill),)}, {5: (torch.full((1, 3), fill),)}, n_parts=3)
    if t is not None:
        for p in th.part_paths(rid, part):
            os.utime(p, (t, t))
    th._prune(rid, part)
    return spec


def _files(d, rid):
    return sorted(p for p in os.listdir(d) if p.startswith(f"{rid}.tail."))


def _save_then(monkeypatch, when, action):
    """torch.save as tail_handoff calls it -- and, for the write whose temp
    path contains `when`, `action()` right after the payload landed in the
    temp file (the other thread runs in that window)."""
    real = th.torch.save
    fired = []

    def save(obj, path, *a, **k):
        real(obj, path, *a, **k)
        if when in str(path) and not fired:
            fired.append(str(path))
            action()

    monkeypatch.setattr(th.torch, "save", save)
    return fired


def test_fnNV4f2_a_sibling_prune_never_takes_a_part_under_write(store, monkeypatch):
    """05:08:42-43 on PP0: parts of the first leg (weg2-8-14..17, 8-14 the
    oldest) are on disk; weg2-8-14's second leg is being written when the
    weg2-8-12 thread publishes and prunes. RED on d93a17316b (the temp file
    is removed, FileNotFoundError), GREEN: the rid under write is the newest."""
    for i, rid in enumerate(("weg2-8-14", "weg2-8-15", "weg2-8-16", "weg2-8-17")):
        _publish(rid, fill=1.0, t=1000.0 + i)
    fired = _save_then(monkeypatch, "weg2-8-14.tail.", lambda: _publish("weg2-8-12", fill=2.0))
    spec = _publish("weg2-8-14", fill=3.0)
    assert fired, "the interleaving did not happen"
    (h,) = [x for x in th.headers_for("weg2-8-14") if x.part == PART]
    bundle = th.verify_part(h)
    assert bundle is not None and h.spec == spec
    assert torch.equal(bundle["fa"][3][0], torch.full((5, 8), 3.0)), "the second leg's payload"
    assert not [p for p in os.listdir(store) if p.endswith(".tmp")], "no temp file left behind"
    # the count rule still holds: the oldest FINISHED rid made room instead
    assert _files(store, "weg2-8-15") == []
    assert _files(store, "weg2-8-12") and _files(store, "weg2-8-16") and _files(store, "weg2-8-17")


def test_two_writers_of_one_part_never_share_a_temp_name(store, monkeypatch):
    """The temp name was `<path>.<pid>.tmp`: a second write of the same part
    in the same process took the first one's temp file away. RED on
    d93a17316b (FileNotFoundError), GREEN: one temp name per write, and the
    payload and header left behind belong together."""
    rid = "weg2-8-14"
    _save_then(monkeypatch, f"{rid}.tail.", lambda: th.write_part(
        _spec(rid), PART, {3: (torch.full((5, 8), 7.0),)}, {5: (torch.full((1, 3), 7.0),)}, n_parts=3))
    h1 = th.write_part(_spec(rid), PART, {3: (torch.full((5, 8), 1.0),)}, {5: (torch.full((1, 3), 1.0),)},
                       n_parts=3)
    (h,) = th.headers_for(rid)
    assert h == h1
    bundle = th.verify_part(h)
    assert bundle is not None, "header and payload of one write (digests agree)"
    assert not [p for p in os.listdir(store) if p.endswith(".tmp")]


def test_temp_names_are_per_write():
    a, b = th._tmp_path("/x/r.tail.pp0-1.pt"), th._tmp_path("/x/r.tail.pp0-1.pt")
    assert a != b and a.endswith(".tmp") and b.endswith(".tmp")
    names = []
    t = threading.Thread(target=lambda: names.append(th._tmp_path("/x/r.tail.pp0-1.pt")))
    t.start()
    t.join()
    assert names[0] not in (a, b) and f".{os.getpid()}." in names[0]


def test_remove_and_the_consumed_receipt_leave_a_write_in_flight_alone(store):
    _publish("weg2-8-20", fill=1.0)
    tmp = store / f"weg2-8-20.tail.pp1-2.pt.{os.getpid()}.9.9.tmp"
    tmp.write_bytes(b"x" * 64)
    th.remove("weg2-8-20")
    assert _files(store, "weg2-8-20") == [tmp.name], "finished parts removed, the write in flight kept"
    _publish("weg2-8-21", fill=1.0)
    tmp2 = store / f"weg2-8-21.tail.pp2-3.pt.{os.getpid()}.9.9.tmp"
    tmp2.write_bytes(b"x" * 64)
    th._remove_consumed("weg2-8-21")
    assert _files(store, "weg2-8-21") == [tmp2.name]


def test_the_budget_rule_keeps_a_rid_under_write(store):
    """H63b budget: a rid whose OLD parts are the oldest but which another
    rank is writing right now (its temp file is fresh) is not a victim. RED
    on d93a17316b (the census skipped the temp file, the rid read as the
    oldest, `remove` took its temp file too)."""
    mib_rows = (1 << 20) // (8 * 4)                      # ~1 MiB per part
    with envs.SGLANG_WEG2_TAIL_KEEP_MIB.override(1):   # smaller than two parts
        for i, rid in enumerate(("weg2-8-30", "weg2-8-31", "weg2-8-32", "weg2-8-33")):
            _publish(rid, fill=1.0, t=1000.0 + i, rows=mib_rows)
        assert all(_files(store, r) for r in ("weg2-8-30", "weg2-8-31", "weg2-8-32", "weg2-8-33"))
        tmp = store / "weg2-8-30.tail.pp1-2.pt.99.1.1.tmp"   # 8-30's second leg, on PP1
        tmp.write_bytes(b"x" * 64)
        _publish("weg2-8-34", fill=1.0, rows=mib_rows)
    assert tmp.exists() and len(_files(store, "weg2-8-30")) == 3, "the rid under write was not pruned"
    assert _files(store, "weg2-8-31") == [], "the oldest finished rid made room"


def test_a_failed_write_leaves_no_temp_file(store, monkeypatch):
    def boom(obj, path, *a, **k):
        with open(path, "wb") as f:
            f.write(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(th.torch, "save", boom)
    with pytest.raises(OSError):
        th.write_part(_spec("weg2-8-40"), PART, {3: (torch.zeros(5, 8),)}, {}, n_parts=3)
    assert os.listdir(store) == []
    assert th._inflight_rids() == set(), "the writer registration is released on failure"
