"""L3FILL-JOINED (30.09., NF y4a ep36, weg2-36-74): the read stopped 146 pages
into a 1070-page prefix on every D rank and on P for 5 cycles because 29
stems were CLAIMED by a writer that never finished ('L3-FILL JOINED stems=29:
a live writer holds the claim'), and nothing named the holder. Now the JOINED
line names slot/age/pid/role/generation/open writers of the first k holders
plus the oldest age, and the census names the oldest open claim."""

import logging
import os
import shutil
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest

from sglang.srt.mem_cache import hicache_storage as hs
from sglang.srt.mem_cache.hicache_storage import HiCacheFile
from sglang.srt.mem_cache.storage.file import hicache_arena as ha
from sglang.srt.mem_cache.storage.file.hicache_arena import ShmArena

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")

TOTAL = 4096


def _backend(root):
    be = object.__new__(HiCacheFile)

    def _path(stem):
        return os.path.join(root, stem + ".bin")

    be._existing_path = _path
    be._stat_stems = lambda stems: {s: os.path.getsize(_path(s)) for s in stems if os.path.exists(_path(s))}
    be._arena_evict_to_disk = lambda arena, want, need=None: 0
    return be


def test_a_fresh_claim_carries_its_pid_and_role(tmp_path):
    arena = ShmArena(str(tmp_path / "kv.bin"), TOTAL, 8)
    [(slot, status, gen)] = arena.claim_slots(["held"], [TOTAL], role=ha.ROLE_HOST_WRITE)
    assert status == 0
    [(s, g, age, pid, role, opened, state)] = arena.claim_info([slot])
    assert (s, g, pid, role, opened, state) == (slot, gen, os.getpid(), ha.ROLE_HOST_WRITE, 1, 1)
    assert 0 <= age < 5000
    # the thread's role is reset after the call: the next claim is "other"
    [(slot2, _, _)] = arena.claim_slots(["plain"], [TOTAL])
    assert arena.claim_info([slot2])[0][4] == 0


def test_the_joined_line_names_the_holder(tmp_path, caplog, monkeypatch):
    root = tmp_path / "store"
    root.mkdir()
    (root / "p146.bin").write_bytes(b"\x5a" * TOTAL)
    arena = ShmArena(str(tmp_path / "kv.bin"), TOTAL, 8)
    [(slot, status, gen)] = arena.claim_slots(["p146"], [TOTAL], role=ha.ROLE_HOST_WRITE)
    assert status == 0                      # a writer that will never finish
    monkeypatch.setattr(hs, "_FILL_JOIN_N", [0, 0])
    with caplog.at_level(logging.INFO, logger=hs.__name__):
        out = HiCacheFile.arena_fill_from_disk(_backend(str(root)), arena, ["p146"], TOTAL,
                                               prefix=True)
    assert out == [None]
    lines = [r.getMessage() for r in caplog.records if "L3-FILL JOINED" in r.getMessage()]
    assert lines, "no JOINED line"
    ln = lines[0]
    assert "holders=[%d/" % slot in ln and "pid%d/host-write/g%d/open1" % (os.getpid(), gen) in ln
    assert "oldest_ms=" in ln and "pids=[%d]" % os.getpid() in ln


def test_the_census_names_the_oldest_open_claim(tmp_path, caplog, monkeypatch):
    arena = ShmArena(str(tmp_path / "kv.bin"), TOTAL, 8)
    [(old, _, gen)] = arena.claim_slots(["old"], [TOTAL], role=ha.ROLE_L3FILL)
    time.sleep(0.05)
    arena.claim_slots(["young"], [TOTAL], role=ha.ROLE_HOST_WRITE)
    oc = arena.oldest_claim()
    assert (oc["slot"], oc["role"], oc["gen"], oc["claimed"], oc["pid"]) == (old, ha.ROLE_L3FILL, gen, 2,
                                                                            os.getpid())
    assert oc["age_ms"] >= 40
    monkeypatch.setenv(ha.ENV_REF_CENSUS_S, "0.05")
    with caplog.at_level(logging.INFO, logger=ha.__name__):
        ha._start_ref_census(arena)
        deadline = time.time() + 3.0
        while time.time() < deadline and not any(
                "ARENA-REF-CENSUS-CLAIM" in r.getMessage() for r in caplog.records):
            time.sleep(0.02)
        ha._CENSUS_THREADS.pop(os.path.realpath(arena.path)).set()
    lines = [r.getMessage() for r in caplog.records if "ARENA-REF-CENSUS-CLAIM" in r.getMessage()]
    assert lines and "claimed=2" in lines[0] and "oldest=%d/" % old in lines[0]
    assert "/l3fill/g%d/open1" % gen in lines[0]
