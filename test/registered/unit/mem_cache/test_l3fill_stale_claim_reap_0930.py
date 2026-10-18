"""L3FILL-JOINED (3) (30.09., NF y4a ep36): 29 stems stayed JOINED for >= 6 s
on D and on P -- a claim whose writer is alive but delivered no byte is never
reaped by #231 (only claims with NO open writer are), and every fill of the
stem is a miss for good. Now such a claim is QUARANTINED from its key after
SGLANG_WEG2_L3FILL_STALE_CLAIM_S (default 5 s) without byte progress and the
fill reads the stem from disk into a fresh slot. Generation-safe: the old
writer's late completion is refused by name (status 6) and can never land in
the stem's new slot nor in the old slot's next owner."""

import logging
import os
import shutil
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest

from sglang.srt.environ import envs
from sglang.srt.mem_cache import hicache_storage as hs
from sglang.srt.mem_cache.hicache_storage import HiCacheFile
from sglang.srt.mem_cache.storage.file.hicache_arena import ShmArena


@pytest.fixture(autouse=True)
def _foreign_writer_is_simulated_in_process():
    """SELF-JOIN (14.10.): the 'live writer' of these tests is this very
    process, so the self-join takeover is off -- it would (rightly) take the
    claim over instead of treating it as a foreign holder."""
    from sglang.srt.environ import envs as _envs

    with _envs.SGLANG_WEG2_L3FILL_SELF_JOIN_TAKEOVER.override(False):
        yield


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


def _setup(tmp_path):
    root = tmp_path / "store"
    root.mkdir()
    for s in ("p0", "p1", "p2"):
        (root / (s + ".bin")).write_bytes(bytes([0x40 + int(s[1])]) * TOTAL)
    arena = ShmArena(str(tmp_path / "kv.bin"), TOTAL, 16)
    [(old, status, gen)] = arena.claim_slots(["p1"], [TOTAL])   # alive, never delivers a byte
    assert status == 0
    return _backend(str(root)), arena, old, gen


def test_a_stale_live_claim_is_quarantined_and_the_stem_read_from_disk(tmp_path, caplog):
    be, arena, old, gen = _setup(tmp_path)
    time.sleep(0.08)
    with envs.SGLANG_WEG2_L3FILL_STALE_CLAIM_S.override(0.05), \
            caplog.at_level(logging.WARNING, logger=hs.__name__):
        out = HiCacheFile.arena_fill_from_disk(be, arena, ["p0", "p1", "p2"], TOTAL, prefix=True)
    new = out[1]
    assert new is not None and new != old            # base: None -- the prefix ended at p1
    assert out[2] is not None                        # base: p2 never read
    assert arena.find_slots(["p1"]) == [(new, 2)]
    assert bytes(arena.slot_view(new, TOTAL)[:4]) == b"\x41" * 4
    [(_, g, _age, _pid, _role, opened, state)] = arena.claim_info([old])
    assert (g, opened, state) == (gen, 1, 4)         # quarantined, generation kept, writer still open
    assert any("L3-FILL STALE-CLAIM-REAP" in r.getMessage() for r in caplog.records)


def test_the_old_writers_late_completion_is_refused_by_generation(tmp_path):
    be, arena, old, gen = _setup(tmp_path)
    time.sleep(0.08)
    with envs.SGLANG_WEG2_L3FILL_STALE_CLAIM_S.override(0.05):
        out = HiCacheFile.arena_fill_from_disk(be, arena, ["p1"], TOTAL, prefix=True)
    new = out[0]
    # the old writer comes back and writes its (other) bytes, then completes
    arena.slot_view(old, TOTAL)[:4] = b"\xee" * 4
    assert arena.complete_slots([old], [gen], [(0, TOTAL)]) == [6]    # named: reaped under it
    assert arena.find_slots(["p1"]) == [(new, 2)]
    assert bytes(arena.slot_view(new, TOTAL)[:4]) == b"\x41" * 4      # the new home is untouched
    # the last one out freed the quarantined slot; its generation moved on
    [(_, g2, _age, _pid, _role, opened, state)] = arena.claim_info([old])
    assert state == 0 and g2 == gen + 1 and opened == 0
    # a second late call of the old writer can never hit the slot's next owner
    [(nxt, st, ngen)] = arena.claim_slots(["other"], [TOTAL])
    if nxt == old:
        assert arena.complete_slots([old], [gen], [(0, TOTAL)]) == [3]
        assert arena.free_if_gen([old], [gen]) == [0]
        assert arena.claim_info([old])[0][6] == 1               # still the new owner's claim


def test_a_young_claim_is_not_reaped(tmp_path):
    be, arena, old, gen = _setup(tmp_path)
    with envs.SGLANG_WEG2_L3FILL_STALE_CLAIM_S.override(30.0):
        out = HiCacheFile.arena_fill_from_disk(be, arena, ["p0", "p1", "p2"], TOTAL, prefix=True)
    assert out[1] is None and out[2] is None
    assert arena.claim_info([old])[0][6] == 1                   # still CLAIMED by its writer


def test_a_join_does_not_count_as_progress(tmp_path):
    """Every reader's JOIN refreshed touched_ms; the stale clock is byte progress."""
    be, arena, old, gen = _setup(tmp_path)
    time.sleep(0.08)
    for _ in range(3):   # readers joining over and over (y4a: 5 cycles x 3 ranks + P)
        arena.claim_slots(["p1"], [TOTAL])
        arena.unclaim([old], [gen])
    assert arena.quarantine_stale([old], [gen], 50) == [1]


def test_the_sweep_frees_a_quarantine_whose_writer_never_returns(tmp_path):
    _be, arena, old, gen = _setup(tmp_path)
    time.sleep(0.08)
    assert arena.quarantine_stale([old], [gen], 50) == [1]
    assert arena.quarantine_sweep(10_000) == 0                 # writer open, backstop not reached
    time.sleep(0.06)
    assert arena.quarantine_sweep(50) == 1
    assert arena.claim_info([old])[0][6] == 0
    assert arena.stats()["claimed"] == 0


def _on_io_thread(fn):
    import threading

    res = {}

    def run():
        res["out"] = fn()

    t = threading.Thread(target=run, name="hicache-prefetch-io-0")
    t.start()
    t.join(30)
    return res["out"]


# --- S23: SELF-JOIN must not take a claim a sibling io thread of THIS process is filling ----------------------------
def _sibling_thread_fill_claim(arena, stem):
    """What a live io worker of this process is in the middle of: its fill claimed ``stem`` (registered live)."""
    hs._FILL_TL.active = True
    try:
        [(slot, status, gen)] = hs._fill_claim(arena, [stem], TOTAL)
    finally:
        hs._FILL_TL.active = False
    assert status == 0
    return slot, gen


def test_s23_self_join_leaves_a_sibling_io_threads_live_claim_alone(tmp_path, monkeypatch, caplog):
    import threading

    monkeypatch.setenv("SGLANG_WEG2_L3FILL_JOIN_IDLE_MS", "1500")
    root = tmp_path / "store"
    root.mkdir()
    (root / "p1.bin").write_bytes(b"\x41" * TOTAL)
    arena = ShmArena(str(tmp_path / "kv.bin"), TOTAL, 16)
    be = _backend(str(root))
    slot, gen = _sibling_thread_fill_claim(arena, "p1")      # the sibling (this test thread) is mid-fill on p1
    done = {}

    def _sibling_finishes():
        time.sleep(0.3)
        arena.slot_view(slot, TOTAL)[:] = b"\x41" * TOTAL
        done["status"] = arena.complete_slots([slot], [gen], [(0, TOTAL)])
        hs._fill_live_drop_thread()

    # the sibling's registration belongs to the thread that made the claim: finish it from a helper that is NOT the
    # reader's thread; the reader runs on its own io-named thread
    ident_owner = threading.get_ident()
    t_sib = threading.Thread(target=_sibling_finishes)
    t_sib.start()
    with envs.SGLANG_WEG2_L3FILL_SELF_JOIN_TAKEOVER.override(True), caplog.at_level(logging.WARNING, logger=hs.__name__):
        out = _on_io_thread(lambda: HiCacheFile.arena_fill_from_disk(be, arena, ["p1"], TOTAL, prefix=True))
    t_sib.join(10)
    assert not any("SELF-JOIN" in r.getMessage() for r in caplog.records), [r.getMessage() for r in caplog.records]
    assert done["status"] == [1], done            # 1 = completed; base: [6] -- the sibling's completion refused, its read cut
    assert out[0] == slot, (out, slot)            # the reader waited and got the sibling's page
    hs._FILL_LIVE.pop(slot, None)
    assert ident_owner


def test_s23_self_join_still_takes_over_a_leftover_claim_of_a_dead_call(tmp_path, monkeypatch, caplog):
    """h16 PP1 (weg2-10-86): our own pid, no live registration -> the old takeover stands."""
    root = tmp_path / "store"
    root.mkdir()
    (root / "p1.bin").write_bytes(b"\x41" * TOTAL)
    arena = ShmArena(str(tmp_path / "kv.bin"), TOTAL, 16)
    be = _backend(str(root))
    [(old, status, gen)] = arena.claim_slots(["p1"], [TOTAL])    # not through a fill: nothing registered live
    assert status == 0
    with envs.SGLANG_WEG2_L3FILL_SELF_JOIN_TAKEOVER.override(True), caplog.at_level(logging.WARNING, logger=hs.__name__):
        out = _on_io_thread(lambda: HiCacheFile.arena_fill_from_disk(be, arena, ["p1"], TOTAL, prefix=True))
    assert out[0] is not None and out[0] != old
    assert any("SELF-JOIN" in r.getMessage() for r in caplog.records)     # by the takeover, not by a later reap
    assert not hs._FILL_LIVE, hs._FILL_LIVE      # the fill gave its live registrations back
