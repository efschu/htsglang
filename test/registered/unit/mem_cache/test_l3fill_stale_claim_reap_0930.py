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


# --- S23 (18.10., h16v4b 1018_013102, D 01:36:23-24): the reap race a peer rank wins ----------------------------------
def _peer_wins_the_reap(arena, delay_s):
    """Wrap ``quarantine_stale``: right after the real quarantine a PEER rank claims the stem first (fresh generation)
    and completes it ``delay_s`` later -- the other two D ranks of the metal run reaped the same claim at the same
    instant; one won."""
    import threading

    real = arena.quarantine_stale
    won = []

    def _q(slots, gens, min_ms):
        st = real(slots, gens, min_ms)
        if any(int(v) == 1 for v in st) and not won:
            [(slot, status, gen)] = arena.claim_slots(["p1"], [TOTAL])
            assert status == 0
            won.append(slot)

            def _fill():
                time.sleep(delay_s)
                arena.slot_view(slot, TOTAL)[:] = b"\x41" * TOTAL
                arena.complete_slots([slot], [gen], [(0, TOTAL)])

            threading.Thread(target=_fill, daemon=True).start()
        return st

    arena.quarantine_stale = _q
    return won


def _on_io_thread(fn):
    import threading

    res = {}

    def run():
        res["out"] = fn()

    t = threading.Thread(target=run, name="hicache-prefetch-io-0")
    t.start()
    t.join(30)
    return res["out"]


def test_s23_the_loser_of_the_reap_race_awaits_the_winners_fill_instead_of_dropping_the_stem(tmp_path, monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_L3FILL_JOIN_IDLE_MS", "150")
    be, arena, old, gen = _setup(tmp_path)
    time.sleep(0.08)
    won = _peer_wins_the_reap(arena, 0.15)
    with envs.SGLANG_WEG2_L3FILL_STALE_CLAIM_S.override(0.05):
        out = _on_io_thread(lambda: HiCacheFile.arena_fill_from_disk(be, arena, ["p0", "p1", "p2"], TOTAL, prefix=True))
    assert won, "the peer must have won the reap"
    assert out[0] is not None and out[1] == won[0] and out[2] is not None, (out, won)   # base: out[1] None, prefix ends
    assert bytes(arena.slot_view(out[1], TOTAL)[:4]) == b"\x41" * 4


def test_s23_a_winner_that_never_completes_still_ends_the_wait_not_the_process(tmp_path, monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_L3FILL_JOIN_IDLE_MS", "100")
    be, arena, old, gen = _setup(tmp_path)
    time.sleep(0.08)
    _peer_wins_the_reap(arena, 60.0)          # never completes inside the test
    with envs.SGLANG_WEG2_L3FILL_STALE_CLAIM_S.override(0.05):
        t0 = time.time()
        out = _on_io_thread(lambda: HiCacheFile.arena_fill_from_disk(be, arena, ["p0", "p1", "p2"], TOTAL, prefix=True))
    assert out[0] is not None and out[1] is None and time.time() - t0 < 20     # bounded: the old miss, no hang
