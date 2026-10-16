"""L3FILL-JOINED (2) (30.09., NF y4a ep36): a prefix read ended at its first
JOINED stem -- another writer's open claim was a miss at once, and one such
page cut a 1070-page prefix at 146. Now the fill waits (bounded by
SGLANG_WEG2_L3FILL_JOIN_WAIT_MS, default 300) for the claim to complete, but
ONLY on a prefetch io thread (hicache-prefetch-io-<k>); the scheduler thread
never waits."""

import os
import shutil
import threading
import time
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest

from sglang.srt.environ import envs
from sglang.srt.mem_cache import hicache_storage as hs
from sglang.srt.mem_cache.hicache_storage import HiCacheFile
from sglang.srt.mem_cache.storage.file.hicache_arena import ShmArena

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")

TOTAL = 4096


@pytest.fixture(autouse=True)
def _foreign_writer_is_simulated_in_process():
    """The 'other writer' of these tests is this very process (its claims
    carry our pid), so the SELF-JOIN takeover is off unless a test turns it
    on to test exactly that."""
    with envs.SGLANG_WEG2_L3FILL_SELF_JOIN_TAKEOVER.override(False):
        yield


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
        (root / (s + ".bin")).write_bytes(b"\x11" * TOTAL)
    arena = ShmArena(str(tmp_path / "kv.bin"), TOTAL, 16)
    [(slot, status, gen)] = arena.claim_slots(["p1"], [TOTAL])   # another writer, mid-fill
    assert status == 0
    return _backend(str(root)), arena, slot, gen


def _run_on(thread_name, fn):
    box = {}
    th = threading.Thread(target=lambda: box.setdefault("r", fn()), name=thread_name)
    th.start()
    th.join(10)
    return box["r"]


def test_the_prefetch_io_thread_waits_for_a_claim_that_completes(tmp_path):
    be, arena, slot, gen = _setup(tmp_path)

    def finish_later():
        time.sleep(0.05)
        assert arena.complete_slots([slot], [gen], [(0, TOTAL)]) == [1]

    threading.Thread(target=finish_later).start()
    with envs.SGLANG_WEG2_L3FILL_JOIN_WAIT_MS.override(300):
        out = _run_on("hicache-prefetch-io-0", lambda: HiCacheFile.arena_fill_from_disk(
            be, arena, ["p0", "p1", "p2"], TOTAL, prefix=True))
    assert out[1] == slot                 # base: None -- the prefix ended at p1
    assert out[0] is not None and out[2] is not None   # base: p2 never read


def test_the_scheduler_thread_never_waits(tmp_path):
    be, arena, slot, gen = _setup(tmp_path)
    with envs.SGLANG_WEG2_L3FILL_JOIN_WAIT_MS.override(300):
        assert hs.fill_join_wait_ms() == 0            # MainThread: no wait
        t0 = time.perf_counter()
        out = HiCacheFile.arena_fill_from_disk(be, arena, ["p0", "p1", "p2"], TOTAL, prefix=True)
        took = time.perf_counter() - t0
    assert out[1] is None and out[2] is None and took < 0.25
    assert _run_on("weg2-flip-lane", hs.fill_join_wait_ms) == 0
    with envs.SGLANG_WEG2_L3FILL_JOIN_WAIT_MS.override(300):
        assert _run_on("hicache-prefetch-io-3", hs.fill_join_wait_ms) == 300


def test_a_claim_that_never_completes_is_a_miss_after_the_bound(tmp_path):
    be, arena, slot, gen = _setup(tmp_path)
    # JOIN-LIVENESS: the idle bound is max(bound, stale age); no stale reap here
    with envs.SGLANG_WEG2_L3FILL_JOIN_WAIT_MS.override(80), \
            envs.SGLANG_WEG2_L3FILL_STALE_CLAIM_S.override(0.0):
        t0 = time.perf_counter()
        out = _run_on("hicache-prefetch-io-1", lambda: HiCacheFile.arena_fill_from_disk(
            be, arena, ["p0", "p1", "p2"], TOTAL, prefix=True))
        took = time.perf_counter() - t0
    assert out[0] is not None and out[1] is None and out[2] is None
    assert 0.07 <= took < 2.0
    assert arena.claim_info([slot])[0][5] == 1       # the other writer's claim untouched


def test_own_claims_are_filled_before_the_join_wait(tmp_path):
    """STORE-SHORT (NF h6 11:17:38, weg2-4-17): two ranks with CROSS claims.
    The peer completes its claim (p1) only once OUR claim (p2) is complete --
    as a peer does that itself waits on our stems before filling its own.
    Before the fix this fill waited out the bound with p2 unfilled, so the
    peer never finished: the prefix ended at p1 (1024 of 3168 pages)."""
    be, arena, slot, gen = _setup(tmp_path)

    def peer():
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            (_s, state), = arena.find_slots(["p2"])
            if int(state) == 2:
                assert arena.complete_slots([slot], [gen], [(0, TOTAL)]) == [1]
                return
            time.sleep(0.005)

    threading.Thread(target=peer).start()
    with envs.SGLANG_WEG2_L3FILL_JOIN_WAIT_MS.override(300):
        out = _run_on("hicache-prefetch-io-0", lambda: HiCacheFile.arena_fill_from_disk(
            be, arena, ["p0", "p1", "p2"], TOTAL, prefix=True))
    assert out[0] is not None and out[1] == slot and out[2] is not None


def test_the_wait_is_priced_on_the_joined_stems():
    with envs.SGLANG_WEG2_L3FILL_JOIN_WAIT_MS.override(300):
        assert _run_on("hicache-prefetch-io-0", lambda: hs.fill_join_wait_ms(1)) == 300
        assert _run_on("hicache-prefetch-io-0", lambda: hs.fill_join_wait_ms(385)) == 770
        assert _run_on("hicache-prefetch-io-0", lambda: hs.fill_join_wait_ms(5000)) == 5000  # cap = stale-claim age (14.10.)
    with envs.SGLANG_WEG2_L3FILL_JOIN_WAIT_MS.override(0):
        assert _run_on("hicache-prefetch-io-0", lambda: hs.fill_join_wait_ms(385)) == 0


def test_a_full_arena_waits_for_room_instead_of_ending_the_prefix(tmp_path):
    """ROOM-WAIT (NF h8 12:04:14, weg2-0-10): the stems that find the arena
    FULL are claimed again once a slot frees (here: 60 ms later), instead of
    ending the prefix (884 of 1853 pages, the files all on disk)."""
    root = tmp_path / "store"
    root.mkdir()
    for s in ("p0", "p1"):
        (root / (s + ".bin")).write_bytes(b"\x22" * TOTAL)
    arena = ShmArena(str(tmp_path / "kv.bin"), TOTAL, 2)
    occ = arena.claim_slots(["o0", "o1"], [TOTAL, TOTAL])     # someone else's pages fill it
    assert [st for _s, st, _g in occ] == [0, 0]
    be = _backend(str(root))

    def free_later():
        time.sleep(0.06)
        arena.free_slots([occ[0][0], occ[1][0]], reason="test")

    threading.Thread(target=free_later).start()
    with envs.SGLANG_WEG2_L3FILL_ROOM_WAIT_MS.override(1000):
        out = _run_on("hicache-prefetch-io-0", lambda: HiCacheFile.arena_fill_from_disk(
            be, arena, ["p0", "p1"], TOTAL, prefix=True))
    assert out[0] is not None and out[1] is not None


def test_no_room_wait_off_the_prefetch_thread():
    with envs.SGLANG_WEG2_L3FILL_ROOM_WAIT_MS.override(1000):
        assert hs.fill_room_wait_ms() == 0
        assert _run_on("hicache-prefetch-io-2", hs.fill_room_wait_ms) == 1000


def _setup_inflight(tmp_path):
    root = tmp_path / "store"
    root.mkdir()
    for s in ("p0", "p2"):                          # p1 is NOT on disk ...
        (root / (s + ".bin")).write_bytes(b"\x33" * TOTAL)
    arena = ShmArena(str(tmp_path / "kv.bin"), TOTAL, 16)
    [(slot, status, gen)] = arena.claim_slots(["p1"], [TOTAL])   # ... another writer is putting it into L2
    assert status == 0
    return _backend(str(root)), arena, slot, gen


def test_a_page_being_written_into_the_arena_is_awaited_not_a_gap(tmp_path):
    """IN-FLIGHT (NF h8 12:40:58, weg2-18-52): D's flip-time publish was
    writing the session's pages into the arena when P's early read walked
    them; not on disk yet -> the walk ended at 47 of 632 pages."""
    be, arena, slot, gen = _setup_inflight(tmp_path)

    def finish_later():
        time.sleep(0.05)
        assert arena.complete_slots([slot], [gen], [(0, TOTAL)]) == [1]

    threading.Thread(target=finish_later).start()
    with envs.SGLANG_WEG2_L3FILL_JOIN_WAIT_MS.override(300):
        out = _run_on("hicache-prefetch-io-0", lambda: HiCacheFile.arena_fill_from_disk(
            be, arena, ["p0", "p1", "p2"], TOTAL, prefix=True))
    assert out[0] is not None and out[1] == slot and out[2] is not None


def test_an_inflight_page_off_the_prefetch_thread_is_still_a_gap(tmp_path):
    be, arena, slot, gen = _setup_inflight(tmp_path)
    with envs.SGLANG_WEG2_L3FILL_JOIN_WAIT_MS.override(300):
        out = HiCacheFile.arena_fill_from_disk(be, arena, ["p0", "p1", "p2"], TOTAL, prefix=True)
    assert out[0] is not None and out[1] is None and out[2] is None


def test_an_inflight_page_that_never_completes_ends_the_prefix_after_the_bound(tmp_path):
    be, arena, slot, gen = _setup_inflight(tmp_path)
    with envs.SGLANG_WEG2_L3FILL_JOIN_WAIT_MS.override(80), \
            envs.SGLANG_WEG2_L3FILL_JOIN_WAIT_PER_PAGE_MS.override(0.0), \
            envs.SGLANG_WEG2_L3FILL_STALE_CLAIM_S.override(0.0):
        t0 = time.perf_counter()
        out = _run_on("hicache-prefetch-io-1", lambda: HiCacheFile.arena_fill_from_disk(
            be, arena, ["p0", "p1", "p2"], TOTAL, prefix=True))
        took = time.perf_counter() - t0
    assert out[0] is not None and out[1] is None and out[2] is None
    assert 0.07 <= took < 2.0
    assert arena.claim_info([slot])[0][5] == 1       # the writer's claim untouched


def test_a_room_wait_claim_lost_to_a_peer_is_awaited_not_dropped(tmp_path):
    """TF2 (NF h13 14:18:00, weg2-10-82): the slot the room wait freed went
    to a PEER rank filling the SAME stem (status 1). Dropped, PP0's mamba
    anchor read missed and #TF sent told=0 (full prefill of 59163)."""
    root = tmp_path / "store"
    root.mkdir()
    (root / "p0.bin").write_bytes(b"\x44" * TOTAL)
    arena = ShmArena(str(tmp_path / "kv.bin"), TOTAL, 1)
    [(occ, st0, _g0)] = arena.claim_slots(["o0"], [TOTAL])       # the arena is FULL
    assert st0 == 0
    be = _backend(str(root))
    box = {}

    def peer():
        time.sleep(0.008)
        arena.free_slots([occ], reason="test")
        [(slot, status, gen)] = arena.claim_slots(["p0"], [TOTAL])  # the peer wins the slot
        assert status == 0
        box["slot"] = slot
        time.sleep(0.08)
        assert arena.complete_slots([slot], [gen], [(0, TOTAL)]) == [1]

    threading.Thread(target=peer).start()
    with envs.SGLANG_WEG2_L3FILL_ROOM_WAIT_MS.override(1000), \
            envs.SGLANG_WEG2_L3FILL_JOIN_WAIT_MS.override(1000):
        out = _run_on("hicache-prefetch-io-0", lambda: HiCacheFile.arena_fill_from_disk(
            be, arena, ["p0"], TOTAL, prefix=True))
    assert out[0] == box["slot"]



# ---- JOIN-LIVENESS (NF h16 a2cfc77f76, P 15:34:27-32, weg2-10-86) ---------------


def _setup_many(tmp_path, n):
    root = tmp_path / "store"
    root.mkdir()
    names = ["q%d" % i for i in range(n)]
    for s_ in names:
        (root / (s_ + ".bin")).write_bytes(b"\x55" * TOTAL)
    arena = ShmArena(str(tmp_path / "kv.bin"), TOTAL, 2 * n + 4)
    claims = arena.claim_slots(names[1:], [TOTAL] * (n - 1))   # a peer holds all but q0
    assert all(st == 0 for _s, st, _g in claims)
    return _backend(str(root)), arena, names, claims


def _complete_one_by_one(arena, claims, every_s):
    def run():
        for slot, _st, gen in claims:
            time.sleep(every_s)
            arena.complete_slots([slot], [gen], [(0, TOTAL)])
    th = threading.Thread(target=run)
    th.start()
    return th


def test_a_holder_that_keeps_completing_is_waited_for_past_the_bound(tmp_path):
    """The peer completes one page every 80 ms for ~0.5 s; the idle bound is
    150 ms. A fixed bound ends the prefix after 150 ms, the live wait gets all."""
    be, arena, names, claims = _setup_many(tmp_path, 7)
    th = _complete_one_by_one(arena, claims, 0.08)
    with envs.SGLANG_WEG2_L3FILL_JOIN_WAIT_MS.override(150), \
            envs.SGLANG_WEG2_L3FILL_JOIN_WAIT_PER_PAGE_MS.override(0.0), \
            envs.SGLANG_WEG2_L3FILL_STALE_CLAIM_S.override(0.0), \
            envs.SGLANG_WEG2_L3FILL_JOIN_LIVENESS.override(True):
        out = _run_on("hicache-prefetch-io-0", lambda: HiCacheFile.arena_fill_from_disk(
            be, arena, names, TOTAL, prefix=True))
    th.join(5)
    assert all(o is not None for o in out), out


def test_the_fixed_bound_with_liveness_off_ends_early(tmp_path):
    be, arena, names, claims = _setup_many(tmp_path, 7)
    th = _complete_one_by_one(arena, claims, 0.08)
    with envs.SGLANG_WEG2_L3FILL_JOIN_WAIT_MS.override(150), \
            envs.SGLANG_WEG2_L3FILL_JOIN_WAIT_PER_PAGE_MS.override(0.0), \
            envs.SGLANG_WEG2_L3FILL_STALE_CLAIM_S.override(0.0), \
            envs.SGLANG_WEG2_L3FILL_JOIN_LIVENESS.override(False):
        out = _run_on("hicache-prefetch-io-0", lambda: HiCacheFile.arena_fill_from_disk(
            be, arena, names, TOTAL, prefix=True))
    th.join(5)
    assert out[0] is not None and any(o is None for o in out)


def test_a_holder_that_never_moves_ends_at_the_idle_bound_not_the_cap(tmp_path):
    be, arena, names, _claims = _setup_many(tmp_path, 3)
    with envs.SGLANG_WEG2_L3FILL_JOIN_WAIT_MS.override(100), \
            envs.SGLANG_WEG2_L3FILL_JOIN_WAIT_PER_PAGE_MS.override(0.0), \
            envs.SGLANG_WEG2_L3FILL_STALE_CLAIM_S.override(0.0), \
            envs.SGLANG_WEG2_L3FILL_JOIN_CAP_S.override(30.0), \
            envs.SGLANG_WEG2_L3FILL_JOIN_LIVENESS.override(True):
        t0 = time.perf_counter()
        out = _run_on("hicache-prefetch-io-0", lambda: HiCacheFile.arena_fill_from_disk(
            be, arena, names, TOTAL, prefix=True))
        took = time.perf_counter() - t0
    assert out[0] is not None and out[1] is None
    assert 0.09 <= took < 2.0


def test_a_stalled_holder_is_reaped_after_the_stale_age_and_read(tmp_path):
    """No progress for the stale age: the caller's stale reap breaks the
    claim and this fill reads the page itself -- fetching back, not a miss."""
    be, arena, names, _claims = _setup_many(tmp_path, 3)
    with envs.SGLANG_WEG2_L3FILL_JOIN_WAIT_MS.override(50), \
            envs.SGLANG_WEG2_L3FILL_JOIN_WAIT_PER_PAGE_MS.override(0.0), \
            envs.SGLANG_WEG2_L3FILL_STALE_CLAIM_S.override(0.2), \
            envs.SGLANG_WEG2_L3FILL_JOIN_LIVENESS.override(True):
        out = _run_on("hicache-prefetch-io-0", lambda: HiCacheFile.arena_fill_from_disk(
            be, arena, names, TOTAL, prefix=True))
    assert all(o is not None for o in out), out


def test_room_wait_keeps_going_while_slots_come_and_reads_them_at_once(tmp_path):
    """Two FULL stems; one slot frees at 60 ms, the next at 230 ms; the room
    bound is 200 ms. The live wait takes both (idle restarts at the first)."""
    root = tmp_path / "store"
    root.mkdir()
    for s_ in ("r0", "r1"):
        (root / (s_ + ".bin")).write_bytes(b"\x66" * TOTAL)
    arena = ShmArena(str(tmp_path / "kv.bin"), TOTAL, 2)
    occ = arena.claim_slots(["o0", "o1"], [TOTAL, TOTAL])
    be = _backend(str(root))

    def free_later():
        time.sleep(0.06)
        arena.free_slots([occ[0][0]], reason="test")
        time.sleep(0.17)
        arena.free_slots([occ[1][0]], reason="test")

    th = threading.Thread(target=free_later)
    th.start()
    with envs.SGLANG_WEG2_L3FILL_ROOM_WAIT_MS.override(200), \
            envs.SGLANG_WEG2_L3FILL_JOIN_LIVENESS.override(True):
        out = _run_on("hicache-prefetch-io-0", lambda: HiCacheFile.arena_fill_from_disk(
            be, arena, ["r0", "r1"], TOTAL, prefix=True))
    th.join(5)
    assert out[0] is not None and out[1] is not None, out


# ---- SELF-JOIN / SELF-EVICT (NF h16 a2cfc77f76, P PP1 15:34:27, weg2-10-86) ------


def test_a_join_on_this_process_own_claim_is_taken_over_at_once(tmp_path):
    """PP1 joined 190 claims its own pid held (holders=[1904/1ms/pid778 ...])
    and waited 5001 ms. A claim this process holds has no other reader to wait
    for: it is taken over and read now -- no wait, no stale age."""
    be, arena, slot, gen = _setup(tmp_path)       # p1 claimed by THIS process, never completed
    with envs.SGLANG_WEG2_L3FILL_JOIN_WAIT_MS.override(5000), \
            envs.SGLANG_WEG2_L3FILL_JOIN_LIVENESS.override(True), \
            envs.SGLANG_WEG2_L3FILL_SELF_JOIN_TAKEOVER.override(True):
        t0 = time.perf_counter()
        out = _run_on("hicache-prefetch-io-0", lambda: HiCacheFile.arena_fill_from_disk(
            be, arena, ["p0", "p1", "p2"], TOTAL, prefix=True))
        took = time.perf_counter() - t0
    assert all(o is not None for o in out), out
    assert took < 1.0, took


def test_self_join_takeover_off_waits_as_before(tmp_path):
    be, arena, slot, gen = _setup(tmp_path)
    with envs.SGLANG_WEG2_L3FILL_JOIN_WAIT_MS.override(80), \
            envs.SGLANG_WEG2_L3FILL_STALE_CLAIM_S.override(0.0), \
            envs.SGLANG_WEG2_L3FILL_SELF_JOIN_TAKEOVER.override(False):
        out = _run_on("hicache-prefetch-io-0", lambda: HiCacheFile.arena_fill_from_disk(
            be, arena, ["p0", "p1", "p2"], TOTAL, prefix=True))
    assert out[0] is not None and out[1] is None


def test_a_stem_twice_in_one_fill_is_claimed_once_and_never_self_joined(tmp_path):
    root = tmp_path / "store"
    root.mkdir()
    for s_ in ("d0", "d1"):
        (root / (s_ + ".bin")).write_bytes(b"\x77" * TOTAL)
    arena = ShmArena(str(tmp_path / "kv.bin"), TOTAL, 8)
    be = _backend(str(root))
    with envs.SGLANG_WEG2_L3FILL_JOIN_WAIT_MS.override(5000), \
            envs.SGLANG_WEG2_L3FILL_SELF_JOIN_TAKEOVER.override(False):
        t0 = time.perf_counter()
        out = _run_on("hicache-prefetch-io-0", lambda: HiCacheFile.arena_fill_from_disk(
            be, arena, ["d0", "d1", "d1"], TOTAL, prefix=True))
        took = time.perf_counter() - t0
    assert out[1] is not None and out[2] == out[1] and out[0] is not None
    assert took < 1.0, "a repeated stem must not wait on its own claim"


def test_the_read_being_filled_is_kept_out_of_the_fill_eviction(monkeypatch):
    seen = {}

    def _cands(arena, want, need, pin_stems):
        seen["keep"] = list(pin_stems)
        return []

    monkeypatch.setattr(HiCacheFile, "_arena_evict_candidates", staticmethod(_cands))
    be = object.__new__(HiCacheFile)
    be.pins = None
    op = types.SimpleNamespace(hash_value=["a", "b"], _weg2_protect_stems=["a.s", "b.s"])
    with envs.SGLANG_WEG2_L3FILL_PROTECT_READ.override(True):
        with hs.fill_protect_scope(be, op):
            HiCacheFile._arena_evict_to_disk(be, types.SimpleNamespace(slots=64), 4, need=4)
        assert hs.fill_protected_stems() == []
    assert seen["keep"] == ["a.s", "b.s"]
    with envs.SGLANG_WEG2_L3FILL_PROTECT_READ.override(False):
        with hs.fill_protect_scope(be, op):
            HiCacheFile._arena_evict_to_disk(be, types.SimpleNamespace(slots=64), 4, need=4)
    assert seen["keep"] == []
