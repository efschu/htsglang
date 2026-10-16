"""H54 PARK-JOIN 1015 (NF h16h 853534a9ff, boots 1014_211215 / 1015_011215).

16 of 16 JOIN-WAIT tears (5 s, still_claimed > 0) fell 3-45 s after a park on D
or a wake of P; the first large P read joined claims that never completed and
cut at the anchor (67-100k tokens recomputed per case; once ArenaChainMismatch
with 'key slot 5573 (state 1)'). Every claim census of boot 011215 that names
the held claims names role HOST-WRITE with no open writer (pid857 = P PP0,
pid2659 = D TP0, barlink P.log Z.110 / D.log Z.136).

The root, by code: a host-write claim completes only when every writer of the
page merged its extents; each writer that merged keeps a reader reference on
the partial slot (``ArenaMHAHostPool.complete_write`` references status <= 2);
a writer that never writes (refused / dropped backup, dormant group) leaves the
claim CLAIMED for good. ``arena_quarantine_stale`` refuses it -- 3 referenced
for as long as a writer's node lives, 0 young while late extents trickle in --
so neither the JOIN-IDLE reap nor the SELF-JOIN take-over can break it.

The fix: a host-write claim that did not complete inside the join wait, on a
page L3 holds, is taken from its key regardless of references and byte
progress (``arena_quarantine_host_write``) and read from L3; a stem a peer
stage took over first is awaited once; the claims a wait leaves are named
(``L3-FILL JOIN-HELD``: refs, merged writers, verdict per slot).
"""

import logging
import os
import shutil
import threading
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest

from sglang.srt.environ import envs
from sglang.srt.mem_cache import hicache_storage as hs
from sglang.srt.mem_cache.hicache_storage import HiCacheFile
from sglang.srt.mem_cache.storage.file import hicache_arena as ha
from sglang.srt.mem_cache.storage.file.hicache_arena import ShmArena

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")

TOTAL = 4096
HALF = TOTAL // 2


def _backend(root):
    be = object.__new__(HiCacheFile)

    def _path(stem):
        return os.path.join(root, stem + ".bin")

    be._existing_path = _path
    be._stat_stems = lambda stems: {s: os.path.getsize(_path(s)) for s in stems if os.path.exists(_path(s))}
    be._arena_evict_to_disk = lambda arena, want, need=None: 0
    return be


def _setup(tmp_path, n=8, slots=32):
    root = tmp_path / "store"
    root.mkdir()
    stems = [f"p{i}" for i in range(n)]
    for s in stems:
        (root / (s + ".bin")).write_bytes(b"\x11" * TOTAL)
    arena = ShmArena(str(tmp_path / "kv.bin"), TOTAL, slots)
    return _backend(str(root)), arena, stems


def _fill_on(be, arena, stems, name="hicache-prefetch-io-0"):
    box = {}

    def run():
        try:
            box["r"] = HiCacheFile.arena_fill_from_disk(be, arena, list(stems), TOTAL, prefix=True, take_ref=True)
        except Exception as exc:  # noqa: BLE001
            box["e"] = exc

    th = threading.Thread(target=run, name=name)
    th.start()
    th.join(30)
    if "e" in box:
        raise box["e"]
    return box["r"]


def _landed(out):
    return sum(1 for s in out if s is not None)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    # metal bounds scaled 1/10 like the cold-start rig (join wait 5 s, idle 1 s, stale 5 s)
    for k, v in {"SGLANG_WEG2_L3FILL_JOIN_WAIT_MS": "300", "SGLANG_WEG2_L3FILL_JOIN_WAIT_MAX_MS": "300",
                 "SGLANG_WEG2_L3FILL_JOIN_IDLE_MS": "100", "SGLANG_WEG2_L3FILL_STALE_CLAIM_S": "0.5",
                 "SGLANG_WEG2_L3FILL_JOIN_CAP_S": "6"}.items():
        monkeypatch.setenv(k, v)
    for k in ("SGLANG_WEG2_L3FILL_HOST_WRITE_TAKEOVER", "SGLANG_WEG2_L3FILL_JOIN_HELD_LOG",
              "SGLANG_WEG2_L3FILL_REF_AT_LAND", "SGLANG_WEG2_L3FILL_JOIN_LIVENESS"):
        monkeypatch.delenv(k, raising=False)
    # the holder of these tests is this very process -- SELF-JOIN off, so it stands for a FOREIGN writer
    with envs.SGLANG_WEG2_L3FILL_SELF_JOIN_TAKEOVER.override(False):
        yield


def _partial_host_write(arena, stems, *, ref=True, merge=True):
    """One writer of a two-writer page: claims host-write, merges its half and -- like
    ``complete_write`` for status <= 2 -- references the partial slot; the other writer never writes."""
    got = arena.claim_slots(list(stems), [TOTAL] * len(stems), role=ha.ROLE_HOST_WRITE)
    slots = [s for s, st, _g in got if st == 0]
    gens = [g for _s, st, g in got if st == 0]
    assert len(slots) == len(stems)
    if merge:
        st = arena.complete_slots(slots, gens, [(0, HALF)])
        assert set(st) == {0}, st                    # partial: the other writer's half is missing
    if ref:
        arena.ref_slots(slots, +1)
    return slots, gens


# ------------------------------------------------------------------ root (C state machine) ------


def test_root_partial_host_write_claim_is_refused_by_quarantine_stale_forever(tmp_path):
    """The h54 root: merged writer references the partial claim -> 3; trickling extents -> 0."""
    _be, arena, stems = _setup(tmp_path, n=2)
    slots, gens = _partial_host_write(arena, stems[:1])
    time.sleep(0.05)
    assert arena.quarantine_stale(slots, gens, 1) == [3]                 # referenced, old: refused
    s2, g2 = _partial_host_write(arena, stems[1:], ref=False)
    assert arena.quarantine_stale(s2, g2, 10_000) == [0]                 # young (a merge just landed)
    assert [int(x[1]) for x in arena.find_slots(stems)] == [1, 1]        # both CLAIMED for good


def test_takeover_quarantines_referenced_host_write_but_never_frees_it_while_held(tmp_path):
    _be, arena, stems = _setup(tmp_path, n=1)
    slots, gens = _partial_host_write(arena, stems)
    assert arena.quarantine_host_write(slots, gens) == [1]
    assert arena.find_slots(stems) == [(-1, 0)]                          # the key is free for L3
    assert arena.claim_census(slots)[0]["state"] == 4                    # QUARANTINE, not freed
    assert arena.slot_refs(slots) == [1]
    arena.quarantine_sweep(0)
    assert arena.claim_census(slots)[0]["state"] == 4, "referenced: the sweep never frees it"
    # the missing writer's late completion is refused by name (6), takes no reference
    assert arena.complete_slots(slots, gens, [(HALF, HALF)]) == [6]
    assert arena.ref_slots(slots, +1) == 0, "a QUARANTINE slot takes no new reference"
    arena.ref_slots(slots, -1)
    arena.quarantine_sweep(0)
    assert arena.claim_census(slots)[0]["state"] == 0                    # last reference gone: FREE


def test_takeover_leaves_other_roles_alone(tmp_path):
    _be, arena, stems = _setup(tmp_path, n=2)
    got = arena.claim_slots(stems, [TOTAL] * 2, role=ha.ROLE_L3FILL)
    assert arena.quarantine_host_write([s for s, _, _ in got], [g for _, _, g in got]) == [4, 4]
    assert [int(x[1]) for x in arena.find_slots(stems)] == [1, 1]


# ------------------------------------------------------------------ the fill ---------------------


def test_S4c_form_referenced_partial_host_write_claim_is_read_from_l3(tmp_path, caplog):
    be, arena, stems = _setup(tmp_path)
    held, _g = _partial_host_write(arena, stems[2:6])
    caplog.set_level(logging.INFO, logger=hs.logger.name)
    t0 = time.perf_counter()
    out = _fill_on(be, arena, stems)
    assert _landed(out) == len(stems), out
    assert time.perf_counter() - t0 < 3.0
    assert [tuple(x) for x in arena.find_slots(stems)] == [(int(s), 2) for s in out]
    assert not set(held) & set(out), "the reader's pages are fresh slots, never the partial claim"
    assert "L3-FILL HOST-WRITE-TAKEOVER" in caplog.text
    assert arena.slot_refs(held) == [1] * 4                               # the writer's node keeps its ref


def test_S4c3_form_trickling_extents_do_not_hold_the_reader(tmp_path):
    """No reference, but late extents keep the claim young (TP merges trickle, one writer dormant)."""
    be, arena, stems = _setup(tmp_path)
    held, gens = _partial_host_write(arena, stems[2:6], ref=False)
    stop = threading.Event()

    def trickle():
        k = 0
        while not stop.is_set():
            q = 64 * (k % 16)
            arena.complete_slots([held[k % 4]], [gens[k % 4]], [(HALF + q, 64)])
            k += 1
            time.sleep(0.02)

    th = threading.Thread(target=trickle, daemon=True)
    th.start()
    try:
        out = _fill_on(be, arena, stems)
    finally:
        stop.set()
        th.join(2)
    assert _landed(out) == len(stems), out


def test_switch_off_keeps_the_cut(tmp_path, monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_L3FILL_HOST_WRITE_TAKEOVER", "0")
    be, arena, stems = _setup(tmp_path)
    _partial_host_write(arena, stems[2:6])
    out = _fill_on(be, arena, stems)
    assert _landed(out) == 2                                              # the h54 cut at the held claim


def test_join_held_line_names_refs_merged_writers_and_verdicts(tmp_path, caplog):
    be, arena, stems = _setup(tmp_path)
    _partial_host_write(arena, stems[2:6])
    caplog.set_level(logging.INFO, logger=hs.logger.name)
    _fill_on(be, arena, stems)
    line = next(r.getMessage() for r in caplog.records if "L3-FILL JOIN-HELD" in r.getMessage())
    assert "held=4 referenced=4 open=0 host_write=4 verdicts=referenced:4" in line, line
    assert "/host-write/" in line and "/refs=1/" in line and "/cov=2048of4096/" in line, line


def test_join_held_line_switch_off(tmp_path, caplog, monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_L3FILL_JOIN_HELD_LOG", "0")
    be, arena, stems = _setup(tmp_path)
    _partial_host_write(arena, stems[2:6])
    caplog.set_level(logging.INFO, logger=hs.logger.name)
    out = _fill_on(be, arena, stems)
    assert _landed(out) == len(stems)
    assert "L3-FILL JOIN-HELD" not in caplog.text


def _peer_took_over_first(arena, stems_of_rows, delay_s=0.08):
    """Hook at the JOIN-HELD point: a PEER stage quarantined the same claims first and holds the stems'
    fresh L3 claims; it completes them ``delay_s`` later (its read from L3)."""
    state = {"done": False}
    orig = hs._note_join_held

    def hook(arena_, rows, verdicts, phase):
        orig(arena_, rows, verdicts, phase)
        if state["done"]:
            return
        state["done"] = True
        arena_.quarantine_host_write([int(s) for _, s, _ in rows], [int(g) for _, _, g in rows])
        mine = arena_.claim_slots([stems_of_rows[int(i)] for i, _, _ in rows], [TOTAL] * len(rows),
                                  role=ha.ROLE_L3FILL)

        def finish():
            time.sleep(delay_s)
            for slot, st, g in mine:
                if st == 0:
                    arena_.slot_view(slot, TOTAL)[:TOTAL] = b"\x11" * TOTAL
                    arena_.complete_slots([slot], [g], [(0, TOTAL)])

        threading.Thread(target=finish, daemon=True).start()

    return hook


def test_a_stem_a_peer_stage_took_over_first_is_awaited_not_cut(tmp_path, monkeypatch):
    be, arena, stems = _setup(tmp_path)
    _partial_host_write(arena, stems[2:6])
    monkeypatch.setattr(hs, "_note_join_held", _peer_took_over_first(arena, stems))
    out = _fill_on(be, arena, stems)
    assert _landed(out) == len(stems), out


def test_three_stages_with_a_dormant_holder_land_the_same_full_depth(tmp_path):
    """Group view (raenge-nie-uneins): every P stage lands the full store depth -- no stage-local
    divergence is created; the read depth stays the group's told."""
    be, arena, stems = _setup(tmp_path, n=12, slots=48)
    _partial_host_write(arena, stems[3:9])
    outs = {}

    def stage(k):
        outs[k] = HiCacheFile.arena_fill_from_disk(be, arena, list(stems), TOTAL, prefix=True, take_ref=True)

    ths = [threading.Thread(target=stage, args=(k,), name=f"hicache-prefetch-io-{k}") for k in range(3)]
    for t in ths:
        t.start()
    for t in ths:
        t.join(30)
    assert sorted(outs) == [0, 1, 2]
    assert [_landed(outs[k]) for k in range(3)] == [len(stems)] * 3, {k: list(v) for k, v in outs.items()}


# ------------------------------------------------------------------ mutants ----------------------


def test_mutant_takeover_with_quarantine_stale_semantics_cuts_again(tmp_path, monkeypatch):
    """Mutant of decision 1 (take over regardless of references / progress): with the old verdict the
    referenced claim stays and the read is cut -- the test above would be red."""
    be, arena, stems = _setup(tmp_path)
    _partial_host_write(arena, stems[2:6])
    monkeypatch.setattr(type(arena), "quarantine_host_write",
                        lambda self, slots, gens: self.quarantine_stale(slots, gens, 100))
    out = _fill_on(be, arena, stems)
    assert _landed(out) == 2


def test_mutant_takeover_of_any_role_steals_a_live_peer_fill(tmp_path, monkeypatch):
    """Mutant of decision 2 (host-write only): taking over a peer's live L3 claim would cost the peer
    its completion (6) -- the role guard keeps the peer's claim and key."""
    _be, arena, stems = _setup(tmp_path, n=2)
    got = arena.claim_slots(stems, [TOTAL] * 2, role=ha.ROLE_L3FILL)
    rows = [(i, s, g) for i, (s, _st, g) in enumerate(got)]
    out = [None, None]
    assert hs._take_over_host_write_joins(arena, stems, rows, TOTAL, out, []) == []
    assert [int(x[1]) for x in arena.find_slots(stems)] == [1, 1]
    assert arena.complete_slots([s for _, s, _ in rows], [g for _, _, g in rows], [(0, TOTAL)]) == [1, 1]
    # the mutant: any role
    got2 = arena.claim_slots(["q0"], [TOTAL], role=ha.ROLE_L3FILL)
    monkeypatch.setattr(type(arena), "quarantine_host_write",
                        lambda self, slots, gens: self.quarantine_stale(slots, gens, 0))
    hs._take_over_host_write_joins(arena, ["q0"], [(0, got2[0][0], got2[0][2])], TOTAL, [None], [])
    assert arena.complete_slots([got2[0][0]], [got2[0][2]], [(0, TOTAL)]) == [6], "the peer lost its fill"


def test_mutant_rejoined_dropped_cuts_the_stage(tmp_path, monkeypatch):
    """Mutant of decision 3 (a stem a peer took over first is awaited): dropped, the stage is cut."""
    be, arena, stems = _setup(tmp_path)
    _partial_host_write(arena, stems[2:6])
    monkeypatch.setattr(hs, "_note_join_held", _peer_took_over_first(arena, stems))
    orig = hs._take_over_host_write_joins

    def dropping(arena_, stems_, joined, total, out, rejoined):
        got = orig(arena_, stems_, joined, total, out, rejoined)
        rejoined.clear()
        return got

    monkeypatch.setattr(hs, "_take_over_host_write_joins", dropping)
    out = _fill_on(be, arena, stems)
    assert _landed(out) == 2


def test_a_fill_off_the_prefetch_thread_never_takes_over(tmp_path):
    """Decision 4 (only after a join wait that ran): the scheduler's own thread, the flip legs, the
    write-behind keep the immediate miss -- a host-write claim in flight there is never taken."""
    be, arena, stems = _setup(tmp_path)
    held, gens = _partial_host_write(arena, stems[2:6])
    box = {}
    th = threading.Thread(target=lambda: box.setdefault("r", HiCacheFile.arena_fill_from_disk(
        be, arena, list(stems), TOTAL, prefix=True, take_ref=True)), name="scheduler-main")
    th.start()
    th.join(30)
    assert _landed(box["r"]) == 2
    assert [int(x[0]) for x in arena.find_slots(stems[2:6])] == held, "the writer keeps its keys"
