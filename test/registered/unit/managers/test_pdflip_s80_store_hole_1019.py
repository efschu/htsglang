"""S80 STORE-HOLE (rc17 boot ..._7d93470b5e_1019_181004, pdflip-8-209, 19.10.).

Metal: D->P flip 18:34:25-28 (R6 reroute). P's LEG1-EARLY read ran dormant during the flip and stopped at
``kv=1337 by=kv_prefix tokens=85568``; awake, ``P-WAKE-PROBE have=85568 store_now=85568 -- the told stands``;
P recomputed 16482 tokens. The planner's reading was "P read before D's sleep flush published" -- but D's flush
had drained and reset before P woke (``#1470 FLUSH-PUBLISH in_flight_after=0``, ``#1068 RESET JOIN`` 18:34:27;
the flip order IS the witness, the wake probe asks after it). The store's KV chain had ended at page 1337 since
18:26:34: 41 pages (85568..88192) CLAIMED by D TP0 with share 0 and no open writer (``ARENA-REF-CENSUS-CLAIM
claimed=41 oldest=1378/.../pid1360/host-write/g3/open0`` from 18:27 to the flip reset), their owners had evicted
their shares unbacked. The wake probe only said "how deep" -- the two classes looked alike.

Now PP0 names the first missing page at the told from the shared arena header (REAL C arena below):
orphan / in_flight / absent / anchor, and the COMPLETE pages the hole strands. The told is unchanged.
RED on the rc17 base: no ``s80_store_hole_census``, no ``S80 STORE-HOLE`` line.
"""
from __future__ import annotations

import logging
import os
import re
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from flliper.srt.managers import pdflip_store_told as T  # noqa: E402
from flliper.srt.mem_cache.storage.file.hicache_arena import ShmArena  # noqa: E402
from flliper.srt.pdflip import short_read as SR  # noqa: E402

PAGE = 64
SLOT = 4096
N_PAGES = 12          # scaled 8-209: 1594 pages asked
HAVE_PAGES = 5        # 1337 pages the store held
HOLE = (5, 6, 7)      # the 41 orphan pages 85568..88192
AFTER = (8, 9, 10)    # COMPLETE pages behind the hole (D's later backups)


class _Tree:
    def __init__(self, have):
        self.have = have
        self.prefetch_loaded_tokens_by_reqid = {}

    def completed_prefetch_tokens(self, rid):
        return self.have


class _Sched:
    def __init__(self, dormant=False):
        self.calls = []
        self.pdflip_dormant = dormant
        self.page_size = PAGE

    def _prefetch_kvcache(self, req, *a, **k):
        self.calls.append(req.rid)
        return "issued"


class _Backend:
    """The store's anchored answer = the leading COMPLETE run of the real arena (capped by ``anchor_pages``:
    a chain may end at a missing Mamba anchor before the KV ends)."""

    def __init__(self, arena, anchor_pages=None):
        self.arena = arena
        self.anchor_pages = anchor_pages
        self.asked = []

    @staticmethod
    def _get_suffixed_key(h):
        return f"{h}.kv"

    def batch_exists_v2(self, keys, transfers, extra):
        self.asked.append(len(keys))
        lead = 0
        for st in self.arena.find_states([self._get_suffixed_key(k) for k in keys]):
            if int(st) != 2:
                break
            lead += 1
        if self.anchor_pages is not None:
            lead = min(lead, self.anchor_pages)
        return types.SimpleNamespace(kv_hit_pages=lead)


def _cc(arena, anchor_pages=None, with_arena=True):
    return types.SimpleNamespace(
        page_size=PAGE, storage_backend=_Backend(arena, anchor_pages),
        mem_pool_host=types.SimpleNamespace(arena=arena if with_arena else None),
        get_hash_str=lambda ids, root, page_size=PAGE: [f"h{i}" for i in range(len(ids) // page_size)],
        _presence_pool_transfers=lambda: [types.SimpleNamespace(name="mamba")],
    )


@pytest.fixture
def arena(tmp_path):
    a = ShmArena(str(tmp_path / "arena-786432.bin"), SLOT, 32)
    yield a
    a.close()


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for env in (T.P_HELD_REREAD_ENV, T.P_HINT_REREAD_ENV, T.P_WAKE_PROBE_ENV, T.S80_HOLE_CENSUS_ENV):
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setattr(T, "_tree_key_probe_armed", lambda: False)
    for d in (SR._HELD, SR._MORE, SR._HINT):
        d.clear()
    yield
    for d in (SR._HELD, SR._MORE, SR._HINT):
        d.clear()


def _complete(arena, pages):
    for p in pages:
        (s, st, g), = arena.claim_slots([f"h{p}.kv"], [SLOT])
        assert st == 0
        assert arena.complete_slots([s], [g], [(0, SLOT)]) == [1]


def _tp0_share0(arena, pages):
    """D TP0 under the token cut: claims the page and merges share 0 (no extents) -- the owners never join."""
    for p in pages:
        (s, st, g), = arena.claim_slots([f"h{p}.kv"], [SLOT])
        assert arena.complete_slots([s], [g], []) == [0]


def _open_writer(arena, pages):
    """A publish still running: the claim's writer has not merged yet."""
    for p in pages:
        (s, st, g), = arena.claim_slots([f"h{p}.kv"], [SLOT])
        assert st == 0


def _dormant_req(rid, n_tokens):
    req = types.SimpleNamespace(rid=rid, origin_input_ids=list(range(n_tokens)), extra_key=None)
    setattr(req, T.WAKE_PROBE_ATTR, True)
    return req


def _wake(cc, rid, caplog, n_tokens=N_PAGES * PAGE + 34):
    """PP0 awake at its told for a read that ran dormant during the flip and landed HAVE_PAGES."""
    have = HAVE_PAGES * PAGE
    SR.note_reap(rid, requested_pages=N_PAGES, hit_pages=HAVE_PAGES, completed_tokens=have, page_size=PAGE)
    s = _Sched(dormant=False)
    s.cache_controller = cc
    with caplog.at_level(logging.INFO, logger=T.logger.name):
        held = T._p_held_short_reread(s, _Tree(have), _dormant_req(rid, n_tokens), rid)
    lines = [r for r in caplog.records if "S80 STORE-HOLE" in r.getMessage()]
    return held, s, lines


def _field(line, key):
    m = re.search(r"\b%s=(\S+)" % key, line.getMessage())
    return m.group(1) if m else None


def test_pdflip_8_209_orphan_hole_is_named_and_the_told_stands(arena, caplog):
    """The metal: KV complete to page 5, TP0 share-0 claims (open0) on 5..7, COMPLETE pages behind."""
    _complete(arena, range(HAVE_PAGES))
    _tp0_share0(arena, HOLE)
    _complete(arena, AFTER)
    held, s, lines = _wake(_cc(arena), "pdflip-8-209", caplog)
    assert held is False and s.calls == []                     # the told stands, unchanged (no re-read)
    assert len(lines) == 1 and lines[0].levelno == logging.INFO
    ln = lines[0]
    assert _field(ln, "class") == "orphan"
    assert _field(ln, "have") == str(HAVE_PAGES * PAGE)
    assert _field(ln, "first_page") == str(HAVE_PAGES)
    assert _field(ln, "hole_pages") == "3"
    assert _field(ln, "open") == "0" and _field(ln, "state") == "1" and _field(ln, "covered") == "0"
    assert _field(ln, "complete_after") == "3"
    assert _field(ln, "complete_after_tokens") == str(3 * PAGE)


def test_read_before_publish_open_writer_is_the_in_flight_tripwire(arena, caplog):
    """A writer still open on the first missing page at the wake: the flip-order witness broken -- WARNING."""
    _complete(arena, range(HAVE_PAGES))
    _open_writer(arena, (HAVE_PAGES,))
    held, s, lines = _wake(_cc(arena), "pdflip-8-209", caplog)
    assert held is False and s.calls == []
    assert len(lines) == 1 and lines[0].levelno == logging.WARNING
    assert _field(lines[0], "class") == "in_flight" and _field(lines[0], "open") == "1"


def test_never_written_page_is_absent(arena, caplog):
    _complete(arena, range(HAVE_PAGES))
    _complete(arena, AFTER)
    _held, _s, lines = _wake(_cc(arena), "pdflip-8-209", caplog)
    assert _field(lines[0], "class") == "absent" and _field(lines[0], "slot") == "-1"
    assert _field(lines[0], "hole_pages") == "3" and _field(lines[0], "complete_after") == "3"


def test_kv_past_the_told_is_an_anchor_end_not_a_hole(arena, caplog):
    _complete(arena, range(9))
    _held, _s, lines = _wake(_cc(arena, anchor_pages=HAVE_PAGES), "pdflip-8-209", caplog)
    assert _field(lines[0], "class") == "anchor"
    assert _field(lines[0], "kv_end") == str(9 * PAGE)


def test_store_holds_more_rereads_and_asks_no_census(arena, caplog):
    """The race the wake probe exists for (pdflip-8-208 / 10-279 class): unchanged, no hole line."""
    _complete(arena, range(9))
    held, s, lines = _wake(_cc(arena), "pdflip-8-208", caplog)
    assert held is True and s.calls == ["pdflip-8-208"]
    assert lines == []


def test_switch_off_no_census(arena, caplog, monkeypatch):
    monkeypatch.setenv(T.S80_HOLE_CENSUS_ENV, "0")
    _complete(arena, range(HAVE_PAGES))
    _tp0_share0(arena, HOLE)
    held, s, lines = _wake(_cc(arena), "pdflip-8-209", caplog)
    assert held is False and lines == []


def test_no_arena_pool_answers_none_and_the_told_is_unchanged(arena, caplog):
    _complete(arena, range(HAVE_PAGES))
    _tp0_share0(arena, HOLE)
    held, s, lines = _wake(_cc(arena, with_arena=False), "pdflip-8-209", caplog)
    assert held is False and s.calls == [] and lines == []


def test_census_failure_never_stops_the_told(arena, caplog, monkeypatch):
    _complete(arena, range(HAVE_PAGES))
    _tp0_share0(arena, HOLE)

    def _boom(stems):
        raise RuntimeError("arena gone")

    monkeypatch.setattr(arena, "find_slots", _boom)
    held, s, lines = _wake(_cc(arena), "pdflip-8-209", caplog)
    assert held is False and s.calls == []
    assert len(lines) == 1 and "census unavailable" in lines[0].getMessage()


def test_census_is_read_only(arena):
    """No claim, no reference, no state change on any slot."""
    _complete(arena, range(HAVE_PAGES))
    _tp0_share0(arena, HOLE)
    _complete(arena, AFTER)
    stems = [f"h{i}.kv" for i in range(N_PAGES)]
    before = (arena.find_slots(stems), arena.stats())
    slots = [s for s, _ in before[0] if s >= 0]
    refs0, census0 = arena.slot_refs(slots), arena.claim_census(slots)
    out = T.s80_store_hole_census(_cc(arena), [f"h{i}" for i in range(N_PAGES)], HAVE_PAGES * PAGE,
                                  N_PAGES * PAGE, PAGE, "pdflip-8-209")
    assert out["cls"] == "orphan"
    assert (arena.find_slots(stems), arena.stats()) == before
    assert arena.slot_refs(slots) == refs0
    strip = [{k: v for k, v in r.items() if k != "age_ms"} for r in census0]
    assert [{k: v for k, v in r.items() if k != "age_ms"} for r in arena.claim_census(slots)] == strip
