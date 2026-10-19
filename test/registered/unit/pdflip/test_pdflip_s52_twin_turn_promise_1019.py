"""S52 (19.10.2026): a late twin waits for the source's own TURN anchor.

Metal (NF rc14, P log ...1019_010742, PP0):
  05:20:07 'TURN-ANCHOR TRACK rid=pdflip-1-28 step=[3072,18976) turn=18624'
  05:20:11 '#TW TWIN-DEFER rid=pdflip-1-30 len=19110 shared=18662 sources=[pdflip-1-28]'
  05:20:14 'TURN-ANCHOR INSERT rid=pdflip-1-28 at=18624 ... value=taken'
  05:20:15 '#TW TWIN-NO-COMMIT rid=pdflip-1-30 at=release sources=[(pdflip-1-28, 18662, 3072, 18624)]'
  05:20:15 '#1416 STORE-TOLD ANCHOR-CLAMP rid=pdflip-1-30 completed=18624 anchored=3072'
  front:   'PDFLIP-USAGE-TRUE rid=pdflip-1-30 prompt=19110 p_computed=16038'
The twin boundary B = floor_page(18662 - 1) = 18624 was exactly the source's turn anchor,
but the promise ledger only knew boundaries booked from ``twin_bounds`` (twins queued when
the source's step was planned) -- 1-30 reached PP0 4 s after that plan: 15552 tokens
recomputed (~3 s of PP0) and the P epoch ended ~2.5 s later.

Pinned (hermetic, CPU; RED on 92dd662831):
  1. a twin arriving after the source planned its step WITH a turn track in (s0, B] is
     held (promise), and released as a TWIN once the source finished and settled;
  2. the metal order: provisional at intake, then at the first pass after the source's
     plan the turn track keeps the twin (base: TWIN-NO-COMMIT at=release);
  3. a turn track past B, or at/below the source's start, promises nothing;
  4. a twin boundary booked on the turn position whose turn slot then fails is declined.
"""

from types import SimpleNamespace

import pytest
import torch

from flliper.srt.pdflip import p_twin_defer as tw
from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")

TOK = 248045            # <|im_start|> (turn_anchor's token in these tests)
MIN = 32                # FLLIPER_PDFLIP_P_TWIN_MIN_TOKENS
N = 120                 # source prompt
TURN = 80               # source's last message starts here -> turn anchor 80 (page 1)
S0 = 8                  # source's start depth (its registered prefix)
SHARED = 90             # twin shares ids[:90] -> B = 89


def _src_ids():
    ids = list(range(1000, 1000 + N))
    ids[TURN] = TOK          # the last message's <|im_start|>
    ids[N - 3] = TOK         # the generation prompt '<|im_start|>assistant\n'
    return ids


SRC = _src_ids()


def _twin_ids(shared=SHARED, total=110, salt=1):
    return SRC[:shared] + [900_000 + salt * 10_000 + i for i in range(total - shared)]


class _Alloc:
    def __init__(self, free):
        self.free_ids = list(free)

    def alloc(self, n):
        if len(self.free_ids) < n:
            return None
        out, self.free_ids = self.free_ids[:n], self.free_ids[n:]
        return torch.tensor(out, dtype=torch.int64)

    def free(self, t):
        pass


class _Sched:
    def __init__(self):
        self.ps = SimpleNamespace(pp_rank=0, pp_size=3, tp_size=1)
        self.waiting_queue = []
        self.chunked_req = None
        self.running_batch = None
        self.page_size = 1
        self.chunked_prefill_size = 16384   # no whole chunk fits (s0, shared]


class _R:
    def __init__(self, rid, ids, head=0):
        self.rid = rid
        self.origin_input_ids = list(ids)
        self.extra_key = None
        self.done = False
        self._prefetch_registered_prefix_len = head
        self.prefix_indices = None
        self.fill_ids = None
        self.output_ids = []
        self.cache_protected_len = 0

    def finished(self):
        return self.done


class _Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def _reset():
    from flliper.srt.pdflip import twin_anchor as ta2

    ta2._reset_for_test()


@pytest.fixture
def pp0(monkeypatch):
    monkeypatch.setenv(tw.ENV, "1")
    monkeypatch.setenv(tw.ENV_MIN_TOKENS, str(MIN))
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "P")
    monkeypatch.setattr(tw, "_twin_anchor_armed", lambda: True, raising=False)
    clock = _Clock()
    monkeypatch.setattr(tw, "_now", clock)
    _reset()
    s = _Sched()
    s.clock = clock
    yield s
    _reset()


def _admit_and_plan(sched, src, turn_slot=True, twin_bounds=()):
    """The source is admitted at S0 and its one step [S0, N) is planned by the REAL
    turn_anchor.note_step (no twin queued then: no twin bounds)."""
    from flliper.srt.pdflip import turn_anchor as ta

    src.prefix_indices = [0] * S0
    src.fill_ids = list(src.origin_input_ids)
    sched.waiting_queue = [q for q in sched.waiting_queue if q is not src]
    sched.chunked_req = src
    alloc = _Alloc([11, 12, 13] if turn_slot else [])
    batch = SimpleNamespace(reqs=[src], req_to_token_pool=SimpleNamespace(mamba_allocator=alloc))
    desc = ta.note_step(batch=batch, desc=None, req=src, row=0, prefix=S0, end=N,
                        track_mask=True, main_track=N - 1, chunk=1, page=1, tok=TOK,
                        twin_bounds=twin_bounds)
    return desc


def _finish_and_settle(sched, src, twin_rid="pdflip-1-30"):
    from flliper.srt.pdflip import turn_anchor as ta

    # the tree's insert of the turn node (written), then the source finishes
    ta.note_insert(src, TURN, prefix_len=TURN, taken=True, prompt=N, step=(S0, N), kind="turn")
    src.done = True
    sched.chunked_req = None
    tw.tick(sched)
    # the finish starts the settle: no release before pp_size passes and settle_ms
    assert tw.release_due(sched, {twin_rid}) == []
    for _ in range(3):
        tw.tick(sched)
    sched.clock.t += 1.0


def test_turn_anchor_is_planned_at_the_last_message_start(pp0):
    src = _R("pdflip-1-28", SRC)
    desc = _admit_and_plan(pp0, src)
    assert desc is not None and sorted(desc.targets) == [TURN]


def test_a_late_twin_waits_for_the_sources_turn_anchor(pp0):
    """1-30's shape with s0 known at intake: base = TWIN-NO-COMMIT at=intake (False)."""
    src = _R("pdflip-1-28", SRC, head=S0)
    _admit_and_plan(pp0, src)
    tw.tick(pp0)                                   # PP0 sees the source in flight: s0 = 8
    twin = _R("pdflip-1-30", _twin_ids())
    pp0.waiting_queue = [twin]
    assert tw.intake_defer(pp0, twin) is True
    assert tw.release_due(pp0, {"pdflip-1-30"}) == []   # still pending on the source
    _finish_and_settle(pp0, src)
    assert tw.release_due(pp0, {"pdflip-1-30"}) == [(twin, True)]
    assert tw.take_pp0_twin(pp0, "pdflip-1-30") is True  # its told goes out absolute


def test_metal_order_provisional_then_kept_by_the_turn_track(pp0):
    """The metal order: at intake the source's start is unknown (provisional); the
    first pass after its plan found no B booking -> base TWIN-NO-COMMIT at=release."""
    src = _R("pdflip-1-28", SRC, head=0)
    pp0.waiting_queue = [src]
    twin = _R("pdflip-1-30", _twin_ids())
    assert tw.intake_defer(pp0, twin) is True     # provisional (head 0 < B)
    pp0.waiting_queue = [twin]
    _admit_and_plan(pp0, src)                     # planned without a B track, turn at 80
    tw.tick(pp0)
    assert tw.release_due(pp0, {"pdflip-1-30"}) == []
    assert tw.is_deferred(pp0, "pdflip-1-30")
    _finish_and_settle(pp0, src)
    assert tw.release_due(pp0, {"pdflip-1-30"}) == [(twin, True)]


@pytest.mark.parametrize("shared", [TURN, TURN - 5])
def test_a_turn_track_past_the_boundary_promises_nothing(pp0, shared):
    """B = shared - 1 < 80: the turn anchor is off the twin's path -> registers at once."""
    src = _R("pdflip-1-28", SRC, head=S0)
    _admit_and_plan(pp0, src)
    tw.tick(pp0)
    twin = _R("pdflip-1-30", _twin_ids(shared=shared))
    pp0.waiting_queue = [twin]
    assert tw.intake_defer(pp0, twin) is False
    assert not tw.is_deferred(pp0, "pdflip-1-30")


def test_a_track_at_or_below_the_sources_start_promises_nothing():
    from flliper.srt.pdflip import twin_anchor as ta2

    _reset()
    ta2._set("pdflip-x", 64, "written")
    assert ta2.deepest_track("pdflip-x", 64, 200) is None    # at s0: there already
    assert ta2.deepest_track("pdflip-x", 10, 63) is None     # past B
    assert ta2.deepest_track("pdflip-x", 10, 64) == 64
    ta2._set("pdflip-x", 128, "declined")
    assert ta2.deepest_track("pdflip-x", 10, 200) == 64      # a declined track writes nothing
    _reset()


def test_a_boundary_on_the_turn_position_is_declined_when_the_turn_slot_fails(pp0):
    from flliper.srt.pdflip import twin_anchor as ta2

    src = _R("pdflip-1-28", SRC, head=S0)
    assert _admit_and_plan(pp0, src, turn_slot=False, twin_bounds=(TURN,)) is None
    assert ta2.status("pdflip-1-28", TURN) == "declined"


def test_switch_off_books_no_turn_track(pp0, monkeypatch):
    """FLLIPER_PDFLIP_TWIN_ANCHOR=0: the ledger stays as before (no turn bookings)."""
    from flliper.srt.pdflip import twin_anchor as ta2

    monkeypatch.setenv("FLLIPER_PDFLIP_TWIN_ANCHOR", "0")
    ta2._reset_for_test()
    src = _R("pdflip-1-28", SRC, head=S0)
    _admit_and_plan(pp0, src)
    assert ta2.status("pdflip-1-28", TURN) is None
