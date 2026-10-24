# SPDX-License-Identifier: Apache-2.0
"""DUAL-GRANT-1024 (Auftrag #223): the dual card KV ledger committed the 5090 to 62 MiB.

Metal: 27B NVFP4 dual, rc41 1ecda02a58,
boot_weg2_dkr27bnvfp4dualbar1drei10241158_1ecda02a58_1024_115913 (P and D share
cards 0,1,2 under MPS). Card 0 (the 5090, ledger card=e773fd938f6d):

  12:08:38 D GROW 65536 -> 212992, ledger have 2214592512 B, card free 2128 MiB
  12:08:43 D SHRINK 212992 -> 204800 (-67108864 B), PP0 GRANT pdflip-0-20
           committed=2231369728 B (the atomic group grant)
  12:08:44 PP0 chunk: card_free 64 -> 62 MiB, CORRIDOR LAW BREACHED (PP0 law
           1055, TP0 law 767), ooms=0, both processes' torch reserved FLAT.

Nothing was double-counted (not the #213 VramCredit class -- no wake, no
credit): ledger free and card free agree to 16 MiB. The ledger promised ALL of
the card. The gap phys_free - ledger_free fell over the boot (2542 -> 360 ->
102 -> 16 -> 0 MiB, 12:20:27) while P's and D's allocator caches grew; the
existing LEDGER-PHYS self-check only booked a NEGATIVE gap (OVER-PROMISE), so a
gap of 0 was "ok" and a full commit put the card at 0.

Fix: ``phys_check`` measures the gap against the card's ACTUATING corridor law
(``CorridorFloor.actuates`` -- a measured transient; an unmeasured fallback
never spends KV pool). Same persistence rule, same ``reconcile``.

DANGER DIRECTIONS, each with a mutant built from the shipped source:
* OOM: without the law term the 12:08:43 grant goes through and the card lands
  at the line's 64 MiB (M1).
* Haenger: the price is exactly law - gap, once; a steady state books nothing
  more, and P's grant is feasible again once D's commit falls (M2 over-prices).
* I1: a card already under its law never drives the budget below what is
  committed.
"""

from __future__ import annotations

import inspect
import os
import re
import tempfile
import textwrap
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest

from flliper.srt.managers import corridor_guard as CG
from flliper.srt.pdflip import card_kv_ledger as K
from flliper.srt.pdflip import dual_p_kv_stage as S
from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

MIB = 1 << 20
FIX = os.path.join(os.path.dirname(__file__), "fixtures", "dual_grant_1024", "excerpt_115913.txt")


def _lines():
    with open(FIX, encoding="utf-8") as f:
        return [l.rstrip("\n") for l in f if l.startswith("[")]


def _one(pat):
    hits = [m for m in (re.search(pat, l) for l in _lines()) if m]
    assert len(hits) == 1, (pat, len(hits))
    return hits[0]


class Metal:
    """Every number of the replay, parsed from the verbatim lines."""

    def __init__(self):
        self.d_contrib = int(_one(r"TP0\] DUAL-TP3PP3 D-KV JOIN card=e773fd938f6d .*contributed=(\d+) B").group(1))
        self.p_contrib = int(_one(r"PP0\] DUAL-TP3PP3 P-KV JOIN card=e773fd938f6d .*contributed=(\d+) B").group(1))
        self.book = int(_one(r"ALLOC-CACHE-BOOK card=e773fd938f6d pp_rank=0: booked (\d+) of").group(1))
        m = _one(r"12:08:38 PP0\] #1530 GRANT-SHORT rid=pdflip-0-20 .*cards=\[0:need=(\d+),have=(\d+),P=(\d+),D=(\d+)")
        self.need, self.have, self.p_comm, self.d_comm = (int(m.group(i)) for i in range(1, 5))
        self.phys_0839 = int(_one(r"12:08:39 TP0\] PDFLIP-VRAM-PEAK .*card_free_mib=(\d+) ").group(1)) * MIB
        self.shrink = int(_one(r"12:08:43 TP0\] DUAL-TP3PP3 D-KV SHRINK 212992 -> 204800 tokens \(-(\d+) B").group(1))
        self.granted = int(_one(r"12:08:43 PP0\] DUAL-TP3PP3 P-KV MAPPED-BY-GRANT .*committed=(\d+) B").group(1))
        self.grow_0838 = int(_one(r"12:08:38 TP0\] DUAL-TP3PP3 D-KV GROW 65536 -> 212992 tokens \(\+(\d+) B").group(1))
        self.bottom = int(_one(r"12:08:44 PP0\] PDFLIP-VRAM-PEAK rank=0 phase=chunk rows=512 .*card_free_mib=(\d+) ")
                          .group(1)) * MIB
        self.law_p = int(_one(r"12:08:46 PP0\] CORRIDOR LAW BREACHED .*BELOW the (\d+) MiB law").group(1))
        self.law_d = int(_one(r"12:08:46 TP0\] CORRIDOR LAW BREACHED .*BELOW the (\d+) MiB law").group(1))
        m = _one(r"12:20:27 PP0\] #1530 GRANT-SHORT .*cards=\[0:need=\d+,have=(\d+),P=(\d+),D=(\d+)")
        self.have_1220, self.p_1220, self.d_1220 = (int(m.group(i)) for i in range(1, 4))
        self.phys_1220 = int(_one(r"12:20:27 PP0\] PDFLIP-VRAM-PEAK .*card_free_mib=(\d+) ").group(1)) * MIB
        # the card-0 bookings before 12:20:27 (the old rule chasing a gap that already stood at 0)
        self.booked_before_1220 = sum(int(m.group(1)) for m in (
            re.search(r"(?:12:10:35 TP0|12:19:10 PP0|12:20:12 PP0)\] .*LEDGER-PHYS BOOKED (\d+) B", l)
            for l in _lines()) if m)


M = Metal()


def _card(d_committed, p_committed=0):
    """Card 0's ledger as the boot built it: D's boot KV, P's boot KV, D's
    allocator-cache record booked at P's join; then the commitments of the line."""
    path = os.path.join(tempfile.mkdtemp(prefix="wkvdg"), "card")
    d = K.CardKvLedger(path, "D")
    p = K.CardKvLedger(path, "P")
    # D keeps its boot pool mapped until P has joined (12:00:52 JOIN "kept mapped", 12:01:44 SHRINK)
    d.contribute(M.d_contrib, committed=M.d_contrib)
    p.contribute(M.p_contrib, committed=0)          # one live pid for both groups (a dead pid is reaped)
    assert p.book_unpriced(M.book) == M.book
    d.release(M.d_contrib - int(d_committed))
    if p_committed:
        assert p.request(int(p_committed))[0] == int(p_committed)
    return path, d, p


def _floor(mib, source="MEASURED-D", uuid="GPU-e773fd938f6d"):
    return CG.CorridorFloor(card_uuid=uuid, group="D", transient_mib=int(mib), reserve_mib=0, source=source)


def _checks(monkeypatch, led, phys_bytes, floor, fn=None, n=S.PHYS_BOOK_AFTER):
    actor = types.SimpleNamespace(ledger=led)
    clock = [0.0]
    fn = fn or S.phys_check
    g = fn.__globals__                       # a mutant runs in its own copy of the module namespace
    monkeypatch.setitem(g, "_now", lambda: clock[0])
    monkeypatch.setitem(g, "phys_free_bytes", lambda: int(phys_bytes))
    monkeypatch.setattr(CG, "corridor_floor_for_current_device", lambda group=None, **_k: floor)
    for _ in range(n):
        fn(actor, led.group)
        clock[0] += S.PHYS_CHECK_S


def _mutant(fixed, back):
    src = textwrap.dedent(inspect.getsource(inspect.unwrap(S.phys_check)))
    assert src.count(fixed) == 1, "the guarded line moved -- re-aim the mutant"
    ns = dict(vars(S))
    exec(compile(src.replace(fixed, back), S.__file__, "exec"), ns)
    return ns["phys_check"]


def _p_grant(p):
    """PP0's atomic grant on this card (pp0_grant's request/give-back shape)."""
    got, _ = p.request(M.need)
    if got < M.need:
        if got:
            p.release(got)
        return False
    return True


def _after_grant_phys(phys_before):
    """Card free after D's 12:08:43 shrink and P's grant mapping (KV maps move both readings equally)."""
    return phys_before + M.shrink - M.granted


# -- the replay itself ---------------------------------------------------------------------


def test_the_metal_ledger_is_rebuilt_exactly_from_the_lines():
    path, _d, _p = _card(M.d_comm)
    st = K.peek(path)
    assert st.budget == M.have + M.d_comm, "pool = have + D committed on the line"
    assert st.free == M.have
    assert M.granted == M.need


def test_base_rule_lets_the_1208_grant_put_the_card_at_the_lines_64_mib(monkeypatch):
    """REPRO: with no actuating floor (the rc41 rule) the gap of 16 MiB is 'ok',
    the grant goes through, and the card lands where the line says."""
    path, d, p = _card(M.d_comm)
    _checks(monkeypatch, d, M.phys_0839, _floor(M.law_d, source=CG.FLOOR_SOURCE_FALLBACK))
    assert K.peek(path).free == M.have, "an unmeasured floor never spends KV pool"
    d.release(M.shrink)
    assert _p_grant(p)
    assert abs(_after_grant_phys(M.phys_0839) - M.bottom) <= 2 * MIB, "reconstructs card_free_mib=64"
    assert _after_grant_phys(M.phys_0839) < (M.law_d << 20)


def test_the_actuating_law_is_priced_and_the_grant_waits(monkeypatch):
    path, d, p = _card(M.d_comm)
    _checks(monkeypatch, d, M.phys_0839, _floor(M.law_d))
    st = K.peek(path)
    assert st.free == M.phys_0839 - (M.law_d << 20), "ledger free = card free above the law"
    d.release(M.shrink)
    assert not _p_grant(p), "P waits (backpressure, it never presses D) instead of taking the card under the law"
    assert K.peek(path).committed["P"] == 0


def test_once_d_falls_the_grant_goes_and_the_card_stays_at_its_law(monkeypatch):
    """Haenger direction: the price does not starve P -- at D's 12:08:34 level
    (65536 tokens = committed - the 12:08:38 grow) the grant fits again."""
    path, d, p = _card(M.d_comm)
    _checks(monkeypatch, d, M.phys_0839, _floor(M.law_d))
    d.release(M.grow_0838)                                       # D back at 65536 tokens
    phys = M.phys_0839 + M.grow_0838
    assert _p_grant(p)
    assert phys - M.granted >= (M.law_d << 20)


def test_steady_state_books_nothing_more(monkeypatch):
    path, d, _p = _card(M.d_comm)
    _checks(monkeypatch, d, M.phys_0839, _floor(M.law_d))
    budget = K.peek(path).budget
    booked = M.have + M.d_comm - budget
    assert booked == (M.law_d << 20) - (M.phys_0839 - M.have), "exactly law - gap, the shortfall"
    _checks(monkeypatch, d, M.phys_0839, _floor(M.law_d), n=10 * S.PHYS_BOOK_AFTER)
    assert K.peek(path).budget == budget


def test_the_1220_gap_of_zero_is_priced_at_ps_law(monkeypatch):
    """12:20:27: ledger have == card free (1810 MiB both) -- the old check calls that 'ok'."""
    path, _d, p = _card(M.d_1220, p_committed=M.p_1220)
    assert p.book_unpriced(M.booked_before_1220) == M.booked_before_1220
    assert K.peek(path).free == M.have_1220, "the 12:20:27 ledger, rebuilt from JOIN + bookings + commits"
    assert (M.have_1220 >> 20) == (M.phys_1220 >> 20), "gap 0 on the line"
    # card_free_mib is a whole-MiB reading (1810) of a ledger free of 1810.125 MiB: the card's free bytes
    # at MiB resolution ARE the ledger's -- feed the ledger's bytes, gap exactly 0
    phys = M.have_1220
    _checks(monkeypatch, p, phys, _floor(M.law_p, source=CG.FLOOR_SOURCE_FALLBACK))
    assert K.peek(path).free == M.have_1220, "old rule (unmeasured): nothing booked at gap 0"
    _checks(monkeypatch, p, phys, _floor(M.law_p, source="MEASURED-P"))
    assert K.peek(path).free == phys - (M.law_p << 20)


def test_a_card_already_under_its_law_never_breaks_i1(monkeypatch):
    path, d, p = _card(M.d_comm, p_committed=0)
    d.release(M.shrink)
    assert _p_grant(p)                                           # the 12:08:43 state: ledger free 48 MiB
    _checks(monkeypatch, d, M.bottom, _floor(M.law_d))
    st = K.peek(path)
    assert sum(st.committed.values()) <= st.budget
    assert st.free == 0
    p.release(M.granted)                                         # 12:08:47 P RELEASE
    d.request(1 * MIB)                                           # no I1 assert on the next commit


def test_no_card_or_no_guard_means_no_floor_term(monkeypatch):
    monkeypatch.setattr(CG, "corridor_floor_for_current_device", lambda group=None, **_k: _floor(767, uuid=""))
    assert S.law_floor_bytes("D") == 0

    def boom(group=None, **_k):
        raise RuntimeError("no nvml")

    monkeypatch.setattr(CG, "corridor_floor_for_current_device", boom)
    assert S.law_floor_bytes("D") == 0
    monkeypatch.setattr(CG, "corridor_floor_for_current_device", lambda group=None, **_k: _floor(767))
    assert S.law_floor_bytes("D") == 767 * MIB


def test_the_user_reserve_is_not_priced_only_the_measured_transient(monkeypatch):
    """The launcher's card-0 floor for P: floor=2855 = transient 1055 MEASURED-P + reserve 1800.
    The ledger prices the 1055 the breach line grades against, never the reserve (Reserven NIE)."""
    m = _one(r"FRONT\] PDFLIP-LAUNCH CORRIDOR-FLOOR card=(\S+) group=P floor=(\d+) .*transient=(\d+) "
             r"source=(\S+) .*reserve=(\d+) actuates=yes")
    uuid, floor, transient, source, reserve = m.group(1), *(m.group(i) for i in range(2, 6))
    assert int(floor) == int(transient) + int(reserve) and int(transient) == M.law_p
    fl = CG.CorridorFloor(card_uuid=uuid, group="P", transient_mib=int(transient), reserve_mib=int(reserve),
                          source=source)
    monkeypatch.setattr(CG, "corridor_floor_for_current_device", lambda group=None, **_k: fl)
    assert S.law_floor_bytes("P") == M.law_p * MIB
    hand = CG.CorridorFloor(card_uuid=uuid, group="P", transient_mib=900, reserve_mib=0,
                            source=CG.FLOOR_SOURCE_ENV)
    monkeypatch.setattr(CG, "corridor_floor_for_current_device", lambda group=None, **_k: hand)
    assert S.law_floor_bytes("P") == 900 * MIB


# -- mutants in the danger directions ------------------------------------------------------


def test_m1_oom_without_the_law_term_the_grant_takes_the_card_to_64_mib(monkeypatch):
    m = _mutant("phys = max(0, int(phys_raw) - int(law))", "phys = int(phys_raw)")
    path, d, p = _card(M.d_comm)
    _checks(monkeypatch, d, M.phys_0839, _floor(M.law_d), fn=m)
    d.release(M.shrink)
    assert _p_grant(p), "the mutant must reproduce the metal grant"
    assert abs(_after_grant_phys(M.phys_0839) - M.bottom) <= 2 * MIB


def test_m2_haenger_double_price_is_caught_by_the_exact_booking(monkeypatch):
    m = _mutant("phys = max(0, int(phys_raw) - int(law))", "phys = max(0, int(phys_raw) - 2 * int(law))")
    path, d, _p = _card(M.d_comm)
    _checks(monkeypatch, d, M.phys_0839, _floor(M.law_d), fn=m)
    booked = M.have + M.d_comm - K.peek(path).budget
    assert booked > 0, "the mutant ran"
    assert booked != (M.law_d << 20) - (M.phys_0839 - M.have), "over-pricing must turn the exact-booking test red"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
