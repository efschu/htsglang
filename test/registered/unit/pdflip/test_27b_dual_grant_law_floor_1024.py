# SPDX-License-Identifier: Apache-2.0
"""DUAL-GRANT-1024 (Auftrag #223): the dual card KV ledger committed the 5090 to 62 MiB.

Metal: 27B NVFP4 dual, rc41 1ecda02a58,
boot_weg2_dkr27bnvfp4dualbar1drei10241158_1ecda02a58_1024_115913 (P and D share
cards 0,1,2 under MPS). Card 0 (the 5090, ledger card=e773fd938f6d):

  12:08:38 D GROW 65536 -> 212992, ledger have 2214592512 B, card free 2128 MiB
  12:08:43 D SHRINK 212992 -> 204800 (-67108864 B), PP0 GRANT pdflip-0-20
           committed=2231369728 B (the atomic group grant)
  12:08:44 PP0 chunk: card_free 64 -> 62 MiB, CORRIDOR LAW BREACHED, ooms=0,
           both processes' torch reserved FLAT.

Not the #213 VramCredit class (no wake, no credit): ledger free and card free
agree to 16 MiB -- the ledger promised ALL of the card. The gap phys_free -
ledger_free fell over the boot (9938 -> 0 MiB) while the allocator caches
grew; the old LEDGER-PHYS self-check only booked a NEGATIVE gap.

Fix (after review dual-grant-1024): ``phys_check`` measures the gap against
the CARD'S law = max over both groups of the transient each MEASURED in this
boot, in this form (PDFLIP-VRAM-PEAK chunk/round windows: P 273, D 380 MiB on
the 5090) -- never the flip form's 4096-row seeds (1055/767). The law part is
logged as LAW-BOOKED, apart from the card-use BOOKED line budget_rest reads.

DANGER DIRECTIONS, each with a mutant built from the shipped source:
* OOM: without the law term the 12:08:43 grant takes the card to 64 MiB (M1).
* Haenger: exact price law - gap, once (M2 double price); the full-context
  grant stays feasible with the boot's law (review pin); a law that makes it
  infeasible even with D at zero is said once, loudly.
* card-uniform law: one card, one price, whichever group ticked last (M3).
* record coupling: the law part never reads as 'unbooked card use' (M4).
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

from flliper.srt.model_executor import vram_peak_window as VPW
from flliper.srt.pdflip import budget_rest as BR
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
        # the boot's OWN measured transients on card 0 (the law's source)
        self.tr_p_0844 = int(_one(r"12:08:44 PP0\] PDFLIP-VRAM-PEAK rank=0 phase=chunk rows=512 .*transient_mib=(\d+) ")
                             .group(1)) * MIB
        self.tr_d = int(_one(r"12:06:49 TP0\] PDFLIP-VRAM-PEAK rank=0 phase=round .*transient_mib=(\d+) ")
                        .group(1)) * MIB
        self.tr_p = int(_one(r"12:19:50 PP0\] PDFLIP-VRAM-PEAK rank=0 phase=chunk .*transient_mib=(\d+) ")
                        .group(1)) * MIB
        # the flip-form seed the first fix priced (CORRIDOR LAW BREACHED lines)
        self.seed_p = int(_one(r"12:08:46 PP0\] CORRIDOR LAW BREACHED .*BELOW the (\d+) MiB law").group(1)) * MIB
        self.seed_d = int(_one(r"12:08:46 TP0\] CORRIDOR LAW BREACHED .*BELOW the (\d+) MiB law").group(1)) * MIB
        m = _one(r"12:20:27 PP0\] #1530 GRANT-SHORT .*cards=\[0:need=\d+,have=(\d+),P=(\d+),D=(\d+)")
        self.have_1220, self.p_1220, self.d_1220 = (int(m.group(i)) for i in range(1, 4))
        self.phys_1220 = int(_one(r"12:20:27 PP0\] PDFLIP-VRAM-PEAK .*card_free_mib=(\d+) ").group(1)) * MIB
        self.booked_before_1220 = sum(int(m.group(1)) for m in (
            re.search(r"(?:12:10:35 TP0|12:19:10 PP0|12:20:12 PP0)\] .*LEDGER-PHYS BOOKED (\d+) B", l)
            for l in _lines()) if m)
        self.bound_tp1 = [l for l in _lines() if "TP1] #1028c BOUND alloc_cache" in l]


M = Metal()


def _card(d_committed, p_committed=0):
    """Card 0's ledger as the boot built it: D's boot KV kept mapped until P joined, P's boot KV, D's
    allocator-cache record booked at P's join; then the commitments of the line."""
    path = os.path.join(tempfile.mkdtemp(prefix="wkvdg"), "card")
    d = K.CardKvLedger(path, "D")
    p = K.CardKvLedger(path, "P")
    d.contribute(M.d_contrib, committed=M.d_contrib)
    p.contribute(M.p_contrib, committed=0)          # one live pid for both groups (a dead pid is reaped)
    assert p.book_unpriced(M.book) == M.book
    d.release(M.d_contrib - int(d_committed))
    if p_committed:
        assert p.request(int(p_committed))[0] == int(p_committed)
    return path, d, p


def _checks(monkeypatch, led, phys_bytes, own, fn=None, n=S.PHYS_BOOK_AFTER, actor=None, lines=None):
    """``n`` throttled checks of ``led``'s group with this process's measured transient ``own`` (bytes)."""
    actor = actor or types.SimpleNamespace(ledger=led)
    clock = [0.0]
    fn = fn or S.phys_check
    phys = phys_bytes if callable(phys_bytes) else (lambda: int(phys_bytes))
    mine = (lambda: (int(own), "round" if led.group == "D" else "chunk"))
    # a mutant runs in its own copy of the module namespace: patch every namespace in play
    for g in {id(x): x for x in (fn.__globals__, vars(S), S.card_law.__globals__)}.values():
        monkeypatch.setitem(g, "_now", lambda: clock[0])
        monkeypatch.setitem(g, "phys_free_bytes", phys)
        monkeypatch.setitem(g, "_own_transient", mine)
    if lines is not None:
        monkeypatch.setattr(S.logger, "warning", lambda m, *a: lines.append(m % a))
        monkeypatch.setattr(S.logger, "error", lambda m, *a: lines.append(m % a))
        monkeypatch.setattr(S.logger, "info", lambda m, *a: lines.append(m % a))
    for _ in range(n):
        fn(actor, led.group)
        clock[0] += S.PHYS_CHECK_S


def _mutant(fn, fixed, back):
    src = textwrap.dedent(inspect.getsource(inspect.unwrap(fn)))
    assert src.count(fixed) == 1, "the guarded line moved -- re-aim the mutant"
    ns = dict(vars(S))
    exec(compile(src.replace(fixed, back), S.__file__, "exec"), ns)
    return ns[fn.__name__]


def _p_grant(p):
    got, _ = p.request(M.need)
    if got < M.need:
        if got:
            p.release(got)
        return False
    return True


def _after_grant_phys(phys_before):
    return phys_before + M.shrink - M.granted


LAW = max(M.tr_p_0844, M.tr_d)        # card 0's law at 12:08 (D 380 round, P 256 chunk)


# -- the replay ---------------------------------------------------------------------------


def test_the_metal_ledger_is_rebuilt_exactly_from_the_lines():
    path, _d, _p = _card(M.d_comm)
    st = K.peek(path)
    assert st.budget == M.have + M.d_comm
    assert st.free == M.have
    assert M.granted == M.need


def test_the_law_is_the_boots_own_measurement_not_the_flip_seed():
    assert M.tr_d == 380 * MIB and M.tr_p == 273 * MIB
    assert M.seed_p == 1055 * MIB and M.seed_d == 767 * MIB
    assert max(M.tr_p, M.tr_d) < min(M.seed_p, M.seed_d), "the flip seeds overprice this form"


def test_unmeasured_card_repro_and_one_loud_line(monkeypatch):
    """REPRO: no window measured yet (law 0) = the rc41 rule: the grant goes through and the card lands at the
    line's 64 MiB -- and the card says once, loudly, that it can be committed to 0."""
    path, d, p = _card(M.d_comm)
    lines = []
    _checks(monkeypatch, d, M.phys_0839, 0, n=3 * S.PHYS_BOOK_AFTER, lines=lines)
    assert K.peek(path).free == M.have
    assert sum(1 for l in lines if S.LAW_ZERO_MARK in l) == 1, lines[:3]
    d.release(M.shrink)
    assert _p_grant(p)
    assert abs(_after_grant_phys(M.phys_0839) - M.bottom) <= 2 * MIB, "reconstructs card_free_mib=64"


def test_the_card_law_is_priced_and_the_grant_waits(monkeypatch):
    path, d, p = _card(M.d_comm)
    _checks(monkeypatch, p, M.phys_0839, M.tr_p_0844, n=1)          # P's measurement reaches the sidecar
    _checks(monkeypatch, d, M.phys_0839, M.tr_d)
    assert K.peek(path).free == M.phys_0839 - LAW, "ledger free = card free above the card's law"
    d.release(M.shrink)
    assert not _p_grant(p), "P waits (it never presses D) instead of taking the card under its law"


def test_once_d_falls_the_grant_goes_and_the_card_stays_at_its_law(monkeypatch):
    path, d, p = _card(M.d_comm)
    _checks(monkeypatch, d, M.phys_0839, M.tr_d)
    d.release(M.grow_0838)                                       # D back at 65536 tokens (12:08:34)
    assert _p_grant(p)
    assert M.phys_0839 + M.grow_0838 - M.granted >= LAW


def test_steady_state_books_nothing_more(monkeypatch):
    path, d, _p = _card(M.d_comm)
    _checks(monkeypatch, d, M.phys_0839, M.tr_d)
    budget = K.peek(path).budget
    assert M.have + M.d_comm - budget == M.tr_d - (M.phys_0839 - M.have), "exactly law - gap"
    _checks(monkeypatch, d, M.phys_0839, M.tr_d, n=10 * S.PHYS_BOOK_AFTER)
    assert K.peek(path).budget == budget


def test_the_1220_gap_of_zero_is_priced_at_the_card_law(monkeypatch):
    path, d, p = _card(M.d_1220, p_committed=M.p_1220)
    assert p.book_unpriced(M.booked_before_1220) == M.booked_before_1220
    assert K.peek(path).free == M.have_1220
    assert (M.have_1220 >> 20) == (M.phys_1220 >> 20), "gap 0 on the line (whole-MiB reading)"
    _checks(monkeypatch, d, M.have_1220, M.tr_d, n=1)
    _checks(monkeypatch, p, M.have_1220, M.tr_p)
    assert K.peek(path).free == M.have_1220 - max(M.tr_p, M.tr_d)


def test_a_card_already_under_its_law_never_breaks_i1(monkeypatch):
    path, d, p = _card(M.d_comm)
    d.release(M.shrink)
    assert _p_grant(p)                                           # 12:08:43: ledger free 48 MiB
    _checks(monkeypatch, d, M.bottom, M.tr_d)
    st = K.peek(path)
    assert sum(st.committed.values()) <= st.budget and st.free == 0
    p.release(M.granted)
    d.request(1 * MIB)


def test_one_card_one_law_whichever_group_ticked_last(monkeypatch):
    out = []
    for order in (("D",), ("D", "P"), ("P", "D")):
        path, d, p = _card(M.d_comm)
        _checks(monkeypatch, p, M.phys_0839, M.tr_p, n=1)        # both groups measured, nothing booked yet
        _checks(monkeypatch, d, M.phys_0839, M.tr_d, n=1)
        for g in order:
            led, own = (d, M.tr_d) if g == "D" else (p, M.tr_p)
            _checks(monkeypatch, led, M.phys_0839, own)
        out.append(K.peek(path).budget)
    assert len(set(out)) == 1, out


def test_a_release_between_the_readings_is_no_phantom_over_promise(monkeypatch):
    """SOLL-2: decommit -> (phys read 1) ... ledger.release -> peek: the second reading closes it."""
    path, d, _p = _card(M.d_comm)
    st = {"n": 0}
    rel = 1879048192

    def phys():
        st["n"] += 1
        if st["n"] % 2 == 1:                                     # the first reading of a check: before the unmap
            return M.phys_0839
        return M.phys_0839 + rel                                 # after it
    d.release(rel)                                               # the ledger already shows the release at peek
    before = K.peek(path).budget
    _checks(monkeypatch, d, phys, 0)                             # law 0: only the race could book
    assert K.peek(path).budget == before


def test_law_booking_is_logged_apart_and_the_3080_record_stays_0(monkeypatch):
    """MUSS-3: budget_rest's ALLOC-CACHE record counts 'LEDGER-PHYS BOOKED' as card use. A pure law
    booking must not switch a 3080 from 0 to the cache growth."""
    path, d, _p = _card(M.d_comm)
    lines = []
    _checks(monkeypatch, d, M.phys_0839, M.tr_d, lines=lines)
    booked = [l for l in lines if "BOOKED" in l]
    assert booked and all("LAW-BOOKED" in l for l in booked), booked
    dlog = "\n".join(M.bound_tp1 + ["[2026-10-24 12:10:35 TP1] " + l for l in booked])
    out, _why = BR.alloc_cache_book_from_boots([("dualbar1drei", dlog)], 3)
    assert out[1] == 0, (out, _why)
    # control: a real card-use booking does switch it (the test discriminates)
    real = dlog + "\n[2026-10-24 12:10:36 TP1] DUAL-TP3PP3 P-KV LEDGER-PHYS BOOKED 1 B of unbooked card use"
    assert BR.alloc_cache_book_from_boots([("dualbar1drei", real)], 3)[0][1] > 0


class _PActor:
    """A P stage's byte table on card 0 (14336 B/token, 58720256 B per 4096-row step), top 266240."""

    def __init__(self, led):
        self.ledger, self.top, self.step = led, 266240, 4096

    def bytes_for(self, tokens):
        return (int(tokens) // self.step) * 58720256


def test_a_law_that_makes_the_top_grant_infeasible_is_said_once(monkeypatch):
    """MUSS-2: with the flip seed (1055) priced on the 12:41:55 card the top grant is infeasible even with
    D at zero -- said once with the numbers; with the boot's own law (380) it is not said."""
    end = _one(r"12:41:55 PP0\] .*LEDGER-PHYS ok .*phys_free=(\d+) budget=(\d+) committed=\{'P': 0, 'D': (\d+)\}")
    phys_end, budget, d_end = (int(end.group(i)) for i in range(1, 4))
    d_idle = int(_one(r"12:02:03 PP0\] .*LEDGER-PHYS ok .*'D': (\d+)\}").group(1))
    for own, said in ((M.seed_p, 1), (M.tr_p, 0)):
        path = os.path.join(tempfile.mkdtemp(prefix="wkvdgi"), "card")
        dl, pl = K.CardKvLedger(path, "D"), K.CardKvLedger(path, "P")
        dl.contribute(budget, committed=d_idle)
        pl.contribute(0, committed=0)
        lines = []
        _checks(monkeypatch, pl, phys_end + (d_end - d_idle), own, n=6 * S.PHYS_BOOK_AFTER,
                actor=_PActor(pl), lines=lines)
        hit = [l for l in lines if S.LAW_INFEASIBLE_MARK in l]
        assert len(hit) == said, hit


def _rank_max(tag):
    """The largest chunk/round transient_mib of rank ``tag`` (e.g. 'TP1') in the fixture, bytes."""
    vals = [int(m.group(1)) for m in (re.search(r"%s\] PDFLIP-VRAM-PEAK .*phase=(?:chunk|round).* transient_mib=(\d+) "
                                                % tag, l) for l in _lines()) if m]
    assert vals, tag
    return max(vals) * MIB


#: card ordinal -> (P rank, D rank) on it (PP-k and TP-k share card k in the dual layout)
CARDS = {0: ("PP0", "TP0"), 1: ("PP1", "TP1"), 2: ("PP2", "TP2")}


def _three_card_laws(monkeypatch):
    """Each card's law from ITS ranks' measured transients, the device keyed like the metal ranks."""
    monkeypatch.setitem(VPW._STATE, "transient_by_dev", {})
    out = {}
    for dev, (pr, dr) in CARDS.items():
        path = os.path.join(tempfile.mkdtemp(prefix="wkv3c"), "card")
        for grp, rank in (("P", pr), ("D", dr)):
            VPW.note_transient(dev, _rank_max(rank), "chunk" if grp == "P" else "round")
            monkeypatch.setattr(VPW, "_current_device", lambda cuda=None, d=dev: d)
            out.setdefault(dev, []).append(S.card_law(types.SimpleNamespace(path=path), grp))
    return {dev: v[-1] for dev, v in out.items()}


def test_measured_transient_is_per_device():
    """Qwen #2: the measuring seam itself -- one process, three devices, three values."""
    saved = dict(VPW._STATE["transient_by_dev"])
    try:
        VPW._STATE["transient_by_dev"].clear()
        VPW.note_transient(0, 380 * MIB, "round")
        VPW.note_transient(1, 324 * MIB, "round")
        VPW.note_transient(1, 100 * MIB, "chunk")                 # a smaller window never lowers it
        assert VPW.measured_transient(0) == (380 * MIB, "round")
        assert VPW.measured_transient(1) == (324 * MIB, "round")
        assert VPW.measured_transient(2) == (0, "")
    finally:
        VPW._STATE["transient_by_dev"].clear()
        VPW._STATE["transient_by_dev"].update(saved)


def test_three_cards_three_laws_each_its_own_max(monkeypatch):
    """Qwen #1: one card, one law -- its OWN max(P, D): 380 / 324 / 324 MiB, never the 5090's 380 on a 3080."""
    laws = _three_card_laws(monkeypatch)
    want = {dev: max(_rank_max(pr), _rank_max(dr)) for dev, (pr, dr) in CARDS.items()}
    assert want == {0: 380 * MIB, 1: 324 * MIB, 2: 324 * MIB}
    assert {dev: v[0] for dev, v in laws.items()} == want
    assert "380MiB" not in laws[1][1] and "380MiB" not in laws[2][1], "law_src names the card's own value"


def test_full_context_need_is_one_request_not_the_stage_top(monkeypatch):
    """(3): P's top need is min(stage top 266240, --max-kv-per-request 262144) = 64 steps, not 65."""
    import flliper.srt.server_args as SA

    monkeypatch.setattr(SA, "get_global_server_args",
                        lambda: types.SimpleNamespace(max_kv_per_request=262144, context_length=262144))
    assert S.full_context_tokens(266240) == 262144
    path = os.path.join(tempfile.mkdtemp(prefix="wkvfc"), "card")
    pl = K.CardKvLedger(path, "P")
    assert S._p_top_need(_PActor(pl), pl) == 64 * 58720256 == 3758096384
    monkeypatch.setattr(SA, "get_global_server_args", lambda: (_ for _ in ()).throw(RuntimeError("no ctx")))
    assert S.full_context_tokens(266240) == 266240, "no server args: the stage top (the stricter level)"


def test_booked_line_names_the_law_too(monkeypatch):
    path, d, _p = _card(M.d_comm)
    lines = []
    _checks(monkeypatch, d, M.have - 100 * MIB, M.tr_d, lines=lines)   # a real card-use gap AND the law
    booked = [l for l in lines if "LEDGER-PHYS BOOKED" in l]
    assert booked and all("law_src=boot-peak" in l for l in booked), booked
    assert BR._PHYS_BOOKED.search("[t TP1] " + booked[0]), "budget_rest still reads the card-use line"


# -- mutants in the danger directions ------------------------------------------------------


def test_m5_process_wide_max_turns_the_three_card_test_red(monkeypatch):
    src = textwrap.dedent(inspect.getsource(VPW.measured_transient))
    fixed = 'cur = _STATE["transient_by_dev"].get(d)'
    assert src.count(fixed) == 1, "the guarded line moved -- re-aim the mutant"
    ns = dict(vars(VPW))
    exec(compile(src.replace(fixed, 'cur = max(_STATE["transient_by_dev"].values(), default=None)'),
                 VPW.__file__, "exec"), ns)
    monkeypatch.setattr(VPW, "measured_transient", ns["measured_transient"])
    laws = _three_card_laws(monkeypatch)
    assert laws[1][0] == 380 * MIB, "the mutant puts the 5090's law on the 3080"


def test_m1_oom_without_the_law_term_the_grant_takes_the_card_to_64_mib(monkeypatch):
    m = _mutant(S.phys_check, "phys = max(0, int(phys_raw) - int(law))", "phys = int(phys_raw)")
    path, d, p = _card(M.d_comm)
    _checks(monkeypatch, d, M.phys_0839, M.tr_d, fn=m)
    d.release(M.shrink)
    assert _p_grant(p), "the mutant must reproduce the metal grant"
    assert abs(_after_grant_phys(M.phys_0839) - M.bottom) <= 2 * MIB


def test_m2_haenger_double_price_is_caught_by_the_exact_booking(monkeypatch):
    m = _mutant(S.phys_check, "phys = max(0, int(phys_raw) - int(law))", "phys = max(0, int(phys_raw) - 2 * int(law))")
    path, d, _p = _card(M.d_comm)
    _checks(monkeypatch, d, M.phys_0839, M.tr_d, fn=m)
    booked = M.have + M.d_comm - K.peek(path).budget
    assert booked > 0, "the mutant ran"
    assert booked != M.tr_d - (M.phys_0839 - M.have)


def test_m3_own_group_only_law_turns_the_card_uniform_test_red(monkeypatch):
    m = _mutant(S.card_law, "best = max(best, b)", "best = max(best, b) if g == group else best")
    monkeypatch.setattr(S, "card_law", m)
    out = []
    for order in (("D",), ("D", "P")):
        path, d, p = _card(M.d_comm)
        _checks(monkeypatch, d, M.phys_0839, M.tr_d, n=1)
        _checks(monkeypatch, p, M.phys_0839, M.tr_p + 200 * MIB, n=1)   # P's measurement above D's
        for g in order:
            led, own = (d, M.tr_d) if g == "D" else (p, M.tr_p + 200 * MIB)
            _checks(monkeypatch, led, M.phys_0839, own)
        out.append(K.peek(path).budget)
    assert len(set(out)) == 2, "the mutant must make the price depend on the tick order"


def test_m4_law_booked_as_card_use_switches_the_3080_record(monkeypatch):
    m = _mutant(S.phys_check, "use = min(int(n), max(0, int(st.free) - int(phys_raw)))", "use = int(n)")
    path, d, _p = _card(M.d_comm)
    lines = []
    _checks(monkeypatch, d, M.phys_0839, M.tr_d, fn=m, lines=lines)
    booked = [l for l in lines if "BOOKED" in l]
    dlog = "\n".join(M.bound_tp1 + ["[2026-10-24 12:10:35 TP1] " + l for l in booked])
    assert BR.alloc_cache_book_from_boots([("dualbar1drei", dlog)], 3)[0][1] > 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
