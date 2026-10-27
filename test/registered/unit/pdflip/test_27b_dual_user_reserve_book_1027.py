# SPDX-License-Identifier: Apache-2.0
"""PDFLIP-USER-RESERVE-BOOK (user rule 24.10.): the user's reserve is the lower bound of the card's free memory
in the dual layout -- booked out of the card ledger budget K_c, per card.

Metal: 27B NVFP4 dual, dualm (marlin_native_inplace), formb99w, 27b789dd0b,
boot_weg2_dkr27bnvfp4dualbar1drei10271355_27b789dd0b_1027_135516.P.log (cards 0/1/2 = PP0/PP1/PP2):

  * the ledger of card c: D contributes 4672 / 3712 / 3520 MiB (P:529-545 'D-KV JOIN'), P 952 / 680 / 544 MiB,
    ALLOC-CACHE-BOOK takes 952 / 0 / 0 (the 5090's record asks 1318, only P's 952 are free at the join);
    K_c = 4672 / 4392 / 4064 MiB;
  * beside the ledger the card kept 1621 / 787 / 749 MiB at the quiet instant (14:05:48) and 1009 / 128 / 32 MiB
    at the tightest (14:07:39, phys_free - ledger_free of the 'P-KV LEDGER-PHYS' lines),
    against the user's reserve of 1800 / 1400 / 1400 MiB (launcher --user-reserve-mib);
  * nobody booked the reserve: the D budget replaces it by the awake-rest record (launcher
    effective_reserve_by_card: the D ranks read 0), the ledger prices only the measured transient (card_law).

DANGER DIRECTIONS, each pinned:
* the rule is kept: a fully committed pool leaves >= the reserve free, per card, exactly (booked = reserve - outside);
* reserve 0 / a card that already keeps the reserve free: the ledger file is byte-identical (nothing touched);
* the deduction never goes below what is committed (I1) -- the rest is OWED and booked when the pool has the bytes
  (D holds the whole pool at the join), once, exactly;
* no collapse: the booking is made ONCE from the record, a foreign consumer that takes memory afterwards (a game)
  moves the budget nowhere (the law-style runtime floor would ratchet it down for good);
* the stated reserve is read for the card whichever group asks (D ranks carry 0 + the masked stated value) and
  keys are compared by ``card_key`` (NVML 'GPU-...' vs torch uuid);
* no record / switch off: nothing booked, said loudly by name (the rule is then NOT kept on that boot);
* the P-262k context still fits every card after the booking (margins in the docstring of ``test_the_262k_margin``).
"""

from __future__ import annotations

import json
import os
import tempfile
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest

from flliper.srt.managers import corridor_guard as CG
from flliper.srt.pdflip import card_kv_ledger as K
from flliper.srt.pdflip import dual_p_kv_stage as S
from flliper.srt.pdflip import form as F
from flliper.srt.pdflip import inventory_view as IV
from flliper.srt.pdflip import launcher as L
from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

MIB = 1 << 20

#: metal boot 10271355, per card ordinal 0/1/2 = 5090 / 3080a / 3080b (MiB)
D_CONTRIB = (4672, 3712, 3520)
P_CONTRIB = (952, 680, 544)
ALLOC_BOOK = (952, 0, 0)                      # applied: min(record 1318/0/0, ledger free at the join)
K_C = (4672, 4392, 4064)                      # the LEDGER-PHYS budget= of the boot
RESERVE = (1800, 1400, 1400)                  # --user-reserve-mib
OUTSIDE = (1009, 128, 32)                     # D_DUAL_OUTSIDE_FREE_MIB: the tightest instant
DEFICIT = (791, 1272, 1368)                   # reserve - outside
P_NEED_262K = (3640, 2600, 2080)              # FA layers 7/5/4 x 520 MiB (dual-p-schnitt-rechnung-1027 §1)
UUID_NVML = ("GPU-31d7ef41-f574-4d0e-21ad-e773fd938f6d", "GPU-62dbbae1-e859-9ccc-f9c2-d9f2443a84f4",
             "GPU-5c648f96-be1d-42d5-0221-34d11ab137f7")
UUID_TORCH = tuple(u[len("GPU-"):] for u in UUID_NVML)


def _env(reserve=RESERVE, outside=OUTSIDE, masked=False):
    """The P rank's env (stated reserve in the published mapping) or a D rank's (0 + the masked stated value)."""
    stated = {u: int(v) for u, v in zip(UUID_NVML, reserve)}
    env = {}
    if masked:
        env[CG.USER_RESERVE_ENV] = json.dumps({u: 0 for u in stated})
        env[CG.USER_RESERVE_MASKED_ENV] = json.dumps(stated)
    else:
        env[CG.USER_RESERVE_ENV] = json.dumps(stated)
    if outside is not None:
        env[S.OUTSIDE_FREE_ENV] = ",".join(str(int(v)) for v in outside)
    return env


def _card(i, d_committed=64):
    """Card ``i`` as the boot built it up to P's join: D's boot KV kept mapped, P's boot KV, the alloc-cache record
    booked; then D's commitment (``d_committed`` MiB; None = D still holds its whole boot pool)."""
    path = os.path.join(tempfile.mkdtemp(prefix="wkvur"), "card")
    d = K.CardKvLedger(path, "D")
    p = K.CardKvLedger(path, "P")
    d.contribute(D_CONTRIB[i] * MIB, committed=D_CONTRIB[i] * MIB)
    p.contribute(P_CONTRIB[i] * MIB, committed=0)
    assert p.book_unpriced(ALLOC_BOOK[i] * MIB) == ALLOC_BOOK[i] * MIB
    assert p.state().budget == K_C[i] * MIB
    if d_committed is not None:
        d.release((D_CONTRIB[i] - int(d_committed)) * MIB)
    return path, d, p


def _bytes(path):
    with open(path, "rb") as f:
        return f.read()


@pytest.fixture
def lines(monkeypatch):
    out = []
    for lvl in ("info", "warning", "error"):
        monkeypatch.setattr(S.logger, lvl, lambda m, *a, _l=lvl: out.append("%s: %s" % (_l.upper(), m % a)))
    return out


# -- the rule is kept ----------------------------------------------------------------------------------


@pytest.mark.parametrize("i", [0, 1, 2])
def test_reserve_set_the_budget_falls_by_the_deficit(i, lines):
    """REPRO of the metal: without the booking a fully committed pool leaves OUTSIDE free (< the reserve). With
    it K_c is smaller by reserve - outside, and a full commit leaves exactly the reserve."""
    path, _d, p = _card(i)
    reserve, booked, owed = S.book_user_reserve(p, i, UUID_TORCH[i], _env())
    assert reserve == RESERVE[i]
    assert (booked, owed) == (DEFICIT[i] * MIB, 0)
    assert K.peek(path).budget == (K_C[i] - DEFICIT[i]) * MIB
    # fully committed pool: what the card keeps free beside the ledger + what the booking left uncommitted
    assert OUTSIDE[i] + DEFICIT[i] == RESERVE[i], "a full commit leaves exactly the reserve free"
    assert any("reserve=%d MiB outside_free=%d MiB" % (RESERVE[i], OUTSIDE[i]) in l for l in lines), lines


def test_the_log_line_names_reserve_deduction_and_the_budget_before_and_after(lines):
    _path, _d, p = _card(0)
    S.book_user_reserve(p, 0, UUID_TORCH[0], _env())
    ln = [l for l in lines if S.RESERVE_BOOK_MARK in l]
    assert len(ln) == 1, ln
    for want in ("card=%s" % UUID_TORCH[0][-12:], "reserve=1800 MiB", "outside_free=1009 MiB",
                 "deduction %d B (791 MiB)" % (791 * MIB), "booked %d B, owed 0 B" % (791 * MIB),
                 "K_c %d -> %d B (4672 -> 3881 MiB)" % (4672 * MIB, 3881 * MIB)):
        assert want in ln[0], (want, ln[0])


def test_cards_differ_each_is_booked_on_its_own_numbers(lines):
    """Three cards, three reserves, three outside values: each budget moves by its own deficit; a card whose outside
    free already covers its reserve moves by nothing."""
    reserve, outside = (1800, 1400, 1400), (1009, 2000, 32)       # the middle card keeps 2000 free by itself
    want = (791, 0, 1368)
    for i in range(3):
        path, _d, p = _card(i)
        _res, booked, owed = S.book_user_reserve(p, i, UUID_TORCH[i], _env(reserve, outside))
        assert (booked, owed) == (want[i] * MIB, 0), i
        assert K.peek(path).budget == (K_C[i] - want[i]) * MIB, i


def test_the_shipped_record_keeps_p_s_262k_context_on_every_card():
    """The booking comes from the SHIPPED record (qwen27b.json, D_DUAL_OUTSIDE_FREE_MIB): with the user's reserve it
    must leave P's full 262144-token context (FA layers 7/5/4 x 520 MiB) a pool on every card, D empty.
    BERECHNET from the metal K_c: 5090 +241, 3080a +520, 3080b +616 MiB (before the booking +1032 / +1792 /
    +1984). A record edit that tightens it past a card's margin turns this red -- by name, before a boot does."""
    from flliper.srt.pdflip import profile_records as PR

    rec = PR.select("qwen27b", "D_DUAL_OUTSIDE_FREE_MIB", fmt="nvfp4").record
    assert tuple(rec.value) == OUTSIDE
    margins = [K_C[i] - S.reserve_deficit_mib(RESERVE[i], rec.value[i]) - P_NEED_262K[i] for i in range(3)]
    assert margins == [241, 520, 616]
    assert all(m > 0 for m in margins)


# -- nothing changes where nothing is owed ------------------------------------------------------------


def test_reserve_zero_leaves_the_ledger_byte_identical(lines):
    path, _d, p = _card(0)
    before = _bytes(path)
    assert S.book_user_reserve(p, 0, UUID_TORCH[0], {}) == (0, 0, 0)
    assert S.book_user_reserve(p, 0, UUID_TORCH[0], {S.OUTSIDE_FREE_ENV: "1009,128,32"}) == (0, 0, 0)  # no reserve stated
    assert _bytes(path) == before
    assert any("reserve=0 MiB" in l and "nothing booked" in l for l in lines), lines


def test_a_card_that_keeps_the_reserve_free_by_itself_is_untouched(lines):
    path, _d, p = _card(0)
    before = _bytes(path)
    assert S.book_user_reserve(p, 0, UUID_TORCH[0], _env(outside=(1800, 1400, 1400)))[1:] == (0, 0)
    assert _bytes(path) == before


def test_without_the_record_nothing_is_booked_and_the_log_says_so(lines):
    path, _d, p = _card(0)
    before = _bytes(path)
    assert S.book_user_reserve(p, 0, UUID_TORCH[0], _env(outside=None)) == (1800, 0, 0)
    assert _bytes(path) == before
    assert any(l.startswith("WARNING") and "UNMEASURED" in l and "NOT booked" in l for l in lines), lines


def test_the_off_switch_books_nothing_and_says_so(lines):
    path, _d, p = _card(0)
    before = _bytes(path)
    env = _env()
    env[S.RESERVE_BOOK_SWITCH_ENV] = "0"
    assert S.book_user_reserve(p, 0, UUID_TORCH[0], env) == (1800, 0, 0)
    assert _bytes(path) == before
    assert any(l.startswith("WARNING") and S.RESERVE_BOOK_SWITCH_ENV in l and "NOT booked" in l for l in lines), lines


# -- the stated reserve, whichever group asks ---------------------------------------------------------


def test_a_d_rank_reads_the_stated_reserve_through_the_masked_mapping():
    """The launcher publishes 0 to the D ranks where D's awake rest is booked and names the stated value in the
    masked mapping (USER_RESERVE_MASKED_ENV): the card ledger is the card's, so the stated one counts."""
    for i in range(3):
        assert S.user_reserve_mib(UUID_TORCH[i], _env()) == RESERVE[i]
        assert S.user_reserve_mib(UUID_TORCH[i], _env(masked=True)) == RESERVE[i]
        assert S.user_reserve_mib(UUID_NVML[i], _env()) == RESERVE[i], "either spelling of the uuid"
    assert S.user_reserve_mib("GPU-other", _env()) == 0
    assert S.user_reserve_mib(UUID_TORCH[0], {CG.USER_RESERVE_ENV: "not json"}) == 0


# -- what the join cannot take is owed, then booked exactly --------------------------------------------


def test_owed_is_booked_exactly_once_when_d_releases_its_boot_pool(lines):
    """The 5090 at the join: D holds its whole boot pool, the alloc-cache booking took P's 952 MiB, the ledger has
    nothing free -- the 791 MiB are OWED, and booked in full when D shrinks (the first P idle pass after the join)."""
    path, d, p = _card(0, d_committed=None)
    assert p.state().free == 0
    reserve, booked, owed = S.book_user_reserve(p, 0, UUID_TORCH[0], _env())
    assert (reserve, booked, owed) == (1800, 0, DEFICIT[0] * MIB)
    assert K.peek(path).budget == K_C[0] * MIB, "I1: nothing below what is committed"
    actor = types.SimpleNamespace(ledger=p, _user_reserve_mib=reserve, _reserve_owed=owed)
    assert S.settle_reserve_owed(actor) == 0 and actor._reserve_owed == owed       # D still holds: waits, no change
    d.release((D_CONTRIB[0] - 64) * MIB)                                           # D unmaps its boot pool
    assert S.settle_reserve_owed(actor) == DEFICIT[0] * MIB
    assert actor._reserve_owed == 0
    assert K.peek(path).budget == (K_C[0] - DEFICIT[0]) * MIB
    assert S.settle_reserve_owed(actor) == 0                                       # settled once, never twice
    assert K.peek(path).budget == (K_C[0] - DEFICIT[0]) * MIB


def test_a_partial_booking_keeps_the_rest_owed(lines):
    """Only part of the deduction fits (D committed 3700 of 4672 on the 5090, 972 free): book those, owe the rest."""
    path, d, p = _card(0, d_committed=3900)
    free = p.state().free // MIB
    assert free == 4672 - 3900
    _reserve, booked, owed = S.book_user_reserve(p, 0, UUID_TORCH[0], _env())
    assert booked == min(free, DEFICIT[0]) * MIB
    assert booked + owed == DEFICIT[0] * MIB
    assert K.peek(path).budget == K_C[0] * MIB - booked


def test_a_foreign_consumer_afterwards_moves_nothing(monkeypatch, lines):
    """No collapse: the booking is made once from the record. A game that takes the card's free memory later
    (phys_free falls to 0 beside the ledger) does not touch the budget -- phys_check books nothing on its account
    for the reserve (the law/over-promise check is the system's own and stands)."""
    path, _d, p = _card(1)
    reserve, _booked, owed = S.book_user_reserve(p, 1, UUID_TORCH[1], _env())
    budget = K.peek(path).budget
    actor = types.SimpleNamespace(ledger=p, _user_reserve_mib=reserve, _reserve_owed=owed)
    free_now = K.peek(path).free
    # the card keeps free_now beside nothing else: a foreign process took everything beside the pool
    clock = [0.0]
    monkeypatch.setattr(S, "_now", lambda: clock[0])
    monkeypatch.setattr(S, "phys_free_bytes", lambda: int(free_now))
    monkeypatch.setattr(S, "_own_transient", lambda: (0, ""))
    for _ in range(3 * S.PHYS_BOOK_AFTER):
        S.phys_check(actor, "P")
        clock[0] += S.PHYS_CHECK_S
    assert K.peek(path).budget == budget


# -- the instrument and the hook -----------------------------------------------------------------------


def _checks(monkeypatch, actor, phys_bytes, n=1, step=None):
    clock = [0.0]
    monkeypatch.setattr(S, "_now", lambda: clock[0])
    monkeypatch.setattr(S, "phys_free_bytes", lambda: int(phys_bytes))
    monkeypatch.setattr(S, "_own_transient", lambda: (0, ""))
    for _ in range(n):
        S.phys_check(actor, "P")
        clock[0] += S.PHYS_CHECK_S if step is None else step
    return clock


def test_phys_check_settles_the_owed_booking(monkeypatch, lines):
    path, d, p = _card(0, d_committed=None)
    _reserve, _booked, owed = S.book_user_reserve(p, 0, UUID_TORCH[0], _env())
    actor = types.SimpleNamespace(ledger=p, _user_reserve_mib=1800, _reserve_owed=owed)
    d.release((D_CONTRIB[0] - 64) * MIB)
    free = K.peek(path).free
    _checks(monkeypatch, actor, free + OUTSIDE[0] * MIB + DEFICIT[0] * MIB)   # the card has all of it free
    assert actor._reserve_owed == 0
    assert K.peek(path).budget == (K_C[0] - DEFICIT[0]) * MIB
    assert any("settle: booked" in l and "owed left 0 B" in l for l in lines), lines


def test_a_group_without_a_reserve_actor_is_untouched(monkeypatch, lines):
    """phys_check of D's actor (or any actor that carries no booking state) is byte-for-byte what it was."""
    path, d, _p = _card(0)
    before = _bytes(path)
    actor = types.SimpleNamespace(ledger=d)
    _checks(monkeypatch, actor, K.peek(path).free + 3000 * MIB)
    assert not any(S.RESERVE_BOOK_MARK in l or S.RESERVE_SHORT_MARK in l for l in lines), lines
    # a check only peeks; the budget and the commitments stand
    st = K.peek(path)
    assert st.budget == K_C[0] * MIB and st.committed["D"] == 64 * MIB


def test_the_short_instrument_names_the_tightest_instant_and_is_rate_limited(monkeypatch, lines):
    path, _d, p = _card(2)
    S.book_user_reserve(p, 2, UUID_TORCH[2], _env())
    actor = types.SimpleNamespace(ledger=p, _user_reserve_mib=1400, _reserve_owed=0)
    free = K.peek(path).free
    # the card keeps 32 MiB free beside the (smaller) ledger: the metal's tightest instant
    clock = _checks(monkeypatch, actor, free + 32 * MIB, n=3)
    short = [l for l in lines if S.RESERVE_SHORT_MARK in l]
    assert len(short) == 1, short                      # 3 checks within 30 s: one line
    assert "reserve=1400 MiB outside_free=32 MiB" in short[0] and "short=1368 MiB" in short[0]
    lines.clear()
    clock[0] += S.RESERVE_LOG_S
    monkeypatch.setattr(S, "_now", lambda: clock[0])
    S.phys_check(actor, "P")
    assert sum(1 for l in lines if S.RESERVE_SHORT_MARK in l) == 1, "again after the log interval"
    lines.clear()
    # the card keeps the reserve: no line
    actor2 = types.SimpleNamespace(ledger=p, _user_reserve_mib=1400, _reserve_owed=0)
    _checks(monkeypatch, actor2, free + 1500 * MIB, n=3)
    assert not any(S.RESERVE_SHORT_MARK in l for l in lines), lines


# -- the launcher side ----------------------------------------------------------------------------------


class _Cards:
    @staticmethod
    def cards():
        return [L.Card(nvml_index=0, uuid=UUID_NVML[0], name="NVIDIA GeForce RTX 5090", total_mib=32607),
                L.Card(nvml_index=1, uuid=UUID_NVML[1], name="NVIDIA GeForce RTX 3080", total_mib=20480),
                L.Card(nvml_index=2, uuid=UUID_NVML[2], name="NVIDIA GeForce RTX 3080", total_mib=20480)]


def _model(quant_method):
    d = tempfile.mkdtemp(prefix="m27b")
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump({"architectures": ["Qwen3_5ForConditionalGeneration"],
                   "quantization_config": {"quant_method": quant_method}}, f)
    return d


def _ns(model, **kw):
    base = dict(dual_share=True, dual_unified_kv="on", model=model, profile=F.PROFILE_QWEN27B)
    base.update(kw)
    return types.SimpleNamespace(**base)


STATED = {u: int(v) for u, v in zip(UUID_NVML, RESERVE)}


def test_launcher_hands_p_the_record_and_names_the_booking(monkeypatch):
    monkeypatch.delenv(S.RESERVE_BOOK_SWITCH_ENV, raising=False)
    out = []
    env = L.dual_outside_free_env(_ns(_model("modelopt")), _Cards.cards(), STATED, out.append)
    assert env == {S.OUTSIDE_FREE_ENV: "1009,128,32"}
    assert len(out) == 1 and out[0].startswith("PDFLIP-USER-RESERVE-BOOK RECORD D_DUAL_OUTSIDE_FREE_MIB fmt=nvfp4"), out
    for want in ("ordinal 0 nvml0 reserve 1800 MiB, outside_free 1009 MiB -> book 791 MiB",
                 "ordinal 1 nvml1 reserve 1400 MiB, outside_free 128 MiB -> book 1272 MiB",
                 "ordinal 2 nvml2 reserve 1400 MiB, outside_free 32 MiB -> book 1368 MiB"):
        assert want in out[0], (want, out[0])


def test_launcher_without_a_stated_reserve_or_other_layouts_is_byte_identical(monkeypatch):
    monkeypatch.delenv(S.RESERVE_BOOK_SWITCH_ENV, raising=False)
    out = []
    zero = {u: 0 for u in UUID_NVML}
    assert L.dual_outside_free_env(_ns(_model("modelopt")), _Cards.cards(), zero, out.append) == {}
    assert L.dual_outside_free_env(_ns(_model("modelopt")), _Cards.cards(), None, out.append) == {}
    assert L.dual_outside_free_env(_ns(_model("modelopt"), dual_share=False), _Cards.cards(), STATED, out.append) == {}
    assert L.dual_outside_free_env(_ns(_model("modelopt"), dual_unified_kv="off"), _Cards.cards(), STATED,
                                   out.append) == {}
    assert out == []                                     # no env, no line


def test_launcher_an_unrecorded_form_books_nothing_and_says_so_loudly(monkeypatch):
    monkeypatch.delenv(S.RESERVE_BOOK_SWITCH_ENV, raising=False)
    out = []
    env = L.dual_outside_free_env(_ns(_model("compressed-tensors")), _Cards.cards(), STATED, out.append)
    assert env == {}
    assert any("PDFLIP-USER-RESERVE-BOOK UNMEASURED" in l and "fmt=int8" in l and "NOT booked" in l for l in out), out


def test_launcher_the_off_switch_is_named_and_reaches_the_rank(monkeypatch):
    monkeypatch.setenv(S.RESERVE_BOOK_SWITCH_ENV, "0")
    out = []
    env = L.dual_outside_free_env(_ns(_model("modelopt")), _Cards.cards(), STATED, out.append)
    assert env == {S.RESERVE_BOOK_SWITCH_ENV: "0"}
    assert any("OFF" in l and "NOT booked" in l for l in out), out


def test_launcher_a_record_of_another_card_count_is_refused(monkeypatch):
    monkeypatch.delenv(S.RESERVE_BOOK_SWITCH_ENV, raising=False)
    with pytest.raises(L.PdFlipLaunchRefused):
        L.dual_outside_free_env(_ns(_model("modelopt")), _Cards.cards()[:2], STATED, lambda _l: None)


def test_the_record_is_a_class_min_record_and_the_env_vector_is_indexed_by_the_stage():
    assert IV.RECORD_POLICY["D_DUAL_OUTSIDE_FREE_MIB"] == IV.CLASS_MIN
    env = {S.OUTSIDE_FREE_ENV: "1009,128,"}
    assert S.outside_free_mib(0, env) == 1009 and S.outside_free_mib(1, env) == 128
    assert S.outside_free_mib(2, env) is None            # empty entry = UNMEASURED
    assert S.outside_free_mib(3, env) is None
    assert S.outside_free_mib(0, {}) is None
