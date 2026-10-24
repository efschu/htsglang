"""WAKE-FLOOR-1024 (Auftrag #213): the flip-wake drives a co-located card
below the 400 MiB near-OOM edge, because the waker's resume and the sleeper's
on-card IPC staging are granted on the SAME free bytes.

Metal, boot ``_1024_080355`` (INT8 drei), the 5090 shared by D TP0 (waker)
and P PP0 (sleeper), fixtures/wake_floor_1024/excerpt_080355.txt:

  08:17:38 TP0 weights_5 claim OVERDRAWN on free 3634 MiB (need 1678)
           PP0 c0 staging 1737384288 B (1656 MiB) malloc'd 4 ms before the map
           -> next RESUME begin reads free_mib=298  (3634 - 1678 - 1656 = 300)
  08:18:59 same on free 3596 -> free_mib=260          (3596 - 1678 - 1656 = 262)

The waker's overdraw (xsn290) pushes consumed past credit, so the sleeper's
overdraw debit (Task #20) read "promised = credit - consumed" as 0 while the
claimed tag was NOT YET MAPPED and still in the free reading -- a double
spend. Fix: claimed-not-mapped bytes are booked as pending until the resume
lands and the sleeper's debit subtracts them (reading the card under the
counter lock); the waker's overdraw is re-graded under that lock too.

Boot ``_1024_100906`` (NVFP4 duo), excerpt_100906.txt: PP0's LATE kv_cache
resume (10326 MiB) on free 10432 while D TP0's last staging (1212 MiB) was
still allocated -> card_free_mib=106. Fix: the late kv resume waits (bounded,
order not reserve) for the peer's booked staging to be freed when the pool
would otherwise land below the floor.

Each guard has an in-file MUTANT in the OOM direction: the shipped source is
copied, the guard term removed, and the metal scenario must then reproduce
the dip -- so the scenario provably detects the defect it names.
"""
from __future__ import annotations

import importlib.util
import inspect
import os
import re
import sys
from pathlib import Path

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.managers import pdflip_memory_saver as PMS  # noqa: E402
from flliper.srt.pdflip import wake_kv as WKV  # noqa: E402

MIB = 1 << 20
NEAR_OOM_MIB = 400
FIX = Path(__file__).parent / "fixtures" / "wake_floor_1024"


# ---------------------------------------------------------------- fixtures --
def _lines(name):
    return [ln for ln in (FIX / name).read_text().splitlines() if not ln.startswith("#")]


def _drei_wakes():
    """Per TP0 wake epoch: grant free / need of weights_5, the next begin's
    free, the D and P floors, PP0's staging bytes -- all parsed, none typed."""
    ls = _lines("excerpt_080355.txt")
    floors = {m[1]: int(m[2]) for m in (re.search(r"group=(\w) floor_mib=(\d+)", x) for x in ls) if m}
    stage = {int(m[1]) for m in (re.search(r"lane=c0 phase=deposit stage=\d+ bytes=(\d+)", x) for x in ls) if m}
    assert len(stage) == 1
    wakes = {}
    cur = None
    for x in ls:
        m = re.search(r"TP0\] PDFLIP-RESUME begin tag=(\w+) need_mib=(\d+) .*epoch=(\S+) free_mib=(\d+)", x)
        if m:
            cur = wakes.setdefault(m[3], {})
            cur.setdefault("begin", {})[m[1]] = (int(m[2]), int(m[4]))
        m = re.search(r"TP0\] PDFLIP-CREDIT-EARLY tag=weights_5 free=(\d+) MiB", x)
        if m and cur is not None:
            cur["grant_free"] = int(m[1])
        m = re.search(r"TP0\] PDFLIP-VRAM-CREDIT .*tag=weights_5 .*credit=(\d+) MiB published=\d+ MiB consumed=(\d+) MiB", x)
        if m and cur is not None:
            cur["credit"], cur["consumed_after"] = int(m[1]), int(m[2])
    breaches = [int(m[1]) for m in (re.search(r"TP0\] CORRIDOR LAW BREACHED.* is (\d+) MiB", x) for x in ls) if m]
    return wakes, floors, stage.pop(), breaches


def test_fixture_reconstructs_both_tp0_dips_exactly():
    wakes, floors, stage_b, breaches = _drei_wakes()
    assert floors == {"P": 1055, "D": 767}
    assert len(wakes) == 2
    observed = []
    for ep, w in wakes.items():
        need, _ = w["begin"]["weights_5"]
        _, after = w["begin"]["weights"]
        # the whole dip is weights_5's map plus ONE staging of PP0's, nothing else
        recon = w["grant_free"] - need - (stage_b >> 20)
        assert abs(recon - after) <= 2, (ep, recon, after)
        assert after < NEAR_OOM_MIB
        observed.append(after)
    assert sorted(observed) == sorted(breaches) == [260, 298]


# ------------------------------------------------------------- card model --
class _Card:
    def __init__(self, free_mib):
        self.free_b = int(free_mib) * MIB

    def free(self):
        return self.free_b

    def alloc(self, nbytes):
        self.free_b -= int(nbytes)

    def release(self, nbytes):
        self.free_b += int(nbytes)


def _book(mod, tmp_path, name, w):
    """The counter as it stood right before TP0's weights_5 claim: the
    sleeper's leg is open, ``credit`` published, earlier tags overdrawn
    (consumed > credit) and already MAPPED."""
    c = mod.VramCredit(name, credit_dir=str(tmp_path))
    ep = "1792829161.11"
    c.begin_leg(ep)
    c.publish("published", w["credit"] * MIB)
    prior = w["consumed_after"] - w["begin"]["weights_5"][0]
    c.claim("prior", prior * MIB, epoch=ep, overdraw=True)
    if hasattr(c, "claim_landed"):
        c.claim_landed("prior", epoch=ep)
    return c, ep


def _waker_first(mod, tmp_path, name):
    """Metal order: TP0's claim, then PP0's staging, then TP0's map."""
    wakes, floors, stage_b, _ = _drei_wakes()
    w = wakes["1792829161.11"]
    need = w["begin"]["weights_5"][0] * MIB
    c, ep = _book(mod, tmp_path, name, w)
    card = _Card(w["grant_free"])
    rec = c.wait_for(
        need, budget_s=2.0, tag="weights_5",
        free_bytes_now=w["begin"]["weights_5"][1] * MIB,   # the begin reading: short
        free_reader=card.free, floor_bytes=floors["D"] * MIB, epoch=ep)
    assert rec["claimed_bytes"] == need                     # the grant itself is right
    staged = c.debit("ipc-stage", stage_b, free_bytes=card.free(),
                     floor_bytes=floors["P"] * MIB, free_reader=card.free) \
        if "free_reader" in inspect.signature(c.debit).parameters else \
        c.debit("ipc-stage", stage_b, free_bytes=card.free(), floor_bytes=floors["P"] * MIB)
    if staged:
        card.alloc(stage_b)
    card.alloc(need)                                        # resume(weights_5) maps
    if hasattr(c, "claim_landed"):
        c.claim_landed("weights_5", epoch=ep)
    return card.free() >> 20, staged, floors


def test_sleeper_staging_cannot_spend_a_claimed_unmapped_tag(tmp_path):
    free_after, staged, floors = _waker_first(PMS, tmp_path, "GPU-wf-a")
    assert not staged, "PP0's staging took the bytes TP0 had claimed (host path expected)"
    assert free_after >= NEAR_OOM_MIB
    assert free_after >= floors["D"]                        # 3596 - 1678 = 1918


def _depositor_first(mod, tmp_path, name):
    """The other interleaving: TP0 read the card (3596), PP0 then booked AND
    allocated its staging, THEN TP0 claims on its stale reading."""
    wakes, floors, stage_b, _ = _drei_wakes()
    w = wakes["1792829161.11"]
    need = w["begin"]["weights_5"][0] * MIB
    c, ep = _book(mod, tmp_path, name, w)
    card = _Card(w["grant_free"])
    stale = card.free()
    assert stale - floors["D"] * MIB >= need               # the waker's decision outside the lock
    assert c.debit("ipc-stage", stage_b, free_bytes=card.free(), floor_bytes=floors["P"] * MIB)
    card.alloc(stage_b)
    c.mark_live(stage_b)
    kw = {"recheck": (card.free, floors["D"] * MIB)} \
        if "recheck" in inspect.signature(c.claim).parameters else {}
    spent = c.claim("weights_5", need, epoch=ep, overdraw=True, **kw)
    if spent["claimed_bytes"] >= need:
        card.alloc(need)
    return card.free() >> 20, spent


def test_waker_overdraw_is_regraded_under_the_counter_lock(tmp_path):
    free_after, spent = _depositor_first(PMS, tmp_path, "GPU-wf-b")
    assert spent["recheck_refused"] is True
    assert spent["claimed_bytes"] == 0
    assert free_after >= NEAR_OOM_MIB


def test_landing_releases_the_pending_book(tmp_path):
    """Perf direction: a landed claim must not keep the sleeper on the host
    path forever -- once the bytes are mapped they leave pending."""
    c = PMS.VramCredit("GPU-wf-c", credit_dir=str(tmp_path))
    c.begin_leg("e1")
    rec = c.claim("weights_5", 1678 * MIB, epoch="e1", overdraw=True)
    assert rec["pending_claim_bytes"] == 1678 * MIB
    # pending: 4000 - 1055 - 1678 = 1267 < 1656
    assert not c.debit("ipc-stage", 1656 * MIB, free_bytes=4000 * MIB, floor_bytes=1055 * MIB)
    assert c.claim_landed("weights_5", epoch="e1") == 1678 * MIB
    assert c.claim_landed("weights_5", epoch="e1") == 0
    assert c.claim_landed("other", epoch="e0-stale") == 0
    # mapped, and later the card freed again: the staging is allowed
    assert c.debit("ipc-stage", 1656 * MIB, free_bytes=4000 * MIB, floor_bytes=1055 * MIB)


def test_begin_leg_clears_pending(tmp_path):
    c = PMS.VramCredit("GPU-wf-d", credit_dir=str(tmp_path))
    c.begin_leg("e1")
    c.claim("weights_5", 1678 * MIB, epoch="e1", overdraw=True)
    c.begin_leg("e2")
    assert not (c.read().get("pending_claims") or {})


# ---------------------------------------------------- duo: late kv resume --
def _duo():
    ls = _lines("excerpt_100906.txt")
    txt = "\n".join(ls)
    floor = int(re.search(r"PP0\] PDFLIP-CREDIT-FLOOR .*floor_mib=(\d+)", txt)[1])
    free = int(re.search(r"PP0\] PDFLIP-DC-BREAKDOWN stage=wake-pre-kv .*card free (\d+) of", txt)[1])
    need = int(re.search(r"PP0\] PDFLIP-WAKE-KV-TIME tag=kv_cache .*need_mib=(\d+)", txt)[1])
    end = int(re.search(r"PP0\] PDFLIP-VRAM-PEAK rank=0 phase=flip leg=resume .*card_free_mib=(\d+)", txt)[1])
    stage = int(re.search(r"TP0\] PDFLIP-SEQ ipc lane=c0 phase=deposit stage=\d+ bytes=(\d+)", txt)[1])
    return floor, free, need, end, stage


def test_duo_fixture_reconstructs_the_106():
    floor, free, need, end, stage = _duo()
    assert (floor, free, need, end) == (1024, 10432, 10326, 106)
    assert free - need == end < NEAR_OOM_MIB
    assert (free + (stage >> 20)) - need >= floor           # after D's release it fits


def _kv_wake(waiter):
    floor, free, need, _end, stage = _duo()
    card = _Card(free)
    book = {"staged": stage, "polls": 0}
    clock = {"t": 0.0}

    def _owed():
        return book["staged"]

    def _sleep(s):
        clock["t"] += s
        book["polls"] += 1
        if book["polls"] == 2:          # D TP0's leg-end drain ~10-20 ms later
            card.release(book["staged"])
            book["staged"] = 0

    free2, waited, owed = waiter(card.free, need * MIB, floor * MIB, _owed,
                                 wait_s=20.0, sleep=_sleep, now=lambda: clock["t"])
    card.alloc(need * MIB)              # resume(kv_cache)
    return card.free() >> 20, waited, owed, floor


def test_late_kv_resume_waits_for_the_peer_staging():
    after, waited, owed, floor = _kv_wake(WKV.wait_for_peer_staging)
    assert owed == 0 and 0 < waited < 1.0
    assert after >= NEAR_OOM_MIB and after >= floor        # 11644 - 10326 = 1318


def test_kv_wait_is_order_not_a_reserve():
    clock = {"t": 0.0}

    def _sleep(s):
        clock["t"] += s

    # nothing owed: no wait at all, even below the floor
    _f, w, _o = WKV.wait_for_peer_staging(lambda: 10432 * MIB, 10326 * MIB, 1024 * MIB,
                                          lambda: 0, sleep=_sleep, now=lambda: clock["t"])
    assert w == 0.0
    # owed but never released: bounded, then it returns (no refusal here)
    _f, w, o = WKV.wait_for_peer_staging(lambda: 10432 * MIB, 10326 * MIB, 1024 * MIB,
                                         lambda: 1212 * MIB, wait_s=0.5, sleep=_sleep,
                                         now=lambda: clock["t"])
    assert 0.5 <= w < 0.6 and o == 1212 * MIB
    # floor already held: no wait
    _f, w, _o = WKV.wait_for_peer_staging(lambda: 12000 * MIB, 10326 * MIB, 1024 * MIB,
                                          lambda: 1212 * MIB, sleep=_sleep, now=lambda: clock["t"])
    assert w == 0.0


# --------------------------------------------------------------- wiring --
def _wu_src():
    from flliper.srt.managers.scheduler_components import weight_updater as wu
    return Path(wu.__file__).read_text()


def test_wiring_resume_lands_charge_reads_kv_waits():
    src = _wu_src()
    # every weights tag: resume, then landed
    i = src.index("pdflip_tms_resume(self.memory_saver_adapter, tag)\n")
    assert "self._pdflip_credit_landed(credit, tag, credit_epoch)" in src[i:i + 400]
    # the parked draft too
    j = src.index("resume=lambda t: pdflip_tms_resume(self.memory_saver_adapter, t))")
    assert "_pdflip_credit_landed(credit, GPU_MEMORY_TYPE_WEIGHTS_DRAFT, epoch)" in src[j:j + 300]
    # the staging debit gets the reader (graded under the lock)
    k = src.index("def _charge(nbytes: int):")
    assert "_reader = self._pdflip_free_bytes" in src[k:k + 1200]
    assert "free_reader=_reader" in src[k:k + 1200]
    # the late kv resume waits for the peer staging BEFORE its fit check
    a = src.index("self._pdflip_kv_wait_peer_staging(_kv_epoch, _kv_free, _kv_need, _kv_floor)")
    b = src.index("_unfit = kv_resume_fit_refusal(_kv_free, _kv_need, _kv_floor)")
    assert a < b


def test_kv_wait_method_reads_this_flips_counter(tmp_path):
    from types import SimpleNamespace

    from flliper.srt.managers.scheduler_components.weight_updater import (
        SchedulerWeightUpdaterManager as WU,
    )
    c = PMS.VramCredit("GPU-wf-e", credit_dir=str(tmp_path))
    c.begin_leg("ep9")
    assert c.debit("ipc-stage", 1212 * MIB, free_bytes=20000 * MIB, floor_bytes=0)
    card = _Card(10432)
    calls = {"n": 0}

    def _free():
        calls["n"] += 1
        if calls["n"] == 3:
            c.refund("ipc-stage", 1212 * MIB)
            card.release(1212 * MIB)
        return card.free()

    stub = SimpleNamespace(_pdflip_credit_reader=lambda e: (c, str(e)), _pdflip_free_bytes=_free)
    got = WU._pdflip_kv_wait_peer_staging(stub, "ep9", 10432 * MIB, 10326 * MIB, 1024 * MIB)
    assert got == 11644 * MIB
    # another flip's counter owes nothing: no wait, the reading comes back as given
    c2 = PMS.VramCredit("GPU-wf-f", credit_dir=str(tmp_path))
    c2.begin_leg("ep9")
    assert c2.debit("ipc-stage", 1212 * MIB, free_bytes=20000 * MIB, floor_bytes=0)
    stub2 = SimpleNamespace(_pdflip_credit_reader=lambda e: (c2, "other-flip"),
                            _pdflip_free_bytes=lambda: 10432 * MIB)
    assert WU._pdflip_kv_wait_peer_staging(stub2, "x", 10432 * MIB, 10326 * MIB, 1024 * MIB) \
        == 10432 * MIB
    # no counter at all (single group): no wait
    stub3 = SimpleNamespace(_pdflip_credit_reader=lambda e: (None, None),
                            _pdflip_free_bytes=lambda: 1 / 0)
    assert WU._pdflip_kv_wait_peer_staging(stub3, "x", 10432 * MIB, 10326 * MIB, 1024 * MIB) \
        == 10432 * MIB


# ------------------------------------------------- mutants (OOM direction) --
def _mutant(module, old, new, tag):
    path = Path(module.__file__)
    src = path.read_text()
    assert src.count(old) == 1, f"mutant anchor gone for {tag}: {old!r}"
    name = f"_wf_mutant_{tag}"
    spec = importlib.util.spec_from_loader(name, loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__file__ = str(path)
    sys.modules[name] = mod
    try:
        exec(compile(src.replace(old, new), str(path), "exec"), mod.__dict__)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return mod


def test_mutant_m1_debit_without_pending_reproduces_the_dip(tmp_path):
    m = _mutant(PMS, "- floor - promised - pending - staged < want",
                "- floor - promised - staged < want", "m1")
    free_after, staged, _ = _waker_first(m, tmp_path, "GPU-wf-m1")
    assert staged and free_after < NEAR_OOM_MIB            # 3596 - 1678 - 1656 = 262


def test_mutant_m2_claim_without_recheck_reproduces_the_dip(tmp_path):
    m = _mutant(PMS, "                        claimed = 0\n                        recheck_refused = True\n",
                "                        pass\n", "m2")
    free_after, spent = _depositor_first(m, tmp_path, "GPU-wf-m2")
    assert spent["claimed_bytes"] > 0 and free_after < NEAR_OOM_MIB


def test_mutant_m3_kv_without_the_staging_wait_reproduces_the_106():
    m = _mutant(WKV, "    while (owed > 0 and free is not None", "    while (False and free is not None", "m3")
    after, waited, _owed, _floor = _kv_wake(m.wait_for_peer_staging)
    assert waited == 0.0 and after < NEAR_OOM_MIB          # 10432 - 10326 = 106


def test_mutant_m4_landed_never_called_is_caught_by_wiring():
    src = _wu_src()
    mutated = src.replace("self._pdflip_credit_landed(credit, tag, credit_epoch)", "pass", 1)
    i = mutated.index("pdflip_tms_resume(self.memory_saver_adapter, tag)\n")
    assert "self._pdflip_credit_landed(credit, tag, credit_epoch)" not in mutated[i:i + 400]
