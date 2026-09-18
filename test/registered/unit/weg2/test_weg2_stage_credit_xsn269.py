"""weg2xsn269 (18.09.2026): group D died in flip 4. D-log line 8792 (no rank
prefix, the memory saver's stderr):

    [torch_memory_saver.cpp] CUresult error: 2 (out of memory)
    func=cu_mem_create line=194            -> exit(1), TP0 gone silently

TP0 had checked its credit (7700 published, 7506 consumed, 194 left after
the claim) and the card's free VRAM (3124 MiB for a 2502 MiB tag) at
00:05:35.19 and entered resume(weights_3). PP0, the depositor on the same
card, allocated its next on-card IPC staging (1.99 GB cudaMalloc, up to two
live per lane, freed only at the leg's end) at 00:05:35.43. The staging was
booked nowhere; the waker's cuMemCreate lost the race.

Now the staging is DEBITED from the card's credit under the same lock the
waker claims under (VramCredit.debit / refund), the bounce refuses a
staging the balance does not cover (host path for that tag), and every
freed staging refunds. These tests drive the credit file, the staging
allocator with a counting fake, and the updater's charge factory.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers.weg2_memory_saver import VramCredit  # noqa: E402
from sglang.srt.weg2 import weight_exchange_bounce as bx  # noqa: E402

MIB = 1 << 20


class _Ops:
    def __init__(self):
        self.live = {}
        self.next = 0x1000

    def raw_malloc(self, device, nbytes):
        p = self.next
        self.next += int(nbytes)
        self.live[p] = int(nbytes)
        return p

    def raw_free(self, ptr):
        self.live.pop(int(ptr))


@pytest.fixture(autouse=True)
def _clean_stage():
    with bx._SEQ_CACHE_LOCK:
        bx._SEQ_STAGE.clear()
    yield
    with bx._SEQ_CACHE_LOCK:
        bx._SEQ_STAGE.clear()


def test_debit_takes_only_what_the_waker_was_not_promised(tmp_path):
    c = VramCredit("GPU-xsn269-a", credit_dir=str(tmp_path))
    c.begin_leg("flip-4")
    assert c.debit("ipc-stage", 1990 * MIB) is False          # nothing published yet
    c.publish("weights_0", 3880 * MIB)
    assert c.debit("ipc-stage", 1990 * MIB) is True
    st = c.read()
    assert st["consumed_bytes"] == 1990 * MIB and st["staged_bytes"] == 1990 * MIB
    # the waker now sees 1890 MiB, not 3880: a 2502 MiB claim is NOT covered
    r = c.claim("weights_3", 2502 * MIB, require_full=True)
    assert r["covered"] is False and r["claimed_bytes"] == 0
    c.refund("ipc-stage", 1990 * MIB)
    r = c.claim("weights_3", 2502 * MIB, require_full=True)
    assert r["covered"] is True and r["claimed_bytes"] == 2502 * MIB


def test_refund_never_exceeds_what_was_staged_nor_goes_negative(tmp_path):
    c = VramCredit("GPU-xsn269-b", credit_dir=str(tmp_path))
    c.begin_leg("flip-4")
    c.publish("weights_0", 100 * MIB)
    assert c.debit("ipc-stage", 60 * MIB)
    c.refund("ipc-stage", 500 * MIB)                          # more than staged
    st = c.read()
    assert st["staged_bytes"] == 0 and st["consumed_bytes"] == 0
    # a debit on a leg nobody opened writes nothing
    d = VramCredit("GPU-xsn269-c", credit_dir=str(tmp_path))
    assert d.debit("ipc-stage", 1) is False


def test_stage_alloc_refuses_when_the_charge_refuses_and_refunds_on_free():
    ops = _Ops()
    ledger = {"consumed": 0}

    def charge(n):
        if ledger["consumed"] + n > 4000 * MIB:
            return None
        ledger["consumed"] += n

        def refund():
            ledger["consumed"] -= n
        return refund

    lines = []
    key = ("nonce", "c0_s0")
    p = bx._stage_alloc(ops, 0, key, 1990 * MIB, lines.append, "c0", charge=charge)
    assert ops.live[p] == 1990 * MIB and ledger["consumed"] == 1990 * MIB
    p2 = bx._stage_alloc(ops, 0, ("nonce", "c0_s1"), 1990 * MIB, lines.append, "c0", charge=charge)
    assert ledger["consumed"] == 3980 * MIB
    # a third staging of the same slot frees the old one first (refund) and
    # is then charged again -- the balance holds at two
    p3 = bx._stage_alloc(ops, 0, key, 1990 * MIB, lines.append, "c0", charge=charge)
    assert p not in ops.live and p3 in ops.live and ledger["consumed"] == 3980 * MIB
    # beyond the balance: refused BEFORE any malloc, nothing charged
    with pytest.raises(RuntimeError) as exc:
        bx._stage_alloc(ops, 0, ("nonce", "c1_s0"), 100 * MIB, lines.append, "c1", charge=charge)
    assert "refused by the card's VRAM credit" in str(exc.value)
    assert ledger["consumed"] == 3980 * MIB and len(ops.live) == 2
    n = bx.release_stage_buffers("nonce", log=lines.append)
    assert n == 2 and not ops.live and ledger["consumed"] == 0
    del p2


def test_a_failed_malloc_refunds_the_charge():
    class _Oom(_Ops):
        def raw_malloc(self, device, nbytes):
            raise RuntimeError("cudaMalloc rc=2 out of memory")

    ledger = {"consumed": 0}

    def charge(n):
        ledger["consumed"] += n
        return lambda: ledger.__setitem__("consumed", ledger["consumed"] - n)

    with pytest.raises(RuntimeError):
        bx._stage_alloc(_Oom(), 0, ("n", "c0_s0"), 10, lambda *_a: None, "c0", charge=charge)
    assert ledger["consumed"] == 0


def test_without_a_charge_the_staging_is_unbooked_as_before():
    ops = _Ops()
    p = bx._stage_alloc(ops, 0, ("n", "c0_s0"), 64, lambda *_a: None, "c0")
    assert bx._SEQ_STAGE[("n", "c0_s0")]["refund"] is None and p in ops.live
    assert bx.release_stage_buffers("n") == 1


def test_updater_charge_books_against_the_leg_credit_and_refuses_without_balance(tmp_path):
    from sglang.srt.managers.scheduler_components import weight_updater as wu
    c = VramCredit("GPU-xsn269-d", credit_dir=str(tmp_path))
    c.begin_leg("flip-4")
    c.publish("weights_0", 2000 * MIB)
    m = SimpleNamespace(_weg2_leg_credit=c)
    charge = wu.SchedulerWeightUpdaterManager._weg2_stage_charge(m)
    assert callable(charge)
    refund = charge(1990 * MIB)
    assert callable(refund) and c.read()["consumed_bytes"] == 1990 * MIB
    assert charge(1990 * MIB) is None                          # balance 10 MiB: refused
    refund()
    assert c.read()["consumed_bytes"] == 0
    # no credit on this leg -> no charge (unbooked staging, the old form)
    assert wu.SchedulerWeightUpdaterManager._weg2_stage_charge(SimpleNamespace()) is None
    assert wu.SchedulerWeightUpdaterManager._weg2_stage_charge(SimpleNamespace(_weg2_leg_credit=None)) is None


def test_the_bounce_threads_the_charge_to_the_staging_allocator():
    src = open(bx.__file__).read()
    i = src.index("def run_sequential_units")
    assert "stage_charge=None" in src[i:i + 3000]
    j = src.index("_ipc_base = _stage_alloc(", i)
    assert "charge=stage_charge" in src[j:j + 400]
    from sglang.srt.managers.scheduler_components import weight_updater as wu
    usrc = open(wu.__file__).read()
    k = usrc.index("last = bx.run_sequential_units(")
    assert "stage_charge=self._weg2_stage_charge()" in usrc[k:k + 1500]
    assert "self._weg2_leg_credit = credit" in usrc
