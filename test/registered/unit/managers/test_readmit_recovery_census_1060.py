"""#1060: the re-admission recovery census must BUCKET and must PRINT.

MATCHED TO THIS EDIT'S FAILURE CLASS, which is not "does it compile". The
class that cost boot 29 a whole window is an instrument that is wired and
never executes, and the class that cost boot 28 its measurement is an emitter
gated on an observation count. Both look identical to `py_compile` and to an
import smoke. So this drives the arithmetic directly and asserts the two
properties a boot cannot distinguish afterwards:

* every observation lands in EXACTLY ONE outcome bucket (denominator law), and
* the WHY-EMPTY discriminator separates "a host-pool clear fell between this
  rid's two consults" from "it did not" -- the whole reason the census exists.

DANGER-DIRECTION MUTANT included: if the clear-epoch stamp were dropped (the
plausible regression -- it is the only per-rid state the census keeps), every
empty match would score `clear_unknown` and the census would answer nothing
while still printing a full line. `test_mutant_without_epoch_stamp_is_blind`
asserts that shape is DISTINGUISHABLE from the healthy one, so a future
refactor cannot silently produce it.
"""

import sglang.srt.managers.schedule_batch as sb


class _Pool:
    def __init__(self, size=100, avail=100, epoch=0):
        self.size = size
        self._avail = avail
        self._clear_epoch = epoch

    def available_size(self):
        return self._avail


class _CC:
    def __init__(self, pool):
        self.mem_pool_host = pool


class _Root:
    def __init__(self, children):
        self.children = children


class _Tree:
    def __init__(self, pool, children=None, storage=True, prefetch=()):
        self.cache_controller = _CC(pool)
        self.root_node = _Root({} if children is None else children)
        self.enable_storage = storage
        self.ongoing_prefetch = {r: object() for r in prefetch}


class _Req:
    def __init__(self, rid, host_hit=0, prefix=()):
        self.rid = rid
        self.host_hit_length = host_hit
        self.prefix_indices = prefix


def _reset():
    sb._1060_STATE.clear()
    sb._1060_EPOCH_BY_RID.clear()
    del sb._1060_LAST_CENSUS_T[:]


def test_one_observation_one_bucket():
    """Denominator law: seen == the sum of the three outcome buckets."""
    _reset()
    pool = _Pool()
    tree = _Tree(pool)
    sb._1060_note_readmit_recovery(_Req("a", host_hit=1234), tree)
    sb._1060_note_readmit_recovery(_Req("b", prefix=(1, 2, 3)), tree)
    sb._1060_note_readmit_recovery(_Req("c"), tree)
    s = sb._1060_STATE
    assert s["seen"] == 3, s
    assert s["outcome_recovered_host"] == 1, s
    assert s["outcome_recovered_device"] == 1, s
    assert s["outcome_empty"] == 1, s
    assert (
        s["outcome_recovered_host"]
        + s["outcome_recovered_device"]
        + s["outcome_empty"]
        == s["seen"]
    ), s


def test_clear_between_is_the_discriminator():
    """An empty match AFTER a clear scores clear_between; without one it does not.

    This is the measurement the boot is for: it is what separates "the host
    pool clear destroyed the copy the read-through was promised" from "the copy
    was never there".
    """
    _reset()
    pool = _Pool(epoch=7)
    tree = _Tree(pool)
    # First consult of this rid: no previous epoch, so the honest answer is
    # UNKNOWN -- never a free 'no clear'.
    sb._1060_note_readmit_recovery(_Req("r1"), tree)
    assert sb._1060_STATE["empty_clear_unknown"] == 1, sb._1060_STATE
    # Second consult, no clear in between.
    sb._1060_note_readmit_recovery(_Req("r1"), tree)
    assert sb._1060_STATE["empty_no_clear_between"] == 1, sb._1060_STATE
    # A cutover clears the host pool, then the rid is consulted again.
    pool._clear_epoch = 8
    sb._1060_note_readmit_recovery(_Req("r1"), tree)
    assert sb._1060_STATE["empty_clear_between"] == 1, sb._1060_STATE
    assert sb._1060_STATE["outcome_empty"] == 3, sb._1060_STATE


def test_tier_flags_are_independent_and_may_overlap():
    _reset()
    pool = _Pool(size=100, avail=100)  # every host slot free == the clear shape
    tree = _Tree(pool, children={}, storage=True, prefetch=("x",))
    sb._1060_note_readmit_recovery(_Req("x"), tree)
    s = sb._1060_STATE
    assert s["tier_tree_empty"] == 1, s
    assert s["tier_host_pool_all_free"] == 1, s
    assert s["tier_storage_enabled"] == 1, s
    assert s["tier_prefetch_registered"] == 1, s
    # Counters are only materialised when bumped, so an absent key IS zero.
    assert s.get("probe_unreadable", 0) == 0, s


def test_unreadable_probe_is_counted_not_scored_as_absent():
    """'Unknown' may never be reported as 'empty' -- the indicator law."""
    _reset()

    class _Broken:
        cache_controller = None
        root_node = None
        enable_storage = False

    sb._1060_note_readmit_recovery(_Req("y"), _Broken())
    assert sb._1060_STATE["probe_unreadable"] >= 1, sb._1060_STATE
    assert sb._1060_STATE.get("tier_tree_empty", 0) == 0, sb._1060_STATE


def test_census_prints_at_zero_observations(caplog):
    """seen=0 must be a PRINTED measurement, not a missing line (boot 28)."""
    _reset()
    with caplog.at_level("WARNING"):
        sb._1060_emit_census("teardown")
    text = caplog.text
    assert "#1060 READMIT-RECOVERY CENSUS (teardown)" in text, text
    assert "seen=0" in text, text


def test_mutant_without_epoch_stamp_is_blind():
    """DANGER DIRECTION: drop the per-rid epoch stamp and the census answers nothing.

    Asserted as a DISTINGUISHABLE shape rather than merely 'still runs': with
    the stamp gone every empty match scores `clear_unknown`, and both real
    buckets stay at zero. That is the failure mode a refactor would produce,
    and this test is what makes it loud instead of silent.
    """
    _reset()
    pool = _Pool(epoch=3)
    tree = _Tree(pool)
    sb._1060_note_readmit_recovery(_Req("m"), tree)
    pool._clear_epoch = 4
    sb._1060_EPOCH_BY_RID.clear()  # the mutant: the stamp did not survive
    sb._1060_note_readmit_recovery(_Req("m"), tree)
    s = sb._1060_STATE
    assert s["empty_clear_unknown"] == 2, s
    assert s.get("empty_clear_between", 0) == 0, s
    assert s.get("empty_no_clear_between", 0) == 0, s
