"""#1342 B4: THE WHOLE CHAIN, ONCE, ON DOUBLES, UNDER THE BOOT'S OWN ARGV.

THIS FILE IS THE ANSWER TO A COUNT.  On 2026-09-11 five defects of ONE form
fell in this lane, in one day:

  1. six emitters with no production caller at all (#1342 S3);
  2. the carrier router whose exchange branch no argv could select (S1);
  3. the step-6c grader parked behind an early return this rig always takes
     (S1b -- cost boot weg2xsn18);
  4. `emit_plan_line` wired to a producer that returns a different type
     (S2b -- same boot, 39 logged AttributeErrors);
  5. the delegation reading `plan.raw_descs` from a `LegPlan` that has only
     `descs` -- FOUND BY THIS FILE, at the desk, with no window spent.

One form: BUILT, BUT THE CHAIN NEVER EXECUTED ONCE END TO END.  Every one of
them was individually unit-tested and green.  Three boots were spent
discovering three of them, one at a time, because the Plan of Record's B4
("execution smoke on doubles for BOTH paths, red on the tree before B2/B3")
stood OPEN while B5 -- the boot -- ran three times.

SO THIS IS NOT A COMPONENT TEST AND MUST NOT BECOME ONE.  It is ONE run of the
real production chain, from the wake entry point to the graded artifacts, with
doubles only at the edges that need a GPU (the device-ops layer) or a live model
(the plan producer).  Every frame between them is the product's own.

THE ARGV IT RUNS UNDER is the S6I order's, exactly, because that is the
configuration all three boots used and the one every defect above hid behind:

    SGLANG_WEG2_WEIGHT_SOURCE = exchange
    SGLANG_WEG2_XCHG_INJECT   = shadow
    --enable-weights-cpu-backup ON   (the launcher passes it unconditionally)
    --enable-memory-saver ON

which means the carrier is `tms-backup`, the disk refill does NOT run, and the
grader must still be reached.  That sentence is the whole point.

WHAT IT SHOWS AT THE END -- the graded artifacts, by name:
  * `WEG2-XCHG-INJECT ... verdict=`      (grading item (a), per leg)
  * `WEG2-XCHG-INJECT-SUMMARY ... verdict=`  (grading item (a), per boot)
  * `bounce.bin` on disk at `(depth+1) * slot_bytes` (grading item (c))

SCOPE BOUNDARY, STATED RATHER THAN BLURRED: the fourth graded artifact, census
(d)'s `WEG2-XCHG-PLAN dir=.../plan_id=`, is emitted one frame UPSTREAM of the
wake, inside the plan derivation -- which needs a live model inventory this run
substitutes.  It is covered by `test_weg2_xchg_plan_emit_site_1342.py` against
the producer's real type.  Claiming this run proved it too would be the same
overreach that put "the chain is named" into a boot order.
"""

import dataclasses
import logging

import pytest

from sglang.srt.managers.scheduler_components import weight_updater as wu
from sglang.srt.weg2 import weight_exchange as wx
from sglang.srt.weg2 import weight_exchange_bounce as bx
from sglang.srt.weg2 import weight_exchange_region as xr
from sglang.srt.weg2 import weight_exchange_shadow as sh
from sglang.srt.weg2 import xchg_bounce as xb

# Relative imports, the same shape the neighbouring smoke uses: this directory
# is a package and the doubles are shared rather than re-written per file -- a
# second FakeDeviceOps would be exactly the drift this file is about.
from .test_weg2_xchg_bounce_execution_smoke_1273 import (  # noqa: E402
    DEPTH,
    SLOT_BYTES,
    _all_descs,
    _seed_source,
)
from .test_weg2_xchg_transport_1273 import (  # noqa: E402
    FakeDeviceOps,
    _fresh_boot,
)

WU = wu.SchedulerWeightUpdaterManager


class _ServerArgs:
    """The S6I boot's own flags, not a minimal set.

    `enable_weights_cpu_backup=True` is the load-bearing one: the weg2 launcher
    passes it in `common_flags` on every boot, which is what makes the carrier
    `tms-backup` and what hid defects 2 and 3.
    """

    enable_memory_saver = True
    enable_weights_cpu_backup = True
    enable_draft_weights_cpu_backup = False
    speculative_draft_model_path = None
    model_path = "/models/main"


class _Runner:
    def __init__(self):
        self.model = None
        self.model_config = None


class _Worker:
    def __init__(self):
        self.model_runner = _Runner()


def _seed_destinations_as_a_correct_refill_would(ops, descs):
    """Put the CORRECT bytes in the destination before the shadow compares.

    THIS STANDS IN FOR THE TMS RESTORE, and it is the precondition the graded
    arm actually runs under: by the time the step-6c compare fires, the weights
    are already correct -- the restore (or the disk refill) wrote them -- and
    what the compare grades is whether the EXCHANGE would have assembled the
    same bytes.  So a faithful double must have destination == source here, and
    `MATCH` is then the meaningful outcome.

    WHY IT IS NEEDED AT ALL, recorded because the first version of this run got
    it wrong and read MISMATCH: `_seed_source` tiles the card with a pattern
    keyed on ABSOLUTE offset, so source and destination bytes legitimately
    differ until something copies them.  In `shadow` mode the leg writes
    NOTHING (that is the whole point of the mode), so an unseeded destination
    makes every piece mismatch -- a defect of the double, not of the product,
    and exactly the kind of false red that would discredit this run.

    Done with an explicit copy rather than by first running an AUTHORITATIVE
    leg: the shadow assertion must not depend on the authoritative path being
    correct, or a single bug would hide in both halves at once.
    """
    import ctypes
    for d in descs:
        for r in range(int(d.rows)):
            src = ops.real(int(d.src_ptr)) + int(d.src_off) + r * int(d.spitch)
            dst = ops.real(int(d.dst_ptr)) + int(d.dst_off) + r * int(d.dpitch)
            ctypes.memmove(dst, src, int(d.run_bytes))


def _real_legplan(descs):
    """A REAL `LegPlan`, built from the PRODUCER's field set.

    THE ROOT OF DEFECTS 4 AND 5 WAS THE DOUBLE, NOT THE TYPE, so this helper is
    the corrective: it constructs the producer's actual dataclass, and it
    asserts the field set first, so the day `LegPlan` changes shape this run
    fails LOUDLY instead of drifting into agreement with the consumer.

    Every previous fake in this slice was shaped after what the CONSUMER read
    (`raw_descs`, `plan_id`), which is precisely why two type mismatches
    survived a green suite and reached the metal.
    """
    names = {f.name for f in dataclasses.fields(sh.LegPlan)}
    assert "descs" in names
    assert "raw_descs" not in names, (
        "LegPlan grew `raw_descs`; the consumer/producer mismatch this run "
        "exists to catch has changed shape -- re-read the delegation"
    )
    facts = sh.LegPlanFacts(
        chunk_layers=8, chunk_count=8,
        family_tags=("weights_0",), waves=(("weights_0",),),
        cards=(0, 1, 2), classes=("down_proj",), source=sh.PLAN_SOURCE)
    return sh.LegPlan(facts=facts, descs=tuple(descs), card=0,
                      tags=("weights_0",), card_digest=0xabc)


@pytest.fixture()
def chain(tmp_path, monkeypatch):
    """The S6I ARGV, a region on tmp_path, and doubles ONLY at the two edges."""
    boot = _fresh_boot()
    monkeypatch.setenv(wx.WEIGHT_SOURCE_ENV, wx.WEIGHT_SOURCE_EXCHANGE)
    monkeypatch.delenv(wx.INJECT_ENV, raising=False)          # default = shadow
    monkeypatch.setenv(xr.ENV_REGION_BOOT, boot)
    # the launcher's published bounce geometry, through its own publisher
    terms = xb.bounce_terms(
        bytes_per_direction=24_000_000, n_layers=8,
        widest_layer_bytes=4_000_000, pairs=3, depth=DEPTH,
        slot_bytes=SLOT_BYTES,
    )
    monkeypatch.setenv(xb.ENV_BOUNCE_TERMS, xb.publish_terms(terms))
    # EDGE 1: the shared-memory root. The chain hardcodes xr.SHM_ROOT, so the
    # constant is redirected rather than a parameter threaded through -- noted
    # as a seam the product does not currently offer.
    monkeypatch.setattr(xr, "SHM_ROOT", str(tmp_path), raising=True)
    # EDGE 2: the device-ops layer (needs a GPU) and the plan producer (needs a
    # live model inventory). NOTHING between the wake entry and the verdict is
    # faked.
    ops = FakeDeviceOps(str(tmp_path), 0)
    _seed_source(ops)
    descs_for_seed = _all_descs()
    _seed_destinations_as_a_correct_refill_would(ops, descs_for_seed)
    monkeypatch.setattr(WU, "_weg2_xchg_device_ops", lambda self: ops,
                        raising=True)
    descs = _all_descs()
    monkeypatch.setattr(
        WU, "_weg2_shadow_plan",
        lambda self, hook, g, r, agreed=None: (_real_legplan(descs), ""),
        raising=True)
    monkeypatch.setattr(WU, "_weg2_server_args",
                        lambda self: _ServerArgs(), raising=True)
    monkeypatch.setattr(WU, "_weg2_group_name", lambda self: "D", raising=True)
    monkeypatch.setattr(WU, "_weg2_rank", lambda self: 0, raising=True)
    monkeypatch.setattr(WU, "_weg2_device_index", lambda self: 0, raising=True)
    bx.reset_inject_verdicts()
    mgr = wu.SchedulerWeightUpdaterManager(
        tp_worker=_Worker(), draft_worker=None, tp_cpu_group=None,
        memory_saver_adapter=None, flush_cache=lambda *a, **k: True,
        is_fully_idle=lambda *a, **k: True,
    )
    return mgr, boot, tmp_path, terms


def test_the_chain_runs_once_and_produces_the_graded_artifacts(chain, caplog):
    """ONE RUN, and the assertions at the end are the graded artifacts.

    RED before the `plan.descs` fix: the delegation read `plan.raw_descs`, which
    a `LegPlan` does not have, so the chain died one frame past the grader with
    an AttributeError the observer swallowed into a single NO-COMPARE line --
    and NO-COMPARE is a FAIL in the grading plan, not a neutral.
    """
    mgr, boot, root, terms = chain
    caplog.set_level(logging.INFO)

    # -- LINK 0: THE GRADER IS REACHED FROM THE WAKE PATH. -------------------
    #
    # This assertion is inside THIS test on purpose, and mutant M15 is why.
    # An earlier version of this run called `_weg2_xchg_shadow_compare()`
    # directly and asserted the artifacts -- which proves "IF called, it
    # works" and says NOTHING about "it is called".  Removing the call site
    # from the wake path (i.e. restoring defect 3, the one that cost boot
    # weg2xsn18) left this run GREEN.  A chain smoke that is blind to the
    # first link of the chain is not a chain smoke.
    #
    # Structural rather than executed: `resume_memory_occupation` needs a live
    # model runner, a process group and a device.  What can be checked without
    # them is that the call EXISTS on that path, inside the `family_complete`
    # guard, and AFTER the reload -- which is the whole of what defect 3 got
    # wrong.
    import inspect
    wake_src = inspect.getsource(WU.resume_memory_occupation)
    assert "self._weg2_xchg_shadow_compare()" in wake_src, (
        "the grader has NO call site on the wake path: defect 3 is back, and "
        "no amount of artifact-passing below would be reached on a real boot"
    )
    assert (wake_src.index("self._weg2_xchg_shadow_compare()")
            > wake_src.index("self._weg2_wake_reload_weights()")), (
        "the grader runs before the reload: it would grade unsettled bytes"
    )
    assert "_weg2_xchg_shadow_compare" not in inspect.getsource(
        WU._weg2_wake_reload_weights), (
        "the grader is parked in the refill path again (defect 3's exact shape)"
    )

    # -- the carrier this ARGV selects, asserted before anything runs ---------
    assert mgr._weg2_wake_weight_carrier() == WU.CARRIER_TMS_BACKUP, (
        "the premise of this run is the tms-backup carrier"
    )

    # -- STEP 1: the refill. It must do NOTHING here, by design. -------------
    mgr._weg2_wake_reload_weights()

    # -- STEP 2: the wake path's grader call, which is what S1b moved. --------
    mgr._weg2_xchg_shadow_compare()

    text = caplog.text

    # -- ARTIFACT 1: the per-leg verdict line, grading item (a) ---------------
    inject = [l for l in text.splitlines() if "WEG2-XCHG-INJECT " in l]
    assert inject, (
        "no WEG2-XCHG-INJECT line: the chain did not reach the verdict.\n"
        + text[-3000:]
    )
    assert "mode=shadow" in inject[0], inject[0]
    assert "verdict=MATCH" in inject[0], (
        "the leg compared but did not MATCH, or compared nothing "
        "(NO-COMPARE is a FAIL, not a neutral): " + inject[0]
    )

    # -- ARTIFACT 2: the per-boot summary, grading item (a) -------------------
    summary = [l for l in text.splitlines()
               if "WEG2-XCHG-INJECT-SUMMARY" in l]
    assert summary, "no WEG2-XCHG-INJECT-SUMMARY line"
    assert "legs_no_compare=0" in summary[-1], summary[-1]
    assert "legs_mismatch=0" in summary[-1], summary[-1]
    assert "verdict=MATCH" in summary[-1], summary[-1]

    # -- ARTIFACT 3: bounce.bin on disk, grading item (c) --------------------
    import os
    path = bx.bounce_path(boot, shm_root=str(root))
    assert os.path.exists(path), f"bounce.bin absent at {path}"
    # THE SIZE IS DERIVED FROM THE PUBLISHED TERMS, never from this file's
    # constants -- the same rule the grading plan states for the boot ("if
    # depth or slot_mib differ on the build, recompute from THAT boot's
    # PUBLISHED line").  `leg_geometry` is the ONE reader of the ARM's priced
    # decision, so asking it is asking the authority rather than re-deriving.
    # (The first version of this assertion hard-coded (DEPTH+1)*SLOT_BYTES and
    # went red against a correct 12,000,000 B file, which is the denominator
    # mistake this comment exists to prevent.)
    slot_bytes, depth = bx.leg_geometry(terms)
    expect = (int(depth) + 1) * int(slot_bytes)
    assert os.path.getsize(path) == expect, (
        f"bounce.bin is {os.path.getsize(path)} B, expected "
        f"(depth+1)*slot_bytes = {expect} from the PUBLISHED terms "
        f"(slot_bytes={slot_bytes}, depth={depth})"
    )


def test_the_chain_declines_on_stock_without_reaching_the_transport(chain,
                                                                   caplog,
                                                                   monkeypatch):
    """The same ONE run on the arm that must produce nothing.

    `stock` released no weights, so there is nothing to grade and the chain must
    stop at the grader's gate -- not at the transport, and not with a
    NO-COMPARE line that would read as a failed comparison.
    """
    mgr, boot, root, terms = chain
    caplog.set_level(logging.INFO)

    class _Stock(_ServerArgs):
        enable_memory_saver = False

    monkeypatch.setattr(WU, "_weg2_server_args", lambda self: _Stock(),
                        raising=True)
    assert mgr._weg2_wake_weight_carrier() == WU.CARRIER_STOCK
    mgr._weg2_wake_reload_weights()
    mgr._weg2_xchg_shadow_compare()
    assert "WEG2-XCHG-INJECT" not in caplog.text
    import os
    assert not os.path.exists(bx.bounce_path(boot, shm_root=str(root)))


def test_the_chain_is_silent_on_the_ring_arm(chain, caplog, monkeypatch):
    """The default boot: no exchange armed, so not one byte and not one line."""
    mgr, boot, root, terms = chain
    caplog.set_level(logging.INFO)
    monkeypatch.delenv(wx.WEIGHT_SOURCE_ENV, raising=False)
    assert wx.exchange_armed() is False
    mgr._weg2_wake_reload_weights()
    mgr._weg2_xchg_shadow_compare()
    assert "WEG2-XCHG-INJECT" not in caplog.text
    import os
    assert not os.path.exists(bx.bounce_path(boot, shm_root=str(root)))
