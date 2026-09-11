"""#1347: THE TRIPWIRE FOR THE FOUR ORPHANED #631 TEST MODULES.

WHAT THIS REPLACES, and why a tripwire rather than a skip.

Four test modules could not be COLLECTED on this tree.  They imported symbols
that #969 deleted, so every gate baseline on this branch ended
``rc=3 / VERDICT: INCONCLUSIVE`` -- not because anything was wrong with the
product, but because four modules raised at import and the solo re-run could not
adjudicate them either.  An automated NEW=0/GONE=0 verdict was structurally
impossible while they sat there, and that cost half a verdict cycle on
2026-09-11.  Measured, not asserted: `gate_tier2_partitioned.py:614-616`,
``if inconclusive and rc == 0: rc = 3``.

A ``skipif`` would have been the wrong answer twice over.  #910's lesson is that
a skip can DARKEN a working test; here it darkens nothing -- there is nothing
behind these imports to run -- so it would only make ``rc=3`` permanent under a
friendlier name.  #905's answer is the right one: when a falsifier's
counterfactual can no longer be constructed, the falsifier is replaced by ONE
tripwire that fails the day the mechanism returns and NAMES the arms owed again.

THE RETIRED MECHANISMS, each with the authority that retired it:

* ``phase_flip_resident_carry`` (the whole module, 911 LOC) -- deleted by
  ``069f98c5f3`` *"[#969 CUT K] Delete the resident carry: the cutover is a
  re-entry, not object surgery"*.  That is also standing user design: a flip
  nulls everything and re-admits through HiCache, and object surgery at the
  cutover is the root class of roughly fifteen boot killers.
* ``phase_flip_resident_carry.reseed_decode_input_relay`` -- the only subject of
  the deleted ``test_phase_flip_decode_relay_631.py``.
* ``phase_flip_resident_carry.harvest_resident_batches`` -- drove three tests in
  ``test_phase_flip_spec_seam_631.py``, two of them #905-shape falsifiers.
* ``IN_FLIGHT_CHUNKED_ALLOWANCE`` and the defect-M arming ceiling -- retired in
  the product's own words at ``phase_flip_draft_bootstrap.py:456-461``.
* ``SchedulerPPMixin.pp_flip_drain_tensor_dicts`` -- the kind-blind drain that
  ate an owed output (PP1, 07:33:30Z).  ``docs/dev/631/HANDOFF_656.md:1348``
  says *"Do not re-enable ``pp_flip_drain_tensor_dicts`` as written."*  Its
  CORRECTED successor ``pp_flip_drain_leftover_dicts`` is live and already
  covered by six test modules, so retiring the old test lost no coverage.
  NOTE FOR THE RECORD: this was NOT a rename.  The successor's own docstring
  calls it *"CORPSE S DONE CORRECTLY"* and explains that the old one was
  kind-blind and discarded what it took, while the new one demultiplexes first.
  Same family, different contract -- so the old test could not simply be
  re-pointed at the new name.

WHAT THIS FILE IS NOT: a test of the product's behaviour.  It asserts ABSENCE,
which is exactly the shape a tripwire needs -- it costs nothing while the
absence holds and it fires the moment it stops holding.
"""

from __future__ import annotations

import importlib

import pytest


RETIRED_MODULE = "sglang.srt.managers.phase_flip_resident_carry"


def test_the_resident_carry_module_stays_deleted():
    """If this module returns, three retired test bodies are owed again.

    OWED ON RED, by name -- restore from git history and re-point at whatever
    the returning module actually holds:
      * ``test_phase_flip_decode_relay_631.py`` (7 tests over
        ``reseed_decode_input_relay``) -- deleted in #1347, recoverable at
        ``04cd920add``.
      * ``test_phase_flip_spec_seam_631.py`` -- the three
        ``harvest_resident_batches`` tests, including the two falsifiers that
        pinned the seam's reach APART from the harvest.  The surviving seven
        tests in that file need no change.
      * ``test_phase_flip_draft_bootstrap_631.py`` -- the arming-ceiling refusal
        and the two-ceilings-agree pin.

    AND THE DESIGN QUESTION THAT COMES BACK WITH IT: object surgery at the
    cutover is what #969 CUT K removed on purpose.  A returning resident carry
    is a design decision to re-open, not a test to fix.
    """
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(RETIRED_MODULE)


def test_the_kind_blind_drain_stays_gone_from_the_mixin():
    """``pp_flip_drain_tensor_dicts`` must not return to ``SchedulerPPMixin``.

    OWED ON RED: ``test_pp_proxy_stamp_631.py`` (deleted in #1347, recoverable
    at ``04cd920add``) pinned the stamp-based void-proxy decision against this
    drain.  If the drain returns, that file's five ``pp_flip_drain_tensor_dicts``
    assertions are owed again -- and before restoring them, read
    ``HANDOFF_656.md:1348``, which forbids re-enabling it AS WRITTEN because it
    is kind-blind and ate an output that was still owed.

    The live contract is the successor's, and it is asserted here too so this
    test cannot pass by the whole family having disappeared.
    """
    from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin

    assert not hasattr(SchedulerPPMixin, "pp_flip_drain_tensor_dicts"), (
        "the kind-blind drain is back; see HANDOFF_656.md:1348 before using it"
    )
    # THE OTHER HALF OF THE TRIPWIRE: absence is only meaningful while the
    # successor is present.  If both vanished, the drain-at-disarm promise would
    # be unheld and this file would have gone quietly green on nothing.
    assert hasattr(SchedulerPPMixin, "pp_flip_drain_leftover_dicts"), (
        "the SUCCESSOR is gone too -- the drain-at-disarm promise (#757) is "
        "now unheld, and six test modules that cover it are about to fail"
    )


def test_the_arming_ceiling_stays_retired():
    """No second copy of the resident ceiling on the draft arming leg.

    OWED ON RED: the two tests #1347 removed from
    ``test_phase_flip_draft_bootstrap_631.py`` -- the cap refusal and the
    two-ceilings-agree pin -- plus a decision about WHICH bound is authoritative,
    because two guards asserting two different bounds is the defect #682 fixed
    with a longer fuse.

    Asserted on the module rather than on a string, so a renamed constant does
    not slip past: what must stay absent is a public allowance constant on the
    arming module.
    """
    bootstrap = importlib.import_module(
        "sglang.srt.managers.phase_flip_draft_bootstrap"
    )
    assert not hasattr(bootstrap, "IN_FLIGHT_CHUNKED_ALLOWANCE"), (
        "the in-flight-chunked allowance is back on the arming leg; "
        "phase_flip_draft_bootstrap.py:456-461 says why it was retired"
    )
    # The arming entry point itself must still exist -- otherwise this file is
    # asserting the absence of a ceiling on a module that no longer arms, which
    # would be green-by-vacancy.
    assert hasattr(bootstrap, "arm_draft_bootstrap")
