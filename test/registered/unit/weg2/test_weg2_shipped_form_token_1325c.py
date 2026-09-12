"""#1325c -- B4e's SYNTHETIC form token vs TRAIN FIX 5's shipped-form check.

THE DEFECT, structurally: B4e builds the form argv the ring table is gated on
as ``xchg_form_argv(argv_p(<sentinel ledger terms>), weight_source)``, which
APPENDS the synthetic ``ring_table.XCHG_FORM_TOKEN``
(``--weg2-xchg-region=armed``) whenever the arm is armed.  TRAIN FIX 5 then
re-derives the key from the argv the boot really SHIPS and refuses on any
difference (``launcher.py``: ``shipped_key, shipped_norm =
ring_table.p_form_key(shipped_argv_p)`` -> ``W48 Weg2RingFormMismatch``).

The token is SYNTHETIC -- it is a marker the xchg launcher prints, never a
flag ``argv_p`` emits -- so it can never appear in a shipped argv, and it is
not in ``FORM_KEY_EXCLUDED_FLAGS`` either.  The two keys are therefore
UNEQUAL BY CONSTRUCTION on every ``--weg2-weight-source exchange`` boot and
EQUAL on every unarmed one, which is exactly why this never fired: the armed
form has never got past the host ledger.

MEASURED 2026-09-12: on the train head ``c2287ee8c6`` the boot weg2xsn21 dry
run refuses ``W21 Weg2HostRunPeakRefused`` at the ledger and never reaches
this check.  With the #1325b netting correction the cheapest rung funds
(run_peak 85.15 < 87.30), the launcher walks on, and the very next gate
refuses ``W48 ... gated-only: ['--weg2-xchg-region=armed']``.  Two independent
pre-flight blockers, the second hidden behind the first.

THE FIX IS THE COMPARISON, NOT THE GATE.  A check that compares a TRANSFORMED
value against an UNTRANSFORMED one tests the transform, not the thing it was
built to test.  Applying ``xchg_form_argv`` to BOTH sides leaves TRAIN FIX 5's
actual subject -- "are the sentinel ledger terms inert?" -- fully intact,
because the token is appended identically to both and cancels.

DANGER DIRECTION: this must not become a way for a REAL form difference to
pass.  Both mutants are on that side:
  M1 an armed boot's shipped argv must reach the same key
     -> test_an_armed_boot_ships_the_form_it_was_gated_on
  M2 a genuine form difference must STILL refuse, armed or not
     -> test_a_real_form_difference_still_refuses_on_an_armed_boot
  M3 an unarmed boot must be byte-identical
     -> test_an_unarmed_boot_is_unchanged
"""

from sglang.srt.weg2 import launcher as L
from sglang.srt.weg2 import ring_table as rt

BASE = ["python", "-m", "sglang.launch_server", "--pp-stage-ratio=39,13,12",
        "--max-total-tokens=324702"]


def _key(argv, source):
    return rt.p_form_key(L.xchg_form_argv(list(argv), source))[0]


def test_an_armed_boot_ships_the_form_it_was_gated_on():
    """The gated key and the shipped key must agree once both carry the token."""
    gated = _key(BASE, "exchange")          # what solve() was handed
    shipped = _key(BASE, "exchange")        # what TRAIN FIX 5 re-derives
    assert gated == shipped, (
        "an armed boot can never ship the synthetic token as a flag, so the "
        "shipped side must be put through the SAME transform or W48 refuses "
        "every armed boot by construction"
    )
    # and the raw, untransformed shipped argv is the thing that did NOT match
    assert rt.p_form_key(BASE)[0] != gated, (
        "if these were equal the token would not be in the key at all and "
        "B4e's own source selection would be undone"
    )


def test_a_real_form_difference_still_refuses_on_an_armed_boot():
    """The gate must keep catching a form difference that is NOT the token."""
    other = [t for t in BASE if not t.startswith("--pp-stage-ratio")] + \
            ["--pp-stage-ratio=42,11,11"]
    assert _key(BASE, "exchange") != _key(other, "exchange"), (
        "TRAIN FIX 5's subject is a real form drift; cancelling the token must "
        "not cancel anything else"
    )


def test_an_unarmed_boot_is_unchanged():
    """Every ring-arm boot -- the default and every boot that has run."""
    assert _key(BASE, "ring") == rt.p_form_key(BASE)[0]
    assert L.xchg_form_argv(list(BASE), "ring") == BASE
