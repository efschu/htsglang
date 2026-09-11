# SPDX-License-Identifier: Apache-2.0
"""#1333 -- ONE authority for the per-card carrier size, and the prose that lied.

THE QUESTION, as the operator filed it: ``diagonal_carrier_bytes`` versus the
bounce lane's inline sizing are TWO authorities for ONE size, and plan
``PLAN_S6_BOUNCE_0911.md`` says *"one reader; no second ledger"*.  The
upstream-minimal law makes DELETE the default and puts the burden of proof on
the repair, so this file carries the proof -- at file:line, against the tree at
``04cd920add`` (RE-STAMP 8) -- that there were THREE numbers and not two, that
only one of them is read by the product, and that the deleted one stated an
invariant the code does not hold.

THE THREE NUMBERS, measured:

  1. ``xchg_bounce.staging_bytes_per_card(slot)`` = ``SLOTS_PER_PAIR x slot``
     (``xchg_bounce.py:255``).  THE CHARGE.  Bound to ``BounceTerms``
     (``:102-109``) -> ``bounce_terms()`` (``:190``) -> the ARM line's
     ``path_a_staging_mib=`` (``:309``), and read by the rank that enforces it
     (``weight_updater.py:2495``, whose docstring is titled "ONE READER OF ONE
     NUMBER").  This is the authority.
  2. ``weight_exchange_transport.ONCARD_DEPOSIT_BYTES_MAX`` =
     ``ONCARD_SLOTS_MAX (8) x ONCARD_SLOT_BYTES_MAX``.  THE DEPOSIT'S SHAPE
     MAXIMUM.  A DIFFERENT number on purpose -- ``test_weg2_xchg_transport_1273``
     says so in as many words ("deliberately different numbers now: 8 x slot vs
     SLOTS_PER_PAIR x slot") -- and it bounds the ROW AREA, not the charge.
  3. ``weight_exchange_transport.diagonal_carrier_bytes(slot_bytes=)`` =
     ``xr.SLOTS_PER_PAIR x slot``.  THE SECOND BOOKKEEPING.  Numerically (1),
     computed in another module off another ``SLOTS_PER_PAIR`` literal
     (``weight_exchange_region.py:132`` vs ``xchg_bounce.py:57``), with **ZERO
     production call sites** -- at the pin its only references are its own
     definition, its own name inside ``oncard_host_path``'s docstring, and three
     test calls.

WHY DELETING (3) IS NOT MERELY TIDYING -- IT NAMED A SIZE NOTHING GUARANTEES.
Its docstring called itself *"the per-card diagonal carrier's size"*.  The
carrier the product actually allocates is ``plan.slots x plan.slot_bytes``
(``HostBounce.__init__``/``OnCardBounce.__init__``, reached from
``weight_exchange_transport.py:3187``/``:3295`` with ``diag_slots`` from
``require_oncard_slots``, bounded by ``ONCARD_SLOTS_MAX = 8``), and under
store-and-forward ``plan.slots`` IS ``batches``, returned UNCLAMPED on purpose
so ``deposit_refusal_reason`` can name it (``plan_oncard_slot_bytes`` docstring,
``:2095-2107``).  So the carrier is ``2 x slot`` only in the case the budget
happens to fund; a function asserting it is always ``2 x slot`` was restating
the CHARGE while claiming to size the CARRIER.  That is the same one-payload-
two-sizes shape AMENDMENT 5 deleted ``host_ledger.xchg_bounce_bytes`` and
``host_ledger.xchg_bounce_bytes_per_card`` for -- see the comment still standing
at ``weight_updater.py:2485``: *"Both functions are now deleted and the
arithmetic has a single owner"*.  (3) survived that cleanup as a leftover.

AND THE PROSE WAS WRONG ABOUT ITS OWN REFUSAL, in three places at the pin:
``diagonal_carrier_bytes`` (transport), ``staging_bytes_per_card``
(xchg_bounce:266-268) and the comment at ``weight_updater.py:2497-2498`` all
said *"a plan needing more than SLOTS_PER_PAIR batches is refused BY NAME
(DEPOSIT_REASON_BATCHES)"*.  ``deposit_refusal_reason`` refuses ``batches`` only
against ``slots_max``, whose default is ``ONCARD_SLOTS_MAX = 8``, and the ONE
production caller (``weight_exchange_shadow.py:1986``) does not pass it.  On the
product's own path a 3-batch deposit is refused as ``DEPOSIT_REASON_UNFUNDED``.
The only test that showed otherwise passed ``slots_max=xr.SLOTS_PER_PAIR``
explicitly -- a claim made true by injecting a value the product never supplies,
which is the sibling of this strand's "a double shaped after the consumer".
Both refusals are correct and both are BY NAME; what was wrong is WHICH lever a
reader is sent to: ``ONCARD_SLOTS_MAX`` is a transport row-area knob, while the
lever that actually fires is the ledger charge.
"""

from __future__ import annotations

import ast
import inspect
import os
from pathlib import Path

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import host_ledger as hl  # noqa: E402
from sglang.srt.weg2 import weight_exchange_region as xr  # noqa: E402
from sglang.srt.weg2 import weight_exchange_transport as tp  # noqa: E402
from sglang.srt.weg2 import xchg_bounce as xb  # noqa: E402

WEG2_DIR = Path(xb.__file__).resolve().parent

#: The ONE owner of the PER-CARD ``SLOTS_PER_PAIR x slot_bytes``.
CARRIER_SIZE_OWNER = "staging_bytes_per_card"

#: EVERY function allowed to multiply ``SLOTS_PER_PAIR`` by a byte count, each
#: with the reason it is not a second authority.  An allowlist and not a
#: threshold, so a THIRD site is a FAILURE and not a warning: the previous two
#: sightings of this class (``host_ledger.xchg_bounce_bytes`` and
#: ``...bytes_per_card``) were each found only after a boot had been priced
#: against the wrong half.
CARRIER_SIZE_ALLOWED = {
    # THE owner: one card's share, the number a rank enforces.
    "staging_bytes_per_card": "the per-card charge itself",
    # The GROUP-WIDE term, which is ``pairs x`` the per-card one.  Same
    # authority one level up, in the one dataclass the launcher and the rank
    # both rebuild from published scalars (``ENV_BOUNCE_TERMS``), so it is the
    # container of the number rather than a second derivation of it.
    "bounce_terms": "pairs x the per-card charge, inside BounceTerms",
}


def _function(path: Path, name: str) -> ast.FunctionDef:
    """The named function's AST node, so a pin can ask what the code DOES."""
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {path}")

# ===========================================================================
# (1) THE DELETION, asserted as an ABSENCE rather than assumed.
# ===========================================================================


def test_the_second_carrier_authority_is_GONE():
    """RED at 04cd920add: ``diagonal_carrier_bytes`` is still defined.

    Absence is asserted the way B4i asserted ``residual_unattributed_mib``'s:
    on the MODULE, so a re-introduction under the same name fails here rather
    than on the next boot's ledger line.
    """
    assert not hasattr(tp, "diagonal_carrier_bytes"), (
        "the second authority is back; the per-card carrier charge has exactly "
        f"one owner, {CARRIER_SIZE_OWNER}"
    )
    # ...and no DEFINITION or CALL survives in the module.  Deliberately not a
    # ban on the STRING: the deletion leaves a tombstone comment naming what
    # went and why, which is how this tree records a removal, and a test that
    # forbade the name outright would forbid the explanation with it.  (My
    # first version did exactly that and went red on my own comment.)
    # Asked of the AST and not of the text, so the tombstone may quote the old
    # SIGNATURE without counting as a call -- a grep for ``name(`` cannot tell
    # those apart, and the looser grep is what a prose ban degrades into.
    tree = ast.parse((WEG2_DIR / "weight_exchange_transport.py").read_text())
    assert not [n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef)
                and n.name == "diagonal_carrier_bytes"], "definition survives"
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "id", getattr(n.func, "attr", None))
             == "diagonal_carrier_bytes"]
    assert not calls, f"a call survives at line(s) {[c.lineno for c in calls]}"


def test_the_surviving_authority_is_the_one_the_product_reads():
    """The positive half: the owner exists, and its reader names it.

    Deleting the unread copy is only correct if the READ one is the survivor.
    The rank's enforced budget and the launcher's charged term must be the same
    call, which is what ``weight_updater``'s own docstring claims -- pinned here
    against the source so the claim cannot rot into prose.
    """
    assert callable(getattr(xb, CARRIER_SIZE_OWNER))
    slot = 128 * xb.MIB
    assert xb.staging_bytes_per_card(slot) == xb.SLOTS_PER_PAIR * slot
    # The charge is bound to the ARM's own terms object, not computed beside it.
    terms = xb.bounce_terms(bytes_per_direction=27_120 * xb.MIB, n_layers=64,
                            widest_layer_bytes=756_323_776, pairs=xr.N_CARDS,
                            slot_bytes=slot)
    assert terms.staging_per_card == xb.staging_bytes_per_card(slot)
    assert terms.staging_bytes == xr.N_CARDS * xb.staging_bytes_per_card(slot)
    # The rank-side budget reader calls it by name (source pin, #1273 S6 form).
    wu = (Path(hl.__file__).resolve().parents[1] / "managers"
          / "scheduler_components" / "weight_updater.py")
    fn = _function(wu, "_weg2_shadow_host_budget")
    # WHAT REACHES THE RETURN, not what the body MENTIONS.  MUTANT M8 -- the
    # rank enforcing the RETIRED `ONCARD_DEPOSIT_BYTES_MAX` (8 x slot), i.e.
    # literally the 4x disagreement AMENDMENT 5 retired -- SURVIVED a substring
    # pin, because the method's own comment still names the owner while
    # explaining the retirement.  Same lesson as seat 5's M9 and B4f's M5: a pin
    # must name what reached the site.
    # EVERY callee inside a return expression, not just the outermost one --
    # the real return is ``int(xb.staging_bytes_per_card(...))``, so a pin that
    # read only the top node saw ``int`` and would have passed on anything.
    called = {
        (n.func.attr if isinstance(n.func, ast.Attribute)
         else getattr(n.func, "id", None))
        for r in ast.walk(fn)
        if isinstance(r, ast.Return) and r.value is not None
        for n in ast.walk(r.value) if isinstance(n, ast.Call)
    }
    assert CARRIER_SIZE_OWNER in called, called
    names = {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)}
    assert "ONCARD_DEPOSIT_BYTES_MAX" not in names, (
        "the rank is back on the retired 8x ceiling while the ledger charges "
        "SLOTS_PER_PAIR x slot -- the 4x disagreement on one payload"
    )
    assert "diagonal_carrier_bytes" not in {*names, *called}


def test_no_SECOND_function_in_weg2_computes_the_carrier_size():
    """THE RATCHET, and it is an AST question with an allowlist of ONE.

    Two deletions of this exact class have already been paid for.  A ratchet is
    the only thing that makes the third one impossible rather than unlikely:
    any function under ``srt/weg2`` whose body returns ``SLOTS_PER_PAIR x <x>``
    must be the owner.
    """
    offenders: set[str] = set()
    seen: set[str] = set()
    for path in sorted(WEG2_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            for mul in (n for n in ast.walk(node)
                        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mult)):
                names = {
                    (o.attr if isinstance(o, ast.Attribute) else o.id)
                    for o in ast.walk(mul)
                    if isinstance(o, (ast.Name, ast.Attribute))
                }
                # A BYTE product, not an index one: ``slot_index``'s
                # ``pair * SLOTS_PER_PAIR + slot`` addresses a row and is not a
                # size, so the ratchet asks for a SIZE and skips it.
                #
                # MEASURED DEFECT IN MY OWN FIRST VERSION, kept as the comment
                # it earned: filtering on a ``bytes``-named OPERAND alone let
                # ``diagonal_carrier_bytes`` through, because its product is
                # ``SLOTS_PER_PAIR * validated * MIB`` and none of those three
                # names contains "bytes" -- a ratchet that could not see the one
                # site it was written for.  The unit shows up EITHER in an
                # operand or in the function's own name, so both count.
                if "SLOTS_PER_PAIR" not in names:
                    continue
                unit = any(("bytes" in n.lower() or n == "MIB") for n in names)
                if not (unit or "bytes" in node.name.lower()):
                    continue
                seen.add(node.name)
                if node.name not in CARRIER_SIZE_ALLOWED:
                    offenders.add(f"{path.name}:{node.lineno} {node.name}")
    assert not offenders, (
        f"a second owner of SLOTS_PER_PAIR x <bytes>: {sorted(offenders)}; "
        f"allowed: {sorted(CARRIER_SIZE_ALLOWED)}"
    )
    # THE RATCHET MUST BE ABLE TO SEE: an allowlist that matched nothing would
    # pass by finding nothing, which is the #1300 shape one tool over.
    assert seen == set(CARRIER_SIZE_ALLOWED), (seen, set(CARRIER_SIZE_ALLOWED))


# ===========================================================================
# (2) THE PROSE, corrected against what the code does.
# ===========================================================================


def test_the_bounce_file_docstring_points_at_the_SURVIVING_owner():
    """RED at the pin: it points at ``diagonal_carrier_bytes``.

    A cross-reference to a deleted function is the dangling half of the same
    defect -- it sends the next reader to a size nobody computes.
    """
    doc = tp.oncard_host_path.__doc__ or ""
    assert "diagonal_carrier_bytes" not in doc, doc
    assert CARRIER_SIZE_OWNER in doc, doc


def test_every_host_ledger_reference_in_the_refusal_docstring_RESOLVES():
    """RED at the pin: it cites ``host_ledger.xchg_bounce_bytes_per_card``.

    That function was deleted by AMENDMENT 5 (``weight_updater.py:2485`` says
    so), and the citation survived pointing at nothing.  Generalised on purpose:
    EVERY ``host_ledger.<name>`` named in this docstring must exist, so the next
    deletion cannot leave a second dangling pointer.
    """
    doc = tp.deposit_refusal_reason.__doc__ or ""
    cited = {
        tok.split("host_ledger.")[1].split("`")[0].strip("(): ")
        for tok in doc.split()
        if "host_ledger." in tok
    }
    missing = sorted(n for n in cited if n and not hasattr(hl, n))
    assert not missing, f"dangling host_ledger references: {missing}"


def test_a_three_batch_deposit_is_refused_as_UNFUNDED_on_the_PRODUCT_default():
    """The corrected claim, graded WITHOUT injecting ``slots_max``.

    This is the replacement for
    ``test_more_batches_than_slots_is_refused_BY_NAME_and_never_sized_up``,
    which asserted ``DEPOSIT_REASON_BATCHES`` while passing
    ``slots_max=xr.SLOTS_PER_PAIR`` -- a value the ONE production caller
    (``weight_exchange_shadow.py:1986``) does not pass.  Renamed rather than
    edited in place so the reversal reads as one GONE and one NEW with its
    argument attached.

    Both refusals are BY NAME and the deposit is never sized up either way; the
    correction is WHICH ONE, because the two name different levers.
    """
    slot = 128 * xb.MIB
    budget = xb.staging_bytes_per_card(slot)
    graded = dict(slot_bytes=slot, budget_bytes=budget,
                  mode=tp.ONCARD_MODE_HOST)
    # Two batches fit the charge exactly.
    assert tp.deposit_refusal_reason(batches=2, slots=2, **graded) == ""
    # Three do not -- and on the product's default the word is UNFUNDED.
    assert tp.deposit_refusal_reason(batches=3, slots=3, **graded) == \
        tp.DEPOSIT_REASON_UNFUNDED
    # BATCHES is reachable, but only above the ROW AREA's own max, which is the
    # lever that word actually names.
    assert tp.deposit_refusal_reason(
        batches=tp.ONCARD_SLOTS_MAX + 1, slots=tp.ONCARD_SLOTS_MAX + 1,
        **graded) == tp.DEPOSIT_REASON_BATCHES
    # The production caller does NOT narrow slots_max -- the premise of the
    # whole correction, pinned at the source so it cannot drift silently.
    # ASKED OF THE AST.  MUTANT M6 -- the production caller narrowing
    # `slots_max` back to SLOTS_PER_PAIR -- SURVIVED my first version, which
    # took the call text with `split(")")[0]` and so stopped at the `)` inside
    # `int(host_bounce_budget_bytes)`, before the argument it was looking for.
    # A nested-paren call cannot be read by a text split; this is the same
    # extractor-blindness class as #1342's caller-detection regex.
    shadow = WEG2_DIR / "weight_exchange_shadow.py"
    calls = [
        n for n in ast.walk(ast.parse(shadow.read_text()))
        if isinstance(n, ast.Call)
        and getattr(n.func, "attr", getattr(n.func, "id", None))
        == "deposit_refusal_reason"
    ]
    assert calls, "the production caller is gone; the premise needs re-checking"
    for call in calls:
        kw = {k.arg for k in call.keywords}
        assert "slots_max" not in kw, (
            f"the production caller at line {call.lineno} narrows slots_max "
            f"again; the charge, not the row area, is what binds here"
        )


@pytest.mark.parametrize("owner_doc", ["staging_bytes_per_card"])
def test_the_owners_own_prose_names_the_refusal_that_fires(owner_doc):
    """The same wrong sentence stood in the SURVIVOR's docstring too.

    Fixing only the deleted copy would have left the misdirection in the one
    place a reader is now sent to.
    """
    doc = " ".join((getattr(xb, owner_doc).__doc__ or "").split())
    # THE WRONG CLAIM, verbatim as it stood at 04cd920add.  Pinned as the
    # SENTENCE and not as the token, because the corrected prose must be free to
    # name ``DEPOSIT_REASON_BATCHES`` in order to say it is NOT the lever here.
    assert "batches per leg is refused BY NAME (``DEPOSIT_REASON_BATCHES``)" \
        not in doc, "the owner still sends the reader to the row-area lever"
    assert "DEPOSIT_REASON_UNFUNDED" in doc, doc
    # And if it mentions the other word at all, it must say which knob that is.
    if "DEPOSIT_REASON_BATCHES" in doc:
        assert "ONCARD_SLOTS_MAX" in doc, doc


def test_the_two_SLOTS_PER_PAIR_literals_are_pinned_equal():
    """The drift the deletion leaves behind, named rather than hoped away.

    ``xchg_bounce`` is deliberately dependency-free (B1: "weg2/xchg_bounce.py
    pure"), so it carries its OWN ``SLOTS_PER_PAIR = 2`` beside
    ``weight_exchange_region``'s.  Purity is the right call and the literal
    stays duplicated; what must not stay unpinned is their EQUALITY, because the
    charge is computed off one and the handshake rows off the other.
    """
    assert xb.SLOTS_PER_PAIR == xr.SLOTS_PER_PAIR == 2
    # And the geometry that reads the region's copy agrees with the charge.
    assert len(xr.all_diagonal_sem_names("s1333")) == \
        xr.N_CARDS * xr.SLOTS_PER_PAIR * 2
    assert inspect.getsource(xb.staging_bytes_per_card).count("SLOTS_PER_PAIR") >= 1
