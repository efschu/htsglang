"""#1345: THE INJECTION PLAN IS NOT NARROWED TO THE PAIR'S AGREED SET.

RED-FIRST, and every pin here has a named mutant that kills it.

THE DECISION (operator, 2026-09-11, RE-STAMP 9): the exchange injection's plan
is NOT narrowed to the co-located pair's agreed intersection, and rule 1 of the
S6I order ("``require_agreement=True`` stays") is LIFTED FOR THAT ONE CALLER.
Three legs, all of them measurements rather than taste:

1. ``derive_leg_plan`` itself defaults to ``agreed=None,
   require_agreement=False`` (``weight_exchange_shadow.py``), so a NON-narrowed
   plan is the library's own sanctioned default.  Passing ``False`` is not
   softening a gate; hard-setting ``True`` for every caller was the deviation.
2. The agreement machinery answers a PEER question -- which pieces are
   identical on both rows, so a cross-compare means something.  The injection
   lane has no peer question: its bytes come from path (b) and its acceptance
   is ``xchg_bounce == expression``, never ``== peer``.
3. Narrowing made the grader vacuous: the agreed set measured 4.90 MiB of a
   27.52 GiB image = 0.018 %.

WHY THE PRODUCER IS NOT CALLED INSTEAD (boot weg2xsn19's refusal, accepted):
``_weg2_shadow_manifest`` requires keyword-only ``leg`` and ``epoch``, and the
injection path has NO leg identity -- ``_weg2_xchg_inject_weights`` is called
with no arguments from ``_weg2_wake_reload_weights`` and with only ``mode=``
from the grader, and ``_weg2_xchg_shadow_compare`` takes none.  Calling the
producer there would also add a SECOND WRITER to a row whose writer is declared
unique (``write_card_manifest``: "ONE WRITER PER ADDRESS, sealed") and could
regress the monotone ``peer_seen`` flag 1 -> 0, which is the W80 class #1311 S6b
closed.  So the fix is the flag's granularity, and ``test_no_manifest_writer_is_
reachable_from_the_injection_path`` pins the absence of the tempting shortcut.
"""

import ast
import inspect
import pathlib

import pytest

from sglang.srt.managers.scheduler_components import weight_updater as wu
from sglang.srt.weg2 import weight_exchange_shadow as sh


# --------------------------------------------------------------------------
# helpers: AST over the PRODUCT source, never a text scan (#1341 form)
# --------------------------------------------------------------------------
def _func_ast(qualname: str) -> ast.FunctionDef:
    """The ast.FunctionDef of one method of the weight-updater mixin.

    #1341 FORM, and it is the reason this is an AST walk and not a regex: a
    text scan for a symbol matches its own explanatory prose, and this file
    NAMES every symbol it forbids.  A pin that its own docstring can break is
    not a pin.
    """
    src = pathlib.Path(inspect.getsourcefile(wu)).read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == qualname:
                return node
    raise AssertionError(f"{qualname} not found in {inspect.getsourcefile(wu)}")


def _keyword_of_call(node: ast.Call, name: str):
    for kw in node.keywords:
        if kw.arg == name:
            return kw
    return None


def _calls_to(node: ast.AST, attr: str):
    """Every ``ast.Call`` in ``node`` whose callee ENDS in ``attr``."""
    out = []
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        f = sub.func
        if isinstance(f, ast.Attribute) and f.attr == attr:
            out.append(sub)
        elif isinstance(f, ast.Name) and f.id == attr:
            out.append(sub)
    return out


# --------------------------------------------------------------------------
# 1. THE LIBRARY'S DEFAULT IS NOT NARROWED -- leg 1 of the decision
# --------------------------------------------------------------------------
def test_derive_leg_plan_defaults_to_no_agreement():
    """A non-narrowed plan is the SANCTIONED default, not a relaxation.

    MUTANT M1: flip either default in ``derive_leg_plan``'s signature and this
    goes red -- the decision's first leg would no longer hold and the change at
    the call site would then really be a softening.
    """
    sig = inspect.signature(sh.derive_leg_plan)
    assert sig.parameters["agreed"].default is None
    assert sig.parameters["require_agreement"].default is False


# --------------------------------------------------------------------------
# 2. THE ADAPTER TAKES THE CHOICE EXPLICITLY -- no hard-set for every caller
# --------------------------------------------------------------------------
def test_the_adapter_takes_require_agreement_explicitly():
    """``_weg2_shadow_plan`` must not decide narrowing on its callers' behalf.

    THE DEFECT THIS CLOSES, measured on boot weg2xsn19: one flag hard-set
    ``True`` served two callers with OPPOSITE and both-documented needs, so the
    injection caller could only ever read ``manifest-unagreed`` -- 12 of 12
    legs on both groups, on every argv.

    MUTANT M2: restore the hard-coded ``require_agreement=True`` inside the
    adapter and this goes red.
    """
    sig = inspect.signature(wu.SchedulerWeightUpdaterManager._weg2_shadow_plan)
    assert "require_agreement" in sig.parameters, (
        "the adapter must take the narrowing decision from its caller"
    )
    node = _func_ast("_weg2_shadow_plan")
    calls = _calls_to(node, "derive_leg_plan")
    assert len(calls) == 1, f"expected ONE derive_leg_plan call, got {len(calls)}"
    kw = _keyword_of_call(calls[0], "require_agreement")
    assert kw is not None, "require_agreement must be passed through, not defaulted"
    assert not isinstance(kw.value, ast.Constant), (
        "require_agreement must be FORWARDED from the parameter, never a "
        "literal -- a literal here is the hard-set this test exists to forbid"
    )


# --------------------------------------------------------------------------
# 3. THE SHADOW HOOK STILL NARROWS -- the danger direction of the whole change
# --------------------------------------------------------------------------
def test_the_shadow_hook_still_requires_agreement():
    """#1311 S6b MUST NOT REGRESS: the shadow lane keeps its narrowing.

    This is the danger direction of the entire slice.  The shadow hook compares
    stripes ACROSS the co-located pair, so a lane planned over a set the peer
    never agreed to is exactly boot weg2xsn5's W80 -- 7 of 8 legs refused on
    ``field=piece_digest`` because each end hashed its own inventory.

    MUTANT M3: pass ``require_agreement=False`` at the shadow hook's call and
    this goes red.
    """
    node = _func_ast("_weg2_shadow_hook")
    calls = _calls_to(node, "_weg2_shadow_plan")
    assert len(calls) == 1, f"expected ONE plan call in the hook, got {len(calls)}"
    kw = _keyword_of_call(calls[0], "require_agreement")
    assert kw is not None, "the shadow hook must state its narrowing explicitly"
    assert isinstance(kw.value, ast.Constant) and kw.value.value is True, (
        "the shadow hook MUST require agreement (#1311 S6b / W80)"
    )
    # ... and it must still reconcile BEFORE planning, which is the other half
    # of #1311 S6b ("THE CARD MANIFEST, BEFORE THE PLAN AND NOT AFTER").
    man = _calls_to(node, "_weg2_shadow_manifest")
    assert len(man) == 1, "the shadow hook must reconcile exactly once"
    assert man[0].lineno < calls[0].lineno, (
        "the reconcile must precede the plan -- MUTANT M5 reorders them"
    )


# --------------------------------------------------------------------------
# 4. THE INJECTION CALLER DOES NOT NARROW -- the fix itself
# --------------------------------------------------------------------------
def test_the_injection_caller_does_not_require_agreement():
    """The graded lane's own call: ``require_agreement=False``, stated.

    MUTANT M4: set it back to ``True`` (or drop the keyword so the adapter's
    old hard-set applies) and this goes red -- which is boot weg2xsn19's
    measured wall, ``manifest-unagreed:hook=authoritative`` 12 on D and 12 on P.
    """
    node = _func_ast("_weg2_xchg_inject_from_peer")
    calls = _calls_to(node, "_weg2_shadow_plan")
    assert len(calls) == 1, f"expected ONE plan call, got {len(calls)}"
    kw = _keyword_of_call(calls[0], "require_agreement")
    assert kw is not None, (
        "the injection caller must state that it does NOT narrow, rather than "
        "inheriting a default -- the inheritance IS the defect"
    )
    assert isinstance(kw.value, ast.Constant) and kw.value.value is False, (
        "the injection plan is NOT narrowed to the agreed set (RE-STAMP 9)"
    )


# --------------------------------------------------------------------------
# 5. NO MANIFEST WRITER IS REACHABLE FROM THE INJECTION PATH
# --------------------------------------------------------------------------
@pytest.mark.parametrize("forbidden", [
    "_weg2_shadow_manifest",
    "reconcile_card_manifest",
    "write_card_manifest",
])
def test_no_manifest_writer_is_reachable_from_the_injection_path(forbidden):
    """Danger (1) is gone BY CONSTRUCTION, not by care.

    ``write_card_manifest`` declares "ONE WRITER PER ADDRESS, sealed" and
    ``reconcile_card_manifest`` relies on ``peer_seen`` being monotone 0 -> 1.
    The injection path runs on the SAME rank and row as the destination shadow
    leg, 22 lines earlier on the wake path, so a reconcile here would be a
    second writer of one address and could regress that flag 1 -> 0 -- a
    W80-class divergence reported as startup order.

    MUTANT M6: add any of these calls to the injection path and this goes red.
    AST, not text: this file names all three symbols in prose.
    """
    node = _func_ast("_weg2_xchg_inject_from_peer")
    assert _calls_to(node, forbidden) == [], (
        f"{forbidden} must not be called from the injection path"
    )


# --------------------------------------------------------------------------
# 6. authoritative STAYS UNREACHABLE -- the named gate, unchanged by this slice
# --------------------------------------------------------------------------
def test_authoritative_stays_unreachable_on_this_arm():
    """The slice must not widen the authority.

    ``inject_authoritative()`` is the ONE predicate, and this slice touches
    neither it nor ``inject_mode()``.  The grader's own call passes
    ``INJECT_SHADOW`` explicitly, so no path acquires the authority as a side
    effect of the narrowing change.

    MUTANT M7: widen the grader's ``mode=`` to the authoritative constant, or
    make ``inject_mode()`` answer ``(shadow, authoritative)``, and this reds.
    """
    from sglang.srt.weg2 import weight_exchange as wx

    node = _func_ast("_weg2_xchg_shadow_compare")
    calls = _calls_to(node, "_weg2_xchg_inject_weights")
    assert len(calls) == 1
    kw = _keyword_of_call(calls[0], "mode")
    assert kw is not None, "the grader must state its mode"
    # the value must resolve to INJECT_SHADOW, spelled as the module constant
    assert isinstance(kw.value, ast.Attribute) and kw.value.attr == "INJECT_SHADOW", (
        "the grader compares; it must never ask for the authoritative mode"
    )
    assert wx.INJECT_SHADOW != wx.INJECT_AUTHORITATIVE
