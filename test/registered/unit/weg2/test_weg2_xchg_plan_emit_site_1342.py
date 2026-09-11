"""#1342 S2b: the acceptance line is emitted where the XchgPlan LIVES.

WHY THIS FILE EXISTS -- boot weg2xsn18 printed, 21x on D and 18x on P:

    WEG2-XCHG-PLAN emit failed: AttributeError: 'LegPlan' object has no
      attribute 'log_line' @ weight_exchange.py:1241

I had wired `emit_plan_line` into `_weg2_shadow_plan`, which returns
`derive_leg_plan`'s **LegPlan**, while `emit_plan_line` -> `XchgPlan.log_line`
needs an **XchgPlan**.  Census (d)'s `plan_id` therefore read 0 -- my
instrument failing, not the lane.

THE ROOT IS THE DOUBLE, NOT THE TYPE, and that is why this file leads with the
producer.  My hermetic test monkeypatched `_weg2_shadow_plan` with a `_FakePlan`
carrying the **CONSUMER's** field set (`raw_descs`, `plan_id`, `src_group`...),
so the double satisfied the emitter and misrepresented the producer.  A fake
built from the consumer's shape CANNOT, by construction, catch a
producer/consumer mismatch -- and this was the SECOND instance in one slice (the
first: a `BounceResult` handed to something wanting `InjectVerdict`).

WHY NOT A CONVERSION, AND WHY NOT A SHARED `log_line` -- measured, not asserted.
The two dataclasses share exactly ONE field, `descs`:

    XchgPlan : byte_matrix descs dst_group plan_id raw_descs skipped_tags
               src_group tag_bytes waves
    LegPlan  : agreed_* card card_digest derive_ms descs facts manifest
               planned population tags unagreed_* undescribed unplanned*
    shared   : descs

`XchgPlan.log_line` reads `waves`, `raw_descs`, `descs`, `plan_id`,
`oncard_bytes`, `cross_bytes`, `zerofill_bytes`, `min_piece_bytes`.  A LegPlan
has NONE of those but `descs`, and in particular **no plan_id at all**.  So a
conversion would have to INVENT a plan identity, and giving LegPlan its own
`log_line` would print a `plan_id` it does not have.  Both fabricate the one
field census (d) exists to read.  Rejected on that ground, not on taste.

THE ACTUAL FIX: an XchgPlan DOES exist in production -- `derive_leg_plan` builds
one at `weight_exchange_shadow.py:3282` and then narrows it into a LegPlan.  My
wiring was one frame too late.  The line is emitted where the object lives.

AND THE TWO `WEG2-XCHG-PLAN` SHAPES STAY DISTINCT, deliberately: `card=` is the
shadow's own per-card line off the LegPlan (healthy, 18 P / 21 D on weg2xsn18)
and `dir=`/`plan_id=` is the acceptance line off the XchgPlan.  The module's own
comment at weight_exchange_shadow.py:97 already drew that distinction.
"""

import dataclasses
import inspect

from sglang.srt.weg2 import weight_exchange as wx
from sglang.srt.weg2 import weight_exchange_shadow as sh


# ===========================================================================
# THE FALSIFIER, BUILT FROM THE PRODUCER.  This is the half that was missing.
# ===========================================================================


def test_the_emitter_is_driven_by_the_producers_real_type():
    """A REAL XchgPlan from build_plan's own constructor reaches log_line.

    Built from `dataclasses.fields(XchgPlan)` rather than from a hand-written
    literal, so the day the producer grows a field this test constructs it too
    instead of quietly drifting -- which is the defect this file is about.
    """
    names = {f.name for f in dataclasses.fields(wx.XchgPlan)}
    # every field the emitter reads must exist on the producer's type
    for needed in ("waves", "raw_descs", "descs", "plan_id"):
        assert needed in names, f"XchgPlan lost {needed}, which log_line reads"
    plan = wx.XchgPlan(
        descs=(), raw_descs=(), waves=(("weights_0",),),
        byte_matrix=((0,),), plan_id="0xfeedface",
        src_group="D", dst_group="P",
    )
    line = wx.emit_plan_line(plan, direction="d2h",
                             logger=_CollectingLogger())
    assert line.startswith("WEG2-XCHG-PLAN ")
    assert "plan_id=0xfeedface" in line
    assert "dir=d2h" in line


class _CollectingLogger:
    def __init__(self):
        self.lines = []

    def info(self, fmt, *a):
        self.lines.append(fmt % a if a else fmt)


def test_a_legplan_is_not_an_xchgplan_and_never_will_be():
    """THE MEASUREMENT that rejects both conversion and a shared log_line.

    Pinned as a test so a future reader cannot re-propose either without
    tripping over the field sets.  If LegPlan ever genuinely grows `plan_id`
    and the rest, this test fails and the decision gets re-made ON PURPOSE
    rather than by accident.
    """
    xf = {f.name for f in dataclasses.fields(wx.XchgPlan)}
    lf = {f.name for f in dataclasses.fields(sh.LegPlan)}
    assert xf & lf == {"descs"}, (
        f"the two plan types now share more than `descs`: {sorted(xf & lf)} -- "
        f"re-decide the conversion question deliberately"
    )
    assert "plan_id" not in lf, (
        "LegPlan grew a plan_id; the 'a conversion would have to invent the "
        "identity' argument no longer holds and must be revisited"
    )
    assert not hasattr(sh.LegPlan, "log_line"), (
        "LegPlan grew log_line: it would print a plan_id it does not have"
    )


def test_the_legplan_keeps_its_own_distinct_line():
    """`card=` is the shadow's line and must not be confused with `dir=`.

    Both are `WEG2-XCHG-PLAN`; they answer different questions and weg2xsn18
    showed the `card=` one working (18 P / 21 D) while `dir=` was absent.  A
    grader counting the prefix alone would have read that as healthy.
    """
    assert sh.PLAN_LINE_PREFIX == "WEG2-XCHG-PLAN"
    src = inspect.getsource(sh.LegPlan)
    assert "card={self.card}" in src.replace("f\"", "\"")


# ===========================================================================
# THE SITE.  Structural, because build_plan needs a live model inventory.
# ===========================================================================


def test_the_acceptance_line_is_emitted_where_build_plan_runs():
    """RED on 34cc29d8e3: nothing emits it at the XchgPlan's own frame."""
    src = inspect.getsource(sh.derive_leg_plan)
    assert "emit_plan_line" in src, (
        "derive_leg_plan builds an XchgPlan and never emits its acceptance line"
    )


def test_the_emit_is_after_the_plan_exists_and_cannot_kill_the_derivation():
    """Ordering and blast radius, both pinned.

    The emit must come AFTER `build_plan` returns, and the derivation must
    still never raise -- an instrument that took a leg down would be the
    observer taking authority, which this module's own contract forbids.
    """
    src = inspect.getsource(sh.derive_leg_plan)
    i_build = src.index("wx.build_plan(")
    i_emit = src.index("emit_plan_line(")
    assert i_emit > i_build, "the line is emitted before the plan exists"

    # THE WRAPPING IS CHECKED ON THE AST, NOT BY CHARACTER DISTANCE.  The first
    # version of this assertion scanned a +-400 character window for
    # `except BaseException` and went red the moment the explaining comment grew
    # past the window -- a test that measures the length of a comment instead of
    # the structure of the code.  `ast` answers the actual question: is the emit
    # call INSIDE a Try handler?
    import ast
    import textwrap
    tree = ast.parse(textwrap.dedent(src))
    wrapped = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        if not node.handlers:
            continue
        for sub in ast.walk(ast.Module(body=node.body, type_ignores=[])):
            if (isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Attribute)
                    and sub.func.attr == "emit_plan_line"):
                wrapped = True
    assert wrapped, (
        "the emit is not inside a try/except: a failing instrument could kill "
        "the derivation, and this derivation's contract is that it never raises"
    )


def test_the_wrong_site_no_longer_emits():
    """The site that raised 39 times on weg2xsn18 must not try any more.

    `_weg2_shadow_plan` only ever holds a LegPlan, so an emit there is a
    guaranteed AttributeError. Pinned so the removal is not undone.
    """
    from sglang.srt.managers.scheduler_components import weight_updater as wu
    src = inspect.getsource(
        wu.SchedulerWeightUpdaterManager._weg2_shadow_plan)
    assert "emit_plan_line" not in src, (
        "the LegPlan-only site still tries to emit the XchgPlan line"
    )
