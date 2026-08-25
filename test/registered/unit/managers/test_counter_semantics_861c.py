# SPDX-License-Identifier: Apache-2.0
"""#861c F2: one number was asked two questions. The classification is pinned.

CLASS: **a counter whose semantics are correct for its author's question and
wrong for a second caller's.** Not "a wrong number" -- `uncached_prompt_tokens`
is exactly right for the break-even it was written for. The defect is that a
DIFFERENT decision reused it.

    ECONOMICS  "would PP be cheaper than TP for this backlog?"
               A cached token is read at a layout-independent cost, so it
               cancels on both sides of the inequality.        -> uncached
    EXISTENCE  "must a prefill pass happen somewhere before this
               request can decode?"
               A cached token still needs an extend pass to place its KV in
               the device pool and enter the running batch.    -> raw

SIBLINGS SWEPT: every consumer on the flip-decision and admission paths reads
ONE field, `PhasePolicyInputs.pending_prefill_tokens`, so the sweep is by
DECISION CLASS rather than by line. The existence-class sites found and routed:
`idle` (x2), the bare-truthiness "is there prefill", `starved` (the dwell-floor
bypass), and `Scheduler._layout_admits_prefill` / the TP arm of
`_layout_admits`. Everything else -- break-even N, the price bands, the
LAYOUT-ECONOMY holds -- is economics and deliberately UNCHANGED.

FUTURE-CHECK: `test_existence_class_sites_do_not_read_the_economics_number`
below parses the policy with `ast` and fails if an existence-class site goes
back to the raw field.

THE SPECIMEN, W37-C: six requests queued, every prompt token prefetched into
HiCache, `pending_prefill_tokens=0`, 18 flips, ZERO completions, 0 % GPU,
`avail=468981`.
"""

import ast
import inspect
import types


from sglang.srt.managers.phase_policy import PhasePolicyInputs


def make_inputs(**kw):
    base = dict(phase="tp", pending_prefill_tokens=0, running_bs=0, now=0.0)
    base.update(kw)
    return PhasePolicyInputs(**base)


# ------------------------------------------------------- the two semantics


def test_work_exists_is_true_for_a_fully_cached_backlog():
    """THE SPECIMEN. Economics says 0 (correctly); existence must say yes."""
    inp = make_inputs(pending_prefill_tokens=0, admissible_prefill_tokens=5988)
    assert inp.pending_prefill_tokens == 0, "the economics number stays honest"
    assert inp.work_exists() is True


def test_work_exists_is_false_on_a_genuinely_empty_queue():
    assert make_inputs().work_exists() is False


def test_work_exists_takes_the_max_so_it_can_never_lower_a_verdict():
    """The admission term is additive-only: it can turn 'no work' into 'work',
    never the reverse. An unsupplied field reproduces today's behaviour."""
    assert make_inputs(pending_prefill_tokens=4096).work_exists() is True
    assert (
        make_inputs(pending_prefill_tokens=4096, admissible_prefill_tokens=0)
        .work_exists()
        is True
    )


def test_the_field_defaults_so_every_stand_in_keeps_working():
    inp = make_inputs(pending_prefill_tokens=1)
    assert inp.admissible_prefill_tokens == 0


# ------------------------------------------------- the admission term itself


class FakeReq:
    def __init__(self, n, protected=0):
        self.origin_input_ids = list(range(n))
        self.cache_protected_len = protected


def admissible(queue):
    from sglang.srt.managers.scheduler import Scheduler

    sched = types.SimpleNamespace(waiting_queue=queue)
    return Scheduler._admissible_prefill_tokens(sched)


def test_admission_term_counts_cached_prompts_in_full():
    """`cache_protected_len` is the DEVICE tree prefix (schedule_batch.py:919).
    After the #856 seam drops the tree it is 0, so a re-admitted request reads
    its full prompt -- which is the honest ROW requirement for the read-through
    that must place it back in the device pool."""
    assert admissible([FakeReq(998), FakeReq(1996), FakeReq(2993)]) == 5987


def test_admission_term_subtracts_what_is_already_in_the_device_tree():
    assert admissible([FakeReq(1000, protected=400)]) == 600


def test_admission_term_is_zero_on_an_empty_queue():
    assert admissible([]) == 0


def test_admission_term_never_goes_negative():
    assert admissible([FakeReq(10, protected=99)]) == 0


def test_admission_term_survives_a_request_with_no_ids():
    bad = types.SimpleNamespace(origin_input_ids=None, cache_protected_len=0)
    assert admissible([bad]) == 0


# --------------------------------------------------------- the future-check


ECONOMICS_ONLY_MARKERS = (
    "break-even",
    "price",
    "LAYOUT-ECONOMY",
)


def test_existence_class_sites_do_not_read_the_economics_number():
    """FUTURE-CHECK for the class.

    The four existence-class sites in `phase_policy` must go through
    `work_exists()` or take the explicit max. Parsed with `ast` so that the
    prose in this file and in the module -- which necessarily quotes the
    defect -- cannot read as the defect.
    """
    import sglang.srt.managers.phase_policy as pp

    src = inspect.getsource(pp)
    tree = ast.parse(src)

    # `idle = running_bs == 0 and <X>` -- X must not be a bare comparison of
    # pending_prefill_tokens to 0.
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if "idle" not in targets and "starved" not in targets:
            continue
        seg = ast.get_source_segment(src, node) or ""
        if "work_exists" in seg or "admissible_prefill_tokens" in seg:
            continue
        offenders.append(f"{targets} at line {node.lineno}: {seg.splitlines()[0]}")
    assert not offenders, (
        "existence-class site(s) reading the economics number again:\n"
        + "\n".join(offenders)
    )


def test_work_exists_is_the_single_definition():
    """The defect was two questions sharing one expression; the fix is worth
    nothing if the next consumer re-derives the answer a third way."""
    import sglang.srt.managers.phase_policy as pp

    src = inspect.getsource(pp)
    assert src.count("def work_exists") == 1


def test_can_fail_the_future_check_sees_a_reverted_site():
    """CAN-FAIL: the exact shape the check must catch."""
    tree = ast.parse("idle = inp.running_bs == 0 and inp.pending_prefill_tokens == 0")
    node = tree.body[0]
    seg = "idle = inp.running_bs == 0 and inp.pending_prefill_tokens == 0"
    assert "work_exists" not in seg and "admissible_prefill_tokens" not in seg
    assert isinstance(node, ast.Assign)


# ----------------------------------------------- #861d-2 SYMMETRY PINS
#
# CLASS RULE, added after the THIRD instance in 24h of "a term answers a
# different question than the decision needs" (F2 blind, demand blind, demand
# inverted): every existence/demand term needs BOTH pins -- it must FIRE when
# work exists and be SILENT when none does. One pin alone is how a blind term
# and an inverted term both pass their own tests.


def test_demand_fires_on_the_d2_wedge_specimen():
    """7 queued, 0 running, GPU 0%, no first token for 589s. Nothing is served
    by staying, so the flip must be demanded."""
    inp = make_inputs(
        pending_prefill_tokens=0, admissible_prefill_tokens=5988, running_bs=0
    )
    assert inp.demand_prefill_tokens() == 5988


def test_demand_is_SILENT_on_the_d3_pingpong_specimen():
    """THE INVERSE DEFECT. Same queue, but two requests DECODING: 18 armings
    chopped the bundle mid-flight, epochs 13/14 in six minutes, COMPLETIONS 0.
    Drain, THEN flip -- an arm may not destroy its own justification (W30)."""
    inp = make_inputs(
        pending_prefill_tokens=0, admissible_prefill_tokens=5988, running_bs=2
    )
    assert inp.demand_prefill_tokens() == 0


def test_demand_is_silent_when_there_is_genuinely_nothing():
    assert make_inputs(running_bs=0).demand_prefill_tokens() == 0


def test_demand_still_fires_for_an_ordinary_uncached_backlog():
    inp = make_inputs(pending_prefill_tokens=32768, running_bs=0)
    assert inp.demand_prefill_tokens() == 32768


def test_verdict_and_message_come_from_one_read():
    """#713, violated by d3: 18 lines reading "pending prefill 0 tok > 0" --
    a verdict of >0 printed beside a 0. The arm must show the number it used."""
    import inspect

    import sglang.srt.managers.phase_policy as pp

    src = inspect.getsource(pp)
    assert "demand_tokens = inp.demand_prefill_tokens()" in src
    assert "_shown" in src, "the message must print the read the verdict used"


def test_demand_is_the_single_definition():
    import inspect

    import sglang.srt.managers.phase_policy as pp

    assert inspect.getsource(pp).count("def demand_prefill_tokens") == 1
