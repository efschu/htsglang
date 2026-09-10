"""#1317b -- the window re-issue must not re-match the chunked request.

BOOT weg2sn6c, 2026-09-10 08:23:32Z, rid f12d8ce7bc764978. The window loop
worked (`WINDOW-REISSUE` matched 4,095 -> 12,287 -> 20,479 on all three D
ranks), then D died on all ranks:

    ValueError: pool memory leak detected! [full] total=699584,
    available=651402, evictable=48176, withheld=0 ... (deficit of 6 row(s))

and the sixth re-issue's own line named the six:
`WINDOW-REISSUE n=6 ... verdict=declined:anchor matched=24570 total=48011`
against a 24,576-token window. 24,576 - 24,570 = 6.

ROOT (not a checker false positive -- real rows were orphaned).
`Scheduler._prefetch_kvcache`'s first act was
`req.init_next_round_input(self.tree_cache, cow_mamba=False)`, which runs
`match_prefix` and OVERWRITES `req.prefix_indices`
(`schedule_batch.py:1946-1969`). For the three ORIGINAL callers -- intake,
disagg-prefill intake, deferral retry -- the request is in the WAITING queue
and that is correct. For the CHUNKED request it is corruption, because
`prefix_indices` is that request's live row accounting: `extend_range` is
indexed against `len(self.chunked_req.prefix_indices)`
(`scheduler.py:8554`). A re-match returning fewer rows than the standing
chunk wrote leaves those rows in NO bucket the leak law knows -- not
`available`, not `evictable` (the node is not a leaf), not `protected` (the
lock went with the replaced tensor).

The tree already knew the class: the W38/#1245 comment sitting directly above
that call exists to "stop an asynchronously-completed host hit moving one
rank's prefix_indices alone (the W27 width divergence)". Same field, same
hazard, one caller later.

MUTANTS on the danger direction (each turns this file red ALONE):
  M1  restoring the unconditional re-match
      -> test_the_rematch_is_reachable_only_under_the_flag
  M2  the seam dropping `rematch=False`
      -> test_the_window_reissue_does_not_rematch
  M3  a pre-admission caller silently losing its re-match
      -> test_the_pre_admission_callers_keep_their_rematch
  M4  the leak law being "relaxed" instead of the writer fixed
      -> test_the_leak_law_still_counts_every_bucket
"""

import ast
import inspect
import textwrap


def _fn(name):
    from sglang.srt.managers import scheduler as sched

    tree = ast.parse(inspect.getsource(sched))
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return n
    raise AssertionError(f"{name} not found in scheduler.py")


# --------------------------------------------------------------------------
# The specimen's arithmetic, pinned as a witness
# --------------------------------------------------------------------------

# Boot weg2sn6c's own numbers, from the D log.
SPEC_TOTAL, SPEC_AVAIL, SPEC_EVICT, SPEC_WITHHELD = 699584, 651402, 48176, 0
SPEC_WINDOW = 24576          # solve_window(30518, 4096, 1, 4096, 37)
SPEC_LAST_MATCH = 24570      # WINDOW-REISSUE n=6 matched=
SPEC_REISSUES = 6            # n=6


def test_the_specimen_deficit_is_exactly_the_rows_the_rematch_dropped():
    """The two independent readings must agree, or this root is the wrong one.

    The leak law's shortfall and the window's own match shortfall are measured
    by different instruments in different modules; that they are the SAME six
    rows is the whole argument.
    """
    law_deficit = SPEC_TOTAL - (SPEC_AVAIL + SPEC_EVICT + SPEC_WITHHELD)
    match_shortfall = SPEC_WINDOW - SPEC_LAST_MATCH
    assert law_deficit == 6
    assert match_shortfall == 6
    assert law_deficit == match_shortfall


def test_the_window_is_the_one_the_solver_gives_group_d():
    """So the 24,576 above is derived, not copied out of the log."""
    from sglang.srt.weg2 import host_ledger

    assert host_ledger.solve_window(30518, 4096, 1, 4096, 37) == SPEC_WINDOW


def test_the_earlier_rounds_were_page_aligned_and_the_last_one_was_not():
    """Why the boot survived five re-issues and died on the sixth: the first
    matches landed on k*4096 - 1 and dropped nothing; only the final one landed
    off the page boundary, and that is the round whose rows went missing."""
    for aligned in (4095, 12287, 20479):
        assert (aligned + 1) % 4096 == 0
    assert (SPEC_LAST_MATCH + 1) % 4096 != 0


# --------------------------------------------------------------------------
# The fix
# --------------------------------------------------------------------------

def test_the_rematch_is_reachable_only_under_the_flag():
    """M1. An unconditional `init_next_round_input(tree_cache=...)` in
    `_prefetch_kvcache` is the boot killer. Every call to it in that function
    must sit under `if rematch`."""
    f = _fn("_prefetch_kvcache")
    all_calls, guarded = [], []
    for node in ast.walk(f):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "init_next_round_input"
        ):
            all_calls.append(node.lineno)
    for node in ast.walk(f):
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "rematch"
        ):
            for n2 in ast.walk(node):
                if (
                    isinstance(n2, ast.Call)
                    and isinstance(n2.func, ast.Attribute)
                    and n2.func.attr == "init_next_round_input"
                ):
                    guarded.append(n2.lineno)
    assert all_calls, "the re-match vanished entirely -- the waiting-queue callers need it"
    assert set(all_calls) == set(guarded), (
        f"an UNGUARDED re-match survives at {sorted(set(all_calls) - set(guarded))}"
    )


def test_the_signature_carries_the_flag_defaulting_to_the_old_behaviour():
    """A default of True keeps every existing caller byte-identical, so the
    change is additive at the seam and not a global behaviour swap."""
    from sglang.srt.managers.scheduler import Scheduler

    sig = inspect.signature(Scheduler._prefetch_kvcache)
    assert "rematch" in sig.parameters
    assert sig.parameters["rematch"].default is True


def test_the_window_reissue_does_not_rematch():
    """M2. The seam owns no match: the adder does. The re-issue must say so."""
    g = _fn("_weg2_issue_next_window")
    passed = None
    for n in ast.walk(g):
        if (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "_prefetch_kvcache"
        ):
            kw = {k.arg: getattr(k.value, "value", None) for k in n.keywords}
            passed = kw.get("rematch")
    assert passed is False, "the window re-issue must pass rematch=False"


def test_the_pre_admission_callers_keep_their_rematch():
    """M3. The three original callers are all pre-admission -- intake, the
    disagg-prefill intake, and the deferral retry -- and for them the re-match
    IS the correct behaviour. A blanket rematch=False would break the
    store-read path for every waiting request."""
    from sglang.srt.managers import scheduler as sched

    tree = ast.parse(inspect.getsource(sched))
    others = []
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef) or fn.name == "_weg2_issue_next_window":
            continue
        for n in ast.walk(fn):
            if (
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == "_prefetch_kvcache"
            ):
                others.append((fn.name, {k.arg for k in n.keywords}))
    assert others, "the pre-admission callers disappeared"
    for name, kw in others:
        assert "rematch" not in kw, (
            f"{name} now overrides rematch; the pre-admission callers must keep the default"
        )
    assert {n for n, _ in others} == {
        "_add_request_to_queue",
        "_retry_deferred_prefetches",
    }


def test_the_reason_is_recorded_where_the_call_is():
    """The next reader of this line must find the boot and the numbers, not a
    bare flag. `rematch=False` without its measurement is how this defect
    comes back."""
    src = inspect.getsource(_pf_source())
    for token in ("weg2sn6c", "f12d8ce7bc764978", "24,570", "extend_range", "8554"):
        assert token in src, f"the re-match guard lost its provenance token {token!r}"


def _pf_source():
    from sglang.srt.managers.scheduler import Scheduler

    return Scheduler._prefetch_kvcache


# --------------------------------------------------------------------------
# The leak law is the instrument, not the defect
# --------------------------------------------------------------------------

def test_the_leak_law_still_counts_every_bucket():
    """M4. The checker was RIGHT: six real rows were orphaned. So the law must
    keep summing all of its buckets -- `protected` above all, which is what
    accounts for a legitimately in-flight chunk. Relaxing this law to make the
    boot survive would have hidden a real row leak, which is the #832/#856
    shape the message itself cites."""
    from sglang.srt.managers.scheduler_components import invariant_checker

    src = inspect.getsource(invariant_checker)
    for bucket in ("available", "evictable", "protected", "withheld"):
        assert bucket in src, f"the leak law lost its {bucket} bucket"
    # and the deficit message must keep naming that no row census can close it
    assert "no row census can close a deficit" in src


def test_an_in_flight_chunk_is_accounted_by_protected_not_by_a_deficit():
    """The reason this fix is at the WRITER and not at the checker: a request
    that is legitimately mid-prefill holds LOCKED rows, and locked rows are
    `protected`. The law already has a home for in-flight work, so a deficit
    is never 'a window in flight' -- it is always rows nobody owns."""
    from sglang.srt.managers.scheduler_components import invariant_checker as ic

    cls = next(
        obj for name, obj in vars(ic).items()
        if isinstance(obj, type) and hasattr(obj, "_check_pool_invariant")
    )
    src = textwrap.dedent(inspect.getsource(cls._check_pool_invariant))
    # the law's own sum, and `protected` is the bucket an in-flight chunk lands in
    assert "protected" in src
    assert "available" in src and "evictable" in src and "withheld" in src
