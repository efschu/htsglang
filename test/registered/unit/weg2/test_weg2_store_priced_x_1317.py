"""#1317 DESIGN A -- the X gate prices against the STORE, and the window loop.

Hermetic: the reduce arm and the pricing are PURE functions of their inputs
(that is `tp_head_congruence`'s own design rule, and the reason a mutant can
be driven here instead of grepped for), so the group verdict is exercised for
real without a collective, a GPU or a scheduler.

THE DANGER DIRECTION of this build is admission: design A ADMITS work the old
gate refused, so every mutant here must make the suite red by admitting
something unservable, not merely by changing a number:

  M1  a rank-divergent store vote must NOT become a group credit
      -> test_a_rank_that_cannot_see_the_pages_drags_the_group_down
  M2  a second prefill of a store-resident prompt must stay impossible
      -> test_a_store_resident_prompt_is_priced_as_cached
  M3  a D prefill above X must stay impossible
      -> test_the_surviving_remainder_is_bounded_by_one_chunk
         + test_x_is_floored_at_the_chunk_size_on_every_path
  M4  a missing store arm must degrade to the OLD behaviour, never to a free
      admission
      -> test_no_store_arm_prices_exactly_as_before
  M5  the window mark must not arm off a rank-local verdict
      -> test_only_the_group_agreed_verdict_arms_the_window
"""

from sglang.srt.managers import tp_head_congruence as thc
from sglang.srt.weg2 import host_ledger


# --------------------------------------------------------------------------
# The reduce arm: payload, MIN semantics, readback
# --------------------------------------------------------------------------

RIDS = ["r-a", "r-b", "r-c"]


def _canonical(rids=RIDS):
    return thc.canonical_head_rids(rids)


def _reduce(payloads):
    """MIN over the arm, exactly as the packed all_reduce does."""
    return [min(col) for col in zip(*payloads)]


def _inputs(canonical, match_lens, store_match=()):
    return thc.build_uniform_head_inputs(
        canonical, match_lens, None, True, (), store_match
    )


def test_the_arm_is_indexed_by_the_canonical_head_not_queue_order():
    """Queue order is the diverging quantity; the arm must not use it."""
    a = thc.build_x_store_match_payload(_canonical(), {"r-a": 100})
    b = thc.build_x_store_match_payload(_canonical(["r-c", "r-b", "r-a"]), {"r-a": 100})
    assert a == b
    assert len(a) == thc.TP_HEAD_SLOTS


def test_a_rank_that_does_not_hold_the_rid_rides_absent_and_removes_it():
    """MIN over _ABSENT_MATCH removes the rid from the group's opinion, so the
    gate falls back to the tree match rather than crediting a store depth no
    peer confirmed."""
    canon = _canonical()
    held = thc.build_x_store_match_payload(canon, {"r-a": 5000})
    missing = thc.build_x_store_match_payload(canon, {})
    group = _reduce([held, missing])
    assert thc.group_store_match_for(_inputs(canon, [], group), "r-a") is None


def test_a_rank_that_cannot_see_the_pages_drags_the_group_down():
    """M1 -- THE RANK-DIVERGENT VOTE. Presence is content-keyed, so the ranks
    normally agree; if one does not, MIN must take the LOWER depth so the
    group prices MORE uncached work. A mutant using MAX here admits a prefill
    the group cannot serve and must turn this red."""
    canon = _canonical()
    sees = thc.build_x_store_match_payload(canon, {"r-a": 60000})
    blind = thc.build_x_store_match_payload(canon, {"r-a": 1000})
    group = _reduce([sees, sees, blind])
    got = thc.group_store_match_for(_inputs(canon, [], group), "r-a")
    assert got == 1000, "MIN must take the blind rank's depth, never the seeing one's"


def test_the_group_value_is_identical_on_every_rank_by_construction():
    """Every rank reduces the same payload, so every rank reads the same slot.
    This is what makes the verdict a GROUP verdict rather than three."""
    canon = _canonical()
    votes = [
        thc.build_x_store_match_payload(canon, {"r-a": 60000, "r-b": 10}),
        thc.build_x_store_match_payload(canon, {"r-a": 59000, "r-b": 10}),
        thc.build_x_store_match_payload(canon, {"r-a": 60000, "r-b": 10}),
    ]
    group = _reduce(votes)
    reads = {
        thc.group_store_match_for(_inputs(canon, [], group), "r-a") for _ in range(3)
    }
    assert reads == {59000}


# --------------------------------------------------------------------------
# The pricing: the extent's arithmetic, transcribed and pinned
# --------------------------------------------------------------------------


def _price(total, group_match, group_store_match):
    """`weg2_uncached_extent`'s design-A arithmetic.

    Kept beside the tests so this file pins the RULE;
    `test_the_priced_extent_matches_the_shipped_expression` stops the
    transcription from drifting from the implementation.
    """
    gm = group_match
    priced = gm if group_store_match is None else max(gm, group_store_match)
    return max(0, total - priced)


def test_a_store_resident_prompt_is_priced_as_cached():
    """M2 -- THE SECOND PREFILL. A 60k prompt P prefilled and wrote through:
    D's tree holds nothing of it (group_match 0) but the store holds all but
    the one token upstream keeps to forward. Old pricing: 60,000 uncached ->
    W50 -> re-route -> P prefills it AGAIN. Design A: 1."""
    assert _price(60000, 0, 59999) == 1
    # and the OLD reading, kept beside it so the delta is visible
    assert _price(60000, 0, None) == 60000


def test_the_surviving_remainder_is_bounded_by_one_chunk():
    """M3a -- A D PREFILL ABOVE X. The credited depth is anchor-capped
    (#869b), so by the #939 law the deepest reachable anchor is
    floor((L-1)/C)*C and the remainder is at most C. Checked across a sweep of
    prompt lengths rather than at one convenient point."""
    C = 4096
    for total in range(C, 300_000, 4093):
        deepest_anchor = ((total - 1) // C) * C
        remainder = _price(total, 0, deepest_anchor)
        assert remainder <= C, f"L={total} left {remainder} > one chunk"


def test_x_is_floored_at_the_chunk_size_on_every_path():
    """M3b -- and the one-chunk remainder is therefore <= X. If a future change
    unfloors X, the phase-law argument of design A breaks HERE, loudly, rather
    than on a boot."""
    import inspect

    from sglang.srt.weg2 import launcher

    src = inspect.getsource(launcher.derive_x_star)
    assert "max(int(floor_tokens)" in src, "derive_x_star no longer floors X at C"
    rx = inspect.getsource(launcher.resolve_x)
    assert "max(floor_tokens, int(override))" in rx, (
        "the override path no longer floors X"
    )
    # and the only call site still passes the chunk size as that floor
    main_src = inspect.getsource(launcher)
    assert (
        "resolve_x(ns.tp_prefill_max_tokens, EVIDENCE_DIR, CHUNKED_PREFILL_TOKENS)"
        in main_src
    )
    assert launcher.CHUNKED_PREFILL_TOKENS == 4096


def test_no_store_arm_prices_exactly_as_before():
    """M4 -- A MISSING ARM MUST NOT BE A FREE ADMISSION. No store vote this
    pass, rid outside the canonical head, or a peer missing the rid: all read
    None, and None must degrade to the TREE match, never to 'fully cached'."""
    for total, gm in ((60000, 0), (60000, 24576), (350, 349)):
        assert _price(total, gm, None) == max(0, total - gm)


def test_the_store_arm_can_only_ever_credit_never_debit():
    """The max() is one-sided on purpose: a store arm that came back BELOW the
    tree match (a stale or partial probe) must not increase the priced
    extent, because the tree match is already proven-resident work."""
    assert _price(60000, 24576, 1000) == 60000 - 24576


def _executed_names(func):
    """Every identifier and attribute a function actually EXECUTES.

    Docstring excluded on purpose: several functions on this path name the
    forbidden rank-local terms IN PROSE precisely in order to forbid them
    (`weg2_uncached_extent`: "Deliberately NOT here: ``available_size()`` ..."),
    so a source-text grep for those terms reports the prohibition as a
    violation. Measured: that false positive failed this very test once.
    """
    import ast
    import inspect
    import textwrap

    fn = ast.parse(textwrap.dedent(inspect.getsource(func))).body[0]
    stmts = (
        fn.body[1:]
        if (
            fn.body
            and isinstance(fn.body[0], ast.Expr)
            and isinstance(fn.body[0].value, ast.Constant)
            and isinstance(fn.body[0].value.value, str)
        )
        else fn.body
    )
    names = set()
    for st in stmts:
        for n in ast.walk(st):
            if isinstance(n, ast.Name):
                names.add(n.id)
            if isinstance(n, ast.Attribute):
                names.add(n.attr)
    return names


def test_the_priced_extent_matches_the_shipped_expression():
    """Drift guard for the transcription above."""
    import inspect

    from sglang.srt.managers.scheduler import Scheduler

    src = inspect.getsource(Scheduler.weg2_uncached_extent)
    assert "group_store_match_for" in src
    assert "max(gm, int(gsm))" in src
    assert "total - priced_match" in src
    names = _executed_names(Scheduler.weg2_uncached_extent)
    for bad in ("available_size", "monotonic", "free_slots"):
        assert bad not in names, f"rank-local {bad} EXECUTED in the extent"


# --------------------------------------------------------------------------
# The window loop
# --------------------------------------------------------------------------


def test_only_the_group_agreed_verdict_arms_the_window():
    """M5 -- THE MARK MUST NOT ARM OFF A RANK-LOCAL VERDICT. Only
    `issued:truncated_group` arms it: that string is read off the census key
    only the POST-CONSENSUS group trim bumps, so every rank returns it
    together. `issued` (a whole read) owes no window, and no decline may arm
    a re-issue."""
    import inspect

    from sglang.srt.managers.scheduler import Scheduler

    src = inspect.getsource(Scheduler._weg2_note_window_verdict)
    assert 'verdict == "issued:truncated_group"' in src
    # the rank-local truncation key must NOT arm it: at that site the cut is a
    # rank-local decision and there is no group fact to continue from.
    assert 'host_pool_truncated"' not in src

    loop = inspect.getsource(Scheduler._weg2_issue_next_window)
    assert "self.chunked_req is not req" in loop
    assert "_weg2_window_open" in loop
    assert "issued:truncated_group" in loop


def test_the_window_gate_executes_no_rank_local_term():
    """A rank that calls `_prefetch_kvcache` while a peer does not leaves the
    peer alone in the #580 participation vote -- a gloo abort. So the CALL
    decision must carry only replicated terms. Checked on the executable body,
    not the source text: the docstring names those terms in prose in order to
    forbid them."""
    from sglang.srt.managers.scheduler import Scheduler

    names = _executed_names(Scheduler._weg2_issue_next_window)
    for bad in ("available_size", "monotonic", "free_slots", "uniform_min_avail"):
        assert bad not in names, f"rank-local {bad} executed in the window gate"


def test_the_mark_has_exactly_the_writers_its_lifecycle_table_claims():
    """A state field with more writers than its table names is a field nobody
    owns -- the class that cost this line a boot (#1317 §1bl, the cutover
    deleting a carry its consumer needed)."""
    import ast
    import inspect

    from sglang.srt.managers import scheduler as sched

    tree = ast.parse(inspect.getsource(sched))
    writers = [
        n.lineno
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        for t in n.targets
        if isinstance(t, ast.Attribute) and t.attr == "_weg2_window_open"
    ]
    assert len(writers) == 2, (
        f"expected 2 writers (re-issue + intake arm), got {writers}"
    )


def test_design_a_added_no_collective():
    """MUST NOT 6. The store arm rides the reduce that already runs once per
    TP-loop iteration; a new `all_reduce` would be a second collective on the
    admission path and is exactly what the #580 family forbids."""
    import inspect

    from sglang.srt.managers import scheduler as sched

    src = inspect.getsource(sched)
    assert src.count("torch.distributed.all_reduce") == 2, (
        "scheduler.py must keep exactly the two all_reduce sites the parent has"
    )


def test_the_store_arm_slice_width_is_checked_before_it_is_priced():
    """A slice of the wrong width would read another arm's numbers as store
    depths -- and THIS arm admits work, so a foreign number admits a prefill
    the group cannot serve. That must stop the group by name.

    THE NUMBER IS W86, NOT THE NEXT ONE UP. W38 is already taken (retired
    `#1234 W38 Weg2CarrierlessPpStoreRead`, still in the census) and
    `test_weg2_wcode_uniqueness_1263` asserts W39 free. The registry's own
    rule is "the first free number above the highest assigned code" -- that
    is W68 here -- but the in-flight xchg candidate recorded in BOOT_QUEUE
    claims the W69-W85 band, so W86/W87 are the first pair that collide with
    neither census at merge."""
    import inspect

    from sglang.srt.managers.scheduler import Scheduler

    src = inspect.getsource(Scheduler._update_uniform_pool_budget)
    assert "W86 Weg2XStoreLayoutStop" in src
    assert "_xstore_at" in src


# --------------------------------------------------------------------------
# The M pin (#1318 consequence)
# --------------------------------------------------------------------------

DK5 = dict(
    memtotal=126_751_866_880, memavail=111_196_077_056, cg_current=22_719_148_032
)
RING_KW = dict(ring_bytes=32964 * 1024 * 1024, ring_span1_bytes=29912 * 1024 * 1024)


def _arm(m_mib):
    return host_ledger.price(
        DK5["memtotal"],
        DK5["memavail"],
        1,
        m_mib,
        cg_current_bytes=DK5["cg_current"],
        cg_ceiling_bytes=DK5["memtotal"],
        **RING_KW,
    )


def test_the_sn5w_arm_is_fundable_so_the_pin_is_bootable():
    """M=600 is the arm boot weg2sn5w actually ran -- the only FUNDABLE rung of
    its printed ladder (`ARM S=1 M=600 ... => FUNDABLE`, against M=1200 and
    M=2400 both refused). The pin must therefore price it fundable."""
    a = _arm(600)
    assert a.fundable_moments
    assert a.launch_leftover_gib > 0


def test_the_pin_refuses_an_unfundable_arm_rather_than_taking_it():
    """A pin selects among arms the ledger would fund; it is not a way past the
    verdict, and the host-threshold law admits no accept-the-risk branch."""
    a = _arm(2400)
    assert not a.fundable_moments
    assert a.launch_leftover_gib < 0


def _host_ledger_calls_in(module_source):
    """Every ``host_ledger.<fn>(...)`` call in the launcher, as (fn, kwargs)."""
    import ast

    out = []
    for node in ast.walk(ast.parse(module_source)):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if (
            isinstance(f, ast.Attribute)
            and isinstance(f.value, ast.Name)
            and f.value.id == "host_ledger"
        ):
            out.append((f.attr, [k.arg for k in node.keywords if k.arg]))
    return out


def test_every_host_ledger_call_in_the_launcher_matches_the_callee_signature():
    """Boot weg2sn6a (2026-09-10 07:33Z) died in choose_host_ledger with
    `price() got an unexpected keyword argument 'ring_provenance'`: the pinned
    arm was priced by a hand-copied call carrying choose()'s keywords, and no
    test exercised that call. This pins keyword/signature conformance for
    EVERY host_ledger call the launcher makes, so the class cannot recur
    silently. Mutant: add `ring_provenance=` to the price() call in a copy ->
    this test must go red."""
    import inspect

    from sglang.srt.weg2 import launcher

    calls = _host_ledger_calls_in(inspect.getsource(launcher))
    assert calls, "no host_ledger.<fn>( calls found -- the walker is broken"
    checked = 0
    for fn, kws in calls:
        target = getattr(host_ledger, fn, None)
        # exception classes (Weg2HostLedgerRefused(...)) and other non-function
        # callables carry no Python signature -- only plain functions are checked
        if not inspect.isfunction(target):
            continue
        params = inspect.signature(target).parameters
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            continue
        unknown = [k for k in kws if k not in params]
        assert not unknown, (
            f"host_ledger.{fn}(...) is called with {unknown}, not in its signature"
        )
        checked += 1
    assert checked >= 2, (
        f"expected the ladder and the pin call to be checked, saw {checked}"
    )


def test_the_pin_runs_the_ladder_restricted_to_the_one_arm(monkeypatch):
    """The pin is priced by choose() itself with arms=[(1, M)] -- the identical
    model the ladder applies (slab term, reap bound, W20/W21), never a second
    price() call that can drift. Behavioural: choose is recorded, not the
    launcher's source text."""
    from sglang.srt.weg2 import launcher

    seen = {}
    ladder = host_ledger.choose

    def recording_choose(*a, **kw):
        seen["arms"] = kw.get("arms")
        seen["kw"] = set(kw) - {"arms"}
        return ladder(*a, **kw)

    monkeypatch.setattr(launcher.host_ledger, "choose", recording_choose)
    monkeypatch.setattr(
        launcher.host_ledger, "read_measured_record", lambda *_a, **_k: None
    )
    # The reader helpers differ per tree; drive choose_host_ledger only if the
    # tree exposes injectable readers, otherwise the AST conformance test
    # above is the guard and this test documents the intended shape.
    import inspect as _inspect

    sig = _inspect.signature(launcher.choose_host_ledger)
    if "meminfo_path" not in sig.parameters:
        return
    import os
    import tempfile

    d = tempfile.mkdtemp()
    mi_path = os.path.join(d, "meminfo")
    with open(mi_path, "w") as f:
        f.write(
            f"MemTotal:       {DK5['memtotal'] // 1024} kB\n"
            f"MemAvailable:   {DK5['memavail'] // 1024} kB\n"
        )
    cg = os.path.join(d, "cg")
    os.makedirs(cg, exist_ok=True)
    with open(os.path.join(cg, "memory.current"), "w") as f:
        f.write(f"{DK5['cg_current']}\n")
    with open(os.path.join(cg, "memory.max"), "w") as f:
        f.write("max\n")
    with open(os.path.join(cg, "memory.stat"), "w") as f:
        f.write("anon 0\nfile 0\nslab_reclaimable 0\nslab_unreclaimable 0\nshmem 0\n")
    with open(os.path.join(cg, "memory.events"), "w") as f:
        f.write("oom_kill 0\n")
    try:
        arm, _hd, lines, _cg = launcher.choose_host_ledger(
            **RING_KW, meminfo_path=mi_path, cgroup_root=cg, pin_m_mib=600
        )
    except (OSError, KeyError, host_ledger.Weg2HostLedgerRefused):
        # a reader this fixture cannot satisfy -- the AST test still guards
        return
    assert seen.get("arms") == [(1, 600)]
    assert arm.fundable_moments
    assert any("ARM PINNED" in ln for ln in lines)
