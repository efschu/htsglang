"""1537 -- the GROUP COLLECTIVES of the scheduler's admission path, pinned.

WHY THIS FILE EXISTS. `test_pdflip_store_priced_x_1317.test_design_a_added_no_collective`
pins "scheduler.py has exactly 2 `torch.distributed.all_reduce` sites" and has
been red since 17.09. (5 sites: 4121, 6559, 6733, 10708, 10792 on candidate 3,
bce16a6ddf). That pin cannot be re-counted to 5 and left: the number says
nothing about WHICH sites, in which function, behind which guard, or who calls
the two wrappers that carry two of them. The family it guards is the one the
code comments call RAENGE-NIE-UNEINS / rank-local-test-before-a-group-
collective (#580/#607-E/#610/#791b): a condition that differs per rank in front
of a collective takes it on some ranks only, and the group hangs.

WHAT IS PINNED (all by AST over the shipped source -- no GPU, no process group):

  P1  every group collective in scheduler.py, by (function, op, count)
  P2  every reference to the three collective-carrying helpers
      (`_pdflip_group_min_flags`, `_pdflip_group_min_ints`,
      `_uniform_timeout_ballot`) anywhere under python/flliper/srt, by
      (file, enclosing function)  -- the wrapper hides ~16 call sites
  P3  the guard chain in front of each of the five `all_reduce` calls
  P4  the order and the guards of the collective-bearing calls in
      `get_next_batch_to_run` (one scheduler iteration, every rank)
  P5  the guard in front of the optional collective of the X-gate refusal path
  P6  behaviour with a recording stand-in for torch.distributed: each helper
      takes exactly ONE collective per call, its number does not depend on the
      VALUES passed, and `_room_ok` (d_seat_vram) enters it as a function of
      replicated inputs only
  P7  `_chunked_rest` (the 1528 input) reads exactly the two fields its
      docstring calls replicated

A mutant that adds a site, moves one behind a rank-dependent `if`, adds a
caller, reorders the iteration, or lets a rank-local value decide the entry
turns one of these red (mutation probe in done/1537-bericht.md).

WHAT THIS DOES NOT PROVE. That the guards it records are rank-uniform at run
time. It records WHAT the guard is; the replication of its terms is argued in
the report with line evidence, not proven here.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import pathlib
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402


def _srt_root() -> pathlib.Path:
    spec = importlib.util.find_spec("flliper.srt")
    assert spec is not None and spec.submodule_search_locations
    return pathlib.Path(list(spec.submodule_search_locations)[0])


SRT = _srt_root()
SCHED_PATH = SRT / "managers" / "scheduler.py"

_TREES: dict = {}


def _tree(path: pathlib.Path) -> ast.Module:
    key = str(path)
    if key not in _TREES:
        _TREES[key] = ast.parse(path.read_text(encoding="utf-8"))
    return _TREES[key]


def _parents(tree: ast.AST) -> dict:
    out = {}
    for p in ast.walk(tree):
        for c in ast.iter_child_nodes(p):
            out[c] = p
    return out


def _chain(n: ast.AST) -> str:
    parts = []
    while isinstance(n, ast.Attribute):
        parts.append(n.attr)
        n = n.value
    if isinstance(n, ast.Name):
        parts.append(n.id)
    return ".".join(reversed(parts))


def _find_fn(tree: ast.AST, name: str) -> ast.FunctionDef:
    hits = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == name
    ]
    assert len(hits) == 1, f"{name}: {len(hits)} definitions in the module"
    return hits[0]


def _guards(node: ast.AST, fn: ast.AST, parent: dict) -> list:
    """The control-flow ancestors of ``node`` inside ``fn``, outermost first."""
    out = []
    n = parent.get(node)
    while n is not None and n is not fn:
        if isinstance(n, ast.If):
            out.append("If:" + ast.unparse(n.test))
        elif isinstance(n, (ast.For, ast.While)):
            out.append(type(n).__name__)
        elif isinstance(n, ast.Try):
            out.append("Try")
        n = parent.get(n)
    return list(reversed(out))


# --------------------------------------------------------------------------
# P1  every group collective in scheduler.py
# --------------------------------------------------------------------------

_OPS = {
    "all_reduce", "all_gather", "all_gather_object", "broadcast",
    "broadcast_object_list", "barrier", "reduce_scatter", "all_to_all",
    "gather", "scatter", "reduce", "send", "recv", "broadcast_pyobj",
    "all_gather_into_tensor", "reduce_scatter_tensor", "batch_isend_irecv",
}

#: (enclosing function, call) -> number of call sites, scheduler.py @ bce16a6ddf
EXPECTED_COLLECTIVES = {
    ("Scheduler._uniform_timeout_ballot", "torch.distributed.all_reduce"): 1,
    ("Scheduler._form_a_tp_exchange", "broadcast_pyobj"): 1,
    ("Scheduler._form_a_tp_gather", "torch.distributed.all_gather_object"): 1,
    ("Scheduler._process_and_broadcast_mm_inputs", "torch.distributed.broadcast_object_list"): 2,
    ("Scheduler._pdflip_group_min_flags", "torch.distributed.all_reduce"): 1,
    ("Scheduler._pdflip_group_min_ints", "torch.distributed.all_reduce"): 1,
    ("Scheduler._update_uniform_pool_budget", "torch.distributed.all_reduce"): 2,
    ("Scheduler.handle_rpc_request", "barrier"): 1,
}


def _collective_sites() -> dict:
    out: dict = {}

    class V(ast.NodeVisitor):
        def __init__(self):
            self.stack = []

        def _fn(self, n):
            self.stack.append(n.name)
            self.generic_visit(n)
            self.stack.pop()

        visit_FunctionDef = _fn
        visit_AsyncFunctionDef = _fn
        visit_ClassDef = _fn

        def visit_Call(self, n):
            c = _chain(n.func)
            last = c.split(".")[-1]
            if last in _OPS and (
                c.startswith("torch.distributed")
                or c.startswith("dist.")
                or c in ("barrier", "broadcast_pyobj")
                or (c.startswith("self.") and "group" in c)
            ):
                key = (".".join(self.stack), c)
                out[key] = out.get(key, 0) + 1
            self.generic_visit(n)

    V().visit(_tree(SCHED_PATH))
    return out


def test_p1_scheduler_py_has_exactly_the_known_group_collectives():
    got = _collective_sites()
    assert got == EXPECTED_COLLECTIVES, (
        "scheduler.py's group collectives changed. A new or moved collective "
        "on the admission path must run on every rank in the same order "
        "(RAENGE-NIE-UNEINS). Update EXPECTED_COLLECTIVES only together with "
        "the argument that its entry condition is rank-uniform.\n"
        f"added/changed: {sorted(set(got.items()) - set(EXPECTED_COLLECTIVES.items()))}\n"
        f"gone/changed: {sorted(set(EXPECTED_COLLECTIVES.items()) - set(got.items()))}"
    )


def test_p1_the_five_all_reduce_sites_are_five():
    n = sum(
        v for (fn, c), v in _collective_sites().items() if c == "torch.distributed.all_reduce"
    )
    assert n == 5, f"{n} torch.distributed.all_reduce sites in scheduler.py (was 5 at bce16a6ddf)"


# --------------------------------------------------------------------------
# P2  who references the three helpers that carry a collective
# --------------------------------------------------------------------------

_HELPERS = {"_pdflip_group_min_flags", "_pdflip_group_min_ints", "_uniform_timeout_ballot"}

#: (file relative to srt/, enclosing function) -> helper names, @ bce16a6ddf.
#: A reference through getattr(self, "<name>"), `Scheduler.<name>` or
#: `sched._<name>` all count (they are the same function).
EXPECTED_HELPER_REFS = {
    ("managers/scheduler.py", "Scheduler._abort_on_running_timeout"): {"_uniform_timeout_ballot"},
    ("managers/scheduler.py", "Scheduler._abort_on_waiting_timeout"): {"_uniform_timeout_ballot"},
    ("managers/scheduler.py", "Scheduler._pdflip_drain_prefetch_revokes"): {"_pdflip_group_min_ints"},
    ("managers/scheduler.py", "Scheduler._pdflip_hold_refetch"): {"_pdflip_group_min_flags"},
    ("managers/scheduler.py", "Scheduler._pdflip_post_wake_settle_tick"): {"_pdflip_group_min_flags"},
    ("managers/scheduler.py", "Scheduler._pdflip_release_dormant_hold"): {"_pdflip_group_min_flags"},
    ("managers/scheduler.py", "Scheduler._pdflip_answer_x_refusals"): {"_pdflip_group_min_flags"},
    ("pdflip/d_park_runtime.py", "park_tick"): {"_pdflip_group_min_flags"},
    ("pdflip/d_park_runtime.py", "_capacity_requeue"): {"_pdflip_group_min_flags"},
    ("pdflip/d_park_runtime.py", "_lift_holds_when_idle"): {"_pdflip_group_min_flags"},
    ("pdflip/d_park_runtime.py", "displace_for_age"): {"_pdflip_group_min_flags"},
    ("pdflip/d_seat_rewake.py", "_group_min"): {"_pdflip_group_min_flags"},
    ("pdflip/d_seat_vram.py", "_agree_reserve"): {"_pdflip_group_min_ints"},
    ("pdflip/d_seat_vram.py", "_group_floor_tokens"): {"_pdflip_group_min_ints"},
    ("pdflip/d_seat_vram.py", "_group_room_below"): {"_pdflip_group_min_ints"},
    ("pdflip/d_seat_vram.py", "_room_ok"): {"_pdflip_group_min_ints"},
}


def _helper_refs() -> dict:
    out: dict = {}
    for p in sorted(SRT.rglob("*.py")):
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        rel = p.relative_to(SRT).as_posix()

        class V(ast.NodeVisitor):
            def __init__(self):
                self.stack = []

            def _fn(self, n):
                self.stack.append(n.name)
                self.generic_visit(n)
                self.stack.pop()

            visit_FunctionDef = _fn
            visit_AsyncFunctionDef = _fn
            visit_ClassDef = _fn

            def _hit(self, name):
                out.setdefault((rel, ".".join(self.stack)), set()).add(name)

            def visit_Attribute(self, n):
                if n.attr in _HELPERS:
                    self._hit(n.attr)
                self.generic_visit(n)

            def visit_Name(self, n):
                if n.id in _HELPERS:
                    self._hit(n.id)

            def visit_Constant(self, n):
                if isinstance(n.value, str) and n.value in _HELPERS:
                    self._hit(n.value)

        V().visit(tree)
    return out


def test_p2_the_collective_carrying_helpers_have_exactly_the_known_callers():
    got = _helper_refs()
    # the helpers' own definitions are `def` names, not references; nothing to drop
    assert got == EXPECTED_HELPER_REFS, (
        "the set of functions that reach a group all_reduce through "
        "_pdflip_group_min_flags / _pdflip_group_min_ints / _uniform_timeout_ballot "
        "changed. Each new caller must be shown to enter the helper on every "
        "rank (or on none) in the same iteration.\n"
        f"new: {sorted(set(got) - set(EXPECTED_HELPER_REFS))}\n"
        f"gone: {sorted(set(EXPECTED_HELPER_REFS) - set(got))}\n"
        f"changed: {sorted(k for k in got if k in EXPECTED_HELPER_REFS and got[k] != EXPECTED_HELPER_REFS[k])}"
    )


# --------------------------------------------------------------------------
# P3  the guard chain in front of each all_reduce
# --------------------------------------------------------------------------

_WRAPPER_GUARD = [
    "Try",
    "If:tp_size > 1 and group is not None and torch.distributed.is_initialized()",
]


def _all_reduce_guards(fname: str) -> list:
    tree = _tree(SCHED_PATH)
    parent = _parents(tree)
    fn = _find_fn(tree, fname)
    return sorted(
        (c.lineno, _guards(c, fn, parent))
        for c in ast.walk(fn)
        if isinstance(c, ast.Call) and ast.unparse(c.func) == "torch.distributed.all_reduce"
    )


def test_p3_the_wrapper_reduces_are_guarded_by_uniform_terms_only():
    for fname in ("_pdflip_group_min_flags", "_pdflip_group_min_ints"):
        sites = _all_reduce_guards(fname)
        assert [g for _, g in sites] == [_WRAPPER_GUARD], (fname, sites)


def test_p3_the_wrappers_return_before_the_reduce_on_an_empty_payload():
    tree = _tree(SCHED_PATH)
    for fname in ("_pdflip_group_min_flags", "_pdflip_group_min_ints"):
        fn = _find_fn(tree, fname)
        early = [
            ast.unparse(n.test)
            for n in fn.body
            if isinstance(n, ast.If) and any(isinstance(b, ast.Return) for b in n.body)
        ]
        assert early == ["not vals"], (fname, early)


def test_p3_the_timeout_ballot_has_no_branch_but_two_early_returns():
    tree = _tree(SCHED_PATH)
    fn = _find_fn(tree, "_uniform_timeout_ballot")
    assert [g for _, g in _all_reduce_guards("_uniform_timeout_ballot")] == [[]]
    early = [
        ast.unparse(n.test)
        for n in fn.body
        if isinstance(n, ast.If) and any(isinstance(b, ast.Return) for b in n.body)
    ]
    assert early == [
        "not local_verdicts",
        "grp is None or torch.distributed.get_world_size(grp) <= 1",
    ]


def test_p3_the_admission_reduce_is_unconditional_and_the_realize_round_follows_the_skew():
    sites = _all_reduce_guards("_update_uniform_pool_budget")
    assert [g for _, g in sites] == [[], ["If:_usable_skew"]], (
        "the packed per-iteration MIN must be the first reduce and sit behind no "
        "branch; the realize round (H97, 10792) may sit behind exactly one `if`, "
        "whose term is derived from the already-reduced vector"
    )
    first, second = sites[0][0], sites[1][0]
    assert first < second


def test_p3_the_skew_that_gates_the_realize_round_comes_from_the_reduced_vector():
    tree = _tree(SCHED_PATH)
    fn = _find_fn(tree, "_update_uniform_pool_budget")
    assigns = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "_usable_skew" for t in n.targets)
    ]
    assert len(assigns) == 1
    txt = ast.unparse(assigns[0].value)
    assert txt.startswith("tp_match_floor.skewed_rids(_usable_group,")
    # its inputs: the group usable map (decoded from `t`) and the group MAX decoded from `t`
    assert "tp_match_floor.decode_group_max(_head_canonical" in txt
    assert "t[_usable_max_at" in txt


# --------------------------------------------------------------------------
# P4  one scheduler iteration: order and guards of the collective-bearing calls
# --------------------------------------------------------------------------

_WATCH = {
    "process_pending_chunked_abort", "process_pending_pdflip_park", "_pdflip_d_park_tick",
    "_pdflip_post_wake_settle_tick", "round_boundary", "_abort_on_waiting_timeout",
    "_abort_on_running_timeout", "pre_schedule", "update_dcp_admission_state",
    "_update_uniform_pool_budget",
}

EXPECTED_ITERATION = [
    ("process_pending_chunked_abort", []),
    ("process_pending_pdflip_park", []),
    ("_pdflip_d_park_tick", ["If:getattr(self, 'pdflip_d_parked', None)"]),
    ("_pdflip_post_wake_settle_tick", ["If:getattr(self, 'pdflip_post_wake_settle', None)"]),
    ("round_boundary", []),
    ("_abort_on_waiting_timeout", []),
    ("_abort_on_running_timeout", []),
    ("pre_schedule", ["If:self.kv_session_offload is not None"]),
    ("update_dcp_admission_state", ["If:self.kv_session_offload is not None"]),
    ("_update_uniform_pool_budget", []),
]


def test_p4_the_collective_bearing_calls_of_one_iteration_keep_order_and_guards():
    tree = _tree(SCHED_PATH)
    parent = _parents(tree)
    fn = _find_fn(tree, "get_next_batch_to_run")
    calls = []
    for c in ast.walk(fn):
        if isinstance(c, ast.Call):
            f = c.func
            nm = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else None)
            if nm in _WATCH:
                calls.append((c.lineno, nm, _guards(c, fn, parent)))
    got = [(nm, g) for _, nm, g in sorted(calls)]
    assert got == EXPECTED_ITERATION, (
        "the order/guards of the group-collective-bearing calls in "
        "get_next_batch_to_run changed; every rank must run them in this order, "
        "and a guard here must be a replicated term"
    )


def test_p4_the_prefill_admission_calls_that_reach_a_collective_keep_their_guards():
    tree = _tree(SCHED_PATH)
    parent = _parents(tree)
    fn = _find_fn(tree, "_get_new_batch_prefill_raw")
    got = {}
    for c in ast.walk(fn):
        if isinstance(c, ast.Call):
            f = c.func
            nm = f.attr if isinstance(f, ast.Attribute) else None
            if nm in ("_pdflip_answer_x_refusals", "_pdflip_d_park_admission"):
                got[nm] = _guards(c, fn, parent)
    assert got == {
        "_pdflip_d_park_admission": [],
        "_pdflip_answer_x_refusals": ["If:_x_refused"],
    }, got


# --------------------------------------------------------------------------
# P5  the optional collective of the X-gate refusal path
# --------------------------------------------------------------------------


def test_p5_the_x_refusal_vote_is_entered_only_behind_the_replicated_precondition():
    tree = _tree(SCHED_PATH)
    parent = _parents(tree)
    fn = _find_fn(tree, "_pdflip_answer_x_refusals")
    sites = [
        (c.lineno, _guards(c, fn, parent))
        for c in ast.walk(fn)
        if isinstance(c, ast.Call) and ast.unparse(c.func).endswith("_pdflip_group_min_flags")
    ]
    assert [g for _, g in sites] == [
        ["If:any((_pdflip_rvp.capacity_park_precondition(r) for r in refused))"]
    ], sites
    # and the precondition reads only request fields + two process-level switches
    rvp = _tree(SRT / "pdflip" / "resume_via_p.py")
    pre = _find_fn(rvp, "capacity_park_precondition")
    doc = pre.body[0].value if isinstance(pre.body[0], ast.Expr) else None
    attrs = sorted(
        {
            n.value
            for n in ast.walk(pre)
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and n is not doc
        }
    )
    assert attrs == [
        "",
        "D",
        "FLLIPER_PDFLIP_GROUP",
        "_pdflip_store_short_fallback",
        "multimodal_inputs",
        "stream",
    ], attrs


# --------------------------------------------------------------------------
# P6  behaviour with a recording stand-in for torch.distributed
# --------------------------------------------------------------------------


@pytest.fixture()
def recorder(monkeypatch):
    calls = []

    def _all_reduce(t, op=None, group=None, **kw):
        calls.append((int(t.numel()), group))

    monkeypatch.setattr(torch.distributed, "all_reduce", _all_reduce)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 3)
    return calls


def _sched_cls():
    from flliper.srt.managers.scheduler import Scheduler

    return Scheduler


def _fake(tp_size=3, group=object()):
    return types.SimpleNamespace(ps=types.SimpleNamespace(tp_size=tp_size), tp_cpu_group=group)


@pytest.mark.parametrize("name", ["_pdflip_group_min_flags", "_pdflip_group_min_ints"])
def test_p6_a_wrapper_takes_one_collective_whatever_the_values(recorder, name):
    fn = getattr(_sched_cls(), name)
    for vals in ([True, True], [False, False], [True, False], [0, 5, 7]):
        before = len(recorder)
        fn(_fake(), vals)
        assert len(recorder) - before == 1, (name, vals)
        assert recorder[-1][0] == len(vals)


@pytest.mark.parametrize("name", ["_pdflip_group_min_flags", "_pdflip_group_min_ints"])
def test_p6_a_wrapper_takes_no_collective_on_an_empty_payload_one_rank_or_no_group(recorder, name):
    fn = getattr(_sched_cls(), name)
    fn(_fake(), [])
    fn(_fake(tp_size=1), [True])
    fn(_fake(group=None), [True])
    assert recorder == []


def test_p6_the_timeout_ballot_takes_one_collective_whatever_the_verdicts(recorder):
    fn = _sched_cls()._uniform_timeout_ballot
    for vals in ([True], [False, False, False], [True, False]):
        before = len(recorder)
        fn(_fake(), vals)
        assert len(recorder) - before == 1
    fn(_fake(), [])
    fn(_fake(group=None), [True])
    assert len(recorder) == 3


def _room_env(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_D_MEM_RECHECK_ROUNDS", "8")


def _room_sched(free_ids):
    """A D rank whose LOCAL free ids differ; the group MIN is recorded."""
    seen = []

    def gmin(vals):
        seen.append(list(vals))
        return list(vals)

    alloc = types.SimpleNamespace(
        free_pages=torch.tensor(free_ids, dtype=torch.int64),
        release_pages=torch.empty(0, dtype=torch.int64),
    )
    return types.SimpleNamespace(_pdflip_group_min_ints=gmin, server_args=None), alloc, seen


def _room_calls(monkeypatch, free_ids, incoming, rounds=2, need=64):
    from flliper.srt.pdflip import d_seat_vram as V

    _room_env(monkeypatch)
    sched, alloc, seen = _room_sched(free_ids)
    ms = types.SimpleNamespace(pending=1, _round=0)
    fr = V._floor_room_state(ms)
    for _ in range(rounds):
        ms._round = 0
        V._room_ok(sched, ms, fr, alloc, 4096, need, incoming, 1, 64)
    return len(seen)


def test_p6_room_ok_entry_does_not_depend_on_the_rank_local_free_ids(monkeypatch):
    """Two ranks whose LOCAL free-id lists differ (uneven DCP) and who hold the
    same replicated inputs take the same number of collectives."""
    rich = list(range(1, 60))
    poor = [1, 2]
    for incoming in (0, 128):
        assert _room_calls(monkeypatch, rich, incoming) == _room_calls(monkeypatch, poor, incoming)


def test_p6_room_ok_entry_is_a_function_of_incoming_and_the_key(monkeypatch):
    """CHARACTERISATION (1528): with a re-check window > 1, a pure decode round
    (incoming <= 0) under an unchanged key reuses the last group verdict and
    takes NO collective; any incoming demand reads at once. 1528 passes
    `incoming + rest` and a `need` that contains the next chunk of a live
    chunked_req into this predicate, so `rest` must be rank-uniform or two
    ranks disagree on whether this call is a collective. This pins the
    dependency (not that `rest` is uniform -- see P7 and the report)."""
    free = list(range(1, 60))
    assert _room_calls(monkeypatch, free, incoming=0, rounds=2) == 1  # 2nd round cached
    assert _room_calls(monkeypatch, free, incoming=128, rounds=2) == 2  # demand: both read
    # a different `need` is a different key: re-read
    from flliper.srt.pdflip import d_seat_vram as V

    _room_env(monkeypatch)
    sched, alloc, seen = _room_sched(free)
    ms = types.SimpleNamespace(pending=1, _round=0)
    fr = V._floor_room_state(ms)
    V._room_ok(sched, ms, fr, alloc, 4096, 64, 0, 1, 64)
    V._room_ok(sched, ms, fr, alloc, 4096, 64, 0, 1, 64)  # same key, same round: cached
    V._room_ok(sched, ms, fr, alloc, 4096, 64 + 2048, 0, 1, 64)
    assert len(seen) == 2


# --------------------------------------------------------------------------
# P7  the 1528 input
# --------------------------------------------------------------------------


def test_p7_chunked_rest_reads_only_the_prefix_length_and_the_fill_boundary():
    tree = _tree(SRT / "pdflip" / "d_seat_vram.py")
    fn = _find_fn(tree, "_chunked_rest")
    doc = fn.body[0].value if isinstance(fn.body[0], ast.Expr) else None
    names = sorted(
        {
            n.value
            for n in ast.walk(fn)
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and n is not doc
        }
        | {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute) and n.attr != "append"}
    )
    assert names == ["chunked_req", "end", "extend_range", "prefix_indices"], names
    src = ast.unparse(fn)
    for banned in ("tp_rank", "pp_rank", "distributed", "group_min", "get_rank"):
        assert banned not in src, banned
