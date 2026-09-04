#!/usr/bin/env python3
"""CLASS A -- a GROUP-UNIFORM decision computed from RANK-LOCAL state.

THE SHAPE, precisely: a rank reads state only it can see (its own queue, its
own pool, its own match result, its own prefetch status, its own free memory),
turns that into a BRANCH, and inside that branch enters a COLLECTIVE.  When two
ranks read different values they take different branches, and the group either
hangs in a collective one rank never entered (#1158), splits silently (#1176),
votes STOP alone (#1153), or deadlocks on a prefix-match divergence (#944,
told=8192 local=0).

WHY AN AST CHECK AND NOT SEMGREP.  Tried first, does not hold:

  * The relation is CONTROL DEPENDENCE, not adjacency.  Semgrep's
    `pattern-inside: if <... $X ...>: ...` matches only when the collective is
    LEXICALLY nested in the `if`.  Four of the six confirmed instances below
    put the guard in an `elif`, in a `while` test, in a boolean operand of a
    combined test, or in an early `return`/`continue` that makes the REST OF
    THE FUNCTION the guarded region -- the guarded body is then the code
    AFTER the `if`, which no `pattern-inside` can express.
  * The exemption is a WHOLE-FUNCTION property ("does this function route its
    verdict through the agreement point").  `pattern-not-inside` with a
    `def $F(...): ... agreement(...) ...` shape does express it, but semgrep's
    `...` operator does not distinguish "the agreement call happens BEFORE the
    collective" from "after", so a function that reduces AFTER acting would be
    silently exempted -- which is exactly the #1176 defect.
  * The rank-local classification is a NAME-ORIGIN judgement over a curated
    vocabulary plus attribute chains (`self.waiting_queue`,
    `req.prefix_indices`, `...allocator.available_size()`).  Semgrep
    metavariable-regex can approximate it, but it cannot then propagate
    through the local `x = self.waiting_queue` alias that four of the sites
    use, so it reads the guard as a bare local and misses it.

So: `ast` walks the CFG-shaped skeleton (dominating tests, early exits,
boolean operands), keeps a per-function alias map from local names to the
rank-local expressions they were assigned, and exempts on ORDERED evidence
(agreement call must lexically precede the collective).

USAGE
  python3 rank_local_gates_collective.py [PATH ...]        # report
  python3 rank_local_gates_collective.py --self-test       # can-fire proof
Exit 1 when any finding is reported (so it can gate).
"""

from __future__ import annotations

import argparse
import ast
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

# --------------------------------------------------------------------------
# (1) GROUP ACTS -- entering one of these commits this rank to a rendezvous
#     that every peer must also enter, with the same shape, in the same order.
# --------------------------------------------------------------------------
COLLECTIVE_CALLS: Set[str] = {
    # torch.distributed / ProcessGroup surface
    "all_reduce",
    "all_gather",
    "all_gather_object",
    "all_gather_into_tensor",
    "_all_gather_base",
    "reduce_scatter",
    "reduce_scatter_tensor",
    "reduce",
    "broadcast",
    "broadcast_object_list",
    "broadcast_pyobj",
    "barrier",
    "monitored_barrier",
    "all_to_all",
    "all_to_all_single",
    "scatter_object_list",
    "gather_object",
    # point-to-point: a send whose peer does not recv (or vice versa) is the
    # same failure with one fewer participant (#1153, #1180).
    "send",
    "recv",
    "isend",
    "irecv",
    "batch_isend_irecv",
    "send_object",
    "recv_object",
    "send_object_list",
    "recv_object_list",
    "point_to_point_pyobj",
    # fork-local wrappers that ARE collectives (verified: each contains a
    # torch.distributed call or a barlink transport rendezvous)
    "poll_and_all_reduce_attn_cp_tp_group",
    "bounded_wait",
    "_pp_sync",
}

# --------------------------------------------------------------------------
# (1b) THE SECOND ACT FAMILY -- the one the collective-only lens MISSES.
#
#     Measured on this tree: 90 collective sites in managers/ + mem_cache/, of
#     which only ONE was rank-local-gated.  The confirmed defects of this class
#     (#1176, #1153, #1158, #1173) do NOT enter a collective inside the guarded
#     branch.  They ARM a flip, VOID a slot, ADMIT a request, or RAISE a group
#     STOP -- and the collective that hangs is entered LATER, by the peers who
#     read the state differently.  The act is the commitment; the collective is
#     only where the disagreement becomes visible.
#
#     So the rule carries two act families with two precision profiles.
# --------------------------------------------------------------------------
GROUP_ACT_CALLS: Set[str] = {
    # arm / cutover -- committing the group to a layout change
    "arm_flip",
    "arm_phase_flip",
    "phase_flip_arm",
    "arm_draft_bootstrap",
    "arm_draft_cold_for_admission",
    "reshard_arm",
    "_cutover",
    "build_production_flip_cutover",
    "release_residents_for_cutover",
    # void / retract -- removing a participant the peers still expect
    "void_pp_admission_decision",
    "_release_voided_request",
    "pp_park_voided_batch_member",
    "retract_all",
    "retract_decode",
    "release_req",
    # admission / seats -- letting a request into a batch the peers must match
    "admit_request_into_staging",
    "admit_request_direct",
    "readmit_seam_residents",
    "readmit",
    "add_one_req",
    "add_chunked_req",
}

# Group-STOP exception classes VERIFIED at this tree (grep of `^class .*Error|
# Refused|Contradiction|Mismatch` under managers/ + mem_cache/).  Raising one
# of these from a rank-local predicate kills the group from one rank's reading
# -- #1153 ("a follower refusal is a group STOP, not a void") is exactly this.
GROUP_STOP_EXCEPTIONS: Set[str] = {
    "StoreWitnessContradiction",
    "PPScheduleRefused",
    "PrefetchBallotDigestMismatch",
    "PhasePurityError",
    "PhasePolicyError",
    "ReqPoolRebindRefused",
    "RebindRefused",
    "RebindIncoherent",
    "CorridorBreachRefused",
    "ArmingFloorUnsatisfiable",
    "PpRowDeferCapExceeded",
    "PpChainRecvStalled",
    "RingCommitTimeout",
    "SeamOrderError",
    "ResidentCarryError",
    "PhaseFlipBootError",
    "MambaSlotDoubleFree",
    "KvReshardError",
}

# Names that look like a collective but are a local method on a plain object.
# `.gather` on a tensor, `.reduce` from functools, `.send` on a zmq socket.
NOT_COLLECTIVE_RECEIVERS: Set[str] = {
    "torch",  # torch.gather / torch.scatter are tensor ops, not collectives
    "np",
    "functools",
    "socket",
    "sock",
    "queue",
    "q",
    "self.send_to_tokenizer",
    "logger",
}

# --------------------------------------------------------------------------
# (2) RANK-LOCAL STATE -- things two ranks of one group may legitimately hold
#     DIFFERENT values for at the same instant.  This vocabulary IS the
#     precision lever and it is CURATED, not inferred: every entry is here
#     because a shipped defect read it and branched on it.
# --------------------------------------------------------------------------
RANK_LOCAL_NAMES: Dict[str, str] = {
    # queues and batches -- fed per-rank, drained per-rank (#1158: PP0 dropped
    # a HEALTH_CHECK the followers enqueued -> queue 6 vs 7)
    "waiting_queue": "#1158 queue composition diverges",
    "running_batch": "#823 TP batch-formation divergence",
    "cur_batch": "#821 per-rank current batch",
    "last_batch": "#823 sibling",
    "recv_reqs": "#713 input read before it is state",
    "chunked_req": "#951/#959 survives a retroactive unwind per rank",
    "grammar_queue": "queue sibling",
    "retracted_reqs": "#731 retraction set is per-rank",
    # prefix / match results -- THE #944 axis (told=8192 local=0)
    "prefix_indices": "#1028 only one writer, gated per-rank",
    "prefix_lens": "#645/#944 prefix divergence 2047 vs 10238",
    "cached_tokens": "#873 cached_tokens==0 is a per-rank validator verdict",
    "match_result": "#965 8 fields from ONE match, 3 invalidated",
    "host_hit": "#1157 store-read result per rank",
    "probed_hit": "#1176 witness half",
    "completed_local": "#1176 matched+loaded, per rank",
    "matched": "#1176 already-device-resident, per rank",
    "local_match_lens": "#791 the local half of the asymmetric rule",
    # pool / memory -- per-rank by design under uneven DCP/TP
    "available_size": "#834/#7128 available_size() differs per rank BY DESIGN",
    "evictable_size": "#694 evictable counted per rank",
    "protected_size": "#927 rank-divergent double ownership",
    "free_mib": "#683 free memory is a per-rank NVML read",
    "free_bytes": "#683 sibling",
    "available_bytes": "#1019 per-rank pool headroom",
    "mem_usage": "per-rank",
    "token_to_kv_pool_allocator": "#7128 the pool object itself is per-rank",
    # prefetch / storage -- per-rank IO progress (#1158 ballot mismatch)
    "prefetch_pending": "#1158 PP1 DECLINE while PP0/PP2 ADMIT",
    "prefetch_status": "#915 per-rank prefetch outcome",
    "_retired_prefetch": "#966 per-rank retired set",
    "host_pool": "#911/#915 703472 PP rows vs 30518 TP rows",
    "mem_pool_host": "#915 asymmetric across the flip",
    "staged": "#872/#972 per-rank staging progress",
    "loaded": "#1176 group-MIN read of a per-rank quantity",
    # slots / seats / carriers
    "mamba_pool_idx": "#790/#803 per-rank slot ids",
    "flip_mamba_slots": "#803 had to be agreed by a union collective",
    "req_to_token_pool": "#1040 per-phase index space",
    "resident_reqs": "#1189 per-rank resident counter",
    "parked": "#888b per-rank seat holder",
    "batch_is_full": "#888b latch, cleared on a path this layout forbids",
    # timing / wall clock -- two ranks straddle any deadline (#610)
    "perf_counter": "#610 wall-clock verdict is rank-local BY CONSTRUCTION",
    "deadline": "#610 sibling",
    "elapsed": "#610 sibling",
    "forward_entry_time": "#610 stamped on the rank that processed it",
    "wait_queue_entry_time": "#610 sibling",
}

# Attribute suffixes that make ANY receiver rank-local (`x.available_size()`,
# `self.tree_cache.evictable_size()`).  Kept separate so an alias of an
# unknown object still classifies by the attribute it reads.
RANK_LOCAL_ATTRS: Set[str] = set(RANK_LOCAL_NAMES)

# --------------------------------------------------------------------------
# (3) AGREEMENT POINTS -- where a rank-local value legitimately becomes a
#     group value.  Verified at this tree; each is a reduce/vote whose RESULT
#     is what the branch may then read.
# --------------------------------------------------------------------------
AGREEMENT_CALLS: Set[str] = {
    # scheduler.py:2616 _uniform_timeout_ballot -- MAX all_reduce of a
    # positional verdict; scheduler.py:6806 _update_uniform_pool_budget --
    # packed MIN all_reduce carrying corridor + head order + admit limit +
    # prefetch ballot.
    "_uniform_timeout_ballot",
    "_update_uniform_pool_budget",
    # pp_admission_congruence.py -- the #791 machinery
    "build_pp_admission_decision",
    "reconcile_pp_admission_decision",
    "forwarded_schedule",
    "order_batch_by_schedule",
    "congruent_rids",
    "void_pp_admission_decision",
    "prefix_len_for",
    "uniform_prefix_for",
    "record_return_trip",
    # pure agreement modules (payload builders + verdict readers)
    "build_prefetch_ballot_payload",
    "unpack_prefetch_ballot",
    "prefetch_done_under_ballot",
    "build_head_order_payload",
    "build_admit_limit_payload",
    "uniform_head_order",
    "head_decision",
    "admit_limit_decision",
    "batch_decision",
    "enforcer_gate",
    "congruence_verdict",
    "digest_pair",
    "reduce_pair_result",
    "agreement",
    "scope_for_world",
    # #1153's OWN named exemption: `rank_local_count_veto_applies`
    # (pp_admission_congruence.py:362-379) returns False on a FORWARDED
    # schedule, so the rank-local seat count is DISABLED exactly where it
    # could diverge, and a genuinely unseatable told rid raises
    # PPScheduleRefused instead of being silently skipped.  A guard that
    # already routes through this is the CORRECT pattern; flagging it would
    # make this rule a triage list.
    "rank_local_count_veto_applies",
    # MEASURED EXEMPTION, verified at the tree 2026-09-04.  The name reads
    # rank-local ("local_prefillable=", "running_batch=") but the callee is a
    # GENUINE group agreement: PrefillDelayer._negotiate_should_allow_prefill
    # -> _negotiate_should_allow_prefill_pure -> _gather_info ->
    # torch.distributed.all_gather_into_tensor (prefill_delayer.py:322) over
    # `self._gather_group`, and the verdict is derived from the GATHERED
    # tp0_info ("all"/"none"/"mixed"), never from the local arm.  The
    # rank-local values at schedule_policy.py:2104 are CONTRIBUTIONS to that
    # gather, not the decision.  Flagging it was this rule's own false
    # positive; the honest residual is noted in COVERAGE below (the
    # conditional ENTRY into that gather is a real hazard this rule does not
    # model -- one rank returning before `finalize()` leaves N-1 in the
    # all_gather).
    "negotiate_should_allow_prefill",
}

#: Local names that ARE the result of an agreement helper.  Reading one in a
#: guard is safe; the alias map cannot see through the call on its own.
GROUP_DERIVED_LOCALS: Set[str] = {
    "_count_veto",
    "told",
    "scheduled_extents",
    "scheduled_prefix_len",
}

# A value whose NAME says it already came from the group.  Reading one of
# these in a guard is the CORRECT pattern, not the defect.
GROUP_DERIVED_PREFIXES: Tuple[str, ...] = (
    "uniform_",
    "_uniform_",
    "group_",
    "_group_",
    "told",
    "scheduled_",
    "forwarded_",
    "agreed_",
    "world_",
)


class SimpleLine:
    """Minimal stand-in carrying only `lineno`, for the shared report path."""

    __slots__ = ("lineno",)

    def __init__(self, lineno: int):
        self.lineno = lineno


@dataclass
class Finding:
    file: str
    line: int
    func: str
    collective: str
    guard_line: int
    guard_expr: str
    rank_local_term: str
    why: str
    guard_kind: str
    family: str = "collective"


@dataclass
class _FuncState:
    name: str
    # local name -> (line, rank-local term it was assigned from)
    aliases: Dict[str, Tuple[int, str]] = field(default_factory=dict)
    agreement_lines: List[int] = field(default_factory=list)


def _dotted(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.Call):
        return _dotted(node.func)
    if isinstance(node, ast.Subscript):
        return _dotted(node.value)
    return ""


def _callee(node: ast.Call) -> str:
    f = node.func
    if isinstance(f, ast.Attribute):
        return f.attr
    if isinstance(f, ast.Name):
        return f.id
    return ""


def _receiver(node: ast.Call) -> str:
    f = node.func
    if isinstance(f, ast.Attribute):
        return _dotted(f.value)
    return ""


def _is_collective(node: ast.Call) -> Optional[str]:
    name = _callee(node)
    if name not in COLLECTIVE_CALLS:
        return None
    recv = _receiver(node)
    if recv in NOT_COLLECTIVE_RECEIVERS:
        return None
    # torch.gather/torch.scatter/tensor.gather are elementwise ops
    if name in {"gather", "scatter", "reduce"} and not (
        "group" in recv or "dist" in recv or "pg" in recv
    ):
        return None
    return f"{recv}.{name}" if recv else name


def _is_group_act(node: ast.Call) -> Optional[str]:
    name = _callee(node)
    if name in GROUP_ACT_CALLS:
        recv = _receiver(node)
        return f"{recv}.{name}" if recv else name
    return None


def _raised_group_stop(node: ast.Raise) -> Optional[str]:
    exc = node.exc
    if exc is None:
        return None
    name = _dotted(exc).rsplit(".", 1)[-1]
    return name if name in GROUP_STOP_EXCEPTIONS else None


def _group_derived(text: str) -> bool:
    leaf = text.rsplit(".", 1)[-1]
    if leaf in GROUP_DERIVED_LOCALS or text in GROUP_DERIVED_LOCALS:
        return True
    return any(leaf.startswith(p) or text.startswith(p) for p in GROUP_DERIVED_PREFIXES)


def _rank_local_terms(expr: ast.AST, st: _FuncState) -> List[Tuple[str, str]]:
    """Every rank-local term this expression reads, as (term, why)."""
    out: List[Tuple[str, str]] = []
    for n in ast.walk(expr):
        if isinstance(n, ast.Attribute):
            if n.attr in RANK_LOCAL_ATTRS:
                full = _dotted(n)
                if not _group_derived(full):
                    out.append((full, RANK_LOCAL_NAMES[n.attr]))
        elif isinstance(n, ast.Name):
            if n.id in RANK_LOCAL_NAMES and not _group_derived(n.id):
                out.append((n.id, RANK_LOCAL_NAMES[n.id]))
            elif n.id in st.aliases:
                _, src = st.aliases[n.id]
                if not _group_derived(n.id):
                    out.append((n.id, f"alias of {src}"))
    # dedupe, keep order
    seen: Set[str] = set()
    uniq = []
    for t, w in out:
        if t not in seen:
            seen.add(t)
            uniq.append((t, w))
    return uniq


class _FunctionScan(ast.NodeVisitor):
    """One pass per function body, carrying the stack of dominating tests."""

    def __init__(self, path: str, func: ast.AST, qualname: str):
        self.path = path
        self.qualname = qualname
        self.st = _FuncState(qualname)
        self.findings: List[Finding] = []
        # (line, source-ish text, kind) of tests that dominate the cursor
        self._guards: List[Tuple[int, str, str]] = []
        self._func = func

    # ---- alias + agreement collection (whole body, order preserved) -------
    def _prepass(self) -> None:
        for n in ast.walk(self._func):
            if isinstance(n, ast.Assign) and len(n.targets) == 1:
                tgt = n.targets[0]
                if isinstance(tgt, ast.Name):
                    terms = _rank_local_terms(n.value, self.st)
                    if terms:
                        self.st.aliases[tgt.id] = (n.lineno, terms[0][0])
            elif isinstance(n, ast.Call):
                if _callee(n) in AGREEMENT_CALLS:
                    self.st.agreement_lines.append(n.lineno)

    def run(self) -> List[Finding]:
        self._prepass()
        self._walk_body(getattr(self._func, "body", []))
        return self.findings

    # ---- the control-dependence walk -------------------------------------
    def _walk_body(self, body: List[ast.stmt]) -> None:
        # Early-exit guards: `if <rank-local>: return/continue/break/raise`
        # makes the REST of this body control-dependent on the negation.
        # That is the shape semgrep's pattern-inside cannot see.
        carried: List[Tuple[int, str, str]] = []
        for stmt in body:
            for g in carried:
                self._guards.append(g)
            self._scan_stmt(stmt)
            for _ in carried:
                self._guards.pop()

            if isinstance(stmt, ast.If) and self._exits(stmt.body) and not stmt.orelse:
                carried.append(
                    (stmt.test.lineno, ast.unparse(stmt.test), "early-exit-negation")
                )

    @staticmethod
    def _exits(body: List[ast.stmt]) -> bool:
        if not body:
            return False
        last = body[-1]
        return isinstance(last, (ast.Return, ast.Continue, ast.Break, ast.Raise))

    def _scan_stmt(self, stmt: ast.stmt) -> None:
        if isinstance(stmt, ast.If):
            self._guards.append((stmt.test.lineno, ast.unparse(stmt.test), "if"))
            self._walk_body(stmt.body)
            self._guards.pop()
            if stmt.orelse:
                self._guards.append(
                    (stmt.test.lineno, "not (" + ast.unparse(stmt.test) + ")", "else")
                )
                self._walk_body(stmt.orelse)
                self._guards.pop()
            self._scan_expr(stmt.test)
            return
        if isinstance(stmt, (ast.While,)):
            self._guards.append((stmt.test.lineno, ast.unparse(stmt.test), "while"))
            self._walk_body(stmt.body)
            self._walk_body(stmt.orelse)
            self._guards.pop()
            self._scan_expr(stmt.test)
            return
        if isinstance(stmt, (ast.For, ast.AsyncFor)):
            self._walk_body(stmt.body)
            self._walk_body(stmt.orelse)
            self._scan_expr(stmt.iter)
            return
        if isinstance(stmt, (ast.With, ast.AsyncWith)):
            self._walk_body(stmt.body)
            for item in stmt.items:
                self._scan_expr(item.context_expr)
            return
        if isinstance(stmt, ast.Try):
            self._walk_body(stmt.body)
            for h in stmt.handlers:
                self._walk_body(h.body)
            self._walk_body(stmt.orelse)
            self._walk_body(stmt.finalbody)
            return
        if isinstance(stmt, ast.Raise):
            exc = _raised_group_stop(stmt)
            if exc is not None:
                self._report_at(stmt.lineno, f"raise {exc}", [], "group-stop")
            if stmt.exc is not None:
                self._scan_expr(stmt.exc)
            return
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return  # nested def: its own function, scanned separately
        for child in ast.iter_child_nodes(stmt):
            if isinstance(child, ast.expr):
                self._scan_expr(child)
            elif isinstance(child, ast.stmt):
                self._scan_stmt(child)

    def _scan_expr(self, expr: ast.expr) -> None:
        for n in ast.walk(expr):
            if not isinstance(n, ast.Call):
                continue
            coll = _is_collective(n)
            if coll is not None:
                self._report(n, coll, expr, "collective")
                continue
            act = _is_group_act(n)
            if act is not None:
                self._report(n, act, expr, "group-act")

    # ---- reporting + the exemption ---------------------------------------
    def _report(
        self, call: ast.Call, coll: str, ctx: ast.expr, family: str = "collective"
    ) -> None:
        # Ternary / boolean-operand guards local to THIS expression, plus the
        # dominating stack.
        extra: List[Tuple[int, str, str]] = []
        for n in ast.walk(ctx):
            if isinstance(n, ast.IfExp):
                extra.append((n.test.lineno, ast.unparse(n.test), "ternary"))
            elif isinstance(n, ast.BoolOp):
                for v in n.values:
                    if v is not call and not any(x is call for x in ast.walk(v)):
                        extra.append((v.lineno, ast.unparse(v), "bool-operand"))
        self._report_at(call.lineno, coll, extra, family)

    def _report_at(
        self,
        lineno: int,
        act: str,
        extra_guards: List[Tuple[int, str, str]],
        family: str,
    ) -> None:
        coll = act
        call = SimpleLine(lineno)
        guards = list(self._guards) + list(extra_guards)

        for gline, gtext, gkind in guards:
            try:
                gexpr = ast.parse(gtext, mode="eval").body
            except SyntaxError:
                continue
            terms = _rank_local_terms(gexpr, self.st)
            if not terms:
                continue
            # EXEMPTION, ordered: an agreement call must lexically PRECEDE the
            # collective.  Reducing afterwards is the #1176 defect, not a fix.
            if any(a <= call.lineno for a in self.st.agreement_lines):
                continue
            term, why = terms[0]
            self.findings.append(
                Finding(
                    file=self.path,
                    line=call.lineno,
                    func=self.qualname,
                    collective=coll,
                    guard_line=gline,
                    guard_expr=gtext if len(gtext) < 140 else gtext[:137] + "...",
                    rank_local_term=term,
                    why=why,
                    guard_kind=gkind,
                    family=family,
                )
            )
            return  # one finding per act site


def scan_file(path: str) -> List[Finding]:
    try:
        src = open(path, encoding="utf-8").read()
        tree = ast.parse(src, filename=path)
    except (SyntaxError, UnicodeDecodeError):
        return []
    out: List[Finding] = []
    stack: List[str] = []

    def visit(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qual = ".".join(stack + [child.name])
                out.extend(_FunctionScan(path, child, qual).run())
                stack.append(child.name)
                visit(child)
                stack.pop()
            elif isinstance(child, ast.ClassDef):
                stack.append(child.name)
                visit(child)
                stack.pop()
            else:
                visit(child)

    visit(tree)
    return out


def scan_paths(paths: List[str]) -> List[Finding]:
    out: List[Finding] = []
    for p in paths:
        if os.path.isfile(p):
            out.extend(scan_file(p))
            continue
        for dirpath, dirnames, filenames in os.walk(p):
            dirnames[:] = [d for d in dirnames if d != "__pycache__"]
            for fn in sorted(filenames):
                if fn.endswith(".py"):
                    out.extend(scan_file(os.path.join(dirpath, fn)))
    return out


_SELF_TEST_VIOLATION = '''
import torch

class S:
    def bad_if(self):
        if self.waiting_queue:
            torch.distributed.all_reduce(t, group=self.tp_cpu_group)

    def bad_early_exit(self):
        if not self.waiting_queue:
            return
        torch.distributed.barrier(group=self.pp_group)

    def bad_alias(self):
        q = self.waiting_queue
        if len(q) > 0:
            self.pp_group.broadcast_object_list(d, src=0)

    def bad_while(self):
        while self.token_to_kv_pool_allocator.available_size() < need:
            torch.distributed.all_reduce(t)

    def bad_boolop(self):
        ok = self.prefetch_pending and self.tp_group.all_gather_object(x)

    def clean_uniform(self):
        if self._uniform_corridor_width > 0:
            torch.distributed.all_reduce(t)

    def clean_after_agreement(self):
        v = self._update_uniform_pool_budget()
        if self.waiting_queue:
            torch.distributed.all_reduce(t)

    def clean_no_guard(self):
        torch.distributed.all_reduce(t)
'''


def self_test() -> int:
    """CAN-FIRE PROOF, both directions, on a planted specimen."""
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        red = os.path.join(d, "planted.py")
        open(red, "w").write(_SELF_TEST_VIOLATION)
        got = scan_paths([red])
        names = sorted({f.func for f in got})
        want_red = [
            "S.bad_if",
            "S.bad_early_exit",
            "S.bad_alias",
            "S.bad_while",
            "S.bad_boolop",
        ]
        want_green = ["S.clean_uniform", "S.clean_after_agreement", "S.clean_no_guard"]
        ok = True
        print("=== RED direction: planted violations must be reported ===")
        for w in want_red:
            hit = w in names
            print(f"  {'FIRED ' if hit else 'MISSED'} {w}")
            ok &= hit
        print("=== GREEN direction: correct shapes must NOT be reported ===")
        for w in want_green:
            quiet = w not in names
            print(f"  {'quiet ' if quiet else 'FALSE+'} {w}")
            ok &= quiet
        # removal proof: strip the violations, rule must go silent
        green = os.path.join(d, "clean.py")
        # The removal leg must produce a file that PARSES.  A line filter on
        # "bad_" strips only the five ``def bad_*(self):`` headers and leaves
        # their bodies dangling; the result is a SyntaxError, scan_file()
        # returns [] for it, and the resulting "0 findings" would measure
        # UNPARSABILITY rather than silence -- the exact INDIKATOR-GESETZ
        # failure this rule exists to catch.  Delete the FunctionDefs on the
        # AST instead, and assert the survivor parses before scanning it.
        _tree = ast.parse(_SELF_TEST_VIOLATION)
        for _node in ast.walk(_tree):
            body = getattr(_node, "body", None)
            if isinstance(body, list):
                _node.body = [
                    s
                    for s in body
                    if not (
                        isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and s.name.startswith("bad_")
                    )
                ] or [ast.Pass()]
        keep = ast.unparse(ast.fix_missing_locations(_tree))
        try:
            ast.parse(keep)
        except SyntaxError as exc:  # pragma: no cover - guards the guard
            print(f"  REMOVAL FIXTURE DOES NOT PARSE ({exc.msg}) -- leg is vacuous")
            return 1
        _kept = sorted(
            n.name
            for n in ast.walk(ast.parse(keep))
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
        print(f"  removal fixture parses; retained {_kept}")
        open(green, "w").write(keep)
        n = len(scan_paths([green]))
        print(f"=== REMOVAL: violations deleted -> {n} findings (want 0) ===")
        ok &= n == 0
        print("\nSELF-TEST", "PASS" if ok else "FAIL")
        return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", default=[])
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    paths = a.paths or ["/spinning/wt-weg1/python/sglang/srt/managers"]
    fs = scan_paths(paths)
    for f in fs:
        rel = f.file
        print(f"{rel}:{f.line}  {f.func}")
        print(f"    collective : {f.collective}")
        print(f"    guarded by : {f.guard_kind} @ line {f.guard_line}: {f.guard_expr}")
        print(f"    rank-local : {f.rank_local_term}   ({f.why})")
    print(f"\n{len(fs)} finding(s)")
    return 1 if fs else 0


if __name__ == "__main__":
    sys.exit(main())
