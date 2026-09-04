#!/usr/bin/env python3
"""CLASS B -- a wait that cannot be guaranteed to end.

Two mechanical rules, both runnable as written:

  B1  OBSERVE-NOT-DRIVE   a loop whose only contact with a distributed work
      handle is a REPORTING predicate (is_completed / query / poll / done),
      with no DRIVING call anywhere in the loop body.  This is #630 verbatim:
      "is_completed() REPORTS state, wait() DRIVES the transfer.  With both
      peers polling, neither side advances the exchange and every rank sits
      until its own deadline -- the protective bound WAS the livelock."

  B2  WALL-CLOCK BOUND ON A ROUND-ADVANCING PROTOCOL   a wait/defer whose
      only end is a wall-clock deadline, while the thing being waited on
      advances in ROUNDS (lap / seq / generation / microbatch / pass / frame).
      This is #1180's fix shape inverted: "bound the row-probe defer in RING
      LAPS, not wall-clock seconds ... a wall-clock horizon's lapse was
      unreachable on the weg1b7 specimen, so the guard could not have fired
      on the boot it was built for."  A function that already counts a round
      quantity against a cap does NOT fire.

  B3  NAKED UNBOUNDED WAIT (added 2026-09-04, second pass).  B1 only ever
      inspects LOOPS, and that left the sharpest instance in the tree
      invisible: `grammar_manager._drain_pp_sync_work` is a two-line `for`
      over a work list ending in a bare `p2p_work.work.wait()` -- no poll, no
      predicate, no loop test, so no REPORT call for B1 to see.  It is
      byte-for-byte the shape #973 fixed one file away in
      `_pp_join_comm_work` ("THE WAIT USED TO BE NAKED, and that is what
      boot 2 of window-flip-0828 died on: p2p_work.work.wait() with nothing
      bounding it, on a gloo group whose own timeout is two hours").
      B3 fires on a zero-argument `.wait()` on a work-handle-shaped
      receiver, on `dist.i{send,recv}(...).wait()`, on a zero-argument
      `.join()` on a thread-shaped receiver, and on
      `executor.shutdown(wait=True)` with no timeout.
      EXEMPT: the two canon modules that own the sanctioned unbounded wait
      (mem_cache/hicache_collective.py's ParkedWait and
      distributed/pp_object_recv.py -- there the unbounded wait is ON ITS
      OWN THREAD and the deadline is on the join), and any site inside a
      documented escape branch (`if budget <= 0:`, `if timeout_s <= 0:`,
      `if not liveness_enabled():`).

Usage:
    python3 devtools/classB_wait_rules.py <path> [--rule B1|B2|B3|both|all] [--json]

Exit code 1 if any finding, 0 if clean -- so it can gate.

PRECISION NOTES -- READ BEFORE ACTING ON A COUNT (the #950 lesson: a rule's
own honesty about its false-positive shape is part of the rule).

  B1 is PRECISE inside python/sglang/srt/distributed (8 findings, all read)
  and a TRIAGE LIST outside it (84 tree-wide, measured 2026-09-04).  The
  noise is NAME COLLISION: `poll` / `empty` / `query` / `done` also name
  methods on queues, tensors, dicts and model modules that have nothing to
  do with a distributed Work handle, and the AST cannot type the receiver.
  Every finding prints its receiver text -- a finding whose receiver is not
  a work/event/future/stop-flag is noise.  Do not quote the tree-wide count
  as a defect count.

  B1 also has a KNOWN, DELIBERATE false positive on CUDA events: polling
  `event.query()` is legitimate because the GPU stream advances the copy
  regardless of the poller (barlink_bar1 `_wait_ctl_event`, tp_ar_pipeline
  `_EventPool.acquire`).  The #630 defect is specific to a transport where
  the WAITER must drive -- gloo.  The rule cannot tell those apart; the
  reader must.

  B2 is TIGHT by construction (9 findings tree-wide) because its round
  vocabulary is CURATED to this fork's PP/flip protocol.  That is also its
  blind spot: a round-advancing quantity spelled with a word not on the
  list (`consumed`, `sent_count` in pp_chain_receiver) is invisible.
  Widening the list re-explodes it (first draft: 77 findings, ~60 noise).

  BOTH rules read `time.<f>()` only through a module named `time`/`_time`.
  A wait that does `from time import monotonic` or `import time as _t` is
  invisible to both.  Measured: the first can-fire plant used `_t` and did
  NOT fire until rewritten to the canonical form.

  B3 is PRECISE on the receiver-name axis and BLIND on the type axis: it
  decides "is this a distributed work handle" from the SPELLING of the
  receiver (`work`, `p2p_work.work`, `handle`, `async_handle`, `req`, `w`).
  A wait on a handle named something else -- `self._x.wait()` -- is
  invisible, and a `.wait()` on a threading.Event named `work_ready` is a
  false positive.  Measured 2026-09-04 over python/sglang/srt: 17 findings,
  all 17 read by hand, 11 real (a torch.distributed Work or an unbounded
  thread/executor join) and 6 name collisions.  It is a SHORT list on
  purpose -- read all of it, do not quote the count.

  B3's exemption is SYNTACTIC, not semantic: it looks for an enclosing `if`
  whose test names a deadline identifier compared against 0, or a
  `*_enabled()` call.  A module that spells its escape hatch differently
  will show up as a finding, which is the safe direction.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from typing import Dict, Iterator, List, Optional, Tuple

# ---------------------------------------------------------------------------
# vocabularies
# ---------------------------------------------------------------------------

# Calls that only ASK whether something happened.  None of these advance a
# gloo/NCCL exchange, a queue, or a state machine.
REPORT_CALLS = {
    "is_completed",
    "is_success",
    "query",
    "done",
    "is_set",
    "ready",
    "is_ready",
    "empty",
    "qsize",
    "poll",
    "peek",
    "probe",
    "test",
    "locked",
}

# Calls with positive evidence that they DRIVE progress, plus the local
# helpers that wrap them.  A loop containing any of these is not a B1.
DRIVE_CALLS = {
    "wait",
    "synchronize",
    "recv",
    "send",
    "isend",
    "irecv",
    "barrier",
    "all_reduce",
    "all_gather",
    "broadcast",
    "reduce",
    "advance",
    "_advance",
    "step",
    "run",
    "execute",
    "drain",
    "join",
    "acquire",
    "get",
    "put",
    "sleep",  # a sleeping poll at least yields; see NOTE below
    "consume",
    "consume_up_to",
    "_block_on",
    "_do_recv",
    "_do_send",
    "bounded_wait",
    "bounded_recv",
    "work",
    "forward",
    "launch",
    "flush",
    "commit",
    "post",
    "read",
    "write",
    "copy_",
    "item",
}
# NOTE on "sleep": time.sleep in a poll loop does NOT drive the peer, but it
# does hand the GIL back, and every real B1 specimen on this rig (the #630
# corpse) contained one.  Keeping it in DRIVE would have made the rule blind
# to its own founding specimen, so `sleep` is stripped below when the loop
# ALSO contains a REPORT call on a work-handle-shaped receiver.

WALLCLOCK_FUNCS = {"monotonic", "time", "perf_counter", "monotonic_ns", "time_ns"}

# A wall-clock deadline, however it is spelled.  `bound`/`since`/`lapsed` were
# added after the first draft MISSED the sharpest specimen in the tree
# (`_pp_occupant_horizon_lapsed`, a 90-SECOND bound on the microbatch slot
# ring): it spells its deadline `bound` and its origin `_pp_occupant_since`,
# and its only "horizon" token is in the FUNCTION NAME, which the first draft
# did not scan.
DEADLINE_NAME = re.compile(
    r"(deadline|budget|timeout|expire|expiry|abort_after|horizon|lapse|"
    r"\bbound\b|_bound|\bsince\b|_since|stall_after|_slo|"
    r"_s$|_sec|seconds|elapsed|start$|end$|until)",
    re.I,
)

# The vocabulary of a ROUND-ADVANCING protocol ON THIS FORK.  Deliberately
# CURATED, not generic: the first draft matched `perf_counter` (the clock
# itself), the builtin `round()`, and gc `gen` -- 77 findings, ~60 of them
# noise.  Every token below names a quantity the PP ring / flip protocol
# actually advances, and each was read out of the code that carries it.
ROUND_NAME = re.compile(
    r"("
    r"\blaps?\b|ring_lap|chain_lap"          # #1180's own unit
    r"|\bseq(_num)?\b|sequence_number"        # barlink_host flags[] seq
    r"|generation|binding_generation"          # #911/#1060 pool binding
    r"|\bmb_id\b|microbatch|mbs\b"            # #757/#798 microbatch slot ring
    r"|forward_ct|fwd_ct|\bpass_ct\b"         # #1058 step-count dissent
    r"|occurrence|_occurrences"                # #1180 defer occurrence
    r"|proxy_frame|frame_stamp|\bframe_id\b"  # #631 proxy frame identity
    r"|cutover_gen|flip_gen|\bepoch\b"
    r"|slot_ring|next_mb_id|resume_slot"
    r")",
    re.I,
)

# A cap counted in the round quantity itself -- the #1180 remedy.  Presence of
# one of these means the function already bounds in the right unit and is NOT
# the defect.
ROUND_CAP = re.compile(
    r"(DEFER_CAP|_CAP\b|MAX_LAPS|MAX_ROUNDS|MAX_PASSES|MAX_MESSAGES|"
    r"max_messages|max_laps|max_rounds|max_passes|occurrence_cap|lap_cap)",
)

WAITY_NAME = re.compile(
    r"(wait|recv|receive|join|block|park|defer|poll|await|stall|hold|drain|sync"
    # A BOUND-PREDICATE is a Class-B site even when it does not itself block:
    # it IS the bound the waiter consults.  `_pp_occupant_horizon_lapsed`
    # blocks nothing and decides everything.
    r"|lapse|lapsed|horizon|overdue|expired|_bound|deadline)",
    re.I,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def call_name(node: ast.AST) -> Optional[str]:
    if not isinstance(node, ast.Call):
        return None
    f = node.func
    if isinstance(f, ast.Attribute):
        return f.attr
    if isinstance(f, ast.Name):
        return f.id
    return None


def receiver_text(node: ast.Call) -> str:
    f = node.func
    if isinstance(f, ast.Attribute):
        try:
            return ast.unparse(f.value)
        except Exception:  # pragma: no cover - unparse is total on py3.9+
            return ""
    return ""


def walk_calls(node: ast.AST) -> Iterator[ast.Call]:
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            yield n


def walk_names(node: ast.AST) -> Iterator[str]:
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            yield n.id
        elif isinstance(n, ast.Attribute):
            yield n.attr


def iter_py(path: str) -> Iterator[str]:
    if os.path.isfile(path):
        yield path
        return
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d not in {"__pycache__", ".git", "3rdparty"}]
        for f in files:
            if f.endswith(".py"):
                yield os.path.join(root, f)


# ---------------------------------------------------------------------------
# RULE B1 -- observe, never drive
# ---------------------------------------------------------------------------


def rule_b1(tree: ast.AST, path: str) -> List[Dict]:
    out: List[Dict] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.While, ast.For, ast.AsyncFor)):
            continue
        body_nodes = list(node.body) + ([node.test] if isinstance(node, ast.While) else [])
        reports: List[Tuple[str, str, int]] = []
        drives: List[str] = []
        for sub in body_nodes:
            for c in walk_calls(sub):
                nm = call_name(c)
                if nm is None:
                    continue
                if nm in REPORT_CALLS:
                    reports.append((nm, receiver_text(c), c.lineno))
                if nm in DRIVE_CALLS and nm != "sleep":
                    drives.append(nm)
        if not reports:
            continue
        if drives:
            continue
        out.append(
            {
                "rule": "B1",
                "file": path,
                "line": node.lineno,
                "report_calls": sorted({f"{r}() on {recv or '<local>'}" for r, recv, _ in reports}),
                "why": (
                    "loop polls a REPORTING predicate with no DRIVING call in the "
                    "body -- #630: the bound cannot be the progress"
                ),
            }
        )
    return out


# ---------------------------------------------------------------------------
# RULE B2 -- wall-clock bound on a round-advancing protocol
# ---------------------------------------------------------------------------


def rule_b2(tree: ast.AST, path: str) -> List[Dict]:
    out: List[Dict] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        # (1) wall-clock deadline evidence
        clock_lines = [
            c.lineno
            for c in walk_calls(node)
            if call_name(c) in WALLCLOCK_FUNCS
            and isinstance(c.func, ast.Attribute)
            and isinstance(c.func.value, ast.Name)
            and c.func.value.id in {"time", "_time"}
        ]
        if not clock_lines:
            continue
        # The FUNCTION'S OWN NAME counts as deadline evidence: a wait can spell
        # its bound only in its name (`_pp_occupant_horizon_lapsed`).
        names = list(walk_names(node)) + [node.name]
        if not any(DEADLINE_NAME.search(n) for n in names):
            continue

        # (2) the function actually WAITS or DEFERS -- a blocking/protocol
        #     primitive must appear, or the function itself must be named as a
        #     wait/defer.  A pure measurement function that happens to read a
        #     clock is not a Class-B site.
        blocking = {
            "wait",
            "recv",
            "recv_object",
            "join",
            "acquire",
            "synchronize",
            "is_completed",
            "query",
            "_do_recv",
            "_block_on",
            "bounded_wait",
            "bounded_recv",
            "_pp_recv_typed_dict",
            "_pp_recv_dict_from_prev_stage",
            "check_aborted",
            "raise_if_peer_lost",
        }
        waity = bool(WAITY_NAME.search(node.name)) or any(
            (call_name(c) or "") in blocking for c in walk_calls(node)
        )
        if not waity:
            continue

        # (3) the SUBJECT advances in rounds
        round_names = sorted({n for n in names if ROUND_NAME.search(n)})
        if not round_names:
            continue

        # (4) exemption: the function ALREADY counts the round quantity
        #     against a cap -- that is the #1180 remedy, not the defect.
        src_names = set(names)
        capped = any(ROUND_CAP.search(n) for n in src_names)
        if capped:
            continue

        out.append(
            {
                "rule": "B2",
                "file": path,
                "line": node.lineno,
                "func": node.name,
                "clock_at": clock_lines[:3],
                "round_state": round_names[:6],
                "why": (
                    "wait bounded in WALL-CLOCK while its subject advances in "
                    "ROUNDS, and no cap is counted in the round unit -- #1180"
                ),
            }
        )
    return out


# ---------------------------------------------------------------------------
# RULE B3 -- a naked unbounded wait / join, loop or no loop
# ---------------------------------------------------------------------------

#: Receiver spellings that name a torch.distributed Work handle on this fork.
WORK_RECEIVER = re.compile(
    r"(^|[.\[])(p2p_)?work(s)?($|[.\[])|handle|async_handle|(^|\.)req$|(^|\.)w$"
    r"|(^|\.)fut$|future|_work$",
    re.I,
)
#: Receiver spellings that name a thread / process / executor.
THREAD_RECEIVER = re.compile(
    r"(thread|proc(ess)?$|_p$|^p$|worker|executor|pool)", re.I
)
#: Calls whose RESULT is a Work: `dist.irecv(...).wait()` has no name to match.
WORK_PRODUCERS = {
    "isend",
    "irecv",
    "batch_isend_irecv",
    "all_reduce",
    "all_gather",
    "all_gather_object",
    "broadcast",
    "reduce",
    "reduce_scatter",
    "all_to_all_single",
    "barrier",
    "send_object",
    "recv_object",
}
#: The two modules that OWN the sanctioned unbounded wait (it runs on its own
#: thread there; the deadline lives on the join).  A finding inside them would
#: be the canon reported as its own violation.
CANON_MODULES = ("hicache_collective.py", "pp_object_recv.py")

#: An enclosing `if` that names one of these AND compares against 0, or calls
#: a `*_enabled()` predicate, is a documented escape hatch.
ESCAPE_TEST = re.compile(
    r"(budget|timeout|deadline|bound|abort_after|stall)\w*\s*(<=|<|==|is None)"
    r"|not\s+\w*_enabled\(\)|^\s*not\s+\w+$",
    re.I,
)


def _parented(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child._b3_parent = node  # type: ignore[attr-defined]


def _in_escape_branch(node: ast.AST) -> bool:
    cur = getattr(node, "_b3_parent", None)
    while cur is not None:
        if isinstance(cur, ast.If):
            try:
                test = ast.unparse(cur.test)
            except Exception:  # pragma: no cover
                test = ""
            if ESCAPE_TEST.search(test):
                return True
        cur = getattr(cur, "_b3_parent", None)
    return False


def rule_b3(tree: ast.AST, path: str) -> List[Dict]:
    if any(path.endswith(m) for m in CANON_MODULES):
        return []
    _parented(tree)
    out: List[Dict] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = call_name(node)
        recv = receiver_text(node)
        kind = why = None

        if name == "wait" and not node.args and not node.keywords:
            inner = node.func.value if isinstance(node.func, ast.Attribute) else None
            produced = isinstance(inner, ast.Call) and call_name(inner) in WORK_PRODUCERS
            if produced or WORK_RECEIVER.search(recv):
                kind = "naked-work-wait"
                why = (
                    "zero-argument Work.wait() with nothing bounding it -- #973: "
                    "the group's own timeout is the only end, and it is 2h"
                )
        elif name == "join" and not node.args and not node.keywords:
            if THREAD_RECEIVER.search(recv):
                kind = "naked-thread-join"
                why = (
                    "thread/process join with no timeout -- #885: a joiner that "
                    "cannot give up inherits the wedge it is joining"
                )
        elif name == "shutdown":
            waits = any(
                k.arg == "wait" and getattr(k.value, "value", None) is True
                for k in node.keywords
            )
            timed = any(k.arg in {"timeout", "cancel_futures"} for k in node.keywords)
            if waits and not timed:
                kind = "executor-shutdown-wait"
                why = (
                    "executor.shutdown(wait=True) joins every worker with no "
                    "timeout -- #957: fatal when reached from a signal handler"
                )
        if kind is None:
            continue
        if _in_escape_branch(node):
            continue
        out.append(
            {
                "rule": "B3",
                "file": path,
                "line": node.lineno,
                "func": kind,
                "receiver": recv or "<expr>",
                "why": why,
            }
        )
    return out


# ---------------------------------------------------------------------------


def scan(path: str, rules: str) -> List[Dict]:
    findings: List[Dict] = []
    for f in iter_py(path):
        try:
            src = open(f, "r", encoding="utf-8", errors="replace").read()
            tree = ast.parse(src, filename=f)
        except SyntaxError:
            continue
        if rules in ("B1", "both", "all"):
            findings += rule_b1(tree, f)
        if rules in ("B2", "both", "all"):
            findings += rule_b2(tree, f)
        if rules in ("B3", "all"):
            findings += rule_b3(tree, f)
    return findings


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument(
        "--rule", choices=["B1", "B2", "B3", "both", "all"], default="all"
    )
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    f = scan(a.path, a.rule)
    if a.json:
        print(json.dumps(f, indent=2))
    else:
        for x in sorted(f, key=lambda d: (d["rule"], d["file"], d["line"])):
            print(f"{x['rule']}  {x['file']}:{x['line']}  {x.get('func','<loop>')}")
            print(f"      {x['why']}")
            if x["rule"] == "B1":
                print(f"      polls: {', '.join(x['report_calls'])}")
            elif x["rule"] == "B2":
                print(f"      round state: {', '.join(x['round_state'])}")
            else:
                print(f"      receiver: {x['receiver']}")
        print(f"\n{len(f)} finding(s)")
    return 1 if f else 0


if __name__ == "__main__":
    sys.exit(main())
