"""#552 -- the kvso restore signal, its attribution, and the FIFO it rides on.

WHY THIS FILE EXISTS. Boot K2 armed `--kv-session-offload-resume-under-spec`,
saw the resume seed republished on all three ranks, saw zero host-finishes, and
still had to record the on-device rejoin as UNCORROBORATED -- because its
restore cell was keyed on the string ``restored to device``, which
``KVSessionOffloadManager._close_slot`` emits through ``logger.debug``. Every
boot in the matrix runs at the default ``--log-level info``. That cell was
structurally zero: it could not have reported the rejoin whether or not it
happened, and the matrix harness's own smoke fixture was green because the
fixture line was PARAPHRASED rather than copied from the code.

So the defect is not a missing mechanism, it is an instrument that cannot see
the mechanism. The tests below pin the three things that make it visible and
keep it visible:

  * no matrix signal may be a string only ``logger.debug`` emits (the general
    falsifier -- it would have caught this class in the first round);
  * the INFO-level ``RESTORE complete`` line must match the harness regex and
    must ATTRIBUTE the rejoin (``spec=1``) rather than leave it inferred from
    the earlier seed line;
  * restore order is FIFO by spill time, which nothing but ``dict`` insertion
    order implements.

Hermetic: no GPU, no CUDA, no server, no model. Run with
CUDA_VISIBLE_DEVICES=99.
"""

import ast
import importlib.util
import os
import unittest
from pathlib import Path
from types import SimpleNamespace

_REPO = Path(__file__).resolve().parents[3]
_KVSO_SRC = _REPO / "python" / "sglang" / "srt" / "managers" / "kv_session_offload.py"
_DRIVE_SRC = _REPO / "scripts" / "dev" / "spill_matrix" / "drive.py"


def _load_drive_signals():
    """Import the matrix harness's SIGNALS table without executing its main.

    ``drive.py`` is a script, not a package module, so it is loaded by path.
    It imports only stdlib at module scope.
    """
    spec = importlib.util.spec_from_file_location("_spill_matrix_drive", _DRIVE_SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SIGNALS


def _logging_format_strings_by_level(src_path):
    """Every literal format string the module logs, split into the levels an
    operator can actually see at ``--log-level info`` and the ones they cannot.

    ``self._log`` counts as VISIBLE: it is info on rank 0 and debug elsewhere,
    so a rank-0 log carries it. ``logger.debug`` is the invisible bucket.
    Anything else (info / warning / error) is visible.
    """
    tree = ast.parse(src_path.read_text(), filename=str(src_path))
    visible, invisible = set(), set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        attr = node.func.attr
        target = node.func.value
        is_logger = isinstance(target, ast.Name) and target.id == "logger"
        is_self_log = isinstance(target, ast.Name) and target.id == "self"
        if is_logger and attr in ("debug", "info", "warning", "error", "critical"):
            bucket = invisible if attr == "debug" else visible
        elif is_self_log and attr == "_log":
            bucket = visible
        else:
            continue
        if node.args and isinstance(node.args[0], ast.Constant):
            value = node.args[0].value
            if isinstance(value, str):
                bucket.add(value)
    return visible, invisible


def _literal_prefix(pattern: str) -> str:
    """The leading literal run of a regex, with backslash escapes resolved.

    Signal patterns in the harness are literal text with a regex tail
    (``rid=.*``, ``\\d+``). The leading run is enough to locate the format
    string it was read from.
    """
    out = []
    i = 0
    while i < len(pattern):
        ch = pattern[i]
        if ch == "\\" and i + 1 < len(pattern):
            out.append(pattern[i + 1])
            i += 2
            continue
        if ch in ".*+?[](){}|^$":
            break
        out.append(ch)
        i += 1
    return "".join(out)


class TestNoMatrixSignalIsDebugOnly(unittest.TestCase):
    """THE GENERAL FALSIFIER for the always-zero-cell family."""

    def test_every_signal_found_in_the_manager_is_reachable_at_info(self):
        """A signal whose text the manager only ever passes to ``logger.debug``
        is an always-zero cell at the matrix's own log level.

        Only signals whose literal prefix is actually FOUND in
        ``kv_session_offload.py`` are judged here -- other cells legitimately
        read strings emitted by other modules, and this test must not pretend
        to know about those.

        CAN-FAIL: point H4 back at ``restored to device`` and this goes red
        naming that cell, because the sole emitter of that text is
        ``_close_slot``'s ``logger.debug``.
        """
        signals = _load_drive_signals()
        visible, invisible = _logging_format_strings_by_level(_KVSO_SRC)
        src = _KVSO_SRC.read_text()
        self.assertIn(
            "kv-session-offload: spill slot closed (%s, rpi=%d, region=%d)",
            invisible,
            "the fixture for this test is gone: _close_slot no longer logs at "
            "debug, so the family this test guards has changed shape",
        )
        # THE RULE. A signal that the manager clearly OWNS (its text appears in
        # this source file) but that appears in no info-visible FORMAT string
        # cannot reach a default boot's log. That covers the case which slipped
        # through once already: ``restored to device`` is not a format string at
        # all, it is the ``why`` ARGUMENT ``_close_slot`` substitutes into a
        # ``logger.debug`` template -- so a check that only inspects format
        # strings sees nothing and reports the cell as fine.
        # Signals whose text lives in another module (a server_args refusal, an
        # attention backend line) are not judged here; this test only speaks for
        # what it can read.
        offenders = []
        for cell, entries in signals.items():
            for name, pattern in entries:
                lit = _literal_prefix(pattern)
                if len(lit) < 8:
                    continue
                if lit not in src:
                    continue
                if any(lit in fmt for fmt in visible):
                    continue
                where = (
                    "debug-only"
                    if any(lit in f for f in invisible)
                    else "not a format string"
                )
                offenders.append(f"{cell}/{name}: {pattern!r} ({where})")
        self.assertEqual(
            [],
            offenders,
            "matrix signal(s) keyed on manager text that no info-level log "
            "emits; at the default --log-level info these cells read zero "
            "whether or not the mechanism fired: " + "; ".join(offenders),
        )

    def test_the_signal_the_manager_does_emit_at_info_is_the_one_h4_uses(self):
        """H4 must be bound to the INFO-level restore line, not to a
        paraphrase. Renaming the message in the manager turns this red."""
        signals = _load_drive_signals()
        visible, _ = _logging_format_strings_by_level(_KVSO_SRC)
        pattern = dict(signals["H4"])["restore"]
        lit = _literal_prefix(pattern)
        self.assertTrue(
            any(lit in fmt for fmt in visible),
            f"H4's literal {lit!r} is not in any info-visible format string of "
            f"{_KVSO_SRC.name}",
        )


class TestTheRestoreLineAttributesTheRejoin(unittest.TestCase):
    """The rejoin must be readable off ONE line, not inferred across two."""

    FMT = (
        "kv-session-offload RESTORE complete: rid=%s L=%d (rank %d) "
        "rejoining device batch spec=%d"
    )

    def test_the_manager_still_carries_this_exact_format(self):
        """Binds the rendered lines below to the real source text."""
        visible, _ = _logging_format_strings_by_level(_KVSO_SRC)
        self.assertTrue(
            self.FMT in visible,
            "the restore-complete format changed; update this test AND "
            "scripts/dev/spill_matrix/drive.py together -- they are one pair. "
            "Nearest info-level candidates: "
            + "; ".join(repr(f) for f in visible if "RESTORE complete" in f),
        )

    def test_h4_matches_both_a_spec_and_a_plain_rejoin(self):
        """H4 is the 'did anything restore at all' cell: spec-agnostic."""
        signals = _load_drive_signals()
        pattern = dict(signals["H4"])["restore"]
        for spec in (0, 1):
            line = "[TP0] " + self.FMT % ("abc", 1143, 0, spec)
            self.assertRegex(line, pattern, f"H4 missed spec={spec}")

    def test_h7_discriminates_the_spec_rejoin_from_a_plain_one(self):
        """H7 is #552's own cell and must NOT go green on a plain restore.

        This is the half boot K2 could not establish: the seed line alone
        proves the republish, ``spec=1`` proves the session then rejoined a
        LIVE spec batch.

        CAN-FAIL: drop the ``spec=%d`` argument from the manager's log call and
        the spec=1 assertion below goes red.
        """
        signals = _load_drive_signals()
        pattern = dict(signals["H7"])["restorespec"]
        spec_line = "[TP0] " + self.FMT % ("abc", 1143, 0, 1)
        plain_line = "[TP0] " + self.FMT % ("abc", 1143, 0, 0)
        self.assertRegex(spec_line, pattern)
        self.assertNotRegex(plain_line, pattern)

    def test_h7_also_requires_the_seed_line(self):
        """Both halves of the pair, so a rejoin with no republish cannot pass."""
        signals = _load_drive_signals()
        pattern = dict(signals["H7"])["resumeseed"]
        visible, _ = _logging_format_strings_by_level(_KVSO_SRC)
        seed_fmt = "kv-session-offload MTP RESUME seed published: rid=%s L=%d (rank %d)"
        self.assertIn(seed_fmt, visible)
        self.assertRegex("[TP0] " + seed_fmt % ("abc", 1143, 0), pattern)


def _fifo_manager(order_sink, spilled_rpis):
    """A manager carrying only what ``pre_schedule`` touches on the way to the
    restore loop. No GPU, no scheduler, no __init__."""
    from sglang.srt.managers.kv_session_offload import KVSessionOffloadManager

    mgr = KVSessionOffloadManager.__new__(KVSessionOffloadManager)
    mgr._iter_ct = 0
    mgr.tick_controller = None
    mgr._budget_armed = False
    mgr._dest = None
    mgr._tick_trace = False
    mgr._log = lambda fmt, *a: None
    mgr.spills = {}
    for rpi in spilled_rpis:
        req = SimpleNamespace(
            rid=f"r{rpi}",
            req_pool_idx=rpi,
            finished=lambda: False,
            origin_input_ids=[0],
            output_ids=[],
        )
        mgr.spills[rpi] = SimpleNamespace(
            req=req, batch=None, park_pending=False, region=rpi
        )
    mgr.adopt_born_spilled_prefills = lambda rb: rb
    mgr._maybe_spill_for_fast_lane = lambda rb: None

    def _record(slot, running_batch, last_batch):
        order_sink.append(slot.req.req_pool_idx)
        return running_batch

    mgr._maybe_restore_flow = _record
    return mgr


class TestRestoreOrderIsFifo(unittest.TestCase):
    """``pre_schedule``'s docstring promises FIFO restore; the only thing
    implementing it is ``dict`` insertion order."""

    def test_the_loop_walks_spills_in_spill_order(self):
        """Restores compete for the same freed space, so iteration order IS the
        queue discipline: the eldest spilled session must be offered first.

        CAN-FAIL: wrap the loop's iterable in ``sorted(..., reverse=True)`` (or
        swap the dict for a set) and this goes red -- which is exactly the
        silent elder-starvation this pins against.
        """
        seen = []
        # Insert deliberately NOT in rpi order: FIFO is by SPILL time, and
        # req_pool_idx is an allocator artefact that says nothing about age.
        mgr = _fifo_manager(seen, [7, 2, 5])
        mgr.pre_schedule(SimpleNamespace(), None)
        self.assertEqual([7, 2, 5], seen)

    def test_a_respilled_session_goes_to_the_back(self):
        """A session that restored and spilled again is re-inserted at the
        back -- the intended demotion, and the reason a plain 'sort by rpi'
        would be wrong."""
        seen = []
        mgr = _fifo_manager(seen, [7, 2, 5])
        slot = mgr.spills.pop(7)
        mgr.spills[7] = slot
        mgr.pre_schedule(SimpleNamespace(), None)
        self.assertEqual([2, 5, 7], seen)

    def test_a_park_pending_slot_is_skipped_without_reordering_the_rest(self):
        """#224 parking must not perturb the FIFO of the sessions that stay."""
        seen = []
        mgr = _fifo_manager(seen, [7, 2, 5])
        mgr.spills[2].park_pending = True
        mgr.pre_schedule(SimpleNamespace(), None)
        self.assertEqual([7, 5], seen)


# Quantities that differ BETWEEN RANKS at the same iteration. A collective may
# never be entered under a predicate that reads one of these: two ranks would
# then enter a different NUMBER of collectives in the same iteration, which is
# an NCCL watchdog timeout with a sequence-number mismatch (PG seq N vs N+1),
# not a wrong answer. Under uneven DCP the per-rank pools differ by design, so
# every one of these genuinely diverges here.
_RANK_LOCAL_READS = (
    "available_size",
    "_free_regions",
    "free_regions",
    "_tree_evictable_size",
    "evictable_size",
    "local_avail",
    "local_ratio",
    "local_evict",
    "memory_allocated",
    "mem_get_info",
    "nvml",
)

# Every torch.distributed collective the manager enters, and the method that
# owns it. Pinned as a LIST so a newly added collective fails this test until
# somebody states where it is entered from -- an unreviewed collective is the
# whole risk.
_EXPECTED_COLLECTIVE_OWNERS = {
    "_min_reduce_headroom",
    "_min_reduce_avail",
    "update_dcp_admission_state",
    "_budget_gdn_token_equivalent",
    "_budget_begin_iteration",
}


def _functions_containing_collectives(tree):
    """Map function name -> list of collective call nodes inside it."""
    out = {}
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        calls = [
            n
            for n in ast.walk(fn)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr in ("all_reduce", "all_gather", "broadcast", "barrier")
        ]
        if calls:
            out.setdefault(fn.name, []).extend(calls)
    return out


class TestNoCollectiveIsEnteredOnRankLocalState(unittest.TestCase):
    """THE DESYNC FALSIFIER.

    Production took an NCCL watchdog timeout with the DCP group at seq 15780 on
    one rank and 15781 on another: one rank entered an EXTRA collective. That is
    the failure mode of a conditional collective whose condition reads something
    rank-local, and it is silent until it hangs -- no assertion, no wrong number,
    just a dead server.

    These tests are structural on purpose. A runtime test cannot enter a real
    process group hermetically, but the property that matters is decidable from
    the source: a collective's guard must read replicated state only.
    """

    def _tree(self):
        return ast.parse(_KVSO_SRC.read_text(), filename=str(_KVSO_SRC))

    def test_the_set_of_collective_owning_methods_is_the_reviewed_one(self):
        """A new collective must be declared here before it ships.

        CAN-FAIL: add a ``torch.distributed.all_reduce`` to any other method and
        this goes red naming it.
        """
        found = set(_functions_containing_collectives(self._tree()))
        self.assertEqual(
            _EXPECTED_COLLECTIVE_OWNERS,
            found,
            "the set of methods entering a torch.distributed collective "
            "changed; every entry needs a rank-uniformity argument in its "
            "docstring and a line in this test",
        )

    def test_no_collective_guard_reads_a_rank_local_quantity(self):
        """The guards ENCLOSING each collective, inside its own method.

        The world-size check (``get_world_size(grp) <= 1``) is replicated and
        fine. What must never appear is a per-rank pool number: under uneven DCP
        ``available_size()`` differs by construction, so a guard reading it
        splits the ranks across branches with mismatched collective counts.

        CAN-FAIL: wrap any ``all_reduce`` in
        ``if self.allocator.available_size() > 0:`` and this goes red naming
        the method and the offending read.
        """
        src = _KVSO_SRC.read_text()
        tree = self._tree()
        offenders = []
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(fn):
                if not isinstance(node, ast.If):
                    continue
                has_collective = any(
                    isinstance(c, ast.Call)
                    and isinstance(c.func, ast.Attribute)
                    and c.func.attr
                    in ("all_reduce", "all_gather", "broadcast", "barrier")
                    for stmt in node.body
                    for c in ast.walk(stmt)
                )
                if not has_collective:
                    continue
                test_src = ast.get_source_segment(src, node.test) or ""
                for bad in _RANK_LOCAL_READS:
                    if bad in test_src:
                        offenders.append(f"{fn.name}: guard reads {bad!r}")
        self.assertEqual(
            [],
            offenders,
            "collective(s) entered under a rank-local predicate -- this is the "
            "PG-seq-mismatch hang, not a wrong answer: " + "; ".join(offenders),
        )

    def test_the_spec_tick_collective_is_guarded_only_by_the_spec_algorithm(self):
        """The one collective inside the per-session spill-tick loop.

        ``_min_reduce_avail`` is entered once per spilled session that ticks
        under spec, so its guard controls a per-iteration collective COUNT, not
        just a verdict. It must read ``batch.spec_algorithm`` (replicated) and
        nothing else -- notably not the local ``available_size()`` whose value
        it is about to reduce.
        """
        src = _KVSO_SRC.read_text()
        tree = self._tree()
        sites = [
            fn
            for fn in ast.walk(tree)
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
            and any(
                isinstance(c, ast.Call)
                and isinstance(c.func, ast.Attribute)
                and c.func.attr == "_min_reduce_avail"
                for c in ast.walk(fn)
            )
        ]
        self.assertTrue(sites, "_min_reduce_avail has no call site any more")
        for fn in sites:
            for node in ast.walk(fn):
                if not isinstance(node, ast.If):
                    continue
                calls_it = any(
                    isinstance(c, ast.Call)
                    and isinstance(c.func, ast.Attribute)
                    and c.func.attr == "_min_reduce_avail"
                    for stmt in node.body
                    for c in ast.walk(stmt)
                )
                if not calls_it:
                    continue
                test_src = ast.get_source_segment(src, node.test) or ""
                self.assertIn("spec_algorithm", test_src)
                for bad in _RANK_LOCAL_READS:
                    self.assertNotIn(bad, test_src, f"{fn.name} guard reads {bad}")


class _SpecAlgo:
    def __init__(self, none: bool):
        self._none = none

    def is_none(self):
        return self._none


class _GateProbe(Exception):
    """Raised just past the gate, so 'declined' and 'proceeded' are
    distinguishable without building the whole restore machinery."""


class _Boom:
    def __getattr__(self, _n):
        raise _GateProbe


def _gate_manager(rank, spec_none):
    """Manager fixture for the resume-under-spec capability gate only.

    ``req_to_token_pool`` is the probe: it is the first thing
    ``_maybe_restore_flow`` touches AFTER the gate (the sentinel-boundary scan),
    so reaching it means the gate let the session through.
    """
    from sglang.srt.managers.kv_session_offload import KVSessionOffloadManager

    mgr = KVSessionOffloadManager.__new__(KVSessionOffloadManager)
    mgr.dcp_rank = rank
    mgr.scheduler = SimpleNamespace(
        spec_algorithm=_SpecAlgo(spec_none), waiting_queue=()
    )
    mgr._fast_lane_enabled = False
    mgr._iter_ct = 100
    mgr._log = lambda fmt, *a: None
    mgr.req_to_token_pool = _Boom()
    return mgr


def _run_gate(mgr, slot):
    """Call the real ``_maybe_restore_flow`` and report whether the spec gate
    declined. Anything past the gate hits the probe."""
    sentinel = SimpleNamespace(name="running")
    try:
        out = mgr._maybe_restore_flow(slot, sentinel, None)
    except _GateProbe:
        return "proceeded"
    return "declined" if out is sentinel else "proceeded"


class TestResumeUnderSpecGateIsRankUniformAndNeverSilent(unittest.TestCase):
    """The gate sits on the collective spill/restore path: a rank-split verdict
    here is a hang, not a wrong answer. Every input must be replicated."""

    def setUp(self):
        self._saved = {
            k: os.environ.get(k) for k in ("SGLANG_KVSO_RESUME", "KVSO_RESUME")
        }
        for k in self._saved:
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def _slot(self):
        return SimpleNamespace(
            req=SimpleNamespace(
                rid="r0",
                req_pool_idx=0,
                origin_input_ids=list(range(128)),
                output_ids=[1, 2, 3],
            ),
            batch=None,
            spill_iter=0,
        )

    def test_spec_active_and_flag_off_declines_on_every_rank(self):
        """The default: keep the validated host-finish path, identically on all
        three TP ranks of the production geometry."""
        for rank in (0, 1, 2):
            mgr = _gate_manager(rank, spec_none=False)
            self.assertEqual(
                "declined", _run_gate(mgr, self._slot()), f"rank {rank} diverged"
            )

    def test_spec_active_and_flag_on_proceeds_on_every_rank(self):
        """Armed: the capability path opens, identically on all ranks.

        CAN-FAIL: delete the ``not resume_under_spec_enabled()`` term from the
        gate and this stays green while the previous test goes red -- the two
        together pin that the flag, and only the flag, moves the verdict.
        """
        os.environ["SGLANG_KVSO_RESUME"] = "1"
        for rank in (0, 1, 2):
            mgr = _gate_manager(rank, spec_none=False)
            self.assertEqual(
                "proceeded", _run_gate(mgr, self._slot()), f"rank {rank} diverged"
            )

    def test_spec_off_is_unaffected_by_the_flag_in_either_position(self):
        """NEUTRALITY. With no spec algorithm the gate is not even consulted,
        so arming or not arming the flag must be indistinguishable."""
        verdicts = set()
        for armed in (False, True):
            if armed:
                os.environ["SGLANG_KVSO_RESUME"] = "1"
            else:
                os.environ.pop("SGLANG_KVSO_RESUME", None)
            for rank in (0, 1, 2):
                mgr = _gate_manager(rank, spec_none=True)
                verdicts.add(_run_gate(mgr, self._slot()))
        self.assertEqual({"proceeded"}, verdicts)

    def test_the_gate_reads_only_replicated_state(self):
        """The rank-uniformity ARGUMENT, pinned rather than asserted in a
        comment: the verdict is a pure function of (spec algorithm, flag), and
        ``dcp_rank`` is not one of its inputs."""
        os.environ["SGLANG_KVSO_RESUME"] = "1"
        armed = {_run_gate(_gate_manager(r, False), self._slot()) for r in range(8)}
        os.environ.pop("SGLANG_KVSO_RESUME", None)
        unarmed = {_run_gate(_gate_manager(r, False), self._slot()) for r in range(8)}
        self.assertEqual({"proceeded"}, armed)
        self.assertEqual({"declined"}, unarmed)


class TestTheGuardCommentDoesNotDenyTheMechanism(unittest.TestCase):
    """A comment-invariant pin. The guard's docstring told every auditor that
    on-device MTP resume was an unbuilt follow-up, eight lines above the escape
    that runs it -- which is how a built mechanism gets re-reported as missing.
    """

    def test_the_guard_points_at_the_seeding_function_that_exists(self):
        src = _KVSO_SRC.read_text()
        start = src.index("DEVICE-RESUME UNDER SPEC")
        block = src[start : start + 3000]
        self.assertIn("_seed_resumed_draft_state", block)
        self.assertNotIn("is the follow-up for true on-device MTP", block)

    def test_the_seeding_function_is_actually_defined(self):
        src = _KVSO_SRC.read_text()
        self.assertIn("def _seed_resumed_draft_state(self, slot, L: int)", src)
        self.assertIn("def _publish_resume_seed(self, slot, L: int, seed)", src)


if __name__ == "__main__":
    unittest.main()
