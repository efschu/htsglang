# SPDX-License-Identifier: Apache-2.0
"""#1348 -- WHAT DID THIS BOOT *NOT* EXECUTE, on the exchange lane, per rank.

THE QUESTION, and why it is a different question from every instrument we
already have.  Every other probe on this lane answers "what happened": the
flip decomposition, the policy census, the corridor probes, the seam digest.
Each of them is a statement about code that RAN.  Walls XSN6 through XSN15
were none of those: each sat in a seam that had never executed until the boot
that hit it, so no counter could have shown it and no log line existed to be
absent.  The only instrument that can name them BEFORE a boot pays for them is
the complement -- the lines of the lane's own modules that this boot never
reached.  That list is the list of walls not yet found.

WHAT THIS IS NOT.  It is not a quality measurement and it is not a target.
There is no percentage to maximise here: a line that never runs may be a dead
branch, a refusal nobody triggered, or the next wall.  The output is a
READING LIST, and the reader decides which of the three each entry is.

RELATION TO ``managers/seam_coverage.py``, which is the same MECHANISM for a
different QUESTION and must never run at the same time.  That module measures
which lines are cutover-ONLY versus also-serving, over the whole ``sglang``
package, with coverage.py's dynamic contexts.  This one measures which lines
of a NAMED, SHORT allowlist ran at all, and dumps JSON keyed the way #1292
keys its footprint dumps.  Two ``coverage.Coverage`` objects cannot both own
the tracer in one process -- the second ``start()`` displaces the first and
BOTH data files then silently describe something other than what they claim.
So :func:`arm` REFUSES by name when ``seam_coverage`` is armed, rather than
winning a race it would not report.

THE ARM IS A LAUNCHER FLAG (``--xchg-coverage-diff``), NOT AN AMBIENT
ENVIRONMENT VARIABLE.  :data:`DIR_ENV` below is read by this module, but it is
LAUNCHER OUTPUT in the sense ``build_env`` already uses for
``SGLANG_WEG2_GROUP`` and the host-ring family (R19): the launcher PUBLISHES
it when the flag is set and POPS it when the flag is absent, so a value left
in the operator's own shell can never arm a 2x-10x tracer on an acceptance
boot.  The distinction matters because ``seam_coverage``'s switch IS an
ambient variable, and that is precisely the property this one must not
inherit.

COVERAGE STARTS AT THE FIRST LEG, WHICH IS A REAL BOUND AND IS PRINTED.  The
tracer is installed when the lane's own hook first runs, not at interpreter
start: arming earlier would mean paying for the model load, and there is no
hook at process start this module could own without reaching outside its
remit.  Consequence, stated in the data rather than in prose: a module already
in ``sys.modules`` when the tracer went in had its IMPORT-TIME lines executed
unobserved, so those lines read as "unexecuted" and are an ARTEFACT.  Every
module entry therefore carries ``imported_before_arm``, and the ingest prints
it on the line; a reader who ignores it will over-report exactly the
module-scope lines (imports, ``def``, ``class``, constants) and nothing else.

GUARD DISCIPLINE, taken verbatim from ``seam_coverage`` because the hazard is
identical: this instrument may never be the reason a flip aborts or a boot
dies.  Every entry point is wrapped in a broad ``except``; the FIRST failure
anywhere logs once, flips a module-level dead flag, and every later call --
from any call site -- is an immediate no-op.  A half-working collector is
worse than none, because its dump LOOKS complete and this instrument's entire
output is a claim about completeness.

WRITTEN AT EVERY LEG END, not once at exit, for the reason ``seam_coverage``
checkpoints: a rank that dies between two legs -- deadman, OOM, a wall -- must
still leave the legs it finished on disk.  A dump that only appears after a
clean shutdown is absent on exactly the boots this exists for.

WHAT IT COSTS, MEASURED ON THIS RIG 2026-09-12, never estimated -- and every
one of these three numbers is re-printed per boot on the ``WEG2-COVERAGE
DUMP`` line, so a future reader grades the boot's own figures and not this
docstring's:

* TRACER, on a pure-Python leg double driving the allowlisted modules
  (``xchg_bounce`` sizing + ``weight_exchange_bounce.unit_name``, 400
  iterations, median of 5): **1.68 ms -> 4.36 ms, a factor of 2.60** -- the
  low end of coverage.py's own 2x-10x range, because the allowlist is eleven
  files and not the tree.
* ARM, once per process: **104 ms** in a rank-shaped process, **282 ms** in
  the launcher process (it has more modules loaded, and the import-state scan
  is proportional to ``sys.modules``).
* PER-LEG DUMP, median of 6: **56 ms**, against a flip leg measured in
  seconds. Most of it is ``coverage.get_data()`` flushing the collector; the
  two costs that were ours and avoidable have been removed -- the per-module
  ``sys.modules`` scan (475 ms of the arm, now one pass) and the per-leg
  re-hash of ~1 MB of module source (now cached at arm, 64 ms -> 56 ms).
"""

from __future__ import annotations

import atexit
import hashlib
import json
import logging
import os
import sys
import threading
import time
from typing import Dict, Iterable, List, Optional, Set

logger = logging.getLogger(__name__)

__all__ = [
    "ALLOWLIST",
    "DIR_ENV",
    "NO_OBSERVATION_CODE",
    "SCHEMA",
    "TALLY_REFUSED_CODE",
    "arm",
    "dump_filename",
    "enabled",
    "executable_lines",
    "note_leg_end",
    "note_teardown",
]

#: Schema tag in every dump.  A reader that does not recognise it must refuse
#: rather than guess at the field names -- the same rule the ring table and
#: the footprint dumps already carry.
SCHEMA = "weg2-lane-coverage-1"

#: ``W90 Weg2CoverageNoObservation`` -- the refusal the ingest prints when it
#: has NO reading for a module.  Enumerated against the tree-wide census
#: (``test_weg2_wcode_uniqueness_1263``) AND against a word-bounded textual
#: grep, because the census alone has a documented blind spot that cost a
#: renumber at W88: both instruments read zero for W90 and W91.
NO_OBSERVATION_CODE = "W90 Weg2CoverageNoObservation"

#: ``W91 Weg2CoverageTallyRefused`` -- the count check failed, so the two
#: numbers being differenced are not measurements of the same thing.
TALLY_REFUSED_CODE = "W91 Weg2CoverageTallyRefused"

#: Published by ``weg2/launcher.py``'s ``build_env`` when
#: ``--xchg-coverage-diff`` is set, and POPPED otherwise.  Absent, every entry
#: point below returns before it touches anything and ``coverage`` is never
#: imported: the OFF state is byte-identical, not merely cheap.
DIR_ENV = "SGLANG_WEG2_LANE_COVERAGE_DIR"

#: THE EXCHANGE LANE, ENUMERATED.  A glob over ``srt/weg2`` would widen the
#: instrument's cost and its output every time the package grows a file, and
#: the point of an allowlist is that the scope is a decision somebody made and
#: a test defends (``test_weg2_lane_coverage_1348``).  Repo-relative so the
#: dump is readable on a different checkout than the one that produced it.
#:
#: ``launcher.py`` is here for its ARM/LEDGER half only -- the module runs in
#: the launcher PROCESS, not in a rank, and is measured there
#: (:func:`arm_launcher`).  The ingest restricts its report on this file to
#: :data:`LAUNCHER_ARM_FUNCTIONS`, because the other ~10k lines are argv
#: plumbing whose unexecuted lines answer a question nobody asked.
ALLOWLIST: List[str] = [
    "python/sglang/srt/weg2/weight_exchange.py",
    "python/sglang/srt/weg2/weight_exchange_bounce.py",
    "python/sglang/srt/weg2/weight_exchange_region.py",
    "python/sglang/srt/weg2/weight_exchange_shadow.py",
    "python/sglang/srt/weg2/weight_exchange_transport.py",
    "python/sglang/srt/weg2/host_ledger.py",
    "python/sglang/srt/weg2/xchg_bounce.py",
    "python/sglang/srt/weg2/ring_guard.py",
    "python/sglang/srt/weg2/launcher.py",
    "python/sglang/srt/managers/weg2_memory_saver.py",
    "python/sglang/srt/managers/scheduler_components/weight_updater.py",
]

#: The launcher's arm/ledger surface -- the functions that DECIDE the boot's
#: exchange shape, as opposed to the ones that assemble argv.  Enumerated from
#: the module's own defs; the ingest reports unexecuted lines on
#: ``launcher.py`` only inside these.
LAUNCHER_ARM_FUNCTIONS = (
    "prepare_xchg_env",
    "build_env",
    "_env_knobs",
    "xchg_bounce_arm_pins_host",
    "xchg_bounce_terms_for_arm",
    "choose_host_ledger",
    "teardown_xchg_region",
)

_lock = threading.Lock()
_cov = None  # type: ignore[var-annotated]
_dead = False
_armed = False
_group = ""
_rank = -1
_legs = 0
_data_path: Optional[str] = None
_imported_before_arm: Dict[str, bool] = {}
#: ``rel -> sha256`` of the source as the BOOTING process saw it. Filled at
#: arm; the ingest refuses (SOURCE-DRIFT) if this checkout's source differs.
_sha_cache: Dict[str, str] = {}
_overhead_ms = {"arm": 0.0, "save_total": 0.0, "saves": 0}
_root: Optional[str] = None


def dump_filename(rank: int, group: str = "") -> str:
    """``phase_coverage_{GROUP}_rank{N}.json`` -- #1292's derivation, reused.

    NOT a second naming scheme.  P and D are independently launched process
    groups that share one dump directory, and ``rank`` is unique only WITHIN a
    group's ``torch.distributed`` job; #1292 paid for that once already, with
    P (booted second) silently overwriting D's footprint dump.  The group tag
    comes from ``weg2_memory_saver.weg2_group_name()``, the one thing in the
    tree that tells a rank which Weg-2 group it is in, exactly as
    ``mem_ledger/activation_probe.dump_filename`` reads it.  Outside Weg-2 the
    group is ``""`` and the name degrades to the ungrouped shape.
    """
    tag = f"{group}_" if group else ""
    return f"phase_coverage_{tag}rank{rank}.json"


def enabled() -> bool:
    """Whether this process has been armed. Checked first, on every call."""
    return _armed and not _dead


def _repo_root() -> str:
    """Where ``python/sglang/...`` hangs off, for this checkout.

    Derived from the package's own location rather than from ``os.getcwd()``:
    a rank's working directory is whatever the launcher left it at, and a
    coverage include-list built from the wrong root silently measures nothing
    at all -- which this instrument would then report as a clean sweep.
    """
    global _root
    if _root is None:
        pkg = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../srt
        _root = os.path.dirname(os.path.dirname(os.path.dirname(pkg)))  # repo root
    return _root


def abs_path(rel: str) -> str:
    """Absolute path of an allowlist entry in THIS checkout."""
    return os.path.join(_repo_root(), rel)


def _mark_dead(where: str, exc: BaseException) -> None:
    """First failure anywhere: log once, stay dead, never raise at the caller."""
    global _dead
    if _dead:
        return
    _dead = True
    logger.error(
        "lane_coverage: %s failed (%s: %s) -- the #1348 unexecuted-line "
        "instrument is now permanently disabled for this process. This never "
        "aborts a leg or a boot; it means this rank's coverage dump is "
        "incomplete from here on, and the ingest will say so rather than "
        "report a clean sweep.",
        where,
        type(exc).__name__,
        exc,
    )


def _seam_coverage_armed() -> bool:
    """Is the OTHER coverage instrument already holding the tracer?

    Read from the environment rather than by importing ``seam_coverage``:
    importing it is harmless, but this check runs on the OFF path too and the
    OFF path imports nothing it does not already need.
    """
    return bool(os.environ.get("SGLANG_SEAM_COVERAGE_DIR"))


def arm(
    directory: Optional[str] = None,
    *,
    group: str = "",
    rank: int = -1,
) -> bool:
    """Install the tracer for the allowlist. Idempotent; safe to race.

    Returns whether this process is measuring afterwards.  Called from the
    lane's own per-leg hook, so the first leg pays the arm and every later leg
    pays nothing.  ``directory`` defaults to :data:`DIR_ENV`, which the
    launcher publishes only when ``--xchg-coverage-diff`` is set.
    """
    global _cov, _armed, _group, _rank, _data_path
    if _dead or _armed:
        return enabled()
    directory = directory or os.environ.get(DIR_ENV) or ""
    if not directory:
        return False
    with _lock:
        if _armed or _dead:
            return enabled()
        t0 = time.perf_counter()
        try:
            if _seam_coverage_armed():
                # NOT a silent loss: two Coverage objects in one process means
                # the second start() displaces the first and both data files
                # then describe something other than what they claim.
                logger.error(
                    "lane_coverage: REFUSING to arm -- SGLANG_SEAM_COVERAGE_DIR "
                    "is set, so managers/seam_coverage.py already owns this "
                    "process's tracer. Two coverage.Coverage objects cannot "
                    "both measure; arm exactly one of the two instruments per "
                    "boot. No #1348 dump will be written by this rank."
                )
                return False

            import coverage  # local import: never paid unless armed

            os.makedirs(directory, exist_ok=True)
            includes = [abs_path(rel) for rel in ALLOWLIST]
            loaded_at_arm = _loaded_files()
            for rel in ALLOWLIST:
                _imported_before_arm[rel] = _is_imported(rel, loaded_at_arm)
                try:
                    with open(abs_path(rel), "rb") as fh:
                        _sha_cache[rel] = hashlib.sha256(fh.read()).hexdigest()
                except OSError:
                    _sha_cache[rel] = ""
            cov = coverage.Coverage(
                data_file=None,  # kept in memory; the JSON dump is the product
                include=includes,
                config_file=False,  # never inherit a stray .coveragerc
                branch=False,  # a WHICH-LINES map, not a branch report
                messages=False,
            )
            cov.start()
            _cov = cov
            _armed = True
            _group = group
            _rank = int(rank)
            _data_path = os.path.join(directory, dump_filename(_rank, _group))
            _overhead_ms["arm"] = (time.perf_counter() - t0) * 1000.0
            atexit.register(_save_at_exit)
            logger.info(
                "lane_coverage: ARMED group=%s rank=%d modules=%d dump=%s "
                "arm_ms=%.1f -- this is a MEASUREMENT-RUN instrument "
                "(coverage.py line tracing, 2x-10x on the traced modules); "
                "--xchg-coverage-diff must be absent on an acceptance boot.",
                _group or "?",
                _rank,
                len(ALLOWLIST),
                _data_path,
                _overhead_ms["arm"],
            )
        except Exception as e:  # noqa: BLE001 -- never break the caller
            _mark_dead("arm", e)
            return False
    return True


def arm_launcher(directory: str) -> bool:
    """Arm inside the LAUNCHER process, where ``launcher.py`` actually runs.

    The rank processes never import ``weg2/launcher.py``, so without this the
    ingest could only ever print NO-OBSERVATION for it -- an honest answer, but
    a useless one when the arm/ledger decisions are exactly the half of the
    lane the operator most wants a complement of.  Group ``L`` keeps its dump
    out of every rank's filename space.
    """
    return arm(directory, group="L", rank=0)


def _module_name(rel: str) -> str:
    """``python/sglang/srt/weg2/x.py`` -> ``sglang.srt.weg2.x``."""
    return rel[len("python/") : -len(".py")].replace("/", ".")


def _is_imported(rel: str, loaded: Optional[Set[str]] = None) -> bool:
    """Is this FILE loaded in this interpreter, under ANY module name?

    NOT ``_module_name(rel) in sys.modules``, and the difference is not
    theoretical -- it was measured on the first execution smoke of this
    instrument.  ``weg2/launcher.py`` is started as ``python -m
    sglang.srt.weg2.launcher``, so the running module is registered as
    ``__main__`` and the dotted name is absent; the dump said
    ``module-never-imported`` about the very file the process was executing,
    and the ingest dutifully refused to report the one module the launcher
    arm exists to measure.

    Matching on ``__file__`` catches ``__main__``, and it also catches the
    same file reached under a second name (a re-export, a sys.path quirk),
    which a name lookup cannot.
    """
    if _module_name(rel) in sys.modules:
        return True
    if loaded is None:
        loaded = _loaded_files()
    return os.path.realpath(abs_path(rel)) in loaded


def _loaded_files() -> Set[str]:
    """Realpaths of every loaded module's ``__file__``, built ONCE per call.

    Built as a set rather than scanned per module for a measured reason: the
    per-module scan cost 475 ms on the arm path of the first execution smoke
    (arm_ms 57 -> 532), because eleven allowlist entries each walked ~1500
    loaded modules and called ``realpath`` on every one. The arm sits inside a
    flip, so that is 475 ms of instrument charged to the thing being measured.
    """
    out: Set[str] = set()
    for mod in list(sys.modules.values()):
        f = getattr(mod, "__file__", None)
        if f:
            try:
                out.add(os.path.realpath(f))
            except OSError:
                pass
    return out


def _collect() -> Dict[str, dict]:
    """Per allowlisted module: sha256, import state, and the RAW traced lines.

    RAW, deliberately: mapping a traced line onto its statement needs the
    source parsed, and doing that here would put a parse of eleven modules --
    one of them 10k lines -- inside a flip.  The ingest has the source anyway
    (it must, to compute the complement) and does the mapping there, where it
    costs a boot nothing.  The sha256 is what makes that split safe: a dump
    read against a different source than it was taken on is a SOURCE-DRIFT
    refusal, not a silently wrong line list.
    """
    out: Dict[str, dict] = {}
    data = _cov.get_data() if _cov is not None else None
    measured = set(data.measured_files()) if data is not None else set()
    loaded = _loaded_files()
    for rel in ALLOWLIST:
        path = abs_path(rel)
        entry: dict = {
            "imported": _is_imported(rel, loaded),
            "imported_before_arm": bool(_imported_before_arm.get(rel, False)),
        }
        # HASHED ONCE AT ARM, not once per leg. The source of a running
        # process cannot change under it, and re-hashing ~1 MB of module text
        # at every leg end is instrument cost charged to the flip it measures
        # (measured: it was the bulk of a 63.7 ms per-leg dump).
        entry["sha256"] = _sha_cache.get(rel, "")
        lines: Set[int] = set()
        if data is not None and path in measured:
            lines = set(data.lines(path) or ())
        entry["executed"] = sorted(lines)
        out[rel] = entry
    return out


def _write(reason: str) -> None:
    """Serialise the cumulative reading. Called at every leg end and teardown.

    ``coverage``'s data is cumulative from ``start()``, so rewriting the whole
    file each time IS the merge -- there is no per-leg delta to append and
    reconcile, which is one accounting fewer to get wrong.
    """
    global _legs
    if _cov is None or _data_path is None:
        return
    t0 = time.perf_counter()
    blob = {
        "schema": SCHEMA,
        "group": _group,
        "rank": _rank,
        "pid": os.getpid(),
        "legs": _legs,
        "written_at": reason,
        "instrument": f"coverage.py {_coverage_version()}",
        "overhead_ms": {
            "arm": round(_overhead_ms["arm"], 3),
            "save_total": round(_overhead_ms["save_total"], 3),
            "saves": _overhead_ms["saves"],
        },
        "modules": _collect(),
    }
    tmp = f"{_data_path}.tmp{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(blob, fh)
    os.replace(tmp, _data_path)  # a reader never sees a half-written dump
    _overhead_ms["save_total"] += (time.perf_counter() - t0) * 1000.0
    _overhead_ms["saves"] += 1


def _coverage_version() -> str:
    try:
        import coverage

        return str(coverage.__version__)
    except Exception:  # noqa: BLE001
        return "?"


def note_leg_end(tag: str) -> None:
    """One exchange leg finished on this rank: checkpoint the reading."""
    global _legs
    if _dead or not _armed:
        return
    try:
        _legs += 1
        _write(f"leg:{tag}")
    except Exception as e:  # noqa: BLE001 -- never abort a leg
        _mark_dead(f"note_leg_end({tag})", e)


def note_teardown() -> None:
    """The lane is being torn down: final checkpoint, tracer left running.

    The tracer is NOT stopped here.  Teardown is not necessarily the end of
    the process -- a rank can tear a region down and go on serving -- and a
    stopped tracer that nothing restarts would turn every line after the first
    teardown into a false "unexecuted".  :func:`_save_at_exit` stops it.
    """
    if _dead or not _armed:
        return
    try:
        _write("teardown")
    except Exception as e:  # noqa: BLE001
        _mark_dead("note_teardown", e)


def _save_at_exit() -> None:
    """Final flush at interpreter exit; best-effort, like every path here.

    A SIGKILL skips atexit entirely -- a real limitation, which is why the
    per-leg checkpoints above exist and why this is the belt and not the
    braces.
    """
    if _cov is None or _dead:
        return
    try:
        _write("atexit")
        _cov.stop()
        logger.info(
            "lane_coverage: final dump %s legs=%d save_total_ms=%.1f",
            _data_path,
            _legs,
            _overhead_ms["save_total"],
        )
    except Exception as e:  # noqa: BLE001 -- runs at interpreter exit
        _mark_dead("save_at_exit", e)


def executable_lines(path: str) -> Set[int]:
    """The STATEMENT lines of a source file -- the complement's denominator.

    ``coverage.py``'s own parser, not an ``ast`` walk of our own, and that is
    a correctness requirement rather than a convenience: the denominator has
    to be produced by the same analysis that produced the numerator, or the
    two are not comparable and the tally check below is measuring our
    disagreement with coverage.py instead of the boot's behaviour.

    ``exclude_list = []`` on purpose.  The default excludes ``# pragma: no
    cover`` lines from the report -- reasonable for a coverage GOAL, wrong
    here: the tracer records such a line when it runs, so excluding it from
    the denominator makes ``executed`` a non-subset of ``executable`` and
    breaks the tally on a file nobody touched.
    """
    import coverage
    from coverage.python import PythonFileReporter

    cov = coverage.Coverage(config_file=False)
    cov.config.exclude_list = []
    cov._init()
    return set(PythonFileReporter(os.path.abspath(path), coverage=cov).lines())


def translate(path: str, raw: Iterable[int]) -> Set[int]:
    """Map RAW traced line numbers onto the statement lines they belong to.

    A traced line is not always a statement line: a multi-line call fires on
    its first physical line, a docstring fires but is not a statement.
    Measured on ``xchg_bounce.py``: 46 raw events, 39 after translation, 3 of
    them (the module and two class docstrings) outside the statement set
    entirely.  Dropping those three silently is how a percentage becomes
    unfalsifiable, so the ingest COUNTS them (``off_statement=``) beside the
    number rather than absorbing them into it.
    """
    import coverage
    from coverage.python import PythonFileReporter

    cov = coverage.Coverage(config_file=False)
    cov.config.exclude_list = []
    cov._init()
    fr = PythonFileReporter(os.path.abspath(path), coverage=cov)
    return set(fr.translate_lines(sorted(raw)))


def _reset_for_test() -> None:
    """Tear the module's process-global state down between tests.

    Exists because this module is a singleton by design (one tracer per
    process) and a test suite needs several arms in one interpreter.  Never
    called from product code.
    """
    global _cov, _armed, _dead, _group, _rank, _legs, _data_path, _root
    if _cov is not None:
        try:
            _cov.stop()
        except Exception:  # noqa: BLE001
            pass
    _cov = None
    _armed = False
    _dead = False
    _group = ""
    _rank = -1
    _legs = 0
    _data_path = None
    _imported_before_arm.clear()
    _sha_cache.clear()
    _overhead_ms.update({"arm": 0.0, "save_total": 0.0, "saves": 0})
