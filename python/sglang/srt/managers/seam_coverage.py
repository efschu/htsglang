# SPDX-License-Identifier: Apache-2.0
"""Diagnostic instrument: which LINES actually execute during a phase-flip
cutover, as opposed to during ordinary serving.

WHY THIS EXISTS. Every #631/#855 boot killer to date was discovered by
booting, hitting the seam, and finding out the hard way that some line the
cutover reaches was never exercised by anything before it. This module turns
that discovery process into a MEASUREMENT instead of a lottery: it runs
``coverage.py`` with a dynamic context that flips to ``cutover:<direction>``
for the duration of one cutover (:func:`enter_cutover` / :func:`exit_cutover`,
wired around ``PhaseFlipRuntime._execute`` in ``phase_flip_runtime.py``) and
back to ``serving`` the rest of the time. The resulting per-rank data file
lets a reader (``devtools/seam_surface_report.py``) ask, per line: did this
line EVER run outside a cutover, or only ever inside one? The latter set is
the seam's true execution surface for this boot -- not a guess from reading
the diff, a measurement of what actually ran.

THIS IS A MEASUREMENT-RUN INSTRUMENT, NOT A SERVING FEATURE. It must be
EXACTLY as expensive as ``coverage.py``'s line tracer, which is not free
(coverage.py's own numbers put simple line tracing in the 2x-10x slowdown
range) -- acceptable for a diagnostic boot, never acceptable as a standing
cost on the tree this same file also serves. Every public entry point below
is therefore a no-op, and ``coverage`` is never imported, unless
:data:`DIR_ENV` is set in the process environment. There is no other switch:
absence of the env var is the OFF state, checked first, on every call.

GUARD DISCIPLINE, mirrored from ``mem_ledger/flight_recorder.py``: the
instrument may never be the reason a flip aborts or a boot dies. Every entry
point is wrapped in a broad ``except Exception``; the FIRST failure anywhere
in this module logs once, flips a module-level dead flag, and every
subsequent call -- including from a different call site -- becomes an
immediate no-op. A half-working coverage collector (e.g. one whose
``switch_context`` call raised because ``start`` never actually got a tracer
installed) is worse than no collector: it would produce a data file that
looks complete and silently mislabels its contexts. This module chooses "off
and admits it" (one logged error) over "on and wrong" every time.

SCOPE. Coverage is restricted to the ``sglang`` package (``source_pkgs``),
matching the codegraph index's own v1 scope (``python/sglang``) so a seam
report and an index query describe the same file set. ``config_file=False``
so a stray ``.coveragerc`` or a ``[tool.coverage]`` section in ``pyproject.toml``
-- meant for someone else's `pytest --cov` run -- can never silently change
what this instrument measures or where it writes.

PER-RANK FILES. Three ranks share this tree as three separate processes; the
rank is folded into the data file name (``seam_cutover.rank<N>.coverage``) so
they never clobber each other, exactly as ``flight_recorder``'s
``flight_marks_rank<N>`` files do for the same reason.
"""

from __future__ import annotations

import atexit
import logging
import os
import threading
from typing import Optional

logger = logging.getLogger(__name__)

__all__ = [
    "DIR_ENV",
    "enabled",
    "enter_cutover",
    "exit_cutover",
]

#: Directory for the per-rank ``.coverage`` data files. Absent, every entry
#: point in this module returns immediately and ``coverage`` is never
#: imported -- this instrument's cost on the tree's normal boots must be
#: exactly zero, not "small".
DIR_ENV = "SGLANG_SEAM_COVERAGE_DIR"

#: Dynamic context label used for every line NOT inside a cutover. Chosen so
#: the reader's classification rule is a single string comparison: a line is
#: cutover-only iff every context recorded for it is a ``"cutover:"`` prefix
#: and this label never appears among them.
_SERVING_CONTEXT = "serving"

#: Package(s) measured. Deliberately just the fork's own tree -- stdlib,
#: torch, and every third-party dependency are noise for this question and
#: would multiply the data file size for nothing this reader ever asks.
_SOURCE_PKGS = ("sglang",)

_lock = threading.Lock()
_cov = None  # type: ignore[var-annotated]  # the running coverage.Coverage, once started
_started = False
_dead = False
_data_path: Optional[str] = None


def enabled() -> bool:
    """Whether :data:`DIR_ENV` asks for this instrument at all."""
    return bool(os.environ.get(DIR_ENV))


def _mark_dead(where: str, exc: BaseException) -> None:
    """First failure anywhere in this module: log once, then stay dead.

    Called from inside a broad ``except`` at every entry point. Idempotent by
    construction -- a second failure after the first is expected (the
    collector is already disabled) and must not spam the log a second time.
    """
    global _dead
    if _dead:
        return
    _dead = True
    logger.error(
        "seam_coverage: %s failed (%s: %s) -- the seam execution-surface "
        "instrument is now permanently disabled for this process. This "
        "never aborts a flip or a boot; it only means this run's coverage "
        "data is incomplete from this point on.",
        where,
        type(exc).__name__,
        exc,
    )


def _ensure_started(rank: int) -> bool:
    """Start coverage measurement on first use. Idempotent; safe to race.

    Lazy on purpose: this module has exactly one wired call site
    (``PhaseFlipRuntime._execute``'s enter/exit pair), and there is no other
    boot-time hook to start it from without touching files outside this
    instrument's remit. Starting on the first :func:`enter_cutover` costs
    nothing observable -- a flip is already the rare, multi-millisecond event
    this instrument exists to look inside of.
    """
    global _cov, _started, _data_path
    if _dead:
        return False
    if _started:
        return _cov is not None
    with _lock:
        if _started:
            return _cov is not None
        _started = True
        if not enabled():
            return False
        try:
            import coverage  # local import: never paid unless armed

            directory = os.environ[DIR_ENV]
            os.makedirs(directory, exist_ok=True)
            data_path = os.path.join(
                directory, f"seam_cutover.rank{int(rank)}.coverage"
            )
            cov = coverage.Coverage(
                data_file=data_path,
                source_pkgs=list(_SOURCE_PKGS),
                config_file=False,  # never inherit someone else's .coveragerc
                branch=False,  # line coverage only: this is a WHICH-LINES map
                messages=False,
            )
            cov.start()
            cov.switch_context(_SERVING_CONTEXT)
            _cov = cov
            _data_path = data_path
            atexit.register(_save_at_exit)
            logger.info(
                "seam_coverage: armed for rank %d, data file %s, scope=%s. "
                "This is a measurement-run instrument; SGLANG_SEAM_COVERAGE_DIR "
                "must be unset on an acceptance/serving boot.",
                int(rank),
                data_path,
                ",".join(_SOURCE_PKGS),
            )
        except Exception as e:  # noqa: BLE001 -- never break the caller
            _mark_dead("start", e)
            return False
    return _cov is not None


def _save_at_exit() -> None:
    """Registered once, at successful start. Final flush on process exit.

    Best-effort like everything else here: a SIGKILL skips atexit entirely
    (a real limitation, noted in the reader's output, not hidden), but a
    normal exit -- including an uncaught exception unwinding the process --
    still runs this and saves whatever was collected.
    """
    if _cov is None:
        return
    try:
        _cov.stop()
        _cov.save()
        logger.info("seam_coverage: final save to %s", _data_path)
    except Exception as e:  # noqa: BLE001 -- this runs at interpreter exit
        _mark_dead("save_at_exit", e)


def enter_cutover(tag: str, *, rank: int = 0) -> None:
    """Switch the dynamic context to ``cutover:<tag>`` for the seam ahead.

    ``tag`` is the flip direction (``pp_to_tp`` / ``tp_to_pp``) so the report
    can tell the two seams apart without a second instrument. No-op unless
    :data:`DIR_ENV` is set, and permanently inert after the first internal
    failure (see :func:`_mark_dead`).

    ``rank`` defaults to 0 rather than being resolved internally (e.g. via
    ``torch.distributed``) because the one caller in this tree
    (``PhaseFlipRuntime``) already carries ``self._rank`` as authoritative
    per-rank state -- resolving it a second way here would be exactly the
    kind of rank-identity duplication the fork's device-identity rules warn
    against. Mirrors the ``rank`` keyword on
    ``mem_ledger.flight_recorder.mark``.
    """
    if _dead or not enabled():
        return
    try:
        if not _ensure_started(rank):
            return
        assert _cov is not None
        _cov.switch_context(f"cutover:{tag}")
    except Exception as e:  # noqa: BLE001 -- never abort a flip
        _mark_dead(f"enter_cutover({tag})", e)


def exit_cutover(tag: str, *, rank: int = 0) -> None:
    """Switch the dynamic context back to ``serving`` after the seam.

    Also checkpoints the data file (``coverage.Coverage.save`` may be called
    any number of times while running -- it flushes without stopping) so a
    process that dies between two cutovers -- crash, deadman kill, OOM --
    still leaves a readable file covering every cutover it completed, not
    just the ones before the LAST clean exit.
    """
    if _dead or not enabled():
        return
    try:
        if _cov is None:
            # enter_cutover never ran or never started successfully; nothing
            # to switch back from.
            return
        _cov.switch_context(_SERVING_CONTEXT)
        _cov.save()
    except Exception as e:  # noqa: BLE001 -- never abort a flip
        _mark_dead(f"exit_cutover({tag})", e)
