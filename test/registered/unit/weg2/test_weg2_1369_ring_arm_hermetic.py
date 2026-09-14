"""#1369 -- USER ORDER 2026-09-14 ("DIE 48GB MUESSEN WEG. UND ZWAR PRONTO"), and
on the counter-argument that the ring is a fallback: "auf der festplatte liegt
ein snapshot. das ist auch ein rueckfall. aber wenn es korrekt implementiert
ist braucht es NIEMALS einen rueckfall....!"

DESK12 / Paket C of a four-package coordinated fix: bind the WEIGHTS region's
``enable_cpu_backup`` to ``weight_exchange.weights_cpu_backup_armed()``
(``auto`` default: ``not exchange_armed()``) instead of reading
``server_args.enable_weights_cpu_backup`` unconditionally
(``model_runner.py`` -- was ``:2440-2442`` before this slice). The chain this
closes is measured, file:line + SHA, in
``/spinning/gpu-arb/weg2/ANALYSE_1369_RINGLESER_0913.md``: the launcher passes
``--enable-weights-cpu-backup`` UNCONDITIONALLY (``launcher.py:2656``,
``common_flags``), and neither TMS ``pause``/``resume``
(``tms_csrc/core.cpp``) nor the flag's Python computation ever read
``weight_source()``/``exchange_armed()``/``bounce_lane_armed()`` -- so the
48.672-MiB/card host ring was armed for the weights region on every arm,
including ``exchange``, where the peer-VRAM bounce is already the wake's byte
source (``weight_updater.py`` ``_weg2_xchg_inject_weights``) and the ring's
D2H-at-sleep / H2D-at-wake round trip is a second, unread copy of the same
bytes.

WHY THE C++ SIDE NEEDS NO CHANGE (checked, not assumed -- the class this file
pins is TestDangerDirection*, below): ``core.cpp`` already gates every ring
touch on ``metadata.enable_cpu_backup``, a field fixed ONCE at
``TorchMemorySaver::malloc()`` and read by BOTH ``pause()`` (acquire/D2H) and
``resume()`` (H2D/release) -- neither takes an ``enable_cpu_backup`` parameter
of its own, so the write side and the read side can never disagree for one
allocation's lifetime. Once the Python side never opens the weights region
with ``enable_cpu_backup=True`` under an armed exchange, ``ensure_ring()`` --
the ONLY call site of ``HostBackupRing::open_from_env`` in the whole vendored
tree, which is what opens/mmaps the launcher's pre-``ftruncate``'d per-card
file -- is never reached for that region: no allocation ever asks it to.

TWO DANGER DIRECTIONS, both mandatory per the ticket, both source-scan (no
CUDA context, no device, mirroring ``test_weg2_ring_hotpath_1235.py``'s
rationale: neither is observable without a real weights image, so a source
scan is the check that can actually fail on them):

(i)  the predicate says False, but the ring gets created anyway -- the 48 GiB
     stay and nobody notices except the host counter. Ruled out by pinning
     that ``ensure_ring(`` has exactly one call site anywhere in
     ``tms_csrc``, and that it sits textually AFTER the
     ``if (!metadata.enable_cpu_backup) { continue; }`` guard in the same
     loop iteration.
(ii) the predicate says False and ``resume()`` reads from a ring that was
     never acquired -- nullbytes or garbage served as weights, with no
     symptom but bad text. THIS IS THE WORSE ONE: wrong weights report no
     error, they just produce quietly wrong output. Ruled out by pinning that
     ``pause``/``resume`` take no ``enable_cpu_backup`` parameter of their
     own (so the flag cannot be threaded differently into the two legs for
     one allocation) and that both legs' ring-touching passes gate on the
     SAME ``metadata.enable_cpu_backup`` field.

A THIRD FINDING, named here because it belongs beside the two danger
directions even though the fix is not in this file's boundary: this slice
narrowed the scope of the search to "does another REGION depend on this
flag" (none do -- see the docstring below) and found instead that another
FILE derives the same fact through an independent, now-stale proxy.
``weight_updater.py:911`` (``_weg2_wake_weight_carrier``, NOT owned by this
package -- Paket B / DESK10) computes
``main_carried = bool(getattr(server_args, "enable_weights_cpu_backup",
False))`` -- the RAW launcher flag, which stays unconditionally True even
after this fix, rather than the gated value ``model_runner.py`` now actually
uses. Consequence, walked once here as documentation and NOT asserted as a
regression test (weight_updater.py is outside this ticket's file boundary,
Paket B's to fix): under ``--weg2-weight-source exchange`` with
``SGLANG_WEG2_XCHG_INJECT=shadow`` -- the DOCUMENTED DEFAULT combination,
``weight_exchange.py``'s own docstring: "shadow (the DEFAULT) runs the bounce
legs... the refill stays the authority" -- ``main_carried`` stays True,
``_weg2_wake_weight_carrier`` returns ``CARRIER_TMS_BACKUP`` (exchange is not
authoritative under ``inject=shadow``, so that branch is skipped first), and
``_weg2_wake_reload_weights`` returns without refilling anything -- because it
believes the TMS restore already carried the bytes, when this fix has just
made that restore a no-op on this arm. That is exactly danger direction (ii),
one file over: STILL, wrong weights, no refusal anywhere on this arm/inject
combination until ``weight_updater.py:911`` reads the SAME gated value this
file's :func:`model_runner`-binding tests pin.

REGIONS OTHER THAN WEIGHTS THAT ALSO TAKE AN ``enable_cpu_backup`` KEYWORD,
enumerated so a reader does not have to re-derive that they are unaffected:

* ``adaptive_graph_memory.py`` CUDA-graph capture-pool regions
  (``region_config(tag=tag, enable_cpu_backup=cpu_backup)``) -- an entirely
  separate subsystem (speculative-decode graph capture pools) gated by its
  OWN env var, ``SGLANG_ADAPTIVE_CAPTURE_CPU_BACKUP`` (default False,
  ``envs.py``), never by ``weight_source``/``weights_cpu_backup_armed``. Not
  touched by, and not a reader of, this change.
* ``weight_updater.py:1567`` opens the weights region with a HARDCODED
  ``enable_cpu_backup=False`` for the disk-reload roll-forward path
  (``_weg2_wake_reload_weights``'s ``CARRIER_DISK`` branch) -- this is a
  SEPARATE, later ``malloc()`` for the repacked parameters
  (``process_weights_after_loading`` makes fresh device allocations for
  every repacking scheme), consistent by construction with the carrier
  decision that routed here, and untouched by this change.
"""

from __future__ import annotations

import os
import unittest
from contextlib import contextmanager

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
CSRC = os.path.join(TREE, "python", "sglang", "srt", "weg2", "tms_csrc")


def _read(name: str) -> str:
    with open(os.path.join(CSRC, name)) as f:
        return f.read()


def _code(src: str) -> str:
    """``src`` with ``//`` and ``/* */`` comments blanked (same offsets) --
    the #995 prose-marker trap in source form: a comment that MENTIONS a call
    is not a call. Copied verbatim from ``test_weg2_ring_hotpath_1235.py`` so
    both files agree on what counts as code."""
    out = []
    i, n = 0, len(src)
    while i < n:
        if src.startswith("//", i):
            j = src.find("\n", i)
            j = n if j == -1 else j
            out.append(" " * (j - i))
            i = j
        elif src.startswith("/*", i):
            j = src.find("*/", i + 2)
            j = n if j == -1 else j + 2
            out.append(" " * (j - i))
            i = j
        else:
            out.append(src[i])
            i += 1
    return "".join(out)


def _body(src: str, signature: str) -> str:
    """The text of one function, brace-balanced from its signature."""
    start = src.index(signature)
    depth = 0
    i = src.index("{", start)
    for j in range(i, len(src)):
        if src[j] == "{":
            depth += 1
        elif src[j] == "}":
            depth -= 1
            if depth == 0:
                return src[start : j + 1]
    raise AssertionError(f"unbalanced braces after {signature!r}")


@contextmanager
def _arm(value):
    """``--weg2-weight-source`` for the duration of a block."""
    from sglang.srt.weg2 import weight_exchange as wx

    previous = os.environ.get(wx.WEIGHT_SOURCE_ENV)
    if value is None:
        os.environ.pop(wx.WEIGHT_SOURCE_ENV, None)
    else:
        os.environ[wx.WEIGHT_SOURCE_ENV] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(wx.WEIGHT_SOURCE_ENV, None)
        else:
            os.environ[wx.WEIGHT_SOURCE_ENV] = previous


@contextmanager
def _backup_mode(value):
    """``SGLANG_WEG2_WEIGHTS_CPU_BACKUP`` for the duration of a block."""
    from sglang.srt.weg2 import weight_exchange as wx

    previous = os.environ.get(wx.WEIGHTS_CPU_BACKUP_ENV)
    if value is None:
        os.environ.pop(wx.WEIGHTS_CPU_BACKUP_ENV, None)
    else:
        os.environ[wx.WEIGHTS_CPU_BACKUP_ENV] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(wx.WEIGHTS_CPU_BACKUP_ENV, None)
        else:
            os.environ[wx.WEIGHTS_CPU_BACKUP_ENV] = previous


class TestWeightsCpuBackupArmedPredicate(unittest.TestCase):
    """The predicate itself, all three modes x all three arms."""

    def test_auto_is_not_exchange_armed_on_every_arm(self):
        from sglang.srt.weg2 import weight_exchange as wx

        for arm, expected_ring in (
            (None, True),
            (wx.WEIGHT_SOURCE_RING, True),
            (wx.WEIGHT_SOURCE_SHADOW, True),
            (wx.WEIGHT_SOURCE_EXCHANGE, False),
        ):
            with _arm(arm), _backup_mode(None):
                self.assertIs(
                    wx.weights_cpu_backup_armed(), expected_ring,
                    f"arm={arm!r}: auto must equal `not exchange_armed()`",
                )

    def test_shadow_keeps_its_compare_ground_unlike_exchange(self):
        """The one distinction the whole design rests on: shadow_armed() and
        exchange_armed() are mutually exclusive, and only exchange_armed()
        feeds this predicate -- shadow must not lose the ring beside it."""
        from sglang.srt.weg2 import weight_exchange as wx

        with _arm(wx.WEIGHT_SOURCE_SHADOW), _backup_mode(None):
            self.assertIs(wx.exchange_armed(), False)
            self.assertIs(wx.shadow_armed(), True)
            self.assertIs(wx.weights_cpu_backup_armed(), True,
                          "shadow's ring-backed compare ground must survive")

    def test_on_is_unconditionally_true_on_every_arm(self):
        from sglang.srt.weg2 import weight_exchange as wx

        for arm in (None, wx.WEIGHT_SOURCE_RING, wx.WEIGHT_SOURCE_EXCHANGE,
                    wx.WEIGHT_SOURCE_SHADOW):
            with _arm(arm), _backup_mode(wx.WEIGHTS_CPU_BACKUP_ON):
                self.assertIs(wx.weights_cpu_backup_armed(), True)

    def test_off_is_unconditionally_false_on_every_arm(self):
        from sglang.srt.weg2 import weight_exchange as wx

        for arm in (None, wx.WEIGHT_SOURCE_RING, wx.WEIGHT_SOURCE_EXCHANGE,
                    wx.WEIGHT_SOURCE_SHADOW):
            with _arm(arm), _backup_mode(wx.WEIGHTS_CPU_BACKUP_OFF):
                self.assertIs(wx.weights_cpu_backup_armed(), False)

    def test_an_unrecognised_env_value_refuses_by_name_w107(self):
        """CORRECTED against the landed contract (train tip `edefc42c34`,
        commit '[#1369 step 1/4] weights_cpu_backup_armed()'): unlike
        ``inject_mode``, a non-empty, unrecognised value does NOT silently
        fall back to `auto` -- it refuses by name (W107
        ``Weg2WeightsCpuBackupModeUnknown``), because neither direction of a
        silent guess is obviously safe for THIS predicate (an A/B boot that
        typed `on` and silently got `auto` looks identical to one that ran
        correctly)."""
        from sglang.srt.weg2 import weight_exchange as wx

        with _arm(None), _backup_mode("OFFF"):
            with self.assertRaises(wx.Weg2WeightsCpuBackupModeUnknown) as ctx:
                wx.weights_cpu_backup_armed()
            self.assertIn("W107", str(ctx.exception))

    def test_an_empty_env_value_is_auto_not_a_refusal(self):
        """Absence must not read as a typo: unset/blank is the documented
        default, not W107."""
        from sglang.srt.weg2 import weight_exchange as wx

        with _arm(None), _backup_mode(None):
            self.assertIs(wx.weights_cpu_backup_armed(), True)
        with _arm(wx.WEIGHT_SOURCE_EXCHANGE), _backup_mode(""):
            self.assertIs(wx.weights_cpu_backup_armed(), False)

    def test_explicit_overrides_the_environment_when_recognised(self):
        from sglang.srt.weg2 import weight_exchange as wx

        with _arm(wx.WEIGHT_SOURCE_EXCHANGE), _backup_mode(wx.WEIGHTS_CPU_BACKUP_OFF):
            self.assertIs(
                wx.weights_cpu_backup_armed(explicit=wx.WEIGHTS_CPU_BACKUP_ON),
                True,
                "explicit=on must win over an env value of off",
            )

    def test_explicit_unrecognised_also_refuses_by_name(self):
        """CORRECTED against the landed contract: an unrecognised `explicit`
        is not silently ignored either -- it is read FIRST and, since it is
        non-empty, refuses exactly like a bad environment value would."""
        from sglang.srt.weg2 import weight_exchange as wx

        with _arm(wx.WEIGHT_SOURCE_EXCHANGE), _backup_mode(wx.WEIGHTS_CPU_BACKUP_ON):
            with self.assertRaises(wx.Weg2WeightsCpuBackupModeUnknown):
                wx.weights_cpu_backup_armed(explicit="garbage")

    def test_explicit_empty_string_falls_through_to_the_environment(self):
        """An explicit value that is present but BLANK (the CLI default
        before argv parsing sets it) must defer to the environment, not
        refuse -- only a truly unset/blank explicit does, per the
        function's own contract ("if explicit is not None and
        str(explicit).strip()")."""
        from sglang.srt.weg2 import weight_exchange as wx

        with _arm(wx.WEIGHT_SOURCE_EXCHANGE), _backup_mode(wx.WEIGHTS_CPU_BACKUP_ON):
            self.assertIs(wx.weights_cpu_backup_armed(explicit=""), True)
            self.assertIs(wx.weights_cpu_backup_armed(explicit=None), True)


class TestModelRunnerBinding(unittest.TestCase):
    """The call site: exactly one predicate, ANDed onto the untouched old
    disjunction, never a re-derivation of the axis. The exact-text pin on
    ``model_runner.py``'s decider line lives in
    ``test_weg2_xchg_gate_axis_1273.py``'s
    ``TestTheCompareGroundExistsOnThisArm`` (updated by this same ticket,
    #1369, to state the NEW invariant) -- this class only pins that the two
    files' imports and names agree, so the two cannot drift apart."""

    def test_model_runner_imports_the_predicate_from_weight_exchange(self):
        import inspect

        import sglang.srt.model_executor.model_runner as mr

        src = inspect.getsource(mr.ModelRunner)
        self.assertIn("weights_cpu_backup_armed", src)
        # imported alongside the two names the call site already used --
        # ein-job-ein-mover for the import statement itself, not a second
        # import block elsewhere in the method.
        self.assertIn(
            "from sglang.srt.weg2.weight_exchange import (\n"
            "            RunnerShape,\n"
            "            weights_cpu_backup_armed,\n"
            "            weights_region_tag_for,\n"
            "        )",
            src,
        )


class TestDangerDirectionIRingNeverOpensWhenDisarmed(unittest.TestCase):
    """(i): predicate says False, ring gets created anyway.

    Ruled out structurally: ``ensure_ring()`` -- the sole path to
    ``HostBackupRing::open_from_env`` (the call that opens/mmaps the
    launcher's pre-``ftruncate``'d per-card file) -- has exactly ONE call
    site in the whole vendored tree, and it sits inside the
    ``enable_cpu_backup`` guard of ``pause()``'s pass 1. An allocation whose
    ``enable_cpu_backup`` is False (which is what this ticket's Python-side
    binding now produces for the weights region under an armed exchange)
    never reaches this line, so it never opens the ring -- REGARDLESS of
    whether some OTHER allocation elsewhere in the process has already done
    so (a shared singleton `ring_` does not un-gate a specific allocation's
    own guard, since the guard is checked per-allocation, before
    ``ensure_ring`` is ever called for that allocation).
    """

    @classmethod
    def setUpClass(cls):
        cls.core_raw = _read("core.cpp")
        cls.core = _code(cls.core_raw)
        cls.pause = _body(cls.core, "void TorchMemorySaver::pause(")

    def test_ensure_ring_has_exactly_one_call_site_in_core_cpp(self):
        # One DEFINITION (core.cpp:112) plus one CALL (core.cpp:235) is the
        # whole surface; a second call site anywhere would be a second,
        # possibly unguarded, door.
        self.assertEqual(
            self.core.count("ensure_ring("), 2,
            "ensure_ring( must appear exactly twice: once where it is "
            "defined, once where pause() calls it -- a third occurrence is "
            "an unaudited second door",
        )

    def test_no_other_vendored_file_calls_ensure_ring(self):
        for name in sorted(os.listdir(CSRC)):
            if not name.endswith((".cpp", ".h")) or name in ("core.cpp", "core.h"):
                continue
            src = _code(_read(name))
            self.assertNotIn(
                "ensure_ring(", src,
                f"{name} calls ensure_ring -- a second call site outside "
                "core.cpp's guarded one",
            )

    def test_the_call_site_sits_after_the_enable_cpu_backup_guard(self):
        guard = self.pause.find("if (!metadata.enable_cpu_backup)")
        call = self.pause.find("ensure_ring(metadata.device)")
        self.assertNotEqual(guard, -1, "the guard is gone from pause()")
        self.assertNotEqual(call, -1, "ensure_ring is no longer called from pause()")
        self.assertLess(
            guard, call,
            "ensure_ring() must be reached only after the "
            "enable_cpu_backup guard -- an allocation with the flag False "
            "must never open the ring file at all",
        )

    def test_open_from_env_the_actual_file_open_has_one_caller(self):
        self.assertEqual(
            self.core.count("open_from_env("), 1,
            "core.cpp must call HostBackupRing::open_from_env exactly once, "
            "from inside ensure_ring()",
        )
        ensure_ring_body = _body(self.core, "TorchMemorySaver::ensure_ring(")
        self.assertIn("open_from_env(", ensure_ring_body)


class TestDangerDirectionIIResumeNeverReadsAGhostRing(unittest.TestCase):
    """(ii): predicate says False and resume() reads from a ring that was
    never acquired -- the worse direction, because wrong weights produce no
    error, only quietly wrong text.

    Ruled out structurally: neither ``pause`` nor ``resume`` takes an
    ``enable_cpu_backup`` parameter of its own -- the ONLY place that value
    exists is ``AllocationMetadata::enable_cpu_backup``, fixed once at
    ``TorchMemorySaver::malloc()`` and read (never re-derived) by both legs.
    A single boolean cannot tell pause() "don't acquire" and resume() "do
    read from what was never acquired" for the SAME allocation -- there is
    only one flag, and both legs gate their ring touch on it.
    """

    @classmethod
    def setUpClass(cls):
        cls.core_raw = _read("core.cpp")
        cls.core = _code(cls.core_raw)
        cls.malloc = _body(cls.core, "cudaError_t TorchMemorySaver::malloc(")
        cls.pause = _body(cls.core, "void TorchMemorySaver::pause(")
        cls.resume = _body(cls.core, "void TorchMemorySaver::resume(")

    def test_pause_and_resume_take_no_enable_cpu_backup_parameter(self):
        header = _read("core.h")
        self.assertIn("void pause(const std::string& tag);", header)
        self.assertIn("void resume(const std::string& tag);", header)
        self.assertNotIn("void pause(const std::string& tag, const bool", header)
        self.assertNotIn("void resume(const std::string& tag, const bool", header)

    def test_the_flag_is_fixed_once_at_malloc_and_stored_on_the_metadata(self):
        self.assertIn("const bool enable_cpu_backup", self.malloc)
        # AllocationMetadata construction: the malloc-time value is the ONE
        # place it is written into the struct that pause/resume later read.
        self.assertIn(
            "AllocationMetadata{size, device, tag, AllocationState::ACTIVE, "
            "enable_cpu_backup,",
            self.malloc,
        )

    def test_resumes_h2d_pass_is_gated_on_the_same_stored_field(self):
        # Pass 2 (async H2D per granule) -- the pass that would read a ghost
        # ring if this guard were ever bypassed.
        guard_pos = self.resume.find("if (!metadata.enable_cpu_backup) {\n            continue;")
        copy_pos = self.resume.find("cudaMemcpyAsync((char*)ptr + offset,\n"
                                     "                                             metadata.cpu_backup_granules[g], n,\n"
                                     "                                             cudaMemcpyHostToDevice")
        self.assertNotEqual(guard_pos, -1, "resume()'s H2D guard text changed shape")
        self.assertNotEqual(copy_pos, -1, "resume()'s H2D copy text changed shape")
        self.assertLess(guard_pos, copy_pos,
                        "the H2D-from-granules copy must sit after the "
                        "enable_cpu_backup guard in resume()")

    def test_resumes_release_pass_is_gated_on_the_same_stored_field_too(self):
        # Pass 4 (give the host bytes back) -- releasing a granule that was
        # never acquired would be the free-side twin of the same defect.
        guard_pos = self.resume.find(
            "if (!metadata.enable_cpu_backup || metadata.cpu_backup_granules.empty())")
        release_pos = self.resume.find("ring_->release(metadata.cpu_backup_granules)")
        self.assertNotEqual(guard_pos, -1)
        self.assertNotEqual(release_pos, -1)
        self.assertLess(guard_pos, release_pos)

    def test_pauses_acquire_pass_is_gated_on_the_same_field_name(self):
        guard_pos = self.pause.find("if (!metadata.enable_cpu_backup) {\n            continue;")
        acquire_pos = self.pause.find("ring->acquire(metadata.size")
        self.assertNotEqual(guard_pos, -1)
        self.assertNotEqual(acquire_pos, -1)
        self.assertLess(guard_pos, acquire_pos)


if __name__ == "__main__":
    unittest.main(verbosity=2)
