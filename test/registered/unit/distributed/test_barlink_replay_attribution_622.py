# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""#622: the abort raised at a replay boundary must name the replay window.

THE GAP THESE TESTS PIN
-----------------------
``Bar1CollectiveAborted`` at a graph-replay boundary already said, correctly,
that the aborting kernel is inside the replayed graph and cannot be named.
Two specimens of this crash family (2026-08-05 21:10, 2026-08-07 03:25) end
exactly at that sentence: the only collective either message names is an
8-byte control-plane launch from a window that had already closed, which the
message itself warns is not the culprit.

The kernel still cannot be named -- a replay runs no host code per collective,
so no host-side instrument can see individual kernels inside it. What CAN be
named is the GRAPH, because the host picked it one frame above the launch. The
tests below pin that the pick is recorded there, that it reaches the abort
message, and that recording it touches no device.

The reason each of these is a separate test rather than one end-to-end assert:
the value of this instrument on the NEXT specimen is that the three ranks'
messages can be diffed. That requires the tag to be correct (right window),
present (in the message), and harmless (never the thing that raises).
"""

import unittest
from unittest import mock

import torch

from sglang.srt.distributed.device_communicators import barlink_abort_gate
from sglang.srt.distributed.device_communicators.barlink_bar1 import (
    Bar1CollectiveAborted,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _load_517_harness():
    """Reuse the #517 test's transport builder rather than re-implementing it.

    Loaded by path because this directory is not a package: a re-implemented
    ``_transport`` would be a second definition of the object under test, and
    it would keep these tests passing after the real one changed -- the exact
    failure the #517 module's own docstring gives for not re-implementing
    ``check_aborted``.
    """
    import importlib.util
    import os

    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_barlink_bar1_abort_deferred_517.py",
    )
    spec = importlib.util.spec_from_file_location("_abort_deferred_517", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    def transport(**kwargs):
        """#517's transport, plus the one field its builder has fallen behind on.

        ``_transport`` builds via ``__new__`` and sets each field by hand, so
        it drifts whenever ``__init__`` gains one. It is currently missing
        ``_abort_code_seen`` (``barlink_bar1.py`` __init__), which
        ``_read_status_for_check`` reads -- so seven of that module's own
        tests fail on this tree BEFORE any #622 change. That is a pre-existing
        defect in a test helper, not this window's to fix; setting the field to
        its ``__init__`` default here keeps these tests exercising the real
        ``check_aborted`` rather than forking a second copy of the builder.
        """
        t = mod._transport(**kwargs)
        if not hasattr(t, "_abort_code_seen"):
            t._abort_code_seen = 0
        return t

    return mod._boundary, transport


_boundary, _transport = _load_517_harness()


def _drive_to_abort(t, limit=8):
    """Replay boundaries until the abort surfaces.

    More than one is required and that is the shipped behaviour, not a test
    artefact: the #517 staged read ISSUES the device copy on one boundary and
    reads the value on a later one, so a tripped word is reported at the next
    boundary, not the current one. Bounded so a guard that stopped firing
    fails the test instead of hanging it.
    """
    for _ in range(limit):
        _boundary(t)


class _ExplodingRepr:
    """Caller data whose ``repr`` raises. Not hypothetical: the key is an
    arbitrary runner object and this code runs inside an exception path."""

    def __repr__(self):
        raise RuntimeError("repr exploded")


class _TagTestCase(CustomTestCase):
    def setUp(self):
        barlink_abort_gate.reset_for_test()
        barlink_abort_gate.reset_replay_tag_for_test()

    def tearDown(self):
        barlink_abort_gate.reset_for_test()
        barlink_abort_gate.reset_replay_tag_for_test()


class TestTheTagRecordsTheWindow(_TagTestCase):
    def test_an_unwritten_tag_says_so_instead_of_inventing_a_window(self):
        """Silence must read as silence. A default of "full" or "graph 0"
        would put a plausible, wrong window into a crash report."""
        self.assertEqual(
            barlink_abort_gate.format_current_replay(),
            barlink_abort_gate._REPLAY_UNSET,
        )

    def test_the_kind_and_key_survive_to_the_formatted_line(self):
        barlink_abort_gate.note_replay("full", "ShapeKey(bs=8,tok=8)")
        text = barlink_abort_gate.format_current_replay()
        self.assertIn("full", text)
        self.assertIn("ShapeKey(bs=8,tok=8)", text)

    def test_the_ordinal_appears_only_when_there_is_one(self):
        """A full graph has no segment ordinal; a breakable segment does.
        Printing ``index=-1`` for the former would read as a real segment."""
        barlink_abort_gate.note_replay("full", "k")
        self.assertNotIn("index=", barlink_abort_gate.format_current_replay())
        barlink_abort_gate.note_replay("breakable/seg", None, 7)
        self.assertIn("index=7", barlink_abort_gate.format_current_replay())

    def test_the_last_write_wins_which_is_the_window_that_aborted(self):
        """The abort check runs at the boundary of the MOST RECENT replay, so
        the tag must describe that one and not the first of the step."""
        barlink_abort_gate.note_replay("draft/rung", None, 0)
        barlink_abort_gate.note_replay("draft/rung", None, 1)
        barlink_abort_gate.note_replay("draft/rung", None, 2)
        self.assertIn("index=2", barlink_abort_gate.format_current_replay())

    def test_the_sequence_counter_separates_a_live_tag_from_a_stale_one(self):
        """Without it, a rank that stopped replaying an hour ago and a rank
        that aborted in its current replay produce the same line."""
        for _ in range(5):
            barlink_abort_gate.note_replay("full", "k")
        _, _, _, seq = barlink_abort_gate.current_replay()
        self.assertEqual(seq, 5)
        self.assertIn("replay #5", barlink_abort_gate.format_current_replay())


class TestTheTagIsCheapAndGraphSafe(_TagTestCase):
    """The instrument sits on the decode replay path, so its cost and its
    graph-safety are properties to pin, not intentions to state."""

    def test_the_key_is_stored_by_reference_and_never_formatted_on_the_path(self):
        """Formatting on the hot path would allocate a string per decode step
        for a value read once per crash. The identity check is the pin: a
        stored ``str(key)`` or ``f"{key}"`` would fail it."""
        key = object()
        barlink_abort_gate.note_replay("full", key)
        _, stored, _, _ = barlink_abort_gate.current_replay()
        self.assertIs(stored, key)

    def test_recording_a_replay_touches_no_device(self):
        """A capture-illegal or synchronizing call here would turn an
        instrument into the crash. Any device op would go through one of
        these; none may be reached."""
        with (
            mock.patch.object(
                torch.cuda, "synchronize", side_effect=AssertionError("synchronized")
            ),
            mock.patch.object(
                torch.cuda,
                "current_stream",
                side_effect=AssertionError("touched stream"),
            ),
            mock.patch.object(
                torch.Tensor, "item", side_effect=AssertionError("called .item()")
            ),
        ):
            barlink_abort_gate.note_replay("full", "k", 3)
        self.assertIn("index=3", barlink_abort_gate.format_current_replay())


class TestTheDiagnosticNeverBecomesTheError(_TagTestCase):
    """A triage instrument that raises replaces the fault it was added to
    explain. Every input here is attacker-shaped only in the sense that it is
    real caller data arriving on an already-broken run."""

    def test_a_key_whose_repr_raises_is_reported_not_propagated(self):
        barlink_abort_gate.note_replay("full", _ExplodingRepr())
        text = barlink_abort_gate.format_current_replay()
        self.assertIn("unrepresentable", text)

    def test_an_enormous_key_is_truncated_so_it_cannot_bury_the_message(self):
        barlink_abort_gate.note_replay("full", "x" * 5000)
        text = barlink_abort_gate.format_current_replay()
        self.assertLess(len(text), 400)
        self.assertIn("...", text)


class TestTheAbortMessageCarriesTheWindow(_TagTestCase):
    """The end-to-end property: what a post-mortem actually reads."""

    def test_a_graph_replay_abort_names_the_replay_window(self):
        barlink_abort_gate.note_replay("breakable/seg", "ShapeKey(bs=16)", 4)
        t = _transport(aborted=1)
        t._captured_launches = True
        with self.assertRaises(Bar1CollectiveAborted) as cm:
            _drive_to_abort(t)
        text = str(cm.exception)
        # The pre-#622 sentence is preserved: the kernel is still not named,
        # and the stale last-launch is still flagged as not the culprit.
        self.assertIn("GRAPH-REPLAY window", text)
        self.assertIn("must not be read as", text)
        # ...and the window itself is now in the line.
        self.assertIn("REPLAY WINDOW (#622)", text)
        self.assertIn("breakable/seg", text)
        self.assertIn("ShapeKey(bs=16)", text)
        self.assertIn("index=4", text)

    def test_a_host_path_abort_is_left_exactly_as_it_was(self):
        """The tag answers a question only the empty-window branch asks. A
        host-path abort already names its collective, and appending a replay
        window there would suggest the two are related when they are not."""
        barlink_abort_gate.note_replay("full", "ShapeKey(bs=16)")
        t = _transport(aborted=1)
        with self.assertRaises(Bar1CollectiveAborted) as cm:
            for _ in range(8):
                t._note_launch("all_reduce", 4096)
                t.check_aborted("all_reduce")
        text = str(cm.exception)
        self.assertIn("ran on the host path since the previous check", text)
        self.assertNotIn("REPLAY WINDOW (#622)", text)

    def test_an_unreadable_tag_does_not_stop_the_abort_from_raising(self):
        """Can-fail shape: if the formatter throws, the real error must still
        arrive. This is the failure mode the guard exists for."""
        with mock.patch.object(
            barlink_abort_gate,
            "format_current_replay",
            side_effect=RuntimeError("formatter exploded"),
        ):
            t = _transport(aborted=1)
            t._captured_launches = True
            with self.assertRaises(Bar1CollectiveAborted) as cm:
                _drive_to_abort(t)
        self.assertIn("GRAPH-REPLAY window", str(cm.exception))


class TestTheReplaySitesAreWired(_TagTestCase):
    """The instrument is only worth its cost if it is actually called from
    the replay paths. A unit test of the slot alone would keep passing after
    every call site was removed -- these read the shipped sources instead,
    which is the same pin the #616 window used for its gate sites."""

    _SITES = (
        (
            "sglang/srt/model_executor/runner_backend/full_cuda_graph_backend.py",
            'note_replay("full"',
        ),
        (
            "sglang/srt/model_executor/runner_backend/breakable_cuda_graph_backend.py",
            'note_replay("breakable"',
        ),
        (
            "sglang/srt/model_executor/runner_backend_utils/breakable_cuda_graph"
            "/breakable_cuda_graph.py",
            'note_replay("breakable/seg"',
        ),
        (
            "sglang/srt/speculative"
            "/multi_layer_eagle_draft_extend_cuda_graph_runner.py",
            'note_replay("draft/rung"',
        ),
    )

    def _root(self):
        import os

        import sglang

        return os.path.dirname(os.path.dirname(os.path.abspath(sglang.__file__)))

    def test_every_replay_site_records_its_window(self):
        root = self._root()
        import os

        for rel, needle in self._SITES:
            with self.subTest(site=rel):
                src = open(os.path.join(root, rel)).read()
                self.assertIn(needle, src, f"{rel} no longer records its window")

    def test_the_tag_is_written_before_the_launch_not_after(self):
        """Order is the whole correctness argument: a tag written after
        ``graph.replay()`` returns would still be the PREVIOUS window when the
        abort check on the very next line reads it."""
        import os

        root = self._root()
        rel = "sglang/srt/model_executor/runner_backend/full_cuda_graph_backend.py"
        src = open(os.path.join(root, rel)).read()
        tag_at = src.index('note_replay("full"')
        launch_at = src.index("self._graphs[shape_key].replay()")
        check_at = src.index("barlink_abort_gate.check_after_graph_replay()")
        self.assertLess(tag_at, launch_at, "tag must precede the launch")
        self.assertLess(launch_at, check_at, "check must follow the launch")


class TestTheCaptureCensusReachesTheAbortPath(_TagTestCase):
    """#622 composed with the #619 overlap.

    The collective census is silent on a replay abort by construction -- it
    counts host calls and a replay makes none. The CAPTURE census is the one
    instrument that can describe the named window, and before this it was
    dumped only from the scheduler's periodic tick, which the rank that dies
    first never reaches (the 2026-08-05 specimen has rank 0 raising 10 s
    ahead of the others).
    """

    def test_the_abort_path_dumps_the_capture_census(self):
        from sglang.srt.distributed.device_communicators import barlink_capture_census

        seen = []
        with (
            mock.patch.object(
                barlink_capture_census, "capture_census_enabled", return_value=True
            ),
            mock.patch.object(
                barlink_capture_census,
                "format_local_capture_census",
                side_effect=lambda rank: (
                    seen.append(rank) or "CAPTURE-CENSUS rank %d" % rank
                ),
            ),
        ):
            t = _transport(aborted=1, rank=2)
            t._captured_launches = True
            with self.assertRaises(Bar1CollectiveAborted):
                _drive_to_abort(t)
        self.assertEqual(
            seen, [2], "the abort path must dump THIS rank's capture census"
        )

    def test_a_failing_capture_census_does_not_mask_the_abort(self):
        from sglang.srt.distributed.device_communicators import barlink_capture_census

        with (
            mock.patch.object(
                barlink_capture_census, "capture_census_enabled", return_value=True
            ),
            mock.patch.object(
                barlink_capture_census,
                "format_local_capture_census",
                side_effect=RuntimeError("census exploded"),
            ),
        ):
            t = _transport(aborted=1)
            t._captured_launches = True
            with self.assertRaises(Bar1CollectiveAborted):
                _drive_to_abort(t)


if __name__ == "__main__":
    unittest.main()
