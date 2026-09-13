# SPDX-License-Identifier: Apache-2.0
"""xsn29 D died at prefill-graph capture. The protection existed -- on the
wrong predicate.

WHAT HAPPENED (boot weg2xsn29 @ 19c7502642, three D ranks, exit -9 via D's own
kill_process_tree; no OOM, no host pressure, no W48/W97/W98):

    [12:19:58 TP0] Capture target prefill CUDA graph begin. backend=breakable
    NotImplementedError: Breakable CUDA graph is not compatible with memory
    saver mode                (breakable_cuda_graph_backend.py:213 @ 19c7502642)
    Exception: Capture prefill CUDA graph failed

The collision is old: the weg2 launcher sets SGLANG_MEMORY_SAVER_CUDA_GRAPH=1
for every group (launcher.py:4507 @ 19c7502642, for the R4 capture-pool lever
in weg2_memory_saver.py:1656), and the breakable backend refuses that
combination in its own __init__. It never fired because `is_multimodal` was
disabling the prefill graph for an UNRELATED reason -- the vision tower was
loaded on every boot until #1356 made P and D text-only (2cc618b819). The
xsn28 D log shows the accident:

    [10:37:02] Breakable CUDA graph is incompatible with multimodal model;
               disabling prefill CUDA graph.       -> prefill.backend='disabled'

and xsn29, with --weg2-vision off, shows the same boot reaching
`backend=breakable` for the first time.

THE CLASS is compensator reachability: a guard that holds only because a
neighbouring, unrelated predicate happens to be true is not a guard, and the
day the neighbour changes it is gone -- silently, because nothing ever named
the hazard. So the rule here names the hazard itself. The backend's raise at
:213 stays as the backstop; it should now be unreachable through this path.

NOT DONE, deliberately: setting SGLANG_MEMORY_SAVER_CUDA_GRAPH=0 to get past
it. That switches off R4 (92/102/133 MiB per rank) -- the very lever the
launcher line exists for -- and would be a silent config change on a boot that
is meant to be measured.
"""

import inspect
import os
import pathlib
import unittest
import warnings
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import sglang
from sglang.srt.environ import envs
from sglang.srt.server_args import Backend, ServerArgs
from sglang.test.test_utils import CustomTestCase

_BENIGN = SimpleNamespace(
    attn_cp_size=1, enable_dp_attention=False, moe_a2a_backend="none"
)


def _args(*, enable_memory_saver: bool):
    """A ServerArgs carrying only what the BCG rule list reads, so the test
    exercises the WHOLE list (not just the rule under test) without a model."""
    sa = ServerArgs.__new__(ServerArgs)
    sa.enable_memory_saver = enable_memory_saver
    sa.lora_paths = None
    sa.enable_lora = False
    sa.cuda_graph_config = SimpleNamespace(
        prefill=SimpleNamespace(backend=Backend.BREAKABLE)
    )
    sa.use_mla_backend = lambda: False
    sa.get_model_config = lambda: SimpleNamespace(
        hf_config=SimpleNamespace(architectures=["Qwen3NextForCausalLM"]),
        is_multimodal=False,
    )
    return sa


def _resolve(sa):
    with mock.patch(
        "sglang.srt.arg_groups.overrides.resolved_view", lambda _sa: _BENIGN
    ):
        sa._disable_breakable_cudagraph_if_incompatible()
    return sa.cuda_graph_config.prefill.backend


class TheHazardIsNamedNotInherited(CustomTestCase):
    def test_text_only_plus_memory_saver_graph_disables_the_prefill_graph(self):
        """THE xsn29 CASE: text-only (is_multimodal False), memory saver on,
        SGLANG_MEMORY_SAVER_CUDA_GRAPH=1 as the launcher sets it."""
        with envs.SGLANG_MEMORY_SAVER_CUDA_GRAPH.override(True):
            self.assertEqual(_resolve(_args(enable_memory_saver=True)),
                             Backend.DISABLED)

    def test_it_stays_breakable_when_the_env_var_is_off(self):
        """The rule must not cost the prefill graph to everyone else: the code
        default is False (environ.py:1978), only the weg2 launcher sets 1."""
        with envs.SGLANG_MEMORY_SAVER_CUDA_GRAPH.override(False):
            self.assertEqual(_resolve(_args(enable_memory_saver=True)),
                             Backend.BREAKABLE)

    def test_it_stays_breakable_without_the_memory_saver(self):
        with envs.SGLANG_MEMORY_SAVER_CUDA_GRAPH.override(True):
            self.assertEqual(_resolve(_args(enable_memory_saver=False)),
                             Backend.BREAKABLE)

    def test_the_reason_reaches_the_log_as_a_checkable_literal(self):
        """'incompatible with multimodal model' is composed through %s and is
        NOT in the source; the greppable half is the tail. A boot seat that
        greps the composed half is checking a string that does not exist."""
        src = inspect.getsource(
            ServerArgs._disable_breakable_cudagraph_if_incompatible)
        self.assertIn("disabling prefill CUDA graph.", src)
        self.assertIn("memory saver CUDA graph", src)

    def test_the_rule_predicate_mirrors_what_the_backend_refuses(self):
        """Two bookkeepings of one condition drift. The config rule must read
        the same two things the backend's __init__ reads.

        NOTE: naming the same VARIABLE is not the same as reading it the same
        way -- that is what ONE_READER below actually pins."""
        from sglang.srt.model_executor.runner_backend import (
            breakable_cuda_graph_backend as bcg,
        )

        backend_src = inspect.getsource(bcg.BreakableCudaGraphBackend.__init__)
        self.assertIn("SGLANG_MEMORY_SAVER_CUDA_GRAPH", backend_src)
        self.assertIn("enable_memory_saver", backend_src)

        rule_src = inspect.getsource(
            ServerArgs._disable_breakable_cudagraph_if_incompatible)
        self.assertIn("SGLANG_MEMORY_SAVER_CUDA_GRAPH", rule_src)
        self.assertIn("enable_memory_saver", rule_src)


#: Every spelling worth asking about, plus the unset case as None.
_SPELLINGS = ("1", "true", "TRUE", "yes", "y", "0", "false", "no", "n",
              "t", "f", "on", "off", "", "garbage", None)


def _read(fn):
    try:
        return fn()
    except Exception as exc:  # a raising reader is an outcome too
        return type(exc).__name__


def _with(spelling, fn):
    name = "SGLANG_MEMORY_SAVER_CUDA_GRAPH"
    env = {} if spelling is None else {name: spelling}
    with mock.patch.dict(os.environ, env, clear=False):
        if spelling is None:
            os.environ.pop(name, None)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return _read(fn)


class OneVariableOneReader(CustomTestCase):
    """#1366 follow-up. The boot seat found rule and backstop reading the same
    variable through two different parsers, and the numbers it reported are not
    the ones this box produces -- so the table below is MEASURED here, in the
    test, instead of quoted:

        spelling  get_bool_env_var   envs.EnvBool   verdict
        'yes'     False              True           DIVERGE
        'y'       False              True           DIVERGE
        't' 'on' 'off' '' 'garbage'  False  False   same
        everything else                            same

    Two corrections to the report that motivated this work, both executable:
      * get_bool_env_var does NOT accept yes/y/t/on -- its truthy set is
        exactly ("true", "1") (utils/common.py:1777).
      * envs.EnvBool.get() does NOT raise on an unknown spelling; EnvField.get
        catches the ValueError, warns and returns the declared default
        (environ.py:62-67). No ValueError ever reaches a call site.
    So the divergence is 2 of 16 spellings, not 6 of 10, and it is not an
    exception -- which matters, because the claimed failure mode (the
    resolution exploding before the backstop) does not exist.

    It is still two bookkeepings of one declared variable, which is the defect:
    get_bool_env_var's own first line says `FIXME: move your environment
    variable to sglang.srt.environ`, and the variable IS declared there
    (environ.py:1978). So every reader is moved onto the declaration."""

    def test_no_site_reads_this_variable_through_the_lenient_helper(self):
        """THE POPULATION, not the pair in front of me: at 4a90bf163a this
        variable had TEN readers, six lenient and four strict."""
        root = pathlib.Path(sglang.__file__).resolve().parent
        needle = 'get_bool_env_var("SGLANG_MEMORY_SAVER_CUDA_GRAPH")'
        hits = [str(f.relative_to(root)) for f in root.rglob("*.py")
                if needle in f.read_text(encoding="utf-8", errors="ignore")]
        self.assertEqual(hits, [], f"still reading through two parsers: {hits}")

    def test_the_rule_and_the_backstop_hold_the_very_same_field_object(self):
        """Identity, not resemblance: one object cannot disagree with itself."""
        import sglang.srt.server_args as sa_mod
        from sglang.srt.model_executor.runner_backend import (
            breakable_cuda_graph_backend as bcg,
        )

        self.assertIs(bcg.envs.SGLANG_MEMORY_SAVER_CUDA_GRAPH,
                      sa_mod.envs.SGLANG_MEMORY_SAVER_CUDA_GRAPH)

    def test_rule_and_backstop_agree_on_every_spelling(self):
        """EXECUTION SMOKE over the DIVERGING spellings -- the earlier smoke
        only walked '1', which is the half that never disagreed."""
        from sglang.srt.model_executor.runner_backend import (
            breakable_cuda_graph_backend as bcg,
        )

        for spelling in _SPELLINGS:
            backstop = _with(
                spelling,
                lambda: bool(bcg.envs.SGLANG_MEMORY_SAVER_CUDA_GRAPH.get()))
            rule_disabled = _with(
                spelling,
                lambda: _resolve(_args(enable_memory_saver=True))
                == Backend.DISABLED)
            self.assertEqual(
                backstop, rule_disabled,
                f"spelling {spelling!r}: backstop would refuse={backstop} "
                f"but the rule disabled={rule_disabled}")

    def test_the_two_helpers_diverge_exactly_where_measured(self):
        """The reason-keeper: if upstream ever unifies the helpers, this goes
        red and the note above stops being true."""
        from sglang.srt.utils import get_bool_env_var as lenient

        name = "SGLANG_MEMORY_SAVER_CUDA_GRAPH"
        strict = envs.SGLANG_MEMORY_SAVER_CUDA_GRAPH
        diverging = [sp for sp in _SPELLINGS
                     if _with(sp, lambda: lenient(name))
                     != _with(sp, lambda: strict.get())]
        self.assertEqual(diverging, ["yes", "y"])

    def test_no_spelling_leaves_the_backstop_armed_behind_a_silent_rule(self):
        """The only dangerous direction: rule says keep the graph, backstop
        then kills the rank. It must not exist for ANY spelling."""
        from sglang.srt.utils import get_bool_env_var as lenient

        name = "SGLANG_MEMORY_SAVER_CUDA_GRAPH"
        for spelling in _SPELLINGS:
            would_refuse = _with(spelling, lambda: bool(lenient(name)))
            disabled = _with(
                spelling,
                lambda: _resolve(_args(enable_memory_saver=True))
                == Backend.DISABLED)
            if would_refuse:
                self.assertTrue(
                    disabled,
                    f"spelling {spelling!r} arms the backstop while the rule "
                    f"leaves the prefill graph on -- that is the xsn29 death")


class TheLauncherKeepsItsLever(CustomTestCase):
    def test_launcher_still_sets_the_r4_env_var(self):
        """The fix is in the resolution, not in the launcher: turning the env
        var off would disable R4, which is what the line exists for."""
        from sglang.srt.weg2 import launcher

        src = inspect.getsource(launcher)
        self.assertIn('env["SGLANG_MEMORY_SAVER_CUDA_GRAPH"] = env.get(', src)


class TheLauncherEnvReachesTheRule(CustomTestCase):
    """EXECUTION SMOKE, not a source pin: the value the launcher really puts
    in the group env is the value the rule really reads. A desk test that
    hands the rule its own `True` would pass while the launcher wrote a
    spelling the env parser reads as false."""

    def test_the_env_the_launcher_builds_disables_the_prefill_graph(self):
        from sglang.srt.weg2 import launcher

        env = launcher.build_env(
            tree="/tmp/weg2-smoke-tree", venv="/tmp/weg2-smoke-venv",
            cvd="0,1,2", store_dir="/tmp/weg2-smoke-store", debug_hold=False,
            tag="bcg-rule-smoke", group="D",
        )
        value = env.get("SGLANG_MEMORY_SAVER_CUDA_GRAPH")
        self.assertEqual(value, "1", "launcher.py:4507 is the R4 lever")

        with mock.patch.dict(
            os.environ, {"SGLANG_MEMORY_SAVER_CUDA_GRAPH": value}
        ):
            self.assertEqual(_resolve(_args(enable_memory_saver=True)),
                             Backend.DISABLED,
                             "the launcher's own env string must reach the "
                             "rule as true -- '1' vs 'true' is a real gap")


if __name__ == "__main__":
    unittest.main()
