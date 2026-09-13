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
import unittest
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

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
        the same two things the backend's __init__ reads."""
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
