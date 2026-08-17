"""Without SGLANG_BARLINK* nothing of the BAR1 strand happens.

The strand added a transport, a MoE dispatcher, three JIT extensions and a
handful of new branches in shared files. Every one of them is behind a flag,
and this file is the check on that claim rather than the claim itself.

Four independent angles, because each catches a different way of breaking
it:

1. **Import.** A fresh interpreter that builds the distributed stack, or
   the MoE dispatcher package, must not pull in ``barlink_bar1*``. Those are
   the modules that read env vars, open ``/dev/dmabuf_holder`` and
   JIT-build; reaching them has to require a choice.
2. **MoE backend.** The default ``moe_a2a_backend`` is ``none``, and adding
   ``BAR1EP`` to the enum must not have moved any existing predicate.
3. **Dispatch.** With no transport built, ``_select`` answers ``None`` for
   every op -- the inline gloo plane, exactly as before the strand.
4. **Flags.** Every new environment variable defaults to the previous
   behaviour, and their names all carry the ``SGLANG_BARLINK`` prefix, so
   "unset every SGLANG_BARLINK*" really is the whole off switch.

CPU only.
"""

import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

from sglang.srt.layers.moe.utils import MoeA2ABackend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=25, suite="base-a-test-cpu")


_REPO = Path(__file__).resolve().parents[4]
_COMM = _REPO / "python" / "sglang" / "srt" / "distributed" / "device_communicators"

#: The probe prints one module per line under this prefix, and the parser
#: accepts NOTHING else (#736).
#:
#: The previous parser was ``set(proc.stdout.split())`` -- every whitespace
#: token on stdout became a "module". On a heterogeneous rig the runtime emits
#:
#:     [... 18:52:39] WARNING [cuda.py:962] Detected different devices in the
#:     system: NVIDIA GeForce RTX 3080, NVIDIA GeForce RTX 5090, NVIDIA
#:     GeForce RTX 3080. Please make sure to set `CUDA_DEVICE_ORDER=PCI_BUS_ID`
#:     to avoid unexpected behavior.
#:
#: on stdout, so words like ``Detected``, ``3080,`` and ``WARNING`` entered the
#: set and the assertion ``== set()`` could never hold. The guard was red for a
#: reason that had nothing to do with what it guards -- i.e. it protected
#: nothing on the one rig it matters for, and a real regression would have been
#: indistinguishable from the noise.
#:
#: The #315 lesson: anchor the consumer regex to the structure of the line, not
#: to "whatever the producer happened to print". ``^MODULE:`` at line start
#: cannot be produced by prose.
_MODULE_LINE = re.compile(r"^MODULE:(\S+)$", re.MULTILINE)


def _probe_code(setup: str, needle: str) -> str:
    """Source for a subprocess that reports loaded modules matching ``needle``.

    One module per line, prefixed, so the parser never has to guess which part
    of stdout was the answer.
    """
    return (
        f"{setup}"
        "import sys\n"
        f"for _m in sorted(m for m in sys.modules if {needle!r} in m):\n"
        "    print('MODULE:' + _m)\n"
    )


def _clean_env():
    """The environment without any SGLANG_BARLINK* / bar1 knob."""
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("SGLANG_BARLINK") and "BAR1" not in k
    }
    env["CUDA_VISIBLE_DEVICES"] = "99"
    return env


class TestNothingIsImported(CustomTestCase):
    def _imports(self, code, env):
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
            timeout=600,
        )
        self.assertEqual(
            proc.returncode, 0, msg=f"subprocess failed:\n{proc.stderr[-3000:]}"
        )
        # Anchored, per line: stray stdout (warnings, banners, progress) cannot
        # enter the set. See _MODULE_LINE.
        return set(_MODULE_LINE.findall(proc.stdout))

    def test_distributed_import_does_not_pull_the_bar1_modules(self):
        code = _probe_code(
            "import sglang.srt.distributed.parallel_state\n"
            "import sglang.srt.distributed.device_communicators.barlink\n",
            "bar1",
        )
        self.assertEqual(self._imports(code, _clean_env()), set())

    def test_moe_import_does_not_pull_the_transport(self):
        """``bar1ep`` itself IS imported eagerly, and that is the package's
        convention -- ``token_dispatcher/__init__`` imports ``deepep`` the
        same way, and the name has to be in ``__all__`` to be selectable.

        What must NOT come with it is the transport side: ``barlink_bar1`` and
        its two extension modules. Those are the ones that read env vars,
        open ``/dev/dmabuf_holder`` and JIT-build. bar1ep reaches them only
        from inside ``bar1ep_available()`` / the dispatcher constructor,
        i.e. only once someone has chosen ``--moe-a2a-backend bar1ep``.
        """
        code = _probe_code(
            "import sglang.srt.layers.moe.utils\n"
            "import sglang.srt.layers.moe.token_dispatcher as td\n",
            "barlink_bar1",
        )
        self.assertEqual(self._imports(code, _clean_env()), set())

    def test_the_modules_are_importable_at_all(self):
        """Otherwise the two tests above would pass for the wrong reason."""
        code = _probe_code(
            "import importlib\n"
            "for _t in ('sglang.srt.distributed.device_communicators.barlink_bar1',"
            "'sglang.srt.layers.moe.token_dispatcher.bar1ep'):\n"
            "    importlib.import_module(_t)\n",
            "bar1",
        )
        got = self._imports(code, _clean_env())
        self.assertTrue(any("barlink_bar1" in m for m in got))
        self.assertTrue(any("bar1ep" in m for m in got))


class TestImportParserIsAnchored(CustomTestCase):
    """#736: the guard above must fail for the right reason, and only that one.

    Three angles. The first two are can-fail proofs of the fix; the third
    proves the fix did not simply blind the guard, which is the obvious way to
    make a red test green.
    """

    #: The exact line this rig emits, which is what made the guard falsely red.
    RIG_WARNING = (
        "[2026-08-17 18:52:39] WARNING [cuda.py:962] Detected different "
        "devices in the system: NVIDIA GeForce RTX 3080, NVIDIA GeForce RTX "
        "5090, NVIDIA GeForce RTX 3080. Please make sure to set "
        "`CUDA_DEVICE_ORDER=PCI_BUS_ID` to avoid unexpected behavior."
    )

    def test_the_rig_warning_cannot_enter_the_import_set(self):
        """Planted verbatim: prose on stdout contributes nothing."""
        self.assertEqual(_MODULE_LINE.findall(self.RIG_WARNING), [])
        self.assertEqual(_MODULE_LINE.findall(self.RIG_WARNING + "\n"), [])

    def test_the_old_parser_would_have_failed_on_it(self):
        """CAN-FAIL PROOF for the fix itself.

        Reconstructs the previous implementation and shows it turning the
        warning into 30-odd phantom "modules". Without this, a future cleanup
        could revert to ``.split()`` and nothing would object on a rig that
        happens to be quiet.
        """
        old = set(self.RIG_WARNING.split())
        self.assertGreater(len(old), 20)
        self.assertIn("WARNING", old)
        self.assertIn("Detected", old)

    def test_noise_around_a_real_module_line_is_stripped(self):
        stdout = (
            f"{self.RIG_WARNING}\n"
            "MODULE:sglang.srt.distributed.device_communicators.barlink_bar1\n"
            "some trailing banner text\n"
        )
        self.assertEqual(
            _MODULE_LINE.findall(stdout),
            ["sglang.srt.distributed.device_communicators.barlink_bar1"],
        )

    def test_a_module_name_embedded_in_prose_is_not_counted(self):
        """Only a line that IS a report counts, not one that mentions one."""
        for line in (
            "note: MODULE:barlink_bar1 was mentioned",
            "  MODULE:barlink_bar1",
            "XMODULE:barlink_bar1",
        ):
            self.assertEqual(_MODULE_LINE.findall(line), [], msg=line)

    def test_the_guard_still_reds_on_a_planted_transport_import(self):
        """The anchor must not have made the guard unable to fail.

        A subprocess that genuinely imports the transport is asserted to come
        back NON-empty through the same parser the guard uses -- so the
        ``== set()`` assertions above are load-bearing, not vacuous.
        """
        code = _probe_code(
            "import sglang.srt.distributed.device_communicators.barlink_bar1\n",
            "barlink_bar1",
        )
        got = TestNothingIsImported._imports(self, code, _clean_env())
        self.assertTrue(
            any("barlink_bar1" in m for m in got),
            "parser must still see a real transport import",
        )


class TestMoeBackendDefaultUnchanged(CustomTestCase):
    def test_default_is_none(self):
        self.assertTrue(MoeA2ABackend("none").is_none())
        self.assertFalse(MoeA2ABackend("none").is_bar1ep())
        self.assertFalse(MoeA2ABackend("none").is_deepep())

    def test_bar1ep_did_not_move_deepep(self):
        """``is_deepep`` widened on purpose; ``is_deepep_native`` did not.

        The widening is what routes a bar1ep layer through a dispatcher at
        all. The narrow form is what decides which library gets built, and
        for plain deepep both must still say yes.
        """
        deepep = MoeA2ABackend("deepep")
        self.assertTrue(deepep.is_deepep())
        self.assertTrue(deepep.is_deepep_native())
        self.assertFalse(deepep.is_bar1ep())

        bar1ep = MoeA2ABackend("bar1ep")
        self.assertTrue(bar1ep.is_deepep())
        self.assertFalse(bar1ep.is_deepep_native())
        self.assertTrue(bar1ep.is_bar1ep())

    def test_no_other_backend_answers_bar1ep(self):
        for name in ("none", "deepep", "mooncake", "nixl", "mori"):
            self.assertFalse(MoeA2ABackend(name).is_bar1ep(), msg=name)


class TestSelectWithoutATransport(CustomTestCase):
    def test_every_op_falls_to_the_gloo_plane(self):
        from sglang.srt.distributed.device_communicators.barlink import (
            BarlinkCommunicator,
        )

        c = BarlinkCommunicator.__new__(BarlinkCommunicator)
        c.transport = None
        c._path_dispatcher = None
        for op in (
            "all_reduce",
            "all_gather",
            "reduce_scatter",
            "broadcast",
            "all_to_all",
        ):
            for nbytes in (0, 4096, 10600448):
                self.assertIsNone(BarlinkCommunicator._select(c, op, nbytes))


class TestFlagNamesAndDefaults(CustomTestCase):
    def test_every_new_bar1_knob_is_under_the_barlink_prefix(self):
        """ "Unset every SGLANG_BARLINK*" has to be the complete off switch.

        A knob named anything else would be a second, undocumented way to
        change behaviour -- and the operating recipe only tells people about
        the one prefix.
        """
        offenders = []
        for path in sorted(_COMM.glob("barlink_bar1*.py")) + [
            _REPO / "python/sglang/srt/layers/moe/token_dispatcher/bar1ep.py"
        ]:
            for name in re.findall(
                r"""os\.environ\.get\(\s*["']([A-Z0-9_]+)["']""",
                path.read_text(encoding="utf-8"),
            ):
                if not name.startswith("SGLANG_BARLINK"):
                    offenders.append(f"{path.name}: {name}")
        # TORCH_* / CUDA_* are read by torch's own JIT machinery, not by us;
        # anything else is ours and belongs under the prefix.
        offenders = [
            o
            for o in offenders
            if not o.split(": ")[1].startswith(("TORCH_", "CUDA_", "MAX_JOBS"))
        ]
        self.assertFalse(offenders, msg="\n".join(offenders))

    def test_all_gather_defaults_on_inside_bar1_and_is_switchable(self):
        """Inside the (opt-in) bar1 transport all_gather is on by default.

        Deliberate: with it off the standard run aborts in graph capture,
        which is the bug this was built for. The switch exists so the
        measurement can be taken both ways without a rebuild -- it is a
        knob inside an opt-in path, not a second opt-in.
        """
        from sglang.srt.distributed.device_communicators.barlink_bar1 import (
            BarlinkBar1Transport,
        )

        self.assertIn("all_gather", BarlinkBar1Transport.BARLINK_OPS)
        src = (_COMM / "barlink_bar1.py").read_text(encoding="utf-8")
        self.assertIn('os.environ.get("SGLANG_BARLINK_BAR1_AG", "1")', src)


if __name__ == "__main__":
    unittest.main()
