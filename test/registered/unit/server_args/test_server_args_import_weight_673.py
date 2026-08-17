"""#673 follow-up: what does importing ServerArgs drag in?

THE DEFECT. ``server_args.py`` imported one integer --
``from ...fla.chunk_delta_h import CHUNK_SIZE`` -- and thereby pulled torch,
triton and the whole FLA kernel chain into EVERY process that touches
ServerArgs: the launcher, the tokenizer manager, the detokenizer. The cost was
not only import time. ``fla/utils.py:283`` runs ``get_available_device()`` at
MODULE SCOPE, which reaches ``torch.cuda.is_available()`` and the triton
driver, so processes that must never hold a CUDA context were touching the
driver while parsing arguments -- the #237/#403 second-context family.

WHAT THESE TESTS PIN, and the honest split between them:

* ``test_importing_server_args_does_not_import_fla`` is the fix. It runs in a
  FRESH SUBPROCESS, because ``sys.modules`` in this one is already polluted by
  the test runner -- an in-process assertion would pass or fail for reasons
  that have nothing to do with server_args. It was RED before the constant was
  inlined.
* ``test_chunk_size_matches_the_authoritative_value`` is the containment pin.
  Inlining a constant trades an import for a DUPLICATE, and a duplicate that
  drifts silently is worse than the import was. This imports FLA properly --
  in a test, where the cost is irrelevant -- and fails if the two diverge.
* ``test_torch_and_triton_are_not_server_args_fault`` is an ATTRIBUTION pin,
  and it asserts the CURRENT, UNFIXED state on purpose. torch and triton are
  already in ``sys.modules`` after a bare ``import sglang``, long before
  server_args is reached: ``sglang/__init__.py:29`` ->
  ``sglang/srt/utils/__init__.py:2`` -> ``utils/common.py:87`` (torch) and
  ``:89`` (triton). So "server_args must not import torch" cannot be asserted
  today by any change to server_args, and pretending otherwise would file a
  green test against a red world. When someone makes the package root lazy,
  THIS test fails -- and that failure is the signal to tighten the first test
  to cover torch and triton too.
"""

import os
import subprocess
import sys
import unittest

from sglang.test.test_utils import CustomTestCase

FLA_PACKAGE = "sglang.srt.layers.attention.fla"


def _probe(script: str) -> str:
    """Run ``script`` in a fresh interpreter and return its stdout, stripped.

    A fresh process is the whole point: this suite asks what an import DOES,
    and the answer is only meaningful before anything else has imported it.
    """
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"probe failed ({result.returncode}):\n{result.stdout}\n{result.stderr}"
        )
    return result.stdout.strip().splitlines()[-1]


class TestServerArgsImportWeight(CustomTestCase):
    def test_importing_server_args_does_not_import_fla(self):
        answer = _probe(
            "import sys\n"
            "import sglang.srt.server_args\n"
            f"pulled = sorted(m for m in sys.modules if m.startswith({FLA_PACKAGE!r}))\n"
            "print(pulled)\n"
        )
        self.assertEqual(
            answer,
            "[]",
            "importing server_args pulled FLA kernel modules; the constant is "
            "supposed to be inlined precisely so that every process which "
            "parses arguments does not load a kernel chain that probes the "
            "device at import time",
        )

    def test_chunk_size_matches_the_authoritative_value(self):
        """The containment pin: an inlined constant that drifts is worse than
        the import it replaced, so the duplicate is checked against its
        source."""
        from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE
        from sglang.srt.server_args import FLA_CHUNK_SIZE

        self.assertEqual(
            FLA_CHUNK_SIZE,
            CHUNK_SIZE,
            "server_args.FLA_CHUNK_SIZE has drifted from FLA's own CHUNK_SIZE. "
            "Update the inlined value (and this failure is exactly why it is "
            "pinned rather than trusted).",
        )

    def test_torch_and_triton_are_not_server_args_fault(self):
        """Attribution, asserting today's UNFIXED state deliberately.

        Both are already present after a bare ``import sglang``. When the
        package root is made lazy this test fails, which is the signal to
        tighten the FLA test above into a full torch/triton pin.
        """
        answer = _probe(
            "import sys\n"
            "import sglang\n"
            "print(sorted(m for m in ('torch', 'triton') if m in sys.modules))\n"
        )
        self.assertEqual(
            answer,
            "['torch', 'triton']",
            "a bare `import sglang` no longer pulls torch and triton -- the "
            "package root got lighter, so the FLA test above should now be "
            "tightened to assert their absence after importing server_args too",
        )

    def test_the_owner_of_the_torch_pull_is_named(self):
        """Keeps the attribution honest so the next reader does not re-derive
        it. LADDER RUNG UPDATED: this used to assert that utils/common imported
        BOTH torch and triton at module scope. The triton import is gone (the
        override now arrives through the import hook, see
        test_triton_patch_ordering_673.py), so the rung was tightened rather
        than deleted -- torch is what remains, and it is structural."""
        import pathlib

        import sglang.srt.utils.common as common

        text = pathlib.Path(common.__file__).read_text()
        self.assertIn("\nimport torch\n", text)
        self.assertNotIn("\nimport triton\n", text)


if __name__ == "__main__":
    unittest.main()
