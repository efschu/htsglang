"""#673/#237: the triton override must be applied BEFORE anyone can read it.

``utils/common.py`` used to ``import triton`` at module scope purely to
``setattr(triton, "next_power_of_2", ...)``. That import is why the tokenizer
manager and the detokenizer -- processes that must never touch CUDA -- loaded a
GPU kernel compiler. But the patch cannot simply be deleted: 235 call sites read
``triton.next_power_of_2``, and it overrides a loop with a bit-length shift.

Nor can it be "moved next to the first consumer": with 235 readers there is no
first consumer, and a patch that races its readers is a silent numerics/perf
landmine rather than a crash. So it is applied BY THE IMPORT: a meta-path
finder wraps triton's loader and sets the attribute after the module body runs
and before ``import triton`` returns.

THE ORDERING PIN IS THE POINT OF THIS FILE. ``test_a_reader_importing_triton_
cannot_observe_it_unpatched`` runs a consumer that reads the attribute AT
IMPORT TIME, the way a kernel module does, and fails if it ever sees triton's
own implementation. Its can-fail is real: disarm the hook and it fails, which
is the whole guarantee ("a reader must import triton to read the attribute, and
the patch is installed inside that import").

Every check runs in a FRESH SUBPROCESS. In this process triton is long since
imported and patched by the test runner, so an in-process assertion would pass
no matter what the hook does.
"""

import os
import subprocess
import sys
import textwrap
import unittest

from sglang.test.test_utils import CustomTestCase

PATCH_MODULE = "sglang.srt.utils.triton_patch"


def _hook_path() -> str:
    """File path of the GENERIC post-import hook.

    The isolated probes load THIS by path rather than ``triton_patch``: since
    the triton patch now delegates to the shared hook, importing it by name
    executes the sglang package root (transformers -> torch._dynamo -> triton)
    and the probe would no longer be isolated. The ordering guarantee lives in
    the generic hook, so that is where it is proven -- stdlib-only, therefore
    loadable standalone.
    """
    import sglang.srt.utils.post_import_hook as hook

    return hook.__file__


def _probe(script: str) -> str:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
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


class TestTritonPatchOrdering(CustomTestCase):
    def test_arming_the_hook_does_not_import_triton(self):
        """The entire reason the patch moved: arming must be free.

        The hook module is loaded BY FILE PATH, not by import: importing
        ``sglang.srt.utils.triton_patch`` normally would first execute the
        package root, which pulls transformers and therefore triton -- and the
        probe would then measure the root rather than the hook. The module
        imports nothing but stdlib, so loading it standalone is faithful.
        """
        answer = _probe(f"""
            import importlib.util, sys, pathlib
            path = pathlib.Path({str(_hook_path())!r})
            spec = importlib.util.spec_from_file_location('_hook_probe', path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            mod.install('triton', lambda m: setattr(m, 'next_power_of_2', len))
            print('triton' in sys.modules)
            """)
        self.assertEqual(
            answer,
            "False",
            "arming the patch hook imported triton -- the hook exists exactly "
            "so that installing the override costs nothing",
        )

    def test_a_reader_importing_triton_cannot_observe_it_unpatched(self):
        """THE ordering guarantee, exercised the way a kernel module does it:
        import triton and read the attribute immediately.

        The hook is loaded BY FILE PATH so triton is genuinely not yet
        imported. A normal ``from sglang... import install`` executes the
        package root, which pulls transformers and therefore triton, and
        install() would then patch it RETROACTIVELY -- so this test would pass
        without ever exercising the loader wrapper it exists to prove. Found by
        mutation: the pass-through mutant survived this test until it was
        isolated, and the assert below is what keeps it honest.
        """
        answer = _probe(f"""
            import importlib.util, sys, pathlib
            spec = importlib.util.spec_from_file_location(
                '_hook_probe', pathlib.Path({str(_hook_path())!r}))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            def _patch(triton_module):
                triton_module.next_power_of_2 = lambda n: 1 << (n - 1).bit_length()
                triton_module._patched_by_hook = True
            mod.install('triton', _patch)
            assert 'triton' not in sys.modules, 'probe is not isolated'
            import triton
            print(getattr(triton, '_patched_by_hook', False) and triton.next_power_of_2(100) == 128)
            """)
        self.assertEqual(
            answer,
            "True",
            "a reader that imported triton saw an unpatched next_power_of_2; "
            "the override is installed by the import itself precisely so this "
            "cannot happen",
        )

    def test_without_the_hook_the_override_is_absent(self):
        """The can-fail for the test above: unarmed, triton keeps its own."""
        answer = _probe(f"""
            import triton
            print(triton.next_power_of_2.__module__ != {PATCH_MODULE!r})
            """)
        self.assertEqual(
            answer,
            "True",
            "triton already carried the override without the hook being armed "
            "-- then the ordering test above proves nothing",
        )

    def test_triton_imported_first_is_patched_retroactively(self):
        """A module already in sys.modules cannot be patched by a future
        import, so install() patches it directly. Otherwise arming late would
        silently leave the override off."""
        answer = _probe(f"""
            import triton
            from {PATCH_MODULE} import install
            install()
            print(triton.next_power_of_2.__module__ == {PATCH_MODULE!r})
            """)
        self.assertEqual(answer, "True")

    def test_the_real_boot_path_ends_up_patched(self):
        """Integration: whatever order the package root imports things in,
        a process that imported sglang has patched triton."""
        answer = _probe(f"""
            import sys
            import sglang
            mod = sys.modules.get('triton')
            print(mod is None or mod.next_power_of_2.__module__ == {PATCH_MODULE!r})
            """)
        self.assertEqual(answer, "True")

    def test_install_is_idempotent(self):
        answer = _probe(f"""
            import sys
            from {PATCH_MODULE} import install, is_armed
            install(); install(); install()
            finders = [f for f in sys.meta_path
                       if type(f).__name__ == '_PostImportFinder'
                       and getattr(f, 'name', None) == 'triton']
            print(len(finders) == 1 and is_armed())
            """)
        self.assertEqual(answer, "True")


class TestImportWeightLadder(CustomTestCase):
    """The attribution ladder, tightened one rung (#673 follow-up).

    The previous rung asserted that BOTH torch and triton were imported at
    module scope by ``utils/common.py``. Triton no longer is, so that rung is
    replaced rather than deleted: the file:line attribution stays honest, and
    the remaining owner is named.
    """

    def test_common_no_longer_imports_triton_at_module_scope(self):
        import pathlib

        import sglang.srt.utils.common as common

        text = pathlib.Path(common.__file__).read_text()
        self.assertNotIn(
            "\nimport triton\n",
            text,
            "utils/common.py imports triton at module scope again; the "
            "override is supposed to arrive through the import hook so that "
            "text-only processes never load a kernel compiler",
        )
        self.assertIn("_triton_patch.install()", text)

    def test_torch_is_still_the_module_scope_owner(self):
        """torch remains structural to this module -- named, not fixed."""
        import pathlib

        import sglang.srt.utils.common as common

        text = pathlib.Path(common.__file__).read_text()
        self.assertIn("\nimport torch\n", text)

    def test_torchcodec_is_no_longer_pulled_by_common(self):
        """The measured win: the video-decoder capability probe moved off
        module scope, so torchcodec (and its torch._dynamo -> triton chain) is
        no longer loaded by every process that imports utils/common."""
        answer = _probe("""
            import sys
            import sglang.srt.utils.common
            print('torchcodec' in sys.modules)
            """)
        self.assertEqual(answer, "False")

    def test_the_video_backend_still_resolves(self):
        """Deferring a probe must not change its answer."""
        from sglang.srt.utils.video_decoder import backend

        self.assertIn(backend(), ("torchcodec", "decord"))

    def test_the_legacy_backend_attribute_still_works(self):
        """Existing readers of ``video_decoder._BACKEND`` keep working through
        PEP 562, so deferring the probe is not an API break."""
        answer = _probe("""
            from sglang.srt.utils.video_decoder import _BACKEND
            print(_BACKEND in ('torchcodec', 'decord'))
            """)
        self.assertEqual(answer, "True")

    def test_triton_now_arrives_through_transformers_not_through_us(self):
        """Attribution for what REMAINS, asserting today's unfixed state.

        triton is still in a bare ``import sglang`` -- but now via
        ``transformers`` (masking_utils -> torch._dynamo -> triton), reached
        from the package root's HF patch. Plain ``import torch`` does NOT pull
        it. When the root is made lazy this test fails, which is the signal
        that a triton-free text process has become possible.
        """
        torch_only = _probe("""
            import sys, torch
            print('triton' in sys.modules)
            """)
        self.assertEqual(torch_only, "False", "torch alone now pulls triton")


if __name__ == "__main__":
    unittest.main()
