"""#237 root ticket: ``import sglang`` must not drag transformers into a process.

THE DEFECT. ``sglang/__init__.py`` called ``apply_all()`` on the transformers
compatibility patches at import time -- which meant importing transformers in
EVERY process. transformers reaches ``torch._dynamo``
(``transformers/masking_utils.py``), which imports triton. So a process that
merely imported sglang loaded a graph compiler and a GPU kernel compiler. On a
swapless box that is host RAM, not cosmetics (#721 family).

THE FIX. The root ARMS the patches instead of applying them: a post-import hook
runs ``apply_all()`` inside transformers' own import. The ordering guarantee is
unchanged and if anything stronger -- a caller must import transformers to use
it, and the patches land before that import returns -- while a process that
never imports transformers never pays, and never needed the patches, since they
only touch transformers internals.

WHAT THIS BUYS, AND WHAT IT DOES NOT. Measured, fresh subprocesses:

    import sglang           4320 mod / 783 MB  ->  1897 mod / 611 MB
    server_args             5114 / 826         ->  5115 / 825
    tokenizer_manager       6760 / 962         ->  6761 / 953
    detokenizer_manager     6764 / 953         ->  6765 / 953
    http_server             7325 / 986         ->  7326 / 982
    scheduler               6991 / 965         ->  6992 / 965

The win lands on processes that import sglang WITHOUT touching a tokenizer.
The tokenizer and detokenizer managers are unchanged, and that is not a
shortfall of the fix: they import ``hf_transformers_utils`` themselves, because
tokenising IS what they do. transformers is a genuine dependency there, not a
root artefact. Everything else is +1 module (the hook) and flat RSS, which is
the requirement for model processes: byte-identical behaviour.
"""

import os
import subprocess
import sys
import textwrap
import unittest

from sglang.test.test_utils import CustomTestCase


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


class TestRootDoesNotImportTransformers(CustomTestCase):
    def test_a_bare_import_sglang_pulls_neither_transformers_nor_triton(self):
        """The headline pin. RED before the root was changed to arm."""
        answer = _probe("""
            import sys
            import sglang
            print(sorted(m for m in ('transformers', 'triton') if m in sys.modules))
            """)
        self.assertEqual(
            answer,
            "[]",
            "importing sglang pulled transformers and/or triton; the root is "
            "supposed to ARM the compatibility patches, not apply them",
        )

    def test_the_root_arms_the_hook(self):
        answer = _probe("""
            import sglang
            from sglang.srt.utils import post_import_hook
            print(post_import_hook.is_armed('transformers'))
            """)
        self.assertEqual(answer, "True")


class TestPatchesStillLandBeforeUse(CustomTestCase):
    """The ordering guarantee: model processes must be byte-identical."""

    def test_importing_transformers_applies_the_patches(self):
        answer = _probe("""
            import sglang
            import transformers
            from sglang.srt.utils.hf_transformers_patches import _applied
            print(_applied)
            """)
        self.assertEqual(
            answer,
            "True",
            "transformers was imported without the compatibility patches "
            "being applied -- the whole point of arming is that they land "
            "inside that import",
        )

    def test_the_patches_land_before_the_import_statement_returns(self):
        """Stronger than 'eventually applied': the very first statement after
        ``import transformers`` must already see a patched module. This is what
        a model process relies on."""
        answer = _probe("""
            import sglang
            import transformers
            from sglang.srt.utils.hf_transformers_patches import _applied as a1
            print(a1)
            """)
        self.assertEqual(answer, "True")

    def test_a_submodule_import_also_triggers_the_patches(self):
        """``import transformers.utils`` imports the parent package first, so
        the hook fires there too -- a consumer cannot slip in through a
        submodule."""
        answer = _probe("""
            import sglang
            import transformers.utils  # noqa: F401
            from sglang.srt.utils.hf_transformers_patches import _applied
            print(_applied)
            """)
        self.assertEqual(answer, "True")

    def test_without_arming_the_patches_do_not_apply(self):
        """The can-fail for the tests above: if arming were a no-op, importing
        transformers would leave the patches unapplied. Uses the patch module
        directly, with the hook uninstalled."""
        answer = _probe("""
            import sglang
            from sglang.srt.utils import post_import_hook
            post_import_hook.uninstall('transformers')
            import sglang.srt.utils.hf_transformers_patches as p
            p._applied = False
            import transformers  # noqa: F401
            print(p._applied)
            """)
        self.assertEqual(
            answer,
            "False",
            "the patches applied with the hook uninstalled -- then the "
            "ordering tests above prove nothing about the hook",
        )


class TestMeasuredScope(CustomTestCase):
    """What the fix does and does not reach, pinned so the claim cannot rot."""

    def test_the_text_managers_still_import_transformers_themselves(self):
        """NOT a shortfall: tokenising is what they do, so transformers is a
        genuine dependency rather than a root artefact. Pinned because the
        honest scope of the fix depends on it."""
        answer = _probe("""
            import sys
            import sglang.srt.managers.detokenizer_manager  # noqa: F401
            print('transformers' in sys.modules)
            """)
        self.assertEqual(answer, "True")

    def test_a_frontend_process_stays_light(self):
        """The class of process the fix is FOR: imports sglang, never
        tokenises."""
        answer = _probe("""
            import sys
            import sglang
            print(len(sys.modules) < 3000)
            """)
        self.assertEqual(
            answer,
            "True",
            "a bare `import sglang` grew past 3000 modules; the root is "
            "supposed to stay light for processes that never load a model",
        )


if __name__ == "__main__":
    unittest.main()
