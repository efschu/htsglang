"""Weg 2 S5 / F1: the canonical KV page is a STORE FORMAT choice, not a flip knob.

The blocker this pins (spec WEG2_DESIGN_SPEC_2026-09-06 3.2, re-read in the
tree at ``server_args.py``): ``--phase-flip-canonical-kv-page`` refused at parse
time unless ``--enable-phase-flip``, and ``--enable-phase-flip`` itself demands
``pp_size > 1`` AND ``tp_size == 1``. Weg 2's decode group is
``--tp-size 3 --pp-size 1`` and can therefore never set it -- so the two
groups' KV keys re-append their own geometry
(``hicache_storage.py:_derive_key_suffixes``) and the store, which is Weg 2's
SOLE carrier across a flip, misses 100 % of the time without raising.

F1 renames the flag to ``--hicache-canonical-kv-page`` and moves its
precondition to what the FORMAT actually needs -- ``--page-size 1`` and the
'file' backend -- because the stored bytes depend on the model geometry alone.
The old name hard-removes with a loud parse-time refusal (spec 11.5: no silent
shims), so a boot script carrying it fails rather than silently serving with
the format off.

W10 (``Weg2CanonicalPageMissing``) is the refusal that arms when a group asks
for the format without the page size it is defined on.
"""

import argparse
import unittest

from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


def _group_args(**over):
    """A Weg-2 group launch: no flip flags at all, store format on.

    ``model_path='dummy'`` short-circuits ``__post_init__`` (the pattern the
    sibling #706 args test uses), so the handler under test is called
    explicitly and a refusal can only come from the clause being pinned.
    """
    defaults = dict(
        model_path="dummy",
        hicache_canonical_kv_page=True,
        page_size=1,
        hicache_storage_backend="file",
    )
    defaults.update(over)
    return ServerArgs(**defaults)


class TestTheFormatIsUngatedFromTheFlip(CustomTestCase):
    """RED against the base: neither group shape can carry the format there."""

    def test_the_prefill_group_shape_validates(self):
        """Group P: pp_size=3, tp_size=1, NO --enable-phase-flip."""
        args = _group_args(pp_size=3, tp_size=1)
        args._handle_hicache_canonical_kv_page()
        self.assertTrue(args.hicache_canonical_kv_page)

    def test_the_decode_group_shape_validates(self):
        """Group D: tp_size=3, pp_size=1 -- the shape --enable-phase-flip
        refuses outright (``--enable-phase-flip V1 boots pure PP (tp_size 1)``),
        which is why the format had to be un-gated rather than reused."""
        args = _group_args(tp_size=3, pp_size=1)
        args._handle_hicache_canonical_kv_page()
        self.assertTrue(args.hicache_canonical_kv_page)

    def test_the_flag_parses_without_any_flip_flag(self):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        parsed = parser.parse_args(["--model-path", "m", "--hicache-canonical-kv-page"])
        self.assertTrue(parsed.hicache_canonical_kv_page)

    def test_default_is_off(self):
        """Off must stay indistinguishable from before the flag existed."""
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        parsed = parser.parse_args(["--model-path", "m"])
        self.assertFalse(parsed.hicache_canonical_kv_page)


class TestW10CanonicalPageMissing(CustomTestCase):
    def test_page_size_must_be_one(self):
        """A canonical page is ONE token's attention layers."""
        args = _group_args(page_size=4)
        with self.assertRaises(ValueError) as cm:
            args._handle_hicache_canonical_kv_page()
        self.assertIn("--page-size 1", str(cm.exception))

    def test_backend_must_assemble_partial_writes(self):
        args = _group_args(hicache_storage_backend="mooncake")
        with self.assertRaises(ValueError) as cm:
            args._handle_hicache_canonical_kv_page()
        self.assertIn("--hicache-storage-backend file", str(cm.exception))

    def test_off_validates_whatever_the_page_size(self):
        """CAN-FAIL GUARD: the gate must not fire when the format is off, or
        every stock boot with page_size > 1 would refuse."""
        args = _group_args(hicache_canonical_kv_page=False, page_size=64)
        args._handle_hicache_canonical_kv_page()
        self.assertFalse(args.hicache_canonical_kv_page)


class TestTheOldNameHardRemoves(CustomTestCase):
    """Spec 11.5: no silent shims. A boot script carrying the old flag must
    fail loudly, not start with the format quietly off."""

    def test_the_old_cli_flag_is_refused_by_name(self):
        """MERGE BATCH 1: the refusal moved, the invariant did not.

        S5 registered the old spelling on the parser with a
        ``RemovedFlagAction``.  S0's un-weave removed ELEVEN flip flags and
        refuses all of them from one list, on argv, BEFORE ``parse_args`` --
        and its own test requires that argparse REJECT every removed flag, so
        a registration for one of them is a direct contradiction as well as a
        second bookkeeping of the same list.  S0's gate is the surviving
        authority; it is strictly broader and it keeps the property S5 was
        protecting, because it runs AFTER the ``--config`` merge, so a flag
        arriving from a config file is refused on the same terms as one typed
        on the command line.  What this test pins is unchanged: the old
        spelling fails loudly, and the message names BOTH the old flag and its
        replacement, so no launch line silently boots with the format off.
        """
        from sglang.srt.removed_cli_flags import refuse_removed_flip_flags

        with self.assertRaises(ValueError) as cm:
            refuse_removed_flip_flags(
                ["--model-path", "m", "--phase-flip-canonical-kv-page"]
            )
        msg = str(cm.exception)
        self.assertIn("--hicache-canonical-kv-page", msg)
        self.assertIn("--phase-flip-canonical-kv-page", msg)

    def test_the_old_cli_flag_is_refused_from_a_config_file_too(self):
        """The property S5's parser-side action was bought for.

        A shell gets argparse's "unrecognized arguments" for free; a config
        file or a programmatic caller does not.  The gate runs on the merged
        argv, so the joined ``--flag=value`` form a config merger emits is
        refused as well -- can-fail proof: matching only the bare token would
        walk past this.
        """
        from sglang.srt.removed_cli_flags import refuse_removed_flip_flags

        with self.assertRaises(ValueError) as cm:
            refuse_removed_flip_flags(
                ["--model-path", "m", "--phase-flip-canonical-kv-page=true"]
            )
        self.assertIn("--hicache-canonical-kv-page", str(cm.exception))

    def test_the_old_field_name_is_gone(self):
        self.assertFalse(
            hasattr(ServerArgs(model_path="dummy"), "phase_flip_canonical_kv_page"),
            "a surviving alias field is exactly the silent shim 11.5 refuses",
        )


if __name__ == "__main__":
    unittest.main()
