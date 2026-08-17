# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""#464: the coalescer needs a lever a boot can actually reach.

THE DEFECT THIS PINS. The coalescer shipped with ``coalesce_resume: bool =
False`` on ``KvVmmArena.__init__`` and the comment "the flag exists so the
measurement can be taken, not because the win is assumed". But no construction
site passed it: ``phase_flip_spill.py`` builds both carrier arenas
(``:597``, ``:927``) without the argument, and ``memory_pool.py:2488`` builds
the KV seam owner without it either. So the flag was False in every real boot
and the measurement it exists for could not be taken at all -- a switch with no
actuator, the #679/#681/#684/#715 shape.

The lever is read at the ONE place the flag is stored rather than threaded
through every call site, so a site that does not care needs no edit and cannot
drift out of sync. An explicit argument still wins over the environment: a
caller that has decided is not overridden by ambient state.

DEFAULT OFF, byte-identical. Unset environment must reproduce today's plan
exactly -- that property is what makes this safe to merge ahead of the window.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

from sglang.srt.mem_cache.kv_vmm_backing import resolve_coalesce_resume

_ENV = "SGLANG_VMM_COALESCE_RESUME"


class TestTheLeverDefaultsOff(unittest.TestCase):
    def test_unset_is_off(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(_ENV, None)
            self.assertIs(resolve_coalesce_resume(None), False)

    def test_zero_is_off(self):
        with mock.patch.dict(os.environ, {_ENV: "0"}):
            self.assertIs(resolve_coalesce_resume(None), False)

    def test_empty_is_off(self):
        with mock.patch.dict(os.environ, {_ENV: ""}):
            self.assertIs(resolve_coalesce_resume(None), False)


class TestTheLeverTurnsItOn(unittest.TestCase):
    def test_one_is_on(self):
        with mock.patch.dict(os.environ, {_ENV: "1"}):
            self.assertIs(resolve_coalesce_resume(None), True)

    def test_true_is_on(self):
        with mock.patch.dict(os.environ, {_ENV: "true"}):
            self.assertIs(resolve_coalesce_resume(None), True)


class TestAnExplicitArgumentWins(unittest.TestCase):
    """Ambient state must not overrule a caller that decided."""

    def test_explicit_false_beats_env_on(self):
        with mock.patch.dict(os.environ, {_ENV: "1"}):
            self.assertIs(resolve_coalesce_resume(False), False)

    def test_explicit_true_beats_env_off(self):
        with mock.patch.dict(os.environ, {_ENV: "0"}):
            self.assertIs(resolve_coalesce_resume(True), True)


class TestTheArenaReadsTheLever(unittest.TestCase):
    """Pin the WIRING, not just the helper.

    A resolver nothing calls is the always-absent-marker defect in another
    costume -- the #349 lesson, applied here.
    """

    def test_init_resolves_through_the_helper(self):
        import inspect

        from sglang.srt.mem_cache.kv_vmm_backing import KvVmmArena

        src = inspect.getsource(KvVmmArena.__init__)
        self.assertIn("resolve_coalesce_resume", src)

    def test_the_default_is_none_so_the_env_can_be_seen(self):
        import inspect

        from sglang.srt.mem_cache.kv_vmm_backing import KvVmmArena

        sig = inspect.signature(KvVmmArena.__init__)
        # A hard False default would swallow the environment for every caller
        # that omits the argument -- which is all of them.
        self.assertIsNone(sig.parameters["coalesce_resume"].default)


if __name__ == "__main__":
    unittest.main()
