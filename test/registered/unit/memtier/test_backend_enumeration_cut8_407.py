# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""#407 cut 8: one enumeration of the HiCache storage backends, not three.

WHAT V4 IS. The #407 survey catalogues "HiCache backend names -- a closed
`choices=[...]` list of nine" as vocabulary V4, a second enumeration of
something `StorageBackendFactory._registry` already knows. Cut 8's judgement in
the design is MECHANICAL for backend selection.

THE DRIFT IS NOT HYPOTHETICAL; IT HAD ALREADY HAPPENED. Before this cut the
argument carried THREE lists that disagreed:

* `choices=` -- nine entries;
* the help text -- "Built-in backends: file, mooncake, hf3fs, nixl, aibrix",
  five, missing `eic`, `simm` and `mori`;
* `StorageBackendFactory._registry` -- eight, the authority, since it is what
  `create_backend` actually looks in.

An operator reading `--help` learned five of the eight backends their build
supports. That is the cost of a second enumeration, paid before anyone noticed.

WHY THE LIST IS NOT DERIVED AT IMPORT TIME, which is what the cut table's
wording suggests. `server_args.py` has **zero** module-level imports from
`mem_cache` -- every one is lazy, inside a function -- so argument parsing does
not pull the storage stack or torch. Computing `choices=` at class-definition
time would evaluate at annotation time and break that property for a cosmetic
gain. So the authority moves to the registry where it costs nothing (validation,
which already imports lazily), the literal list stays as an argparse
convenience, and this suite makes the two unable to disagree.
"""

from __future__ import annotations

import unittest

from sglang.srt.mem_cache.storage.backend_factory import StorageBackendFactory
from sglang.srt.server_args import ServerArgs, registered_storage_backends
from sglang.test.test_utils import CustomTestCase


def _arg_metadata(name: str):
    """The ``Arg(...)`` attached to a ServerArgs field.

    Resolved through ``get_type_hints(..., include_extras=True)`` rather than
    ``dataclasses.fields(...).type``: the module uses postponed annotations, so
    ``field.type`` is the STRING "A[Optional[str], Arg(...)]" and has no
    ``__metadata__`` at all. A first version of this suite read it that way and
    failed against correct code, reporting "no choices= to check" for an
    argument that plainly has them.
    """
    import typing

    hints = typing.get_type_hints(ServerArgs, include_extras=True)
    for piece in getattr(hints[name], "__metadata__", ()):
        if getattr(piece, "choices", None) or getattr(piece, "help", None):
            return piece
    raise AssertionError(f"{name} carries no Arg metadata")


def _choices():
    choices = getattr(_arg_metadata("hicache_storage_backend"), "choices", None)
    assert choices, "hicache_storage_backend has no choices= to check"
    return list(choices)


class TestTheEnumerationsAgree(CustomTestCase):
    def test_every_registered_backend_is_selectable(self):
        missing = sorted(set(StorageBackendFactory._registry) - set(_choices()))
        self.assertEqual(
            missing,
            [],
            "backends the factory can build but the CLI will not accept",
        )

    def test_every_choice_is_buildable(self):
        # 'dynamic' is a MODE, not a registered backend: create_backend handles
        # it on its own branch, so it is admissible without being in _registry.
        extra = sorted(set(_choices()) - set(StorageBackendFactory._registry) - {"dynamic"})
        self.assertEqual(
            extra,
            [],
            "choices the CLI accepts that the factory cannot build",
        )

    def test_the_helper_reports_the_registry(self):
        self.assertEqual(
            sorted(registered_storage_backends()),
            sorted(StorageBackendFactory._registry),
        )

    def test_dynamic_stays_selectable(self):
        self.assertIn("dynamic", _choices())


class TestTheHelpTextIsComplete(CustomTestCase):
    """The drift that had already happened, pinned so it cannot return."""

    def _help(self):
        return _arg_metadata("hicache_storage_backend").help

    def test_the_help_names_every_builtin(self):
        help_text = self._help()
        missing = [
            name
            for name in StorageBackendFactory._registry
            if name not in help_text
        ]
        self.assertEqual(
            missing, [], "built-in backends absent from --help"
        )


class TestValidationIsRegistryAuthoritative(CustomTestCase):
    def test_an_unregistered_backend_is_refused_by_name(self):
        args = ServerArgs.__new__(ServerArgs)
        args.hicache_storage_backend = "not_a_backend"
        with self.assertRaises(ValueError) as caught:
            ServerArgs._validate_storage_backend_registered(args)
        message = str(caught.exception)
        self.assertIn("not_a_backend", message)
        # The refusal must list what IS available, from the registry.
        self.assertIn("file", message)

    def test_a_registered_backend_passes(self):
        args = ServerArgs.__new__(ServerArgs)
        args.hicache_storage_backend = "file"
        self.assertIsNone(ServerArgs._validate_storage_backend_registered(args))

    def test_dynamic_passes_without_being_registered(self):
        args = ServerArgs.__new__(ServerArgs)
        args.hicache_storage_backend = "dynamic"
        self.assertIsNone(ServerArgs._validate_storage_backend_registered(args))

    def test_none_passes(self):
        args = ServerArgs.__new__(ServerArgs)
        args.hicache_storage_backend = None
        self.assertIsNone(ServerArgs._validate_storage_backend_registered(args))


if __name__ == "__main__":
    unittest.main()
