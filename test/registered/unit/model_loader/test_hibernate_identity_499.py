"""#499 falsifier: the #89 hibernate manifest identity must be the SAME value
on the park side and on the match side.

The two sides sit on opposite banks of the declaration pipeline:

  * MATCH runs inside ``ServerArgs._handle_load_format``
    (``server_args.py:13332``), which ``__post_init__`` calls at
    ``server_args.py:5939`` -- i.e. BEFORE ``materialize_declarations(self)``
    at ``server_args.py:5984``. Declared-but-not-yet-materialized fields still
    carry their raw value there.
  * PARK runs in the worker (``weight_updater._hibernate_park_weights`` ->
    ``park_weights_to_disk``, writing ``_model_identity(server_args)`` at
    ``hibernate.py:518``) on a fully materialized ``server_args``.

For GGUF -- the only checkpoint class #89 supports -- ``quantization`` is
declared ``"gguf"`` by the ``_gguf_quantization`` post-process pass
(``arg_groups/overrides.py:2091``), and that pass is invoked at the HEAD of
the very handler that then runs the match (``server_args.py:13300``). So the
manifest was written with ``quantization="gguf"`` while every subsequent boot
compared against ``quantization=None``: a park could never match its own
manifest and the fast-restore path was unreachable.

Sibling sweep: exactly two identity fields are declarable at all --
``quantization`` and ``dtype`` (``Arg(resolvable=True)``); every other one is
refused by ``validate_declarations`` (``overrides.py:2176``), so it cannot
take this route. The sweep arm below derives that set from the metadata
rather than hardcoding it, so a field whose ``resolvable`` flag flips later is
covered automatically.

CPU-only, no GPU, no real GGUF file (``check_gguf_file`` is patched, and the
manifest is hand-written into a temp dir; the NVML card-presence gate is a
no-op for a manifest with no ranks).
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from sglang.srt import server_args as server_args_mod
from sglang.srt.arg_groups.arg_utils import resolvable_fields
from sglang.srt.arg_groups.overrides import (
    materialize_declarations,
    run_post_process_pass,
)
from sglang.srt.model_loader import hibernate
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

GGUF_PATH = "/nonexistent/hibernate-499/model.gguf"


def _looks_like_gguf(path) -> bool:
    return str(path).endswith(".gguf")


class _GgufLaunch:
    """A GGUF hibernate launch driven to the exact point of the manifest
    match, without a model, a GPU or a real checkpoint."""

    def __init__(self, hibernate_dir: str, **kwargs):
        # model_path='dummy' short-circuits __post_init__ (server_args.py:5713)
        # so the handler under test can be driven in isolation, in the same
        # pre-materialization state the real __post_init__ has at :5939.
        args = ServerArgs(model_path="dummy", **kwargs)
        args.model_path = GGUF_PATH
        args.load_format = "gguf"
        args.hibernate_dir = hibernate_dir
        args.enable_weights_disk_backup = True
        self.args = args

    def match(self):
        """Run the handler that hosts the manifest match."""
        self.args._handle_load_format()
        return self.args

    def materialize(self):
        """Reach the state the worker (and therefore the park) observes."""
        materialize_declarations(self.args)
        return self.args


class HibernateIdentityTest(unittest.TestCase):
    def setUp(self):
        self._orig = server_args_mod.check_gguf_file
        server_args_mod.check_gguf_file = _looks_like_gguf
        import sglang.srt.utils.hf_transformers_utils as hf_utils

        self._orig_hf = hf_utils.check_gguf_file
        hf_utils.check_gguf_file = _looks_like_gguf
        self._hf_utils = hf_utils
        self.dir = tempfile.mkdtemp(prefix="hib499-")

    def tearDown(self):
        server_args_mod.check_gguf_file = self._orig
        self._hf_utils.check_gguf_file = self._orig_hf

    def _write_manifest(self, identity):
        manifest = {
            "version": hibernate.HIBERNATE_VERSION,
            "identity": identity,
            # No rank entries: the #331 card-presence gate returns True for an
            # empty set, so this test needs no NVML.
            "ranks": {},
        }
        with open(hibernate.manifest_path(self.dir), "w") as f:
            json.dump(manifest, f, default=str)

    def test_identity_is_stable_across_materialization(self):
        """The identity computed at the match point must equal the identity
        computed after materialization -- otherwise a park writes a manifest
        its own next boot cannot recognize."""
        launch = _GgufLaunch(self.dir)
        at_match = hibernate._model_identity(launch.match())
        at_park = hibernate._model_identity(launch.materialize())
        divergent = {
            k: (at_match.get(k), at_park.get(k))
            for k in at_park
            if at_match.get(k) != at_park.get(k)
        }
        self.assertEqual(
            divergent,
            {},
            "identity fields diverge between the match point and the park "
            f"point (field: (match, park)): {divergent}",
        )

    def test_gguf_quantization_is_resolved_on_both_sides(self):
        """Named arm for the #499 mechanism itself: the declared GGUF
        quantization must already be visible at the match point."""
        launch = _GgufLaunch(self.dir)
        at_match = hibernate._model_identity(launch.match())
        self.assertEqual(at_match["quantization"], "gguf")

    def test_park_manifest_matches_the_next_boot(self):
        """End-to-end equivalent of 'park, restart with the same command':
        the manifest a park writes must select the fast restore path."""
        first = _GgufLaunch(self.dir)
        first.match()
        self._write_manifest(hibernate._model_identity(first.materialize()))

        second = _GgufLaunch(self.dir)
        second.match()
        self.assertEqual(
            second.args.load_format,
            "hibernate",
            "a manifest parked by an identical launch was not recognized",
        )

    def test_declaration_of_any_resolvable_identity_field_is_seen(self):
        """Sibling sweep, derived from the Arg metadata: every identity field
        that CAN be declared must read through the declaration overlay at the
        match point, not just ``quantization``."""
        probe_values = {
            "quantization": "gguf",
            "dtype": "float16",
            "kv_cache_dtype": "fp8_e4m3",  # not in the identity; sanity anchor
        }
        declarable = resolvable_fields(ServerArgs)
        identity_fields = set(hibernate._model_identity(ServerArgs(model_path="dummy")))
        covered = sorted(declarable & identity_fields)
        self.assertTrue(
            covered, "expected at least one declarable identity field to exist"
        )
        for field in covered:
            with self.subTest(field=field):
                value = probe_values.get(field)
                self.assertIsNotNone(
                    value,
                    f"identity field {field!r} became declarable; add a probe "
                    "value for it here",
                )
                launch = _GgufLaunch(self.dir)
                launch.match()
                run_post_process_pass(
                    launch.args,
                    lambda _view, f=field, v=value: {f: v},
                )
                at_match = hibernate._model_identity(launch.args)
                at_park = hibernate._model_identity(launch.materialize())
                self.assertEqual(
                    at_match[field],
                    at_park[field],
                    f"identity field {field!r} is declared but read raw at "
                    "the match point",
                )

    def test_park_side_read_is_never_stale(self):
        """Behaviour pin (NOT a falsifier -- it is green against the pre-fix
        file too): the one risk the overlay read introduces is a stale read on
        the PARK side, where the fields are already materialized. Both
        post-resolution mutation shapes are pinned -- a resolvable field
        (``ServerArgs.override`` appends to the stash AND writes the field, so
        the two stay in step) and a non-resolvable one (field only, logged in
        ``_runtime_mutations``; that is how a runtime /update_weights moves
        ``model_path``, ``model_executor/model_runner.py:2559``). A future
        change that lets the overlay drift from the fields turns this red."""
        launch = _GgufLaunch(self.dir)
        launch.match()
        args = launch.materialize()

        args.override("test.runtime", quantization="awq")
        self.assertEqual(hibernate._model_identity(args)["quantization"], "awq")

        args.override("test.update_weights", model_path="/other/model.gguf")
        self.assertEqual(
            hibernate._model_identity(args)["model_path"], "/other/model.gguf"
        )

    def test_no_manifest_still_falls_back_to_a_cold_load(self):
        """The default (first) boot is unchanged: no manifest -> cold load,
        and the handler leaves load_format at 'gguf'."""
        launch = _GgufLaunch(self.dir)
        launch.match()
        self.assertEqual(launch.args.load_format, "gguf")

    def test_a_genuinely_different_launch_still_mismatches(self):
        """Can-discriminate check for the gate: the fix must not turn the
        identity into a constant that matches anything."""
        first = _GgufLaunch(self.dir, tp_size=2)
        first.match()
        self._write_manifest(hibernate._model_identity(first.materialize()))

        other = _GgufLaunch(self.dir, tp_size=3)
        other.match()
        self.assertEqual(other.args.load_format, "gguf")


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "99")
    unittest.main()
