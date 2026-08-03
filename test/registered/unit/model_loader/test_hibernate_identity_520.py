"""#520 falsifier (#499-B): the #89 hibernate identity must survive a field
that a WORKER re-derives after the match.

#499 aligned the two banks of the fingerprint across
``materialize_declarations`` and named the residual it could not reach: a value
resolved in the worker, via ``declare_load_time_override`` /
``ServerArgs.override``, AFTER ``ServerArgs.__post_init__`` has already run the
match. No match-side read can see it -- the process that computes it does not
exist yet. Two live instances:

  * ``ModelRunner._sm80_dtype_fallback`` (``model_executor/model_runner.py:
    1980``) declares ``dtype="float16"`` on a card without bfloat16. The park
    then writes ``float16`` into the manifest while every subsequent boot
    parses ``auto`` -- so a park on the sm75 hetero host (2080 Ti, the #164
    Turing path) can never match its own manifest.
  * ``spec_worker.match_target_context_length``
    (``speculative/eagle_worker_v2.py:1764`` and the standalone / frozen-KV-MTP
    / multi-layer workers) pins ``context_length`` to the target model's
    ``context_len`` on the SHARED ``server_args`` -- the object the park reads
    (``BaseSpecWorker.model_runner`` returns ``self.target_worker.model_runner``,
    ``base_spec_worker.py:384``). Same divergence on every speculative boot.

Fix under test: ``_model_identity`` reads through ``launch_view``
(``arg_groups/overrides.py``), which un-applies the sources on the
IDENTITY-TRANSPARENT registry -- writes that RE-DERIVE a field from state the
fingerprint already pins. Writes that genuinely change what is loaded (a
runtime ``/update_weights`` moving ``model_path``) are NOT on the registry and
must keep showing through; that direction is pinned here too.

Hermetic, CPU-only, no GPU and no real GGUF file. The ONLY thing faked is the
device-capability tuple in the ``_needs_float16_fallback`` arm (named at the
patch site); the source strings, the declaration payloads, the mutation entry
point and both identity reads are production code.
"""

from __future__ import annotations

import ast
import json
import os
import tempfile
import unittest
from typing import Dict, Set, Tuple

from sglang.srt import server_args as server_args_mod
from sglang.srt.arg_groups.overrides import (
    IDENTITY_TRANSPARENT_SOURCES,
    SM80_DTYPE_FALLBACK_SOURCE,
    SPEC_TARGET_CONTEXT_LENGTH_SOURCE,
    declare_load_time_override,
    materialize_declarations,
    resolved_view,
)
from sglang.srt.model_loader import hibernate
from sglang.srt.runtime_context import get_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

GGUF_PATH = "/nonexistent/hibernate-520/model.gguf"

# The target context_len a speculative worker pins onto the shared server_args.
TARGET_CONTEXT_LEN = 262144


def _looks_like_gguf(path) -> bool:
    return str(path).endswith(".gguf")


class _GgufLaunch:
    """A GGUF hibernate launch driven to the exact point of the manifest
    match, without a model, a GPU or a real checkpoint (same fixture shape as
    the #499 falsifier)."""

    def __init__(self, hibernate_dir: str, **kwargs):
        # model_path='dummy' short-circuits __post_init__ so the handler under
        # test can be driven in isolation, in the same pre-materialization
        # state the real __post_init__ has at server_args.py:5939.
        args = ServerArgs(model_path="dummy", **kwargs)
        args.model_path = GGUF_PATH
        args.load_format = "gguf"
        args.hibernate_dir = hibernate_dir
        args.enable_weights_disk_backup = True
        self.args = args

    def match(self):
        self.args._handle_load_format()
        return self.args

    def materialize(self):
        """Reach the state the worker (and therefore the park) observes."""
        materialize_declarations(self.args)
        return self.args


class HibernateIdentity520Test(unittest.TestCase):
    def setUp(self):
        self._orig = server_args_mod.check_gguf_file
        server_args_mod.check_gguf_file = _looks_like_gguf
        import sglang.srt.utils.hf_transformers_utils as hf_utils

        self._orig_hf = hf_utils.check_gguf_file
        hf_utils.check_gguf_file = _looks_like_gguf
        self._hf_utils = hf_utils
        self.dir = tempfile.mkdtemp(prefix="hib520-")
        self._ctx_previous = get_context()._server_args

    def tearDown(self):
        server_args_mod.check_gguf_file = self._orig
        self._hf_utils.check_gguf_file = self._orig_hf
        get_context().set_server_args(self._ctx_previous)

    # -- helpers ---------------------------------------------------------

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

    def _park_after_sm75_fallback(self, launch: _GgufLaunch):
        """Reach the state a park observes on a card without bfloat16, through
        the production declaration path.

        The card itself is not faked: what is driven here is the declaration
        ``ModelRunner.load_model`` issues once ``_needs_float16_fallback``
        returned True (``model_runner.py:1980-1982``) -- same source constant,
        same payload. The predicate itself is exercised against a simulated
        sm75 capability in its own arm below.
        """
        args = launch.materialize()
        get_context().set_server_args(args)
        declare_load_time_override(SM80_DTYPE_FALLBACK_SOURCE, {"dtype": "float16"})
        return args

    def _park_after_spec_context_length(self, launch: _GgufLaunch):
        """Reach the state a park observes on a speculative boot: the spec
        worker pins the target's context_len onto the SHARED server_args
        (``eagle_worker_v2.py:1764``)."""
        args = launch.materialize()
        args.override(
            SPEC_TARGET_CONTEXT_LENGTH_SOURCE, context_length=TARGET_CONTEXT_LEN
        )
        return args

    # -- falsifiers ------------------------------------------------------

    def test_sm75_dtype_rederivation_does_not_move_the_identity(self):
        """FALSIFIER (#520): the worker's fp16 fallback must not change the
        fingerprint -- the match bank cannot see it."""
        launch = _GgufLaunch(self.dir)
        at_match = hibernate._model_identity(launch.match())
        at_park = hibernate._model_identity(self._park_after_sm75_fallback(launch))
        divergent = {
            k: (at_match.get(k), at_park.get(k))
            for k in at_park
            if at_match.get(k) != at_park.get(k)
        }
        self.assertEqual(
            divergent,
            {},
            "identity fields diverge between the match point and a park on a "
            f"card without bfloat16 (field: (match, park)): {divergent}",
        )

    def test_sm75_park_manifest_is_recognized_by_the_next_boot(self):
        """End-to-end equivalent of 'park on the sm75 host, restart with the
        same command': the manifest must select the fast restore path."""
        first = _GgufLaunch(self.dir)
        first.match()
        self._write_manifest(
            hibernate._model_identity(self._park_after_sm75_fallback(first))
        )

        second = _GgufLaunch(self.dir)
        second.match()
        self.assertEqual(
            second.args.load_format,
            "hibernate",
            "a manifest parked on a card without bfloat16 was not recognized "
            "by an identical relaunch",
        )

    def test_spec_context_length_rederivation_does_not_move_the_identity(self):
        """FALSIFIER (#520 sibling): the speculative workers pin
        ``context_length`` to the target's ``context_len`` on the shared
        server_args, so every spec boot parked a value no launch parses."""
        launch = _GgufLaunch(self.dir)
        at_match = hibernate._model_identity(launch.match())
        at_park = hibernate._model_identity(self._park_after_spec_context_length(launch))
        divergent = {
            k: (at_match.get(k), at_park.get(k))
            for k in at_park
            if at_match.get(k) != at_park.get(k)
        }
        self.assertEqual(
            divergent,
            {},
            "identity fields diverge between the match point and a park on a "
            f"speculative boot (field: (match, park)): {divergent}",
        )

    def test_spec_park_manifest_is_recognized_by_the_next_boot(self):
        first = _GgufLaunch(self.dir)
        first.match()
        self._write_manifest(
            hibernate._model_identity(self._park_after_spec_context_length(first))
        )

        second = _GgufLaunch(self.dir)
        second.match()
        self.assertEqual(
            second.args.load_format,
            "hibernate",
            "a manifest parked by a speculative boot was not recognized by an "
            "identical relaunch",
        )

    # -- the hardware predicate (the only faked object) -------------------

    def test_the_fp16_fallback_predicate_fires_on_a_simulated_sm75_card(self):
        """The geometry this ticket is about, at its source.

        NAMED DEVIATION from the real object: ``get_device_capability`` is
        replaced by a constant tuple, because this box has no pre-Ampere card
        (sm86 / sm120 only). Nothing else is substituted -- the predicate under
        test is production code, and it is asked in BOTH directions so an
        always-True stub would fail the sm86 leg.
        """
        from sglang.srt.model_executor import model_runner as model_runner_mod

        original = model_runner_mod.get_device_capability
        try:
            model_runner_mod.get_device_capability = lambda device_id=None: (7, 5)
            self.assertTrue(
                model_runner_mod._needs_float16_fallback(0),
                "sm75 has no bfloat16; the fp16 fallback must fire",
            )
            model_runner_mod.get_device_capability = lambda device_id=None: (8, 6)
            self.assertFalse(
                model_runner_mod._needs_float16_fallback(0),
                "sm86 has bfloat16; the fp16 fallback must NOT fire",
            )
        finally:
            model_runner_mod.get_device_capability = original

    # -- neutrality + can-discriminate controls ---------------------------

    def test_identity_is_unchanged_when_no_rederivation_ran(self):
        """Behaviour pin for every non-sm75, non-speculative launch (green
        both ways on purpose): with an empty transparent overlay, the new read
        is byte-for-byte the #499 read."""
        launch = _GgufLaunch(self.dir, tp_size=2)
        launch.match()
        args = launch.materialize()
        self.assertIsNone(
            getattr(args, "_identity_transparent_supersedes", None),
            "no re-derivation ran, so the supersede register must stay absent",
        )
        cfg = resolved_view(args)
        pre_fix_identity = {
            "model_path": cfg.model_path,
            "quantization": cfg.quantization,
            "load_format_original": "gguf",
            "dtype": str(cfg.dtype),
            "tp_size": cfg.tp_size,
            "dcp_size": getattr(cfg, "dcp_size", None),
            "rank_tp_ratio": getattr(cfg, "rank_tp_ratio", None),
            "rank_gpu_id": getattr(cfg, "rank_gpu_id", None),
            "context_length": cfg.context_length,
        }
        self.assertEqual(hibernate._model_identity(args), pre_fix_identity)

    def test_a_different_launch_dtype_still_mismatches(self):
        """Can-discriminate: normalizing the HARDWARE re-derivation away must
        not deafen the identity to the dtype the USER asked for."""
        first = _GgufLaunch(self.dir, dtype="bfloat16")
        first.match()
        self._write_manifest(
            hibernate._model_identity(self._park_after_sm75_fallback(first))
        )

        other = _GgufLaunch(self.dir, dtype="float32")
        other.match()
        self.assertEqual(other.args.load_format, "gguf")

    def test_a_different_launch_context_length_still_mismatches(self):
        """Can-discriminate for the speculative sibling."""
        first = _GgufLaunch(self.dir, context_length=8192)
        first.match()
        self._write_manifest(
            hibernate._model_identity(self._park_after_spec_context_length(first))
        )

        other = _GgufLaunch(self.dir, context_length=16384)
        other.match()
        self.assertEqual(other.args.load_format, "gguf")

    def test_a_runtime_update_weights_still_invalidates_the_manifest(self):
        """The opaque direction, and the one #499's docstring argues must
        never be normalized away: ``/update_weights`` genuinely changes what is
        loaded, so it MUST move the identity."""
        launch = _GgufLaunch(self.dir)
        launch.match()
        args = launch.materialize()
        args.override("model_runner.update_weights", model_path="/other/model.gguf")
        self.assertEqual(
            hibernate._model_identity(args)["model_path"], "/other/model.gguf"
        )

    def test_transparent_sources_do_not_leak_into_the_live_config(self):
        """The normalization is confined to the fingerprint: the RUNTIME must
        still see the re-derived value, or the worker would run bf16 on a card
        that has none."""
        launch = _GgufLaunch(self.dir)
        launch.match()
        args = self._park_after_sm75_fallback(launch)
        self.assertEqual(args.dtype, "float16")

    # -- maintained sibling sweep ----------------------------------------

    def test_worker_writers_of_identity_fields_are_all_classified(self):
        """MAINTAINED sweep (AST over ``python/sglang/srt``), not a by-eye
        list: every call site that writes an identity field through the
        post-resolution mutation point with a literal source must be
        classified -- either identity-transparent (a re-derivation, normalized
        out of the fingerprint) or explicitly opaque with a reason.

        A new worker-side writer of an identity field turns this red, which is
        the point: that is exactly how #520 came into being.
        """
        opaque: Dict[str, str] = {
            "model_runner.update_weights": (
                "runtime /update_weights: genuinely changes which weights are "
                "loaded, so the manifest MUST stop matching"
            ),
            "tokenizer.update_weights": (
                "the tokenizer-side half of /update_weights; same object, same "
                "reason"
            ),
            "draft_worker.build": (
                "writes draft_server_args, a COPY built for the draft model "
                "(draft_worker_common.py:129) -- never the object the park "
                "reads"
            ),
        }
        identity_fields = set(
            hibernate._model_identity(ServerArgs(model_path="dummy"))
        ) - {"load_format_original"}

        unclassified = sorted(
            f"{site} -> {source} writes {sorted(fields)}"
            for site, source, fields in _identity_field_writers(identity_fields)
            if source not in IDENTITY_TRANSPARENT_SOURCES and source not in opaque
        )
        self.assertEqual(
            unclassified,
            [],
            "these call sites write a hibernate-identity field after "
            "resolution and are neither identity-transparent nor registered "
            "as deliberately opaque:\n  " + "\n  ".join(unclassified),
        )

    def test_the_transparent_registry_has_a_live_call_site(self):
        """REACH: a registry entry that no call site uses normalizes nothing.
        Both entries must be findable in the tree."""
        writers = {source for _site, source, _fields in _identity_field_writers(None)}
        for source in sorted(IDENTITY_TRANSPARENT_SOURCES):
            with self.subTest(source=source):
                self.assertIn(
                    source,
                    writers,
                    f"{source!r} is registered as identity-transparent but no "
                    "call site writes an identity field under it any more",
                )


# ---------------------------------------------------------------------------
# AST scan
# ---------------------------------------------------------------------------

_SRT_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(hibernate.__file__)), ""
)  # .../sglang/srt/

# Receivers that are a ServerArgs. get_parallel().override() and the flag
# groups' override() are a DIFFERENT mutation point (parallel/flag state) and
# never touch the fingerprint.
_SERVER_ARGS_RECEIVERS = {
    "server_args",
    "draft_server_args",
    "dp_server_args",
    "sa",
}


def _receiver_name(func: ast.AST) -> str | None:
    if not isinstance(func, ast.Attribute):
        return None
    value = func.value
    if isinstance(value, ast.Name):
        return value.id
    if isinstance(value, ast.Attribute):
        return value.attr
    return None


def _identity_field_writers(
    identity_fields: Set[str] | None,
) -> list[Tuple[str, str, Set[str]]]:
    """(``file:line``, source, fields) for every post-resolution mutation with
    a LITERAL source that writes one of ``identity_fields`` (all fields when
    None)."""
    found: list[Tuple[str, str, Set[str]]] = []
    for dirpath, _dirs, files in os.walk(_SRT_ROOT):
        for name in files:
            if not name.endswith(".py"):
                continue
            path = os.path.join(dirpath, name)
            try:
                with open(path, encoding="utf-8") as f:
                    tree = ast.parse(f.read(), filename=path)
            except (SyntaxError, UnicodeDecodeError):  # pragma: no cover
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                call = func.attr if isinstance(func, ast.Attribute) else None
                if isinstance(func, ast.Name):
                    call = func.id
                if call == "override":
                    if _receiver_name(func) not in _SERVER_ARGS_RECEIVERS:
                        continue
                elif call != "declare_load_time_override":
                    continue

                source = None
                fields: Set[str] = set()
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        source = source or arg.value
                    elif isinstance(arg, ast.Dict):
                        for key in arg.keys:
                            if isinstance(key, ast.Constant):
                                fields.add(key.value)
                for kw in node.keywords:
                    if kw.arg == "source" and isinstance(kw.value, ast.Constant):
                        source = kw.value.value
                    elif kw.arg is not None:
                        fields.add(kw.arg)
                if source is None:
                    continue
                hit = fields if identity_fields is None else fields & identity_fields
                if hit:
                    rel = os.path.relpath(path, _SRT_ROOT)
                    found.append((f"srt/{rel}:{node.lineno}", source, hit))
    return found


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "99")
    unittest.main()
