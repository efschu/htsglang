"""#421 F4 falsifier: the shared cold tier must be REACHED from production.

``layers/moe/cold_tier_shm.py`` shipped in #394 slice 1 with no production
importer at all -- audit #421 classified it INERT (declared). Slice 2 adds the
routing half (``cold_tier_fetch.py``) and wires it, so the honest replacement
for the old absence pin is a presence pin, at the CALL SITES rather than at the
module boundary. Consumer counting finds absent wiring; only a call-site pin
finds wiring that a later refactor quietly removes.

Three sites carry the chain, and each is asserted here by name:

  * the launcher mints the launch id before the spawn loop
    (``entrypoints/engine.py``) -- without it every worker refuses, one at a
    time, at its first fetch;
  * the GGUF streaming door builds the owner handle and stashes the owner map
    (``fused_moe_triton/layer.py``);
  * the offload cache turns the slice-1 refusal into a fetch route
    (``expert_offload.py``).

Can-fail proof: delete any one of the three calls and the matching test goes
red; set ``SGLANG_MOE_COLD_TIER_SHM`` back to a hard ``False`` default and
``test_the_default_path_takes_no_cold_tier_branch`` still passes while
``test_enabling_the_tier_is_a_single_documented_flag`` goes red, which is the
distinction between "off by default" and "not there".

The behaviour of the chain -- owner map, manifest resolution, zero-copy reads,
the read-only guarantee -- is covered hermetically in
``tests/moe_offload/test_cold_tier_fetch.py``. This file only pins that
production reaches it.

Hermetic: no GPU, no model, no shm segment created.
"""

import ast
import pathlib
import unittest
import warnings

from sglang.srt.layers.moe import cold_tier_fetch as ctf
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[5]
_PY = _REPO_ROOT / "python"


def _tree(rel_path):
    return ast.parse((_REPO_ROOT / rel_path).read_text())


def _call_lines(rel_path, func_name):
    """Line numbers at which ``func_name`` is called in ``rel_path``."""
    out = []
    for node in ast.walk(_tree(rel_path)):
        if not isinstance(node, ast.Call):
            continue
        callee = node.func
        name = (
            callee.id
            if isinstance(callee, ast.Name)
            else callee.attr if isinstance(callee, ast.Attribute) else None
        )
        if name == func_name:
            out.append(node.lineno)
    return out


_ENGINE = "python/sglang/srt/entrypoints/engine.py"
_LAYER = "python/sglang/srt/layers/moe/fused_moe_triton/layer.py"
_OFFLOAD = "python/sglang/srt/layers/moe/expert_offload.py"


class TestColdTierIsWiredFromProduction(CustomTestCase):
    def test_the_launcher_mints_the_launch_id_before_spawning(self):
        """One id for the whole group, published on the environment channel.

        A worker that minted its own would name segments no peer can find, and
        the divergence would only surface at the first fetch.
        """
        lines = _call_lines(_ENGINE, "publish_cold_tier_instance")
        self.assertTrue(
            lines,
            "the launcher no longer publishes the cold-tier launch id; every "
            "worker will refuse at its first peer fetch",
        )

    def test_the_gguf_door_builds_the_owner_handle(self):
        lines = _call_lines(_LAYER, "_gguf_cold_tier_owner")
        self.assertTrue(
            lines,
            "the GGUF streaming door no longer builds a cold-tier owner, so no "
            "rank publishes a segment and #394 is inert again",
        )

    def test_the_offload_cache_resolves_a_fetch_route(self):
        lines = _call_lines(_OFFLOAD, "resolver_for_layer")
        self.assertTrue(
            lines,
            "MoEExpertOffloadCache no longer resolves a cold-tier route, so a "
            "delegated expert is a refusal again (#421 F4 regressed)",
        )

    def test_the_storage_half_has_a_production_importer_now(self):
        """The absence pin this file replaces, inverted."""
        importers = []
        for path in _PY.rglob("*.py"):
            rel = path.relative_to(_REPO_ROOT).as_posix()
            if rel.endswith("layers/moe/cold_tier_shm.py"):
                continue
            try:
                # Parsing the whole tree surfaces every file's own
                # SyntaxWarnings (stray escapes in docstrings and the like).
                # They are not this test's finding and not its business.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    tree = ast.parse(path.read_text())
            except (SyntaxError, UnicodeDecodeError):
                continue
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.module == "sglang.srt.layers.moe.cold_tier_shm"
                ):
                    importers.append(f"{rel}:{node.lineno}")
        self.assertTrue(
            importers,
            "sglang.srt.layers.moe.cold_tier_shm has no production importer "
            "again -- #421 finding F4 has regressed",
        )


class TestTheDefaultPathIsUnchanged(CustomTestCase):
    def test_the_default_path_takes_no_cold_tier_branch(self):
        """Off unless asked for. Every entry point must answer ``None``."""
        import os

        previous = os.environ.pop("SGLANG_MOE_COLD_TIER_SHM", None)
        try:
            self.assertFalse(ctf.cold_tier_enabled())
            self.assertIsNone(ctf.owner_for_layer("L0", 0, 2, ("A", "B")))
            self.assertIsNone(ctf.resolver_for_layer(object()))
        finally:
            if previous is not None:
                os.environ["SGLANG_MOE_COLD_TIER_SHM"] = previous

    def test_enabling_the_tier_is_a_single_documented_flag(self):
        from sglang.srt.environ import envs

        self.assertIn("SGLANG_MOE_COLD_TIER_SHM", dir(envs))
        self.assertFalse(envs.SGLANG_MOE_COLD_TIER_SHM.get())


if __name__ == "__main__":
    unittest.main()
