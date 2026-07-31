# SPDX-License-Identifier: Apache-2.0
"""#355: every KV writer derives its slot bound from ONE helper.

#352 gave the raw ``store_cache`` path a graph-safe index bound and left a
registered finding behind: ``masked_set_kv_buffer_kernel`` -- the kernel every
target-side DCP write goes through -- had no bound at all. Today it is
protected only INDIRECTLY, by #345 having made the compact-row mapping
correct. An escaped compact row (a #345-family regression, a bad reshard, a
scheduler bug) would corrupt a live KV row SILENTLY instead of failing at the
culprit, which is the exact defect class this project hunts.

The fix is not "add a check to that one kernel". Two paths writing the same
buffer with two independently-written bound expressions is how they drift, and
a bound that drifted is a bound that is wrong on exactly the load nobody
tested. So the bound has ONE source, ``kv_store_bound``, and these tests fail
if a writer grows its own:

* bound-source unity -- ``graph_safe_store_bound`` has exactly one production
  caller, and each writer launch takes its bound from ``kv_store_bound``;
* writer audit (#352-style consumer audit, extended) -- every slot-indexed
  kernel launch in the KV write path is either bounded or on an explicit
  allowlist with a reason, so a NEW unbounded writer fails this test rather
  than being discovered by a corrupted answer;
* the check itself -- both masked kernels carry a ``bound`` parameter and a
  ``tl.device_assert``, and the default-on env switch controls whether it is
  lowered.

CPU-only: everything here is source structure and integer math. The card-side
proof that the guard actually fires lives in
``test_masked_kv_bound_falsifier.py``.

    python -m pytest test/registered/mem_cache/test_kv_store_bound_unity.py -v
"""

import ast
import inspect
import textwrap
import unittest
from pathlib import Path

import torch

import sglang.kernels.ops.kvcache.cache_move as cache_move
import sglang.srt.mem_cache.memory_pool as memory_pool
from sglang.srt.environ import envs
from sglang.srt.mem_cache.memory_pool import (
    graph_safe_store_bound,
    kv_bound_check_enabled,
    kv_store_bound,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

MEMORY_POOL_SRC = Path(memory_pool.__file__).read_text()
CACHE_MOVE_SRC = Path(cache_move.__file__).read_text()

# Every kernel launched from the KV write path that indexes a destination row
# by a slot id. The value is the bound mechanism; ``None`` means deliberately
# unbounded and states why. A new writer added to either module without an
# entry here fails ``test_no_unaudited_slot_indexed_writer``.
WRITER_AUDIT = {
    # --- bounded ---
    "store_cache": "size_limit=kv_store_bound(...) -> SGL_DEVICE_ASSERT (#352)",
    "masked_set_kv_buffer_kernel": "bound=kv_store_bound(...) -> tl.device_assert (#355)",
    "set_kv_buffer_prefix_valid_tiled": (
        "bound=kv_store_bound(...) -> tl.device_assert (#355)"
    ),
    # --- deliberately unbounded, with the reason ---
    "store_cache_cpu": None,  # CPU op, no device assert to speak of
    "copy_all_layer_kv_cache_func": None,  # move path: host maybe_detect_oob on both locs
    "copy_all_layer_kv_cache_tiled": None,  # kernel behind copy_all_layer_kv_cache_func
    "copy_all_layer_kv_cache_cpu": None,  # CPU arm of the same
    "move_kv_cache_native": None,  # torch advanced indexing carries its own device assert
    "store_cache_4d": None,  # page-major pool; registered open item, see #355 report
    "store_cache_4d_kernel": None,  # kernel behind store_cache_4d
    "launch_reshape_and_cache_shuffle_5d": None,  # vectorized_5d; registered open item
    "set_mla_kv_buffer_triton": None,  # MLA pool; registered open item
    "set_mla_kv_buffer_triton_fp8_quant": None,  # MLA pool; registered open item
    "set_index_kv_buffer": None,  # MiniMax index cache; registered open item
    "store_kv_index": None,  # MiniMax index cache kernel; registered open item
}

# Names the AST probe cannot tell apart from a writer but which read, wrap or
# compute rather than scatter.
NOT_WRITERS = {
    "store",  # tl.store, the Triton primitive
    "graph_safe_store_bound",
    "kv_store_bound",
    "can_use_store_cache",
    "get_kv_buffer",
    "get_key_buffer",
    "get_value_buffer",
    "get_mla_kv_buffer",
    "get_mla_kv_buffer_triton",
    "get_index_kv_buffer",
    "get_contiguous_buf_infos",
    "get_kv_size_bytes",
    "get_cpu_copy",
    "load_cpu_copy",
    "set_kv_buffer",  # dispatchers; the physical writers are the entries above
    "set_kv_buffer_prefix_valid",
    "set_mla_kv_buffer",
    "move_kv_cache",
    "register_layer_transfer_counter",
    "maybe_detect_oob",
}

BOUNDED_WRITERS = {k for k, v in WRITER_AUDIT.items() if v is not None}


def _launch_names(src: str) -> set:
    """Names of every kernel/op invoked as ``name(...)`` or ``name[grid](...)``."""
    names = set()
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if isinstance(fn, ast.Subscript):  # triton launch: kernel[grid](...)
            fn = fn.value
        if isinstance(fn, ast.Name):
            names.add(fn.id)
        elif isinstance(fn, ast.Attribute):
            names.add(fn.attr)
    return names


def _calls_in_function(src: str, func_name: str) -> set:
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return _launch_names(ast.unparse(node))
    raise AssertionError(f"function {func_name} not found")


def _enclosing_function_of_launch(src: str, kernel: str) -> str:
    """Source of the function that launches ``kernel``."""
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = ast.unparse(node)
        if f"{kernel}[" in body:
            return body
    raise AssertionError(f"no function launches {kernel}")


class TestBoundSourceUnity(CustomTestCase):
    """One helper decides the bound. Two would eventually disagree."""

    def test_graph_safe_store_bound_has_exactly_one_production_caller(self):
        # kv_store_bound is the only place allowed to pick the graph-stable
        # number. If a second writer starts calling graph_safe_store_bound
        # directly it can pass a different row count and the paths drift.
        repo_python = Path(memory_pool.__file__).parents[3]
        callers = []
        for path in repo_python.rglob("*.py"):
            text = path.read_text()
            if "graph_safe_store_bound" not in text:
                continue
            for node in ast.walk(ast.parse(text)):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "graph_safe_store_bound"
                ):
                    callers.append(f"{path.name}:{node.lineno}")
        self.assertEqual(
            len(callers),
            1,
            f"graph_safe_store_bound must be called only by kv_store_bound, got {callers}",
        )
        self.assertIn(
            "graph_safe_store_bound",
            _calls_in_function(MEMORY_POOL_SRC, "kv_store_bound"),
        )

    def test_raw_store_path_takes_its_bound_from_the_helper(self):
        calls = _calls_in_function(MEMORY_POOL_SRC, "_set_kv_buffer_impl")
        self.assertIn("store_cache", calls)
        self.assertIn("kv_store_bound", calls)

    def test_masked_writer_takes_its_bound_from_the_same_helper(self):
        body = _enclosing_function_of_launch(
            MEMORY_POOL_SRC, "masked_set_kv_buffer_kernel"
        )
        self.assertIn("kv_store_bound(", body)

    def test_prefix_valid_writer_takes_its_bound_from_the_same_helper(self):
        body = _enclosing_function_of_launch(
            MEMORY_POOL_SRC, "set_kv_buffer_prefix_valid_tiled"
        )
        self.assertIn("kv_store_bound(", body)

    def test_a_divergent_bound_expression_would_fail_these_tests(self):
        # Falsifier for the tests above: the same AST probe run against a
        # writer that computed its own bound reports no kv_store_bound call.
        drifted = textwrap.dedent(
            """
            def set_kv_buffer(self):
                masked_set_kv_buffer_kernel[(N,)](k, v, self.size + self.page_size)
            """
        )
        body = _enclosing_function_of_launch(drifted, "masked_set_kv_buffer_kernel")
        self.assertNotIn("kv_store_bound(", body)


class TestKvStoreBoundSemantics(CustomTestCase):
    def test_row_count_is_taken_under_the_calling_kernel_stride(self):
        # 3-D per-layer buffer [rows, H, D] flattened by a writer that strides
        # by H*D: the bound is the row count, not the element count.
        buf = torch.zeros(64, 2, 8)
        self.assertEqual(kv_store_bound(64, buf, 2 * 8), 64)
        # A writer striding by half a row can reach twice as many "rows".
        self.assertEqual(kv_store_bound(64, buf, 8), 128)

    def test_off_dial_lane_is_byte_identical_to_the_pre_355_number(self):
        # Buffers allocated at exactly size + page_size: the bound equals the
        # live limit, i.e. what every writer effectively assumed before.
        buf = torch.zeros(4097, 256)
        self.assertEqual(kv_store_bound(4097, buf, 256), 4097)

    def test_dial_lane_widens_to_the_va_reservation(self):
        buf = torch.zeros(522198, 4)  # VA reserve, backed well below it
        self.assertEqual(kv_store_bound(251966, buf, 4), 522198)

    def test_never_narrows_below_the_live_bound(self):
        buf = torch.zeros(400, 4)
        self.assertEqual(kv_store_bound(600, buf, 4), 600)

    def test_delegates_to_graph_safe_store_bound(self):
        buf = torch.zeros(97, 3)
        self.assertEqual(kv_store_bound(11, buf, 3), graph_safe_store_bound(11, 97))

    def test_store_cache_arm_is_byte_identical_to_the_pre_355_expression(self):
        # #352 wrote `graph_safe_store_bound(size_limit, k_cache.view(-1,
        # row_dim).shape[0])` inline. Routing it through kv_store_bound must
        # not change the number for any geometry the pool can hand over --
        # otherwise "no behaviour change on the raw path" is a claim, not a
        # fact. Qwen3.6-27B row: 4 replicated KV heads x head_dim 256.
        for rows, h, d in ((4097, 4, 256), (251966, 1, 256), (64, 8, 128)):
            with self.subTest(rows=rows, h=h, d=d):
                buf = torch.zeros(rows, h, d, dtype=torch.bfloat16)
                row_dim = h * d
                old = graph_safe_store_bound(rows, buf.view(-1, row_dim).shape[0])
                self.assertEqual(kv_store_bound(rows, buf, row_dim), old)


class TestMaskedWriterCarriesTheCheck(CustomTestCase):
    """The kernels themselves: a bound parameter and an assert on it."""

    def _kernel_source(self, jit_fn) -> str:
        # triton.jit wraps the python function; .fn is the original.
        return inspect.getsource(getattr(jit_fn, "fn", jit_fn))

    def test_masked_set_kv_buffer_kernel_asserts_its_loc(self):
        src = self._kernel_source(memory_pool.masked_set_kv_buffer_kernel)
        self.assertIn("bound", src)
        self.assertIn("tl.device_assert", src)
        self.assertIn("loc < bound", src)
        self.assertIn("loc >= 0", src)

    def test_prefix_valid_tiled_asserts_its_loc(self):
        src = self._kernel_source(cache_move.set_kv_buffer_prefix_valid_tiled)
        self.assertIn("bound", src)
        self.assertIn("tl.device_assert", src)
        self.assertIn("loc < bound", src)
        self.assertIn("loc >= 0", src)

    def test_assert_message_names_the_writer(self):
        # A device assert whose message does not say which writer fired is
        # only marginally better than silence.
        for fn in (
            memory_pool.masked_set_kv_buffer_kernel,
            cache_move.set_kv_buffer_prefix_valid_tiled,
        ):
            src = self._kernel_source(fn)
            self.assertIn("loc out of range", src)
        self.assertIn(
            "masked_set_kv_buffer:",
            self._kernel_source(memory_pool.masked_set_kv_buffer_kernel),
        )
        self.assertIn(
            "set_kv_buffer_prefix_valid:",
            self._kernel_source(cache_move.set_kv_buffer_prefix_valid_tiled),
        )

    def test_bound_is_a_runtime_arg_not_a_constexpr(self):
        # A constexpr bound would recompile the kernel per capacity and,
        # worse, invite folding a LIVE size into it. The bound is passed by
        # value exactly like store_cache's size_limit.
        for fn in (
            memory_pool.masked_set_kv_buffer_kernel,
            cache_move.set_kv_buffer_prefix_valid_tiled,
        ):
            params = inspect.signature(fn.fn).parameters
            self.assertIn("bound", params)
            self.assertIs(params["bound"].annotation, inspect.Parameter.empty)

    def test_launches_pass_the_debug_switch(self):
        for src, kernel in (
            (MEMORY_POOL_SRC, "masked_set_kv_buffer_kernel"),
            (MEMORY_POOL_SRC, "set_kv_buffer_prefix_valid_tiled"),
        ):
            body = _enclosing_function_of_launch(src, kernel)
            self.assertIn("debug=kv_bound_check_enabled()", body)


class TestBoundCheckSwitch(CustomTestCase):
    def test_default_is_on(self):
        self.assertTrue(kv_bound_check_enabled())

    def test_env_turns_it_off_and_back_on(self):
        with envs.SGLANG_DISABLE_KV_MASKED_BOUND_CHECK.override(True):
            self.assertFalse(kv_bound_check_enabled())
        self.assertTrue(kv_bound_check_enabled())


class TestWriterAudit(CustomTestCase):
    """#352's consumer audit, extended to the masked writers.

    The #352 finding was not "this kernel is missing a check" but "nobody
    enumerated the writers". This test does the enumeration in code.
    """

    def test_no_unaudited_slot_indexed_writer(self):
        known = set(WRITER_AUDIT)
        launched = _launch_names(MEMORY_POOL_SRC) | _launch_names(CACHE_MOVE_SRC)
        # Only names we already classify are interesting; the point of the
        # test is that a NEW writer name shows up unclassified.
        candidates = {
            n
            for n in launched
            if ("store" in n or "kv_buffer" in n or "kv_cache" in n)
            and n not in known
            and n not in NOT_WRITERS
            and not n.startswith("_")
        }
        self.assertEqual(
            candidates,
            set(),
            "unaudited KV writer(s); add them to WRITER_AUDIT with a bound "
            "mechanism or an explicit reason for having none",
        )

    def test_every_bounded_writer_is_launched_with_the_shared_helper(self):
        for writer in BOUNDED_WRITERS:
            if writer == "store_cache":
                body = MEMORY_POOL_SRC[
                    MEMORY_POOL_SRC.index("def _set_kv_buffer_impl") :
                ]
                body = body[: body.index("\ndef ", 1)]
            else:
                body = _enclosing_function_of_launch(MEMORY_POOL_SRC, writer)
            self.assertIn(
                "kv_store_bound(",
                body,
                f"{writer} does not take its bound from kv_store_bound",
            )

    def test_masked_path_is_reachable_from_the_public_entry(self):
        # Guards against the check being added to a kernel the production
        # DCP write no longer goes through.
        body = _enclosing_function_of_launch(
            MEMORY_POOL_SRC, "masked_set_kv_buffer_kernel"
        )
        self.assertIn("dcp_kv_mask", body)


if __name__ == "__main__":
    unittest.main()
