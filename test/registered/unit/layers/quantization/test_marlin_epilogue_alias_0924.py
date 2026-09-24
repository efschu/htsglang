"""Dense Marlin epilogue alias (sh_red == sh_b): the proof obligations, pinned.

Question (operator 2026-09-24 ~21:10Z; the NF seat confirmed the same lines on
its line): ``jit_kernel/csrc/gemm/marlin/marlin_template.h`` aliases the
reduction buffer onto the B pipeline (``int4* sh_b = sh; int4* sh_red = sh;``),
and the epilogue writes ``sh_red`` (``thread_block_reduce`` / ``write_result``)
right after ``cp_async_wait<0>()`` with no block barrier in between. The NF
line added such a barrier to the MoE twin of this kernel (b8c451a4f8,
SGLANG_MARLIN_EPILOGUE_SYNC). Does the dense kernel need it?

Answer: no byte that is still USED can be hit. The argument rests on seven
structural facts of the template, and this file pins each of them, so an
upstream pull that breaks one fails here instead of silently turning a benign
alias into a live race:

(1) the main loop body is ``fetch_to_registers(k + 1, ...)`` -> [at
    ``k == b_sh_wr_iters - 2``: ``wait_for_stage()``, whose ``__syncthreads()``
    is the last block-wide barrier of the slice] -> ``matmul(k)``. So the
    operands of the last two ``matmul`` calls were loaded BEFORE that barrier
    (``matmul(W-1)`` reads the fragment loaded at ``k = W-2``, before the
    barrier; ``matmul(W-2)`` the one loaded one step earlier), and a later
    write by another warp cannot reach them;
(2) ``matmul`` reads registers only (no ``sh_*``, no ``ldsm``, no cp.async);
(3) the slice ends right after a COMPLETE k-loop (``slice_iters--`` then
    ``break``), so the only shared-memory read after the last barrier is the
    ``k = W-1`` prefetch ``fetch_to_registers(W, (pipe+1) % stages)`` -- a
    fragment for an iteration that does not exist;
(4) that fragment is dead: the next slice starts with ``start_pipes()``, which
    passes a barrier and reloads ``fetch_to_registers(0, 0)`` before any
    ``matmul`` (or the kernel ends);
(5) every other shared buffer (``sh_g_idx``, ``sh_zp``, ``sh_s``, ``sh_a``) is
    laid out past ``max(sh_red_size, sh_b_size)``, and the scale/zero-point/
    group loaders never touch ``sh_b``/``sh_red``;
(6) no cp.async is in flight into ``sh_b`` when the epilogue starts: the main
    loop only issues copies while ``slice_iters >= stages``, and
    ``wait_for_stage`` waits for all but ``stages - 2`` groups;
(7) the next slice's cp.async into ``sh_b`` cannot overtake a reader of
    ``sh_red``: ``write_result`` ends with ``__syncthreads()`` and
    ``barrier_release`` (the non-writing path) begins with one.

What remains is a FORMAL data race -- a lagging warp's dead prefetch may read
bytes a leading warp is writing as reduction partials (compute-sanitizer's
racecheck would report it) -- with no effect on any result. The MoE template
has the same loop shape; the NF metal A/B fn8c2 (barrier ON, 2026-09-20) still
showed the #49 NaNs, whose root turned out to be PDL on sm_120 (0d33570001).

The window test ``test_marlin_epilogue_alias_gpu_0924.py`` hammers the
UNCHANGED kernel (no rebuild) at the running DFlash2-W8 draft's per-rank shapes.

Hermetic: reads the template text, imports nothing from sglang.
"""

import importlib.util
import pathlib
import re
import unittest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _template_path() -> pathlib.Path:
    # find_spec("sglang") locates the package WITHOUT executing its __init__;
    # the template read must be the one the JIT would compile for this tree.
    spec = importlib.util.find_spec("sglang")
    if spec is not None and spec.submodule_search_locations:
        base = pathlib.Path(list(spec.submodule_search_locations)[0])
    else:  # pragma: no cover - layout fallback
        base = pathlib.Path(__file__).resolve().parents[5] / "python" / "sglang"
    return base / "jit_kernel" / "csrc" / "gemm" / "marlin" / "marlin_template.h"


def _strip_comments(src: str) -> str:
    src = re.sub(r"/\*.*?\*/", " ", src, flags=re.S)
    return re.sub(r"//[^\n]*", " ", src)


def _block(src: str, open_idx: int) -> str:
    """Body of the brace block whose ``{`` is at or after ``open_idx``."""
    i = src.index("{", open_idx)
    depth = 0
    for j in range(i, len(src)):
        if src[j] == "{":
            depth += 1
        elif src[j] == "}":
            depth -= 1
            if depth == 0:
                return src[i + 1 : j]
    raise AssertionError("unbalanced braces after offset %d" % open_idx)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _lambda(src: str, name: str) -> str:
    m = re.search(r"auto\s+%s\s*=\s*\[&\]\s*\([^)]*\)\s*\{" % re.escape(name), src)
    assert m, "lambda %s not found in the dense Marlin template" % name
    return _block(src, m.end() - 1)


def _device_fn(src: str, name: str) -> str:
    m = re.search(r"__device__\s+inline\s+void\s+%s\s*\([^)]*\)\s*\{" % re.escape(name), src)
    assert m, "device function %s not found" % name
    return _block(src, m.end() - 1)


class TestDenseMarlinEpilogueAlias(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = _template_path()
        if not path.is_file():
            raise unittest.SkipTest("dense Marlin template not found at %s" % path)
        cls.path = path
        cls.src = _strip_comments(path.read_text())
        cls.flat = _norm(cls.src)
        cls.aliased = ("int4* sh_b = sh;" in cls.flat) and ("int4* sh_red = sh;" in cls.flat)

    def setUp(self):
        if not self.aliased:
            # Nothing to guard: the obligations below exist only because of the
            # alias. Say so rather than pass silently.
            self.skipTest("sh_red is no longer aliased onto sh_b in %s" % self.path)

    def _main_loop(self) -> str:
        m = re.search(r"while\s*\(\s*slice_iters\s*\)\s*\{", self.src)
        self.assertIsNotNone(m, "main loop `while (slice_iters)` not found")
        return _block(self.src, m.end() - 1)

    def _pipe_and_k_loop(self):
        loop = self._main_loop()
        mp = re.search(r"for\s*\(\s*int\s+pipe\s*=\s*0\s*;\s*pipe\s*<\s*stages\s*;\s*\)\s*\{", loop)
        self.assertIsNotNone(mp, "pipe loop not found in the main loop")
        pipe_body = _block(loop, mp.end() - 1)
        mk = re.search(r"for\s*\(\s*int\s+k\s*=\s*0\s*;\s*k\s*<\s*b_sh_wr_iters\s*;\s*k\+\+\s*\)\s*\{", pipe_body)
        self.assertIsNotNone(mk, "k loop not found in the pipe loop")
        k_body = _block(pipe_body, mk.end() - 1)
        after_k = pipe_body[mk.end() - 1 + len(k_body) + 2 :]
        return loop, pipe_body, _norm(k_body), _norm(after_k)

    # (1) load -> barrier -> matmul, one register load per step
    def test_1_k_step_loads_before_the_barrier_and_multiplies_after(self):
        _, _, k_body, _ = self._pipe_and_k_loop()
        self.assertEqual(k_body.count("fetch_to_registers("), 1, k_body)
        self.assertEqual(k_body.count("matmul("), 1, k_body)
        i_fetch = k_body.find("fetch_to_registers(k + 1, pipe % stages)")
        i_if = k_body.find("if (k == b_sh_wr_iters - 2)")
        i_wait = k_body.find("wait_for_stage();")
        i_mm = k_body.find("matmul(k);")
        self.assertTrue(0 <= i_fetch < i_if < i_wait < i_mm, k_body)
        self.assertTrue(k_body.endswith("matmul(k);"), k_body)
        wait_body = _norm(_lambda(self.src, "wait_for_stage"))
        self.assertIn("cp_async_wait<stages - 2>();", wait_body)
        self.assertTrue(wait_body.endswith("__syncthreads();"), wait_body)

    # (2) matmul touches registers only
    def test_2_matmul_reads_no_shared_memory(self):
        body = _lambda(self.src, "matmul")
        self.assertIsNone(re.search(r"\bsh(_\w+)?\b\s*[\[+]|\bsh_\w+", body), "matmul reads shared memory")
        self.assertNotIn("ldsm", body)
        self.assertNotIn("cp_async", body)

    # (3) the slice ends only after a complete k-loop
    def test_3_slice_breaks_right_after_a_complete_k_loop(self):
        _, _, _, after_k = self._pipe_and_k_loop()
        self.assertTrue(
            after_k.startswith("slice_iters--; if (slice_iters == 0) { break; }"), after_k
        )

    # (4) the post-barrier prefetch is dead: start_pipes reloads before any matmul
    def test_4_next_slice_reloads_the_prefetched_fragment(self):
        body = _norm(_lambda(self.src, "start_pipes"))
        i_wait = body.find("wait_for_stage();")
        i_fetch = body.find("fetch_to_registers(0, 0);")
        self.assertTrue(0 <= i_wait < i_fetch, body)
        self.assertNotIn("matmul(", body)
        tail = _norm(self._main_loop())
        self.assertIn("start_pipes();", tail)

    # (5) nothing else lives inside the alias
    def test_5_other_buffers_sit_past_the_aliased_region(self):
        flat = self.flat
        self.assertIn("int4* sh_g_idx = sh_b + (sh_red_size > sh_b_size ? sh_red_size : sh_b_size);", flat)
        self.assertRegex(flat, r"int4\* sh_zp = sh_g_idx \+")
        self.assertRegex(flat, r"int4\* sh_s = sh_zp \+")
        self.assertRegex(flat, r"int4\* sh_a = sh_s \+")
        for name in ("fetch_scales_to_registers", "fetch_zp_to_registers", "init_same_group"):
            body = _lambda(self.src, name)
            self.assertIsNone(re.search(r"\bsh_b\b|\bsh_red\b|\bsh\s*\[", body), name)

    # (6) no cp.async lands in sh_b once the epilogue starts
    def test_6_no_copy_in_flight_at_the_epilogue(self):
        loop = _norm(self._main_loop())
        self.assertIn(
            "fetch_to_shared((pipe + stages - 1) % stages, pipe, slice_iters >= stages);", loop
        )
        m = re.search(r"if\s*\(\s*slice_iters\s*==\s*0\s*\)\s*\{\s*cp_async_wait<0>\(\);", self.src)
        self.assertIsNotNone(m, "epilogue no longer opens with cp_async_wait<0>()")

    # (7) sh_red readers finish before the next slice refills sh_b
    def test_7_next_slice_cannot_overtake_a_sh_red_reader(self):
        wr = _norm(_lambda(self.src, "write_result"))
        self.assertTrue(wr.endswith("__syncthreads();"), wr[-120:])
        rel = _norm(_device_fn(self.src, "barrier_release"))
        self.assertTrue(rel.startswith("__syncthreads();"), rel[:120])
        acq = _norm(_device_fn(self.src, "barrier_acquire"))
        self.assertTrue(acq.endswith("__syncthreads();"), acq[-120:])

    def test_sh_red_is_used_only_by_the_epilogue(self):
        users = set()
        for m in re.finditer(r"auto\s+(\w+)\s*=\s*\[&\]", self.src):
            if re.search(r"\bsh_red\b", _block(self.src, m.end())):
                users.add(m.group(1))
        allowed = {"thread_block_reduce", "global_reduce_fp16", "global_reduce_fp32", "write_result", "write"}
        self.assertTrue(users <= allowed, "sh_red read/written outside the epilogue: %s" % sorted(users - allowed))


if __name__ == "__main__":
    unittest.main()
