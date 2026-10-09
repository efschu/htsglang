"""F0-K (RC1 acceptance 09.10.2026, B1 + m1): the JIT wrappers must name the C++ namespace the kernels really live in.

What broke.  The rename turned ``load_jit``'s default ``wrap_namespace`` from ``"sglang"`` into ``"flliper"`` (python/flliper/kernels/jit/utils.py)
and five diffusion wrappers from ``sglang_<x>::`` into ``flliper_<x>::``.  The C++ kernels are R2 must-keep: they stay in ``namespace sglang``
(``qsa_indexer.cuh``, ``fast_topk.cuh``, ``kernels/jit/include/sgl_kernel_next``) and ``namespace sglang_<x>`` (``csrc/diffusion/*.cuh``).  Every module that
goes through ``flliper.kernels.jit`` (QSA indexer, fast top-k, HC combine, grouped Gemma RMSNorm = the Flash-Next model) stopped compiling;
the first QSA forward of a Flash-Next boot would have killed the rank.  Nothing in the rename kit's checks can see this: the AST comparison
hides strings, the rest inventory counts what is LEFT over, not what was renamed wrongly.

Three layers, all GPU-free:

* ``TestKitKeepsNativeNames``       the kit's rewrite engine leaves exactly these spellings alone (and still renames the rest);
* ``TestWrapperNamespaceParity``    every namespace a Python wrapper names (``wrap_namespace``, ``X::Kernel`` in ``cuda_wrappers`` / ``cpp_wrappers``)
                                    is DECLARED by a C++ file of the tree -- static, no compiler;
* ``TestJitWrappersCompile``        the source ``load_jit`` really hands to the compiler is run through ``nvcc -c`` (no device, no launch) for every
                                    wrapper of the Flash-Next model and of the five diffusion kernels.  Red on the RC1 heads, green after the fix.

The old compiler-visible spelling is written split (``"sg" "lang"``) where it has to appear: the kit's own mechanical pass must leave this file
alone (it is a decision file of the rest inventory as well).
"""

import ast
import contextlib
import importlib
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from flliper.test.ci.ci_register import register_cpu_ci
from flliper.test.test_utils import CustomTestCase

register_cpu_ci(est_time=300, suite="base-a-test-cpu")

ROOT = pathlib.Path(__file__).resolve().parents[4]
PKG = ROOT / "python" / "flliper"
SG = "sg" "lang"          # the C++ namespace of the kernels (R2 must-keep)
NEW = "flli" "per"        # the renamed package name -- what must NOT be a C++ namespace


def _kit():
    p = str(ROOT / "tools" / "release")
    if p not in sys.path:
        sys.path.insert(0, p)
    return importlib.import_module("rename_to_flliper")


# --------------------------------------------------------------------------------------------------
class TestKitKeepsNativeNames(CustomTestCase):
    """The rename engine must leave the C++ namespaces and the persisted identities of RC1 alone, and nothing more."""

    def _rw(self, text, path=None):
        R = _kit()
        out, _rep, _skip = R.rewrite_all(text, True, True, {}, path)
        return out

    def test_wrap_namespace_default_is_kept(self):
        t = 'kwargs.setdefault("wrap_namespace", "%s")\n' % SG
        self.assertEqual(self._rw(t), t)
        self.assertEqual(self._rw('load_jit(x, wrap_namespace="%s")\n' % SG), 'load_jit(x, wrap_namespace="%s")\n' % SG)

    def test_qualified_diffusion_names_are_kept(self):
        for x in ("timestep_embedding", "residual_gate_add", "causal_conv3d_cat_pad", "ltx2_qknorm_split_rope", "norm_scale_shift"):
            t = '"%s_%s::Kernel<{args}>::run"\n' % (SG, x)
            self.assertEqual(self._rw(t), t, x)

    def test_namespace_statement_in_a_docstring_is_kept(self):
        t = "``namespace %s { ... }`` wraps every export\n" % SG
        self.assertEqual(self._rw(t), t)

    def test_persisted_identities_are_kept(self):
        w2 = "WE" "G2"
        for t in ('_NAMESPACE_SEED_TAG = b"%s-kv-namespace-v1\\0"\n' % SG, 'MAGIC = b"%s-L3-INDEX v3"\n' % w2,
                  "one line ``%s-L3-INDEX v3 n=<N>``\n" % w2):
            self.assertEqual(self._rw(t), t, t)

    def test_the_rule_is_narrow(self):
        # the package, the env prefix, a free C++ word and the subsystem word are still renamed
        self.assertEqual(self._rw("import %s.srt.utils\n" % SG), "import %s.srt.utils\n" % NEW)
        self.assertEqual(self._rw("%s_FOO = 1\n" % SG.upper()), "%s_FOO = 1\n" % NEW.upper())
        self.assertEqual(self._rw("weg2_flip_state = 1\n"), "pdflip_flip_state = 1\n")
        self.assertEqual(self._rw("a = '%s kv namespace'\n" % SG), "a = '%s kv namespace'\n" % NEW)


# --------------------------------------------------------------------------------------------------
def _load_jit_calls():
    """-> [(file, lineno, wrapper-leading-namespaces [str], wrap_namespace-or-None)] for every ``load_jit(...)`` call of the tree."""
    out = []
    for base in (PKG / "jit_kernel", PKG / "kernels"):
        for f in sorted(base.rglob("*.py")):
            try:
                tree = ast.parse(f.read_text(encoding="utf-8"))
            except SyntaxError:
                continue
            for n in ast.walk(tree):
                if not (isinstance(n, ast.Call) and getattr(n.func, "id", getattr(n.func, "attr", "")) == "load_jit"):
                    continue
                quals, wrap = [], None
                for kw in n.keywords:
                    if kw.arg in ("cuda_wrappers", "cpp_wrappers") and isinstance(kw.value, (ast.List, ast.Tuple)):
                        for tup in kw.value.elts:
                            if isinstance(tup, ast.Tuple) and len(tup.elts) == 2:
                                kern = tup.elts[1]
                                first = next((c.value for c in ast.walk(kern) if isinstance(c, ast.Constant) and isinstance(c.value, str)), "")
                                # a namespace is lower_snake; ``Kernel::run`` / ``Struct<..>::fn`` is a class scope, not a namespace
                                m = re.match(r"([a-z_][a-z0-9_]*)::", first)
                                if m:
                                    quals.append(m.group(1))
                    if kw.arg == "wrap_namespace" and isinstance(kw.value, ast.Constant):
                        wrap = kw.value.value
                out.append((str(f.relative_to(ROOT)), n.lineno, quals, wrap))
    return out


def _declared_namespaces():
    decl = set()
    for base in (PKG / "jit_kernel", PKG / "kernels"):
        for ext in ("*.cuh", "*.h", "*.hpp", "*.cu", "*.cpp"):
            for f in base.rglob(ext):
                try:
                    decl.update(re.findall(r"^\s*namespace\s+(\w+)\s*\{", f.read_text(encoding="utf-8", errors="replace"), re.M))
                except OSError:
                    pass
    return decl


class TestWrapperNamespaceParity(CustomTestCase):
    def test_every_qualifier_a_wrapper_names_is_declared_by_c_plus_plus(self):
        decl = _declared_namespaces()
        self.assertIn(SG, decl, "the kernels' own namespace must be found in the headers (scan broken?)")
        bad = [(f, ln, q) for f, ln, quals, _w in _load_jit_calls() for q in quals if q not in decl]
        self.assertEqual(bad, [], "wrapper names a C++ namespace no C++ file declares (renamed on one side only)")

    def test_default_wrap_namespace_of_the_kernels_alias_is_declared(self):
        src = (PKG / "kernels" / "jit" / "utils.py").read_text(encoding="utf-8")
        m = re.search(r'setdefault\("wrap_namespace",\s*"(\w+)"\)', src)
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), SG)
        self.assertIn(m.group(1), _declared_namespaces())

    def test_the_package_name_is_no_cpp_namespace(self):
        decl = _declared_namespaces()
        self.assertEqual([d for d in decl if d == NEW or d.startswith(NEW + "_")], [])


# --------------------------------------------------------------------------------------------------
def _nvcc():
    for c in (shutil.which("nvcc"), os.path.join(os.environ.get("CUDA_HOME", ""), "bin", "nvcc"), "/usr/local/cuda/bin/nvcc"):
        if c and os.path.isfile(c):
            return c
    return None


def _tvm_includes():
    try:
        import tvm_ffi.libinfo as li
        return [li.find_include_path(), li.find_dlpack_include_path()]
    except Exception:
        return None


#: (module, wrapper function, args, (major, minor)) -- every ``load_jit`` wrapper of the Flash-Next model (flliper.kernels.jit) and of the five
#: diffusion kernels (qualified ``<ns>::Kernel``).  Arguments are real specialisations of the model code.
CASES = [
    ("flliper.kernels.ops.attention.qsa_indexer", "_jit_qsa_indexer_module", ("bfloat16", 128, True), (8, 6)),
    ("flliper.kernels.ops.elementwise.fast_topk", "_jit_fast_topk_module", (512,), (8, 6)),
    ("flliper.kernels.ops.elementwise.hc_combine", "_jit_hc_combine_module", (4, 4096, "bfloat16"), (8, 6)),
    ("flliper.kernels.ops.layernorm.grouped_gemma_rmsnorm", "_jit_grouped_gemma_rmsnorm_module", (512, "bfloat16"), (8, 6)),
    ("flliper.jit_kernel.timestep_embedding", "_jit_timestep_embedding_module", ("float32",), (8, 6)),
    ("flliper.jit_kernel.diffusion.residual_gate_add", "_jit_residual_gate_add_module", ("bfloat16",), (8, 6)),
    ("flliper.jit_kernel.diffusion.causal_conv3d_cat_pad", "_jit_causal_conv3d_cat_pad_module", ("bfloat16",), (8, 6)),
    ("flliper.jit_kernel.diffusion.ltx2_qknorm_split_rope", "_jit_ltx2_qknorm_split_rope_module", (), (8, 6)),
    ("flliper.jit_kernel.diffusion.norm_scale_shift_native", "_jit_norm_scale_shift_module", (), (12, 0)),
]


class _Stop(Exception):
    pass


def capture_compile_plan(modname, fname, args, arch):
    """Run the REAL wrapper function with ``tvm_ffi.cpp.load_inline`` replaced by a recorder; -> the kwargs ``load_jit`` handed to the compiler."""
    import torch
    import tvm_ffi.cpp as cpp
    from flliper.jit_kernel import utils as ju

    got = {}

    def fake_load_inline(name, **kw):
        got["name"] = name
        got.update(kw)
        raise _Stop()

    mod = importlib.import_module(modname)
    fn = getattr(mod, fname)
    fn = getattr(fn, "__wrapped__", fn)
    args = tuple(getattr(torch, a) if isinstance(a, str) else a for a in args)
    with tempfile.TemporaryDirectory(prefix="f0k-tvm-") as cache, mock.patch.dict(os.environ, {"TVM_FFI_CACHE_DIR": cache}), \
            mock.patch.object(cpp, "load_inline", fake_load_inline), ju.override_jit_cuda_arch(*arch):
        with contextlib.suppress(_Stop):
            fn(*args)
    return got


@unittest.skipUnless(_nvcc() and _tvm_includes(), "needs nvcc (compile only, no device) and tvm_ffi headers")
class TestJitWrappersCompile(CustomTestCase):
    def _compile(self, modname, fname, args, arch):
        plan = capture_compile_plan(modname, fname, args, arch)
        self.assertIn("cuda_sources", plan, "load_jit never reached the compiler for %s.%s" % (modname, fname))
        major, minor = arch
        flags = [f if not re.fullmatch(r"-O\d", f) else "-O0" for f in plan.get("extra_cuda_cflags", [])]
        incs = _tvm_includes() + list(plan.get("extra_include_paths", []))
        with tempfile.TemporaryDirectory(prefix="f0k-nvcc-") as d:
            cu = pathlib.Path(d) / "cuda.cu"
            cu.write_text("\n".join(plan["cuda_sources"]) + "\n", encoding="utf-8")
            cmd = [_nvcc(), "-c", str(cu), "-o", str(pathlib.Path(d) / "cuda.o"), "-Xcompiler", "-fPIC", "-O0", f"-arch=sm_{major}{minor}"]
            cmd += flags + ["-I" + str(pathlib.Path(i).resolve()) for i in incs]
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        return p, plan

    def _check(self, i):
        modname, fname, args, arch = CASES[i]
        p, plan = self._compile(modname, fname, args, arch)
        self.assertEqual(p.returncode, 0, "nvcc rc=%d for %s.%s\n%s" % (p.returncode, modname, fname, (p.stdout + p.stderr)[-1800:]))

    def test_qsa_indexer(self):
        self._check(0)

    def test_fast_topk_512(self):
        self._check(1)

    def test_hc_combine(self):
        self._check(2)

    def test_grouped_gemma_rmsnorm(self):
        self._check(3)

    def test_diffusion_timestep_embedding(self):
        self._check(4)

    def test_diffusion_residual_gate_add(self):
        self._check(5)

    def test_diffusion_causal_conv3d_cat_pad(self):
        self._check(6)

    def test_diffusion_ltx2_qknorm_split_rope(self):
        self._check(7)

    def test_diffusion_norm_scale_shift(self):
        self._check(8)

    def test_a_wrong_namespace_is_rejected_by_the_same_gate(self):
        """The gate has teeth: the exact RC1 defect (wrapper namespace = the package name) must fail nvcc."""
        from flliper.jit_kernel import utils as ju
        modname, fname, args, arch = CASES[1]
        real = ju.load_jit
        import flliper.kernels.jit.utils as kj

        def wrong(*a, **k):
            k["wrap_namespace"] = NEW
            return real(*a, **k)

        with mock.patch.object(kj, "_utils", mock.Mock(load_jit=wrong, **{n: getattr(ju, n) for n in ("cache_once", "is_arch_support_pdl", "make_cpp_args")})):
            p, _plan = self._compile(modname, fname, args, arch)
        self.assertNotEqual(p.returncode, 0)


if __name__ == "__main__":
    unittest.main()
