# SPDX-License-Identifier: Apache-2.0
"""``masked_set_kv_buffer_kernel`` serves every extend length with ONE compile.

y8a D (``..._10021818_d11000a830_1002_181859.D.log``) logged 711
``triton cold module load: masked_set_kv_buffer_kernel`` lines, 91 of them on
TP0 in ~15 min. Root: ``N`` -- the number of rows written this forward -- was a
``tl.constexpr``, so every distinct extend length was its own specialization,
and every first sight of a length loaded a module on the hot path and opened a
barlink JIT cold-build window there. ``N`` only bounds ``pid`` (the grid is
already ``(N,)``); it never sizes a ``tl.arange`` or a block, so it is a
runtime scalar now, and ``do_not_specialize`` also drops Triton's ``==1`` /
``%16`` integer specialization.

* signature -- ``N`` is not a constexpr and is listed in ``do_not_specialize``;
  ``H``, ``D``, ``CHUNK`` and the strides stay constexpr (they size the loop
  and ``tl.arange``);
* numerics -- under ``TRITON_INTERPRET=1`` (CPU) the kernel writes exactly the
  masked rows a torch reference writes, for N in {1, 3, 16, 17, 31}. Runs in a
  subprocess because the interpreter switch must be set before ``triton`` is
  imported.

    python -m pytest test/registered/unit/mem_cache/test_masked_set_kv_runtime_n_1002.py -v
"""

import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

_PYTHON_ROOT = str(Path(__file__).resolve().parents[4] / "python")


class TestNIsARuntimeScalar(unittest.TestCase):
    def setUp(self):
        from sglang.srt.mem_cache import memory_pool

        self.fn = memory_pool.masked_set_kv_buffer_kernel

    def _constexpr_names(self):
        return {self.fn.arg_names[i] for i in self.fn.constexprs}

    def test_n_is_not_constexpr(self):
        self.assertIn("N", self.fn.arg_names)
        self.assertNotIn("N", self._constexpr_names())

    def test_n_is_not_integer_specialized(self):
        self.assertIn("N", self.fn.do_not_specialize)

    def test_shape_args_stay_constexpr(self):
        names = self._constexpr_names()
        for name in (
            "H", "D", "CHUNK", "k_stride_B", "k_stride_H", "v_stride_B", "v_stride_H"
        ):
            self.assertIn(name, names, name)


_INTERP_SCRIPT = textwrap.dedent(
    """
    import os, sys
    os.environ["TRITON_INTERPRET"] = "1"
    sys.path.insert(0, sys.argv[1])
    import torch
    from sglang.srt.mem_cache import memory_pool as mp

    fn = mp.masked_set_kv_buffer_kernel
    H, D, ROWS = 2, 8, 64
    bad = []
    for N in (1, 3, 16, 17, 31):
        g = torch.Generator().manual_seed(N)
        kbuf = torch.zeros(ROWS, H, D)
        vbuf = torch.zeros(ROWS, H, D)
        k = torch.randn(N, H, D, generator=g)
        v = torch.randn(N, H, D, generator=g)
        loc = torch.arange(N, dtype=torch.int64) * 2 + 1
        mask = torch.arange(N) % 2 == 0
        fn[(N,)](
            k, v, kbuf, vbuf, loc, mask, ROWS, N, H, D, 128,
            k.stride(0), k.stride(1), v.stride(0), v.stride(1),
        )
        rk = torch.zeros(ROWS, H, D)
        rv = torch.zeros(ROWS, H, D)
        rk[loc[mask]] = k[mask]
        rv[loc[mask]] = v[mask]
        if not (torch.equal(kbuf, rk) and torch.equal(vbuf, rv)):
            bad.append(N)
    print("INTERP-OK" if not bad else "INTERP-BAD %s" % bad)
    """
)


class TestMaskedWriteMatchesReferenceInInterpreter(unittest.TestCase):
    def test_masked_rows_written_for_several_n(self):
        env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="")
        proc = subprocess.run(
            [sys.executable, "-c", _INTERP_SCRIPT, _PYTHON_ROOT],
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
        self.assertIn("INTERP-OK", proc.stdout, (proc.stdout + proc.stderr)[-4000:])


if __name__ == "__main__":
    unittest.main()
