"""FLA l2norm with the row count as a RUN-TIME bound (SGLANG_FLA_L2NORM_RUNTIME_T).

Measured reason (weg2xsn423, group P): ``l2norm_fwd_kernel`` takes ``T`` (tokens x
heads) and ``NB`` as ``tl.constexpr``, so every new token count is a new Triton
kernel -- compiled or read from the disk cache and loaded on the scheduler
thread inside the launch: 63 cold loads of that one kernel, 3.20 s of load
windows, py-spy PP0 134 of 2263 samples in its ``_init_handles``. Every other
Triton kernel loaded once per rank and variant. Upstream FLA passes T at run
time (``@triton.jit(do_not_specialize=["T"])``).

Pinned here: the run-time variant has T as a plain, unspecialised argument and
the same tile constants; it is off unless the variable is set (and
--p-host-overlap sets it for group P); and, through Triton's interpreter in a
child process (see test_gdn_chunk_h_pad_sentinel_611.py for why a child), it
returns the stock kernel's output BIT FOR BIT for row counts on and off the
tile grid.
"""
import json
import os
import subprocess
import sys
import textwrap
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.layers.attention.fla import l2norm as l2
from sglang.srt.managers import weg2_p_overlap as pov
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-weg2-unit")


def _params(kernel):
    return {p.name: p for p in kernel.params}


class TestKernelShape(unittest.TestCase):
    def test_runtime_variant_takes_t_at_run_time(self):
        p = _params(l2.l2norm_fwd_kernel_rt)
        self.assertFalse(p["T"].is_constexpr)
        for name in ("D", "BT", "BD"):
            self.assertTrue(p[name].is_constexpr, name)
        self.assertNotIn("NB", p)
        # not specialised on its value either (divisibility would still split it)
        self.assertTrue(p["T"].do_not_specialize)

    def test_stock_kernel_is_why(self):
        p = _params(l2.l2norm_fwd_kernel)
        self.assertTrue(p["T"].is_constexpr)
        self.assertTrue(p["NB"].is_constexpr)


class TestSwitch(unittest.TestCase):
    def test_off_unless_set_and_carried_by_p_host_overlap(self):
        old = os.environ.pop(l2.L2NORM_RUNTIME_T_ENV, None)
        try:
            self.assertFalse(l2.l2norm_runtime_t_on())
            os.environ[l2.L2NORM_RUNTIME_T_ENV] = "1"
            self.assertTrue(l2.l2norm_runtime_t_on())
        finally:
            os.environ.pop(l2.L2NORM_RUNTIME_T_ENV, None)
            if old is not None:
                os.environ[l2.L2NORM_RUNTIME_T_ENV] = old
        self.assertEqual(pov.L2NORM_RUNTIME_T_ENV, l2.L2NORM_RUNTIME_T_ENV)
        self.assertEqual(pov.launcher_env_p_host_overlap().get(l2.L2NORM_RUNTIME_T_ENV), "1")


_WORKER = textwrap.dedent("""
    import json, os
    os.environ["TRITON_INTERPRET"] = "1"
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "99")
    import torch
    from sglang.srt.layers.attention.fla import l2norm as l2

    out = {"kernel_type": type(l2.l2norm_fwd_kernel_rt).__name__, "cases": []}
    g = torch.Generator().manual_seed(3)
    for tokens, heads, d in ((1, 1, 128), (3, 2, 128), (8, 2, 128), (9, 2, 128),
                             (37, 4, 128), (64, 1, 64), (5, 3, 100)):
        x = torch.randn(tokens, heads, d, generator=g, dtype=torch.float32)
        os.environ.pop(l2.L2NORM_RUNTIME_T_ENV, None)
        stock = l2.l2norm_fwd(x.clone())
        os.environ[l2.L2NORM_RUNTIME_T_ENV] = "1"
        rt = l2.l2norm_fwd(x.clone())
        ref = x / torch.sqrt((x * x).sum(-1, keepdim=True) + 1e-6)
        out["cases"].append({
            "rows": tokens * heads, "d": d,
            "bit_equal": bool(torch.equal(stock, rt)),
            "shape_equal": list(stock.shape) == list(rt.shape),
            "ref_max_err": float((rt - ref).abs().max()),
        })
    print("RESULT " + json.dumps(out))
""")


class TestInterpreterEquivalence(unittest.TestCase):
    def test_runtime_variant_equals_the_stock_kernel_bit_for_bit(self):
        env = dict(os.environ)
        env["TRITON_INTERPRET"] = "1"
        env.pop(l2.L2NORM_RUNTIME_T_ENV, None)
        proc = subprocess.run([sys.executable, "-c", _WORKER], env=env,
                              capture_output=True, text=True, timeout=600)
        line = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
        self.assertTrue(line, proc.stdout[-2000:] + proc.stderr[-4000:])
        out = json.loads(line[-1][len("RESULT "):])
        # CPU tensors through the REAL kernels: only the interpreter can run them
        self.assertIn("Interpreted", out["kernel_type"])
        self.assertEqual(len(out["cases"]), 7)
        for c in out["cases"]:
            self.assertTrue(c["shape_equal"], c)
            self.assertTrue(c["bit_equal"], c)
            self.assertLess(c["ref_max_err"], 1e-5, c)


if __name__ == "__main__":
    unittest.main()
