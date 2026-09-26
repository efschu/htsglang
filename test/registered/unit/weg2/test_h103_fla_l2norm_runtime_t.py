"""H103: FLA l2norm takes the row count at RUN time on both NF groups, and the
one remaining specialization is loaded at boot.

Measured reason (rc9p, dkrnfbar1rc9p09261011, tree d1c7094ba6): the stock
``l2norm_fwd_kernel`` has ``T`` (tokens x heads) and ``NB`` as ``tl.constexpr``,
so every new extend length is a new Triton kernel -- D log: 15 cold-load
windows of that one kernel after READY (all TP0, D extends, 16.7 s, one of
12.7 s), P log: 78. The 27B strand's run-time variant (6b24aa60da, taken
verbatim) was switched on only for 27B P.

Pinned here, without a GPU:
  * Triton's OWN binder and cache key (``create_function_from_signature`` +
    ``compute_cache_key`` for an sm_120 and an sm_86 target; only the launch is
    mocked) see ONE specialization over many extend lengths with the switch on
    -- and one per length with the stock kernel (red on d1c7094ba6);
  * the boot prewarm's launch has exactly the serving key (so no extend after
    READY loads the kernel), skips by name where it has nothing to do, never
    raises, and sits before the #603b barrier;
  * build_env sets the switch for P and D; the operator's value wins;
  * through Triton's interpreter in a child process: run-time variant ==
    stock kernel BIT FOR BIT.
"""
import ast
import json
import os
import pathlib
import subprocess
import sys
import textwrap
from unittest import mock

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

from sglang.srt.layers.attention.fla import l2norm as l2  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=60, suite="stage-a-weg2-unit")

ENV = "SGLANG_FLA_L2NORM_RUNTIME_T"
# NF GDN: linear_key_head_dim 128, 16 key heads (Qwen3.8-Flash-Next config)
D, HEADS = 128, 16
# every tokens 1..300, the rc9p extend sizes around the 12.7 s window
# (3585 new rows on weg2-22-25) and long P chunks
LENGTHS = list(range(1, 301)) + [511, 512, 513, 1423, 3585, 4096, 16384]
REPO = pathlib.Path(__file__).resolve().parents[4]


def _triton_keys(run_calls, arch):
    """Recompute Triton's cache key for every recorded launch with Triton's
    own binder for a named target (no device needed)."""
    from triton.backends.compiler import GPUTarget
    from triton.compiler.compiler import make_backend
    from triton.runtime import jit as tjit

    backend = make_backend(GPUTarget("cuda", arch, 32))
    keys = {}
    for fn, args, kwargs in run_calls:
        binder = tjit.create_function_from_signature(fn.signature, fn.params, backend)
        _, spec, opts = binder(*args, **kwargs)
        keys.setdefault(fn.fn.__name__, set()).add(tjit.compute_cache_key({}, spec, opts))
    return keys


def _record_launches(fn):
    """Run ``fn`` with every JITFunction launch recorded instead of executed."""
    from triton.runtime import jit as tjit

    calls = []

    def fake_run(self, *args, grid, warmup, **kwargs):
        calls.append((self, args, dict(kwargs)))
        return None

    with mock.patch.object(tjit.JITFunction, "run", fake_run):
        fn()
    return calls


def _serve(lengths, dtype=torch.bfloat16):
    # ChunkGatedDeltaRuleFunction.forward: l2norm_fwd(q), l2norm_fwd(k),
    # q/k [1, tokens, heads, D]
    for tokens in lengths:
        q = torch.empty((1, tokens, HEADS, D), dtype=dtype)
        l2.l2norm_fwd(q)


@pytest.fixture
def runtime_t_on(monkeypatch):
    monkeypatch.setenv(ENV, "1")


# -- the defect and the fix, in Triton's own cache key ---------------------------


@pytest.mark.parametrize("arch", [120, 86])
def test_one_specialization_over_every_extend_length(runtime_t_on, arch):
    keys = _triton_keys(_record_launches(lambda: _serve(LENGTHS)), arch)
    n = sum(len(v) for v in keys.values())
    assert n == 1, {k: len(v) for k, v in keys.items()}
    assert set(keys) == {"l2norm_fwd_kernel_rt"}


def test_stock_kernel_is_why(monkeypatch):
    # the witness: switch off -> one kernel per row count
    monkeypatch.setenv(ENV, "0")
    keys = _triton_keys(_record_launches(lambda: _serve(LENGTHS)), 120)
    assert set(keys) == {"l2norm_fwd_kernel"}
    assert len(keys["l2norm_fwd_kernel"]) == len(set(LENGTHS))


def test_runtime_variant_takes_t_at_run_time():
    p = {k.name: k for k in l2.l2norm_fwd_kernel_rt.params}
    assert not p["T"].is_constexpr and p["T"].do_not_specialize
    assert all(p[n].is_constexpr for n in ("D", "BT", "BD"))
    assert "NB" not in p
    assert l2.L2NORM_RUNTIME_T_ENV == ENV


# -- the boot prewarm -----------------------------------------------------------


class _Cfg:
    linear_key_head_dim = D


def test_prewarm_launch_has_the_serving_key(runtime_t_on):
    from sglang.srt.layers.attention.fla import l2norm_prewarm as pw

    warm = _record_launches(
        lambda: pw.prewarm_l2norm(head_dims=pw.gdn_l2norm_head_dims(_Cfg), dtype=torch.bfloat16,
                                  device="cpu"))
    serve = _record_launches(lambda: _serve(LENGTHS))
    for arch in (120, 86):
        kw = _triton_keys(warm, arch)
        ks = _triton_keys(serve, arch)
        assert kw == ks, (arch, kw, ks)


def test_prewarm_launches_once_per_head_dim_up_to_512():
    from sglang.srt.layers.attention.fla import l2norm_prewarm as pw

    seen = []
    res = pw.prewarm_l2norm(head_dims=(128, 64, 1024), dtype=torch.bfloat16, device="cpu",
                            launch=lambda x: seen.append((tuple(x.shape), x.dtype)))
    assert seen == [((16, 128), torch.bfloat16), ((16, 64), torch.bfloat16)]
    assert res.loaded == 2 and res.head_dims == (128, 64)
    assert "H103 FLA-L2NORM-PREWARM kernel=l2norm_fwd_kernel_rt" in res.line()


def test_prewarm_skips_by_name(monkeypatch):
    from sglang.srt import rank_role
    from sglang.srt.layers.attention.fla import l2norm_prewarm as pw

    boom = mock.Mock(side_effect=AssertionError("launched"))
    monkeypatch.setattr(pw, "prewarm_l2norm", boom)
    monkeypatch.setenv(ENV, "0")
    assert "stock kernel" in pw.run_boot_prewarm(hf_text_config=_Cfg, dtype=torch.bfloat16,
                                                 device="cpu").skipped
    monkeypatch.setenv(ENV, "1")
    monkeypatch.setattr(rank_role, "this_rank_is_form_a_worker", lambda: True)
    assert "Form-A worker" in pw.run_boot_prewarm(hf_text_config=_Cfg, dtype=torch.bfloat16,
                                                  device="cpu").skipped
    monkeypatch.setattr(rank_role, "this_rank_is_form_a_worker", lambda: False)
    assert "no GDN" in pw.run_boot_prewarm(hf_text_config=object(), dtype=torch.bfloat16,
                                           device="cpu").skipped
    boom.assert_not_called()


def test_prewarm_never_raises(monkeypatch, runtime_t_on):
    from sglang.srt import rank_role
    from sglang.srt.layers.attention.fla import l2norm_prewarm as pw

    monkeypatch.setattr(rank_role, "this_rank_is_form_a_worker", lambda: False)
    monkeypatch.setattr(pw, "prewarm_l2norm", mock.Mock(side_effect=RuntimeError("no card")))
    assert pw.run_boot_prewarm(hf_text_config=_Cfg, dtype=torch.bfloat16, device="cpu") is None


def test_the_scheduler_prewarms_before_the_sampling_barrier():
    src = (REPO / "python/sglang/srt/managers/scheduler.py").read_text()
    tree = ast.parse(src)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Scheduler")
    meths = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    delegate = ast.unparse(meths["warm_fla_l2norm"])
    assert "run_boot_prewarm(" in delegate
    caller = [m for m in meths.values()
              if "self.warm_sampling_backend()" in ast.unparse(m) and m.name != "warm_sampling_backend"]
    assert len(caller) == 1
    body = ast.unparse(caller[0])
    assert body.index("self.warm_fla_l2norm()") < body.index("self.warm_sampling_backend()")


# -- the launcher carries the switch to both groups -----------------------------


@pytest.fixture
def launcher_mod(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.delenv("SGLANG_WEG2_GROUP", raising=False)
    monkeypatch.delenv("SGLANG_MOE_EXPERT_MAP", raising=False)
    from sglang.srt.weg2 import launcher

    return launcher


def _build(launcher, **kw):
    return launcher.build_env(tree="/tmp/t", venv="/tmp/v", cvd="0", store_dir="/tmp/s",
                              debug_hold=False, tag="h103", **kw)


def test_build_env_switches_it_on_for_both_groups(launcher_mod):
    assert launcher_mod.L2NORM_RUNTIME_T_ENV == l2.L2NORM_RUNTIME_T_ENV
    for group in ("P", "D"):
        assert _build(launcher_mod, group=group).get(ENV) == "1", group


def test_the_operator_value_wins(launcher_mod, monkeypatch):
    monkeypatch.setenv(ENV, "0")
    assert _build(launcher_mod, group="D").get(ENV) == "0"
    monkeypatch.delenv(ENV)
    assert _build(launcher_mod, group="P", group_env_extra={ENV: "0"}).get(ENV) == "0"


# -- numerics: the interpreter runs both real kernels on the CPU ----------------

_WORKER = textwrap.dedent("""
    import json, os
    os.environ["TRITON_INTERPRET"] = "1"
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "99")
    import torch
    from sglang.srt.layers.attention.fla import l2norm as l2

    out = {"kernel_type": type(l2.l2norm_fwd_kernel_rt).__name__, "cases": []}
    g = torch.Generator().manual_seed(103)
    cases = [(1, 1, 128), (1, 16, 128), (3, 16, 128), (7, 16, 128), (37, 16, 128),
             (129, 16, 128), (5, 3, 100), (64, 1, 64)]
    for dtype in (torch.float32, torch.bfloat16):
        for tokens, heads, d in cases:
            x = (torch.randn(tokens, heads, d, generator=g) * 3).to(dtype)
            os.environ[l2.L2NORM_RUNTIME_T_ENV] = "0"
            stock = l2.l2norm_fwd(x.clone())
            os.environ[l2.L2NORM_RUNTIME_T_ENV] = "1"
            rt = l2.l2norm_fwd(x.clone())
            xf = x.float()
            ref = xf / torch.sqrt((xf * xf).sum(-1, keepdim=True) + 1e-6)
            out["cases"].append({
                "rows": tokens * heads, "d": d, "dtype": str(dtype),
                "bit_equal": bool(torch.equal(stock.view(torch.int16) if dtype == torch.bfloat16 else stock,
                                              rt.view(torch.int16) if dtype == torch.bfloat16 else rt)),
                "shape_equal": list(stock.shape) == list(rt.shape) and stock.dtype == rt.dtype,
                "ref_max_err": float((rt.float() - ref).abs().max()),
            })
    print("RESULT " + json.dumps(out))
""")


def test_runtime_variant_equals_the_stock_kernel_bit_for_bit():
    env = dict(os.environ)
    env["TRITON_INTERPRET"] = "1"
    env.pop(ENV, None)
    proc = subprocess.run([sys.executable, "-c", _WORKER], env=env, capture_output=True,
                          text=True, timeout=900)
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    assert line, proc.stdout[-2000:] + proc.stderr[-4000:]
    out = json.loads(line[-1][len("RESULT "):])
    assert "Interpreted" in out["kernel_type"]
    assert len(out["cases"]) == 16
    for c in out["cases"]:
        assert c["shape_equal"], c
        assert c["bit_equal"], c
        assert c["ref_max_err"] < (1e-5 if "float32" in c["dtype"] else 1e-2), c
