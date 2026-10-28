"""27b-xkv-1028 (port of NF's formhp-hcdfastfix-1028 helper, extended by ``debug``): compile a Triton kernel for a GPU target WITHOUT a GPU, from the arguments its real wrapper passes.

Why: a CPU test under TRITON_INTERPRET=1 runs the kernel as Python and never meets the compiler's rules -- formb164 (20:41:27 TP0) failed the
DFAST variant of the fused HC read with ``NameError('NGK is not defined')`` (a constexpr bound inside a static ``if`` block, invisible to the
compiler, fine for the interpreter) after its interpreter tests had passed. This helper runs the wrapper with the kernel's launch replaced by a
recorder, then hands each recorded launch to ``triton.compile`` (frontend, TTIR/TTGIR, LLVM, PTX, ptxas) for e.g. sm_120: every variant the
wrapper can launch is compiled the way the card would compile it.

Use (in a subprocess WITHOUT TRITON_INTERPRET, CUDA_VISIBLE_DEVICES=""): ``with capture(module, "kernel_name") as rec: wrapper(...)``, then
``compile_all(rec)`` -> a list of (variant, error or None).
"""

from __future__ import annotations

import contextlib
import inspect
from typing import Any, Dict, List, Optional, Tuple

import torch

_DTYPES = {
    torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32", torch.float64: "fp64",
    torch.int8: "i8", torch.int16: "i16", torch.int32: "i32", torch.int64: "i64", torch.uint8: "u8", torch.bool: "i1",
    torch.float8_e4m3fn: "fp8e4nv", torch.float8_e5m2: "fp8e5",
}


class Launch:
    __slots__ = ("fn", "args", "kwargs")

    def __init__(self, fn, args, kwargs):
        self.fn, self.args, self.kwargs = fn, args, kwargs


@contextlib.contextmanager
def capture(module, name: str):
    """Replace ``module.<name>`` (a ``@triton.jit`` function) by a recorder for the duration: ``kernel[grid](*args, **kw)`` appends a
    ``Launch`` and returns without running anything. Yields the list of launches."""
    fn = getattr(module, name)
    rec: List[Launch] = []

    class _Rec:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                rec.append(Launch(fn, args, kwargs))
            return launch

    setattr(module, name, _Rec())
    try:
        yield rec
    finally:
        setattr(module, name, fn)


def _signature(launch: Launch) -> Tuple[Dict[str, str], Dict[str, Any], Dict[tuple, list], Dict[str, Any], str]:
    fn = launch.fn
    params = fn.params
    names = [p.name for p in params]
    vals: Dict[str, Any] = dict(zip(names, launch.args))
    options: Dict[str, Any] = {}
    for k, v in launch.kwargs.items():
        if k in names:
            vals[k] = v
        elif k in ("num_warps", "num_stages", "num_ctas", "maxnreg", "debug"):
            # ``debug`` is a compile option: it decides whether ``tl.device_assert`` lowers to a real device assert (the masked KV
            # writers pass debug=kv_bound_check_enabled(), default True) -- the production variant must be the one compiled here
            options[k] = v
        # launch-only keywords (launch_pdl, ...) do not change the compiled code
    sig: Dict[str, str] = {}
    cx: Dict[str, Any] = {}
    attrs: Dict[tuple, list] = {}
    for i, p in enumerate(params):
        if p.name not in vals:
            v = p.default
            if v is inspect.Parameter.empty:
                raise TypeError(f"{fn.__name__}: no value for {p.name}")
        else:
            v = vals[p.name]
        if p.is_constexpr:
            sig[p.name] = "constexpr"
            cx[p.name] = v
        elif v is None:
            sig[p.name] = "constexpr"
            cx[p.name] = None
        elif isinstance(v, torch.Tensor):
            sig[p.name] = "*" + _DTYPES[v.dtype]
            if v.data_ptr() % 16 == 0:
                attrs[(i,)] = [["tt.divisibility", 16]]
        elif isinstance(v, bool):
            sig[p.name] = "i1"
        elif isinstance(v, int):
            sig[p.name] = "i32" if -(2**31) <= v < 2**31 else "i64"
            if v % 16 == 0 and v != 0:
                attrs[(i,)] = [["tt.divisibility", 16]]
        elif isinstance(v, float):
            sig[p.name] = "fp32"
        else:
            raise TypeError(f"{fn.__name__}: unsupported argument {p.name}={type(v).__name__}")
    key = ",".join(f"{k}={cx[k]}" for k in sorted(cx) if cx[k] is not None)
    return sig, cx, attrs, options, key


def compile_all(launches: List[Launch], cc: int = 120, warp_size: int = 32) -> List[Tuple[str, Optional[str]]]:
    """``triton.compile`` every distinct (signature, constexprs, options) of ``launches`` for ``cuda:cc``. Returns (variant key, None or the
    error text) per distinct variant."""
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    target = GPUTarget("cuda", cc, warp_size)
    seen = set()
    out: List[Tuple[str, Optional[str]]] = []
    for ln in launches:
        sig, cx, attrs, options, key = _signature(ln)
        full = (ln.fn.__name__, tuple(sorted(sig.items())), key, tuple(sorted(options.items())))
        if full in seen:
            continue
        seen.add(full)
        label = f"{ln.fn.__name__}[{key}; {','.join(f'{k}={v}' for k, v in sorted(options.items()))}]"
        try:
            triton.compile(ASTSource(fn=ln.fn, signature=sig, constexprs=cx, attrs=attrs), target=target, options=options)
            out.append((label, None))
        except Exception as exc:  # noqa: BLE001 -- the error IS the result
            out.append((label, f"{type(exc).__name__}: {str(exc)[:400]}"))
    return out
