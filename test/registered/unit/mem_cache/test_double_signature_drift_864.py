"""#864: a test double must be able to accept the call it stands in for.

WHY THIS TEST EXISTS, AND WHY IT IS CPU-ONLY ON PURPOSE.

``test_unified_radix_cache_unittest.py`` replaces ``torch.distributed.all_reduce``
with a local stub so the TP branch of ``check_prefetch_progress`` can run
without ranks. The production call site later grew ``async_op=True``. The stub
did not. From that moment sixteen tests -- ``test_tp_swa_prefetch_adopted_when_peer_present``
and ``test_tp_swa_prefetch_dropped_when_peer_misses``, each over eight SWA
parametrisations -- could only fail, and they failed with a ``TypeError`` from
inside ``mock``'s dispatch, before a single assertion of their own executed.

The reason it survived is the interesting part, and it is why the guard lives
in a separate CPU file rather than next to the tests it protects:

* those sixteen tests need a real accelerator, so ``conftest.py`` turns them
  into skips on every desk machine;
* the desk gate is where a red would be noticed;
* therefore the only machine that could see the failure was the one where
  nobody was looking, and the machine where everybody looks could not see it.

A GPU-only red is invisible to a CPU gate. **Signature compatibility, though,
is decidable by reading -- no device, no ranks, no import of the test module
that would need them.** This file does the reading. It runs in the CPU lane,
so the next drift of this kind is caught by the gate that people actually
watch, on the day it is introduced.

Scope, stated honestly: this proves the double can ACCEPT the call. It does not
prove the double behaves like the real thing. That is the assertions' job, and
those still need a card.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

import ast
import inspect
import re
import unittest
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_TEST_FILE = _HERE / "test_unified_radix_cache_unittest.py"
_PROD_FILE = (
    _HERE.parents[3] / "python" / "sglang" / "srt" / "mem_cache" / "hiradix_cache.py"
)


def _signature_of(path: Path, func_name: str) -> inspect.Signature:
    """Build a Signature for a function we PARSE rather than import.

    Importing the test module would resolve a device and skip; parsing it costs
    nothing and works anywhere.
    """
    P = inspect.Parameter
    tree = ast.parse(path.read_text(errors="replace"))
    fn = next(
        (
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == func_name
        ),
        None,
    )
    if fn is None:
        raise AssertionError(f"{func_name!r} not found in {path.name}")
    a = fn.args
    params = [P(x.arg, P.POSITIONAL_ONLY) for x in a.posonlyargs]
    npos, ndef = len(a.args), len(a.defaults)
    for i, arg in enumerate(a.args):
        params.append(
            P(
                arg.arg,
                P.POSITIONAL_OR_KEYWORD,
                default=P.empty if i < npos - ndef else None,
            )
        )
    if a.vararg:
        params.append(P(a.vararg.arg, P.VAR_POSITIONAL))
    for arg, d in zip(a.kwonlyargs, a.kw_defaults):
        params.append(P(arg.arg, P.KEYWORD_ONLY, default=P.empty if d is None else None))
    if a.kwarg:
        params.append(P(a.kwarg.arg, P.VAR_KEYWORD))
    return inspect.Signature(params)


def _all_reduce_keywords(path: Path) -> set[str]:
    """Every keyword any ``torch.distributed.all_reduce(...)`` call site passes.

    Read from the SOURCE so the guard tracks the production code rather than a
    copy of it that can drift in turn.
    """
    tree = ast.parse(path.read_text(errors="replace"))
    kws: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if isinstance(f, ast.Attribute) and f.attr == "all_reduce":
            owner = ast.unparse(f.value)
            if re.search(r"(?:^|\.)distributed$|^dist$", owner):
                kws.update(k.arg for k in node.keywords if k.arg)
    return kws


class TheAllReduceDoubleMustAcceptTheProductionCall(unittest.TestCase):
    def test_the_production_call_sites_still_pass_async_op(self):
        # If this ever fails, production stopped passing async_op and the guard
        # below is testing a call that no longer exists -- fix the guard, do not
        # delete it.
        kws = _all_reduce_keywords(_PROD_FILE)
        self.assertIn(
            "async_op",
            kws,
            f"hiradix_cache.py all_reduce keywords are {sorted(kws)}",
        )

    def test_the_double_can_bind_every_production_keyword(self):
        sig = _signature_of(_TEST_FILE, "fake")
        kws = _all_reduce_keywords(_PROD_FILE)
        try:
            sig.bind(object(), **{k: object() for k in kws})
        except TypeError as exc:
            self.fail(
                f"the all_reduce double {sig} cannot accept the production call "
                f"(keywords {sorted(kws)}): {exc}. Every test using it fails from "
                f"inside mock's dispatch before its own assertions run -- and only "
                f"on a machine with a card, because the rest skip."
            )


if __name__ == "__main__":
    unittest.main()
