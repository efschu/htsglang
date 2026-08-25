# SPDX-License-Identifier: Apache-2.0
"""#861b: `x or ()` is safe for a list and a landmine for a tensor.

THE SPECIMEN, W37-B, 2026-08-25, first real burst after nine clean flips::

    RuntimeError: Boolean value of Tensor with more than one value is ambiguous
      phase_flip_draft_bootstrap.py:846
      n = len(getattr(req, "prefix_indices", ()) or ())

`Req.prefix_indices` is a torch tensor of cached slot ids. The `or` calls
`bool()` on it, and torch refuses that for anything with more than one element.

WHY IT SURVIVED NINE FLIPS, and this is the part worth keeping: the expression
is FINE for an empty prefix (`bool` of a 0-element tensor is False), FINE for a
ONE-token prefix (`bool` of a 1-element tensor is defined), and fatal from two
tokens up. The 1-token health checks that gate every flip could not reach it.
Its reachability was inverted from what a reader would assume -- a wider test
was not needed, a test with a REALISTIC PREFIX was.

So the specimen test uses a multi-element tensor, and the boundary cases are
pinned beside it, because "0 and 1 pass while 2 raises" is the whole shape of
the defect and a test that only checked the crash would not record it.

Four pins:
  1. THE SPECIMEN: a re-admitted request with a multi-element tensor prefix
     goes through `arm_draft_cold_for_admission` without raising, and its
     prefix draft rows are scrubbed.
  2. THE BOUNDARY: 0 / 1 / many elements, 0-d, empty, list, tuple, None.
  3. ONE DEFINITION: both call sites go through `prefix_len`, so the two
     spellings that drifted apart cannot drift again.
  4. THE CLASS, not just the instance: an `ast` sweep refusing boolean-context
     use of any tensor-bearing attribute in the modules #861 touched.
"""

import ast
import inspect
import types

import pytest
import torch

from sglang.srt.managers.phase_flip_draft_bootstrap import (
    COLD_ARMED_ATTR,
    arm_draft_cold_for_admission,
    draft_cold_reason,
    prefix_len,
    rounds_owed,
)
from sglang.srt.managers.phase_purity import SEAM_READMIT_ATTR

N_SLOTS = 64


class FakeKVPool:
    def __init__(self):
        self.layer_num = 1
        self.start_layer = 0
        self._k = torch.full((N_SLOTS, 4), 7.0)
        self._v = torch.full((N_SLOTS, 4), 9.0)

    def get_key_buffer(self, layer_id):
        return self._k

    def get_value_buffer(self, layer_id):
        return self._v


def make_scheduler(pool, tier_armed):
    req_to_token = torch.arange(N_SLOTS, dtype=torch.int64).reshape(4, N_SLOTS // 4)
    controller = types.SimpleNamespace(draft_tier_armed=lambda direction: tier_armed)
    return types.SimpleNamespace(
        draft_worker=types.SimpleNamespace(
            draft_worker=types.SimpleNamespace(
                draft_runner=types.SimpleNamespace(token_to_kv_pool=pool)
            )
        ),
        req_to_token_pool=types.SimpleNamespace(req_to_token=req_to_token),
        tree_cache=types.SimpleNamespace(cache_controller=controller),
    )


def make_req(rid, req_pool_idx, prefix, seam=True):
    """`prefix` is passed through UNCHANGED -- the point is the type."""
    req = types.SimpleNamespace(
        rid=rid, req_pool_idx=req_pool_idx, prefix_indices=prefix
    )
    if seam:
        setattr(req, SEAM_READMIT_ATTR, 3)
    return req


def batch_of(*reqs):
    return types.SimpleNamespace(reqs=list(reqs))


# ---------------------------------------------------------------- pin 1


def test_specimen_multi_element_tensor_prefix_does_not_raise():
    """RED on c0b3f7c0a3: RuntimeError, Boolean value of Tensor with more than
    one value is ambiguous."""
    pool = FakeKVPool()
    sched = make_scheduler(pool, tier_armed=True)  # armed: the SEAM stamp is the trigger
    prefix = torch.arange(5, dtype=torch.int64)  # a REAL prefix, not a health check
    req = make_req("burst", 1, prefix)

    report = arm_draft_cold_for_admission(sched, batch_of(req))

    assert report["cold"] == 1
    assert report["rows"] == 5
    assert rounds_owed(req) == 1
    assert getattr(req, COLD_ARMED_ATTR) is True
    rows = sched.req_to_token_pool.req_to_token[1, :5]
    assert torch.all(pool.get_key_buffer(0)[rows] == 0)


def test_specimen_reaches_the_reason_probe_too():
    """The other call site on the same attribute, through the same helper."""
    sched = make_scheduler(FakeKVPool(), tier_armed=True)
    req = make_req("burst", 1, torch.arange(5, dtype=torch.int64))
    assert "seam re-admission" in draft_cold_reason(sched, req, True)


# ---------------------------------------------------------------- pin 2


@pytest.mark.parametrize(
    "prefix, expected",
    [
        (None, 0),
        (torch.zeros(0, dtype=torch.int64), 0),  # empty: bool() was False -> passed
        (torch.zeros(1, dtype=torch.int64), 1),  # one:   bool() defined  -> passed
        (torch.zeros(2, dtype=torch.int64), 2),  # two:   bool() RAISES   -> the crash
        (torch.arange(53, dtype=torch.int64), 53),
        (torch.tensor(7), 1),  # 0-d: len() raises, numel() answers 1
        ([], 0),
        ([1, 2, 3], 3),
        ((1, 2), 2),
        (object(), 0),  # unmeasurable -> "no prefix", the safe direction
    ],
)
def test_prefix_len_boundary(prefix, expected):
    assert prefix_len(types.SimpleNamespace(prefix_indices=prefix)) == expected


def test_the_defects_true_shape_measured_not_assumed():
    """THE OLD EXPRESSION HAD TWO FAILURE MODES, NOT ONE, and my first reading
    of it was wrong in both directions. Measured on this torch (2.11.0+cu130),
    `bool(tensor)` is:

        0 elements  -> RuntimeError ("with no values is ambiguous")
        1 element   -> the VALUE's truthiness  (so slot id 0 is False!)
        >=2         -> RuntimeError ("with more than one value is ambiguous")

    So `len(pi or ())` on a tensor prefix:

      * 0 elements  -> RuntimeError. UNREACHABLE in practice, and that matters:
        `draft_cold_reason` runs first, computes the length on the SAFE path and
        returns None for <=0, so a request with an empty prefix `continue`s
        before it ever reaches the crash line.
      * 1 element holding a NON-ZERO slot id -> 1. Correct, by luck.
      * 1 element holding slot id 0 -> `bool` is False -> `or ()` -> len 0 ->
        **the scrub is silently skipped**. Not a crash: a silently unscrubbed
        request that then speculates over the previous occupant's draft rows,
        which is the exact silent-garbage class this module exists to refuse.
      * >=2 elements -> RuntimeError. The W37-B boot death.

    The 1-token health checks that gate every flip sat precisely in the one
    band that does not raise -- and, if their slot id was 0, in the band that is
    silently wrong. A wider test was never needed; a test with a REALISTIC
    prefix was.
    """
    old = lambda pi: len(pi or ())  # noqa: E731 - the defect, verbatim

    with pytest.raises(RuntimeError, match="no values is ambiguous"):
        old(torch.zeros(0, dtype=torch.int64))

    assert old(torch.tensor([9], dtype=torch.int64)) == 1  # correct by luck
    assert old(torch.tensor([0], dtype=torch.int64)) == 0  # SILENTLY WRONG

    with pytest.raises(RuntimeError, match="more than one value is ambiguous"):
        old(torch.zeros(2, dtype=torch.int64))

    # And the helper is right in every one of those four bands.
    ns = types.SimpleNamespace
    assert prefix_len(ns(prefix_indices=torch.zeros(0, dtype=torch.int64))) == 0
    assert prefix_len(ns(prefix_indices=torch.tensor([9], dtype=torch.int64))) == 1
    assert prefix_len(ns(prefix_indices=torch.tensor([0], dtype=torch.int64))) == 1
    assert prefix_len(ns(prefix_indices=torch.zeros(2, dtype=torch.int64))) == 2


def test_the_silent_half_is_fixed_end_to_end():
    """A one-token prefix whose slot id is 0 must now be SCRUBBED. Under the old
    expression this request was marked draft-cold and its draft row left holding
    the previous occupant's bytes -- the failure the mark exists to prevent,
    produced by the code that sets the mark."""
    pool = FakeKVPool()
    sched = make_scheduler(pool, tier_armed=True)
    # req_pool_idx 0 -> req_to_token[0, :1] == slot 0
    req = make_req("zero", 0, torch.tensor([0], dtype=torch.int64))
    report = arm_draft_cold_for_admission(sched, batch_of(req))
    assert report["cold"] == 1
    assert report["rows"] == 1, "the one-row prefix must be scrubbed, not skipped"
    assert torch.all(pool.get_key_buffer(0)[0] == 0)


# ---------------------------------------------------------------- pin 3


def test_both_call_sites_go_through_the_one_helper():
    """Two spellings of "how long is this prefix" is one too many: the copy in
    `arm_draft_cold_for_admission` was the one written with `or ()`, and it was
    the one that crashed."""
    import sglang.srt.managers.phase_flip_draft_bootstrap as mod

    for fn in (mod.draft_cold_reason, mod.arm_draft_cold_for_admission):
        src = inspect.getsource(fn)
        assert "prefix_len(" in src, fn.__name__
        assert "prefix_indices" not in src, (
            f"{fn.__name__} reads prefix_indices directly again; the helper "
            f"exists so the length is computed in exactly one place"
        )


# ---------------------------------------------------------------- pin 4


#: Attributes that hold a torch tensor somewhere on these paths. Boolean
#: context on ANY of them -- `or`, `and`, `if x:`, `not x`, `bool(x)` -- is the
#: #861b landmine: defined for 0 and 1 elements, fatal from 2 up.
TENSOR_BEARING = {
    "prefix_indices",
    "input_ids",
    "out_cache_loc",
    "seq_lens",
    "seq_lens_cpu",
    "topk_index",
    "topk_p",
    "hidden_states",
    "bonus_tokens",
    "free_pages",
    "release_pages",
    "host_indices",
    "device_indices",
    "req_to_token",
    "accept_lens",
    "next_token_ids",
    "kv_indices",
}

#: The modules #861 touched. Named rather than discovered: a module this list
#: forgets is a module the sweep silently blesses.
SWEPT_MODULES = (
    "sglang.srt.managers.phase_flip_draft_bootstrap",
    "sglang.srt.mem_cache.kv_cache_builder",
    "sglang.srt.mem_cache.hicache_phase_binding",
    "sglang.srt.managers.cache_controller",
    "sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller",
)


def _tensor_name(node):
    """The tensor-bearing name this expression resolves to, or None.

    Covers the two shapes the idiom takes: `x.attr` and
    `getattr(x, "attr", default)`.
    """
    if isinstance(node, ast.Attribute) and node.attr in TENSOR_BEARING:
        return node.attr
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and len(node.args) >= 2
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value in TENSOR_BEARING
    ):
        return node.args[1].value
    return None


def _boolean_contexts(tree):
    """Every expression evaluated for truthiness, with its line number."""
    for node in ast.walk(tree):
        if isinstance(node, ast.BoolOp):  # `a or b`, `a and b`
            # The LAST operand of an `or`/`and` is returned, not tested.
            for value in node.values[:-1]:
                yield value, node.lineno
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            yield node.operand, node.lineno
        elif isinstance(node, (ast.If, ast.While, ast.IfExp)):
            yield node.test, node.lineno
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "bool"
            and node.args
        ):
            yield node.args[0], node.lineno


def test_no_tensor_bearing_attribute_is_used_for_truthiness():
    """THE CLASS, not the instance. Parsed with `ast` rather than grepped so
    the prose in this file and in the fixed module -- which necessarily quotes
    the defect verbatim -- does not read as the defect."""
    import importlib

    offenders = []
    for name in SWEPT_MODULES:
        module = importlib.import_module(name)
        tree = ast.parse(inspect.getsource(module))
        for expr, lineno in _boolean_contexts(tree):
            attr = _tensor_name(expr)
            if attr is not None:
                offenders.append(f"{name}:{lineno} truthiness on `{attr}`")
    assert not offenders, "\n".join(offenders)


@pytest.mark.parametrize(
    "source, expected",
    [
        # THE ORIGINAL LINE, verbatim. If the sweep cannot flag this it is a
        # green light that measures nothing.
        ('n = len(getattr(req, "prefix_indices", ()) or ())', "prefix_indices"),
        # The same landmine in plain attribute form -- the OTHER resolver edge,
        # given its own dying mutant so neither branch can rot untested.
        ("n = len(req.prefix_indices or ())", "prefix_indices"),
        ("if batch.input_ids:\n    pass", "input_ids"),
        ("x = not alloc.free_pages", "free_pages"),
        ("x = bool(op.host_indices)", "host_indices"),
        ("x = 1 if req.seq_lens else 2", "seq_lens"),
    ],
)
def test_can_fail_pin4_the_sweep_catches_every_shape(source, expected):
    hits = [_tensor_name(expr) for expr, _ in _boolean_contexts(ast.parse(source))]
    assert expected in hits


def test_the_sweep_does_not_flag_the_safe_shapes():
    """NEGATIVE ARM. A sweep that flags everything is a sweep nobody keeps.

    `a or b` returns `b` untested, `is not None` is not truthiness, and a
    non-tensor attribute is not this defect.
    """
    for source in (
        "x = other or req.prefix_indices",  # last operand is RETURNED, not tested
        "if req.prefix_indices is not None:\n    pass",
        "x = req.output_ids or []",  # a plain list
        "n = prefix_len(req)",
    ):
        hits = [_tensor_name(e) for e, _ in _boolean_contexts(ast.parse(source))]
        assert not [h for h in hits if h], source
