# SPDX-License-Identifier: Apache-2.0
"""#861 fix (0): ONE gate for the draft half, and it closes on three terms.

Before #861 six sites read ``self.has_draft`` directly -- four in
``HiCacheController`` and two in ``HybridCacheController``'s overrides, which
is the lane this rig actually runs. That is the W31/W32/W33 shape: a correct
mechanism a second copy overrides is a mechanism that never runs. These pin the
gate itself and pin that all six sites go through it.

The three terms, each a corruption if skipped:

  1. REGISTERED AT ALL -- the pre-#861 state on a flip boot.
  2. THE ACTIVE PHASE OWNS THE DRAFTER -- a draft backup taken in PP persists
     rows no drafter wrote, under a content-addressed key.
  3. THE BINDING HAS NOT MOVED -- draft host indices are 1-to-1 with the target
     host pool's, and that pool is re-pointed at every #719 rebind.
"""

import inspect


import pytest

from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache import hicache_phase_binding as binding
from sglang.srt.mem_cache import hicache_phase_guard as guard


class Stub:
    """Binds the unbound methods, the same stand-in discipline the scheduler's
    own layout-conformance hook uses. Building a real controller would drag in
    CUDA streams for a predicate that touches none of them."""

    def __init__(self, **kw):
        self.has_draft = True
        self.draft_owner_phase = None
        self.draft_binding_generation = None
        self.draft_identity = None
        self._draft_disarm_warned = set()
        self.__dict__.update(kw)

    armed = HiCacheController.draft_tier_armed
    _warn_draft_disarmed = HiCacheController._warn_draft_disarmed
    component = HiCacheController._draft_component_name


@pytest.fixture(autouse=True)
def _fresh():
    binding.binding_state().reset()
    guard.clear_flip_phase_authority()
    guard.reset_warnings()
    yield
    binding.binding_state().reset()
    guard.clear_flip_phase_authority()
    guard.reset_warnings()


class FakeRuntime:
    def __init__(self, phase):
        self.phase = phase
        self.hicache_seam_active = False


# ---------------------------------------------------------------- term 1


def test_unregistered_draft_is_closed():
    assert Stub(has_draft=False).armed("load") is False


def test_registered_single_phase_instance_is_open():
    """A non-flip deployment must be byte-identical: no phase term, no
    generation term, gate open."""
    assert Stub().armed("load") is True


# ---------------------------------------------------------------- term 2


def test_gate_closes_when_the_active_phase_does_not_own_the_drafter():
    rt = FakeRuntime("pp")  # HELD: the guard keeps only a weakref
    guard.register_flip_phase_authority(rt)
    assert guard.active_phase() == "pp"
    assert Stub(draft_owner_phase="tp").armed("write") is False


def test_gate_opens_in_the_drafters_own_phase():
    rt = FakeRuntime("tp")
    guard.register_flip_phase_authority(rt)
    binding.binding_state().advance("tp", host_pool=object())
    assert guard.active_phase() == "tp"
    assert Stub(draft_owner_phase="tp").armed("write") is True


def test_can_fail_term2_is_not_a_tautology():
    """The same stub with the phase term REMOVED is open in pp -- so the
    assertion above is measuring the term and not the default."""
    rt = FakeRuntime("pp")
    guard.register_flip_phase_authority(rt)
    assert Stub(draft_owner_phase=None).armed("write") is True


def test_seam_closes_the_gate_for_every_binding():
    rt = FakeRuntime("tp")
    rt.hicache_seam_active = True
    guard.register_flip_phase_authority(rt)
    assert Stub(draft_owner_phase="tp").armed("write") is False


# ---------------------------------------------------------------- term 3


def test_gate_closes_on_a_stale_binding_generation():
    """A registration minted at generation g indexes generation g's host slot
    space; consumed at g+1 it addresses a different pool entirely."""
    binding.binding_state().advance("tp", host_pool=object())
    stub = Stub(draft_binding_generation=1)
    assert stub.armed("load") is True
    binding.binding_state().advance("pp", host_pool=object())
    assert stub.armed("load") is False


def test_can_fail_term3_matching_generation_stays_open():
    binding.binding_state().advance("tp", host_pool=object())
    binding.binding_state().advance("pp", host_pool=object())
    assert Stub(draft_binding_generation=2).armed("load") is True


# ------------------------------------------------------- the six consume points


CONSUME_SITES = {
    "sglang.srt.managers.cache_controller": 4,
    "sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller": 2,
}


def test_no_consume_point_reads_has_draft_directly():
    """THE ANTI-SECOND-COPY PIN. `has_draft` may be written, and may be read by
    the gate itself and by the disarm; a TRANSFER site reading it directly is a
    site the gate does not cover, which is precisely how the override lane
    escaped the target tier's own guard three times (W31/W32/W33).

    Parsed with `ast` rather than grepped: prose in a docstring naming the
    defect must not read as the defect.
    """
    import ast
    import importlib

    for module_name, expected in CONSUME_SITES.items():
        module = importlib.import_module(module_name)
        tree = ast.parse(inspect.getsource(module))

        gated = sum(
            1
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "draft_tier_armed"
        )
        assert gated == expected, (
            f"{module_name}: expected {expected} gated consume point(s), "
            f"found {gated}"
        )

        # Every surviving `self.has_draft` READ must live inside the gate or
        # the disarm. Anywhere else it is a transfer condition the gate misses.
        # The gate's own first term, the disarm, and the L3 REGISTRATION
        # picker -- which asks "is there a draft pool to wire funcs for at
        # all", not "may this transfer run". A registration is not a consume
        # point: its result is re-gated at every use.
        allowed = {
            "draft_tier_armed",
            "disarm_draft_kv_pool",
            "_maybe_register_draft_with_storage",
        }
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if fn.name in allowed:
                continue
            for node in ast.walk(fn):
                if (
                    isinstance(node, ast.Attribute)
                    and node.attr == "has_draft"
                    and isinstance(node.ctx, ast.Load)
                ):
                    pytest.fail(
                        f"{module_name}.{fn.name}: ungated self.has_draft read "
                        f"at line {node.lineno}"
                    )


# ---------------------------------------------------------------- drafter key


def test_draft_component_name_carries_the_drafter():
    assert Stub(draft_identity="abc123").component() == "draft-abc123"


def test_draft_component_name_without_identity_is_the_bare_pool_name():
    """A caller that predates the parameter keeps writing exactly the keys it
    wrote before -- the default path stays byte-identical."""
    assert Stub(draft_identity=None).component() == "draft"
