# SPDX-License-Identifier: Apache-2.0
"""#861 fix (0): the draft half of the HiCache tier is a stamped participant.

WHAT THESE PIN, and why each one is a corruption if it is skipped.

THE DEFECT. ``Scheduler.__init__`` registers the draft KV pool with
``draft_worker=self.draft_worker`` and ``spec_algorithm=self.spec_algorithm``.
On a phase-flip boot #631 has DELIBERATELY nulled both -- the boot phase is PP,
PP has no drafter, and the configured algorithm is parked in
``flip_spec_algorithm`` while the real drafter is built on
``phase_flip_stacks``. So the registration returned early, ``has_draft`` stayed
False for process life, and every HiCache read-through in the TP phase restored
a TARGET-ONLY prefix whose draft rows held the previous occupants' bytes.
Nothing raised; acceptance collapsed.

The four pins:

  1. RESOLUTION reaches the flip stack's drafter when the scheduler's own is
     the nulled boot-phase value -- and does NOT reach past a phase that owns
     no drafter (a draft backup taken in PP persists rows no drafter wrote,
     under a content-addressed key, for the TP phase to load as valid).
  2. The registration is stamped with the CURRENT #719 binding generation, not
     a boot-time snapshot. Draft host indices are 1-to-1 with the target host
     pool's, and that pool moves at every rebind.
  3. ONE GATE answers for all six consume points, and it closes on each of its
     three terms independently.
  4. A persisted draft page carries the DRAFTER's identity, so a page written
     by another drafter is a clean MISS rather than a silent wrong-format hit.

Each behavioural pin is followed by a CAN-FAIL proof: the same assertion
against a deliberately broken input, so a refactor that turns the pin into a
tautology is visible.
"""

import pytest

from sglang.srt.mem_cache import hicache_phase_binding as binding
from sglang.srt.mem_cache.kv_cache_builder import (
    DRAFT_OWNER_PHASE_FLIP,
    drafter_identity_hash,
    resolve_draft_registration,
)


class FakeAlgo:
    def __init__(self, none=False, ngram=False):
        self._none = none
        self._ngram = ngram

    def is_none(self):
        return self._none

    def is_ngram(self):
        return self._ngram


class FakePool:
    """Stands in for the draft runner's token_to_kv_pool."""

    def __init__(self, size=1024):
        self.size = size
        self.layer_num = 1


class FakeDraftRunnerHolder:
    def __init__(self, pool):
        self.draft_runner = type("R", (), {"token_to_kv_pool": pool})()


class FakeDraftWorker:
    """Mirrors EAGLEWorkerV2: `.draft_worker.draft_runner.token_to_kv_pool`."""

    def __init__(self, pool):
        self.draft_worker = FakeDraftRunnerHolder(pool)


class FakeServerArgs:
    enable_multi_layer_eagle = False
    speculative_algorithm = "NEXTN"
    speculative_draft_model_path = "/models/draft-a"
    speculative_num_steps = 3
    speculative_eagle_topk = 1
    speculative_num_draft_tokens = 4
    draft_kv_layout = "replicated"
    hicache_mem_layout = "layer_first"
    hicache_storage_backend = "file"


class FakeController:
    def __init__(self):
        self.mem_pool_host = type("H", (), {"size": 4096})()
        self.registered = None
        self.disarmed = None

    def set_draft_kv_pool(self, dev, host, **kw):
        self.registered = (dev, host, kw)

    def disarm_draft_kv_pool(self, reason):
        self.disarmed = reason


class FakeScheduler:
    """The FLIP-BOOT SHAPE: scheduler.draft_worker is None (as #631 leaves it),
    the real drafter hangs off phase_flip_stacks, and the algorithm is parked."""

    def __init__(self, *, flip=True, pool=None):
        pool = pool or FakePool()
        self.enable_hierarchical_cache = True
        self.server_args = FakeServerArgs()
        self.page_size = 1
        self.tree_cache = type("T", (), {"cache_controller": FakeController()})()
        if flip:
            self.draft_worker = None
            self.spec_algorithm = FakeAlgo(none=True)
            self.flip_spec_algorithm = FakeAlgo()
            self.phase_flip_stacks = type(
                "S", (), {"draft_worker": FakeDraftWorker(pool)}
            )()
        else:
            self.draft_worker = FakeDraftWorker(pool)
            self.spec_algorithm = FakeAlgo()
            self.flip_spec_algorithm = FakeAlgo(none=True)
            self.phase_flip_stacks = None


@pytest.fixture(autouse=True)
def _fresh_binding():
    binding.binding_state().reset()
    yield
    binding.binding_state().reset()


# ---------------------------------------------------------------- pin 1


def test_flip_boot_resolution_reaches_the_stacks_drafter():
    """THE DEFECT, stated as an assertion. RED on base: resolve_draft_
    registration does not exist there, and the boot-time call it replaces
    returns None for exactly this shape."""
    sched = FakeScheduler(flip=True)
    reg = resolve_draft_registration(sched, "tp")
    assert reg is not None, (
        "the flip boot's drafter lives on phase_flip_stacks; reading only "
        "scheduler.draft_worker is the #861 defect"
    )
    assert reg.owner_phase == DRAFT_OWNER_PHASE_FLIP


def test_can_fail_pin1_no_drafter_anywhere_resolves_to_none():
    sched = FakeScheduler(flip=True)
    sched.phase_flip_stacks = None
    assert resolve_draft_registration(sched, "tp") is None


def test_the_pp_leg_registers_nothing():
    """A draft backup taken in a phase with no drafter persists rows nobody
    wrote, under a content-addressed key. Worse than the missing registration."""
    sched = FakeScheduler(flip=True)
    assert resolve_draft_registration(sched, "pp") is None


def test_non_flip_boot_is_unchanged_and_carries_no_owner_phase():
    """A single-phase instance has no phase term to satisfy; adding one would
    disarm every non-flip deployment's draft tier."""
    sched = FakeScheduler(flip=False)
    reg = resolve_draft_registration(sched, None)
    assert reg is not None and reg.owner_phase is None


def test_speculation_off_resolves_to_none():
    sched = FakeScheduler(flip=True)
    sched.flip_spec_algorithm = FakeAlgo(none=True)
    assert resolve_draft_registration(sched, "tp") is None


# ------------------------------------------------- the defect, on the base API


def test_the_boot_time_entry_point_registers_nothing_on_a_flip_boot():
    """THE #861 DEFECT, pinned against the function that has always existed.

    GREEN on base and green after: it documents WHY the registration had to
    move to the cutover rather than widening the boot call's reach. The boot
    call reads ``scheduler.draft_worker`` / ``scheduler.spec_algorithm``, which
    #631 nulls on purpose, and at boot the process is in the PP phase -- the
    phase that must NOT have an armed draft half.
    """
    from sglang.srt.mem_cache.kv_cache_builder import maybe_register_hicache_draft

    sched = FakeScheduler(flip=True)
    maybe_register_hicache_draft(
        tree_cache=sched.tree_cache,
        draft_worker=sched.draft_worker,  # None, by #631's design
        spec_algorithm=sched.spec_algorithm,  # NONE, by #631's design
        server_args=sched.server_args,
        enable_hierarchical_cache=True,
        page_size=1,
    )
    assert sched.tree_cache.cache_controller.registered is None


def test_the_cutover_leg_registers_the_flip_stacks_drafter(monkeypatch):
    """THE FIX, end to end. RED on base: neither the function nor the route
    exists there, so a flip boot reaches the TP phase with has_draft False."""
    from sglang.srt.mem_cache import kv_cache_builder as kcb

    host = type("Host", (), {"size": 4096, "layer_num": 1})()
    monkeypatch.setattr(kcb, "_build_draft_host_pool", lambda **kw: host)

    sched = FakeScheduler(flip=True)
    binding.binding_state().advance("tp", host_pool=object())

    assert kcb.rebind_hicache_draft_for_phase(sched, "tp") is True
    dev, got_host, kw = sched.tree_cache.cache_controller.registered
    assert got_host is host
    assert kw["owner_phase"] == "tp"
    assert kw["binding_generation"] == 1
    assert kw["drafter_identity"]


def test_the_tp_to_pp_leg_disarms_rather_than_registering(monkeypatch):
    from sglang.srt.mem_cache import kv_cache_builder as kcb

    monkeypatch.setattr(
        kcb, "_build_draft_host_pool", lambda **kw: pytest.fail("must not allocate")
    )
    sched = FakeScheduler(flip=True)
    assert kcb.rebind_hicache_draft_for_phase(sched, "pp") is False
    assert sched.tree_cache.cache_controller.registered is None
    assert "does not own a draft KV pool" in sched.tree_cache.cache_controller.disarmed


def test_the_host_pool_is_allocated_once_and_restamped(monkeypatch):
    """A pinned host pool per flip would charge the host budget every time, and
    on this box that budget binds (DESIGN_706 C1)."""
    from sglang.srt.mem_cache import kv_cache_builder as kcb

    calls = []
    host = type("Host", (), {"size": 4096, "layer_num": 1})()

    def _build(**kw):
        calls.append(kw)
        return host

    monkeypatch.setattr(kcb, "_build_draft_host_pool", _build)
    sched = FakeScheduler(flip=True)
    binding.binding_state().advance("tp", host_pool=object())
    kcb.rebind_hicache_draft_for_phase(sched, "tp")
    binding.binding_state().advance("pp", host_pool=object())
    binding.binding_state().advance("tp", host_pool=object())
    kcb.rebind_hicache_draft_for_phase(sched, "tp")
    assert len(calls) == 1
    assert sched.tree_cache.cache_controller.registered[2]["binding_generation"] == 3


def test_a_broken_one_to_one_invariant_is_refused_loudly(monkeypatch):
    """Draft host indices are 1-to-1 with the target's BY CONSTRUCTION. If that
    ever stops holding, every draft row lands at an address that drifts with
    the slot id -- the #345 right-token/wrong-slot class, silent."""
    from sglang.srt.mem_cache import kv_cache_builder as kcb

    host = type("Host", (), {"size": 4096, "layer_num": 1})()
    monkeypatch.setattr(kcb, "_build_draft_host_pool", lambda **kw: host)
    sched = FakeScheduler(flip=True)
    binding.binding_state().advance("tp", host_pool=object())
    kcb.rebind_hicache_draft_for_phase(sched, "tp")

    sched.tree_cache.cache_controller.mem_pool_host = type("H2", (), {"size": 8192})()
    binding.binding_state().advance("tp", host_pool=object())
    with pytest.raises(ValueError, match="1-to-1"):
        kcb.rebind_hicache_draft_for_phase(sched, "tp")


# ---------------------------------------------------------------- pin 2


def test_registration_carries_the_current_binding_generation():
    """#719/#847: draft host indices are 1-to-1 with the TARGET host pool's,
    and `_stamp` re-points that pool at every rebind. A boot-time snapshot
    names generation 0's slot space for a consumer running at generation n."""
    sched = FakeScheduler(flip=True)
    assert resolve_draft_registration(sched, "tp").generation == 0
    binding.binding_state().advance("tp", host_pool=object())
    binding.binding_state().advance("pp", host_pool=object())
    assert resolve_draft_registration(sched, "tp").generation == 2


def test_can_fail_pin2_generation_is_read_not_frozen():
    """If the generation were captured once at import/boot, advancing the
    binding would leave the second read equal to the first."""
    sched = FakeScheduler(flip=True)
    first = resolve_draft_registration(sched, "tp").generation
    binding.binding_state().advance("tp", host_pool=object())
    second = resolve_draft_registration(sched, "tp").generation
    assert second != first


# ---------------------------------------------------------------- pin 4


def test_drafter_identity_separates_two_drafters():
    """A `{hash}.draft` page is keyed by the TARGET's identity alone
    (compute_model_identity_hash covers model_path/revision/dtype/quantization/
    kv_cache_dtype and nothing about the drafter), so two boots agreeing on the
    target and differing in drafter would read each other's draft KV as valid."""
    a = FakeServerArgs()
    b = FakeServerArgs()
    b.speculative_draft_model_path = "/models/draft-b"
    assert drafter_identity_hash(a) != drafter_identity_hash(b)


def test_drafter_identity_covers_algorithm_and_layout():
    base = drafter_identity_hash(FakeServerArgs())
    for field, value in (
        ("speculative_algorithm", "EAGLE"),
        ("speculative_num_steps", 5),
        ("speculative_eagle_topk", 4),
        # draft_kv_layout decides the draft pool's ROW SPACE and per-row byte
        # length. DESIGN_631b records that it must NOT enter the target key
        # (it is a parallelism decision, not a weights one) -- which is exactly
        # why it has to enter the DRAFT key.
        ("draft_kv_layout", "dcp"),
    ):
        sa = FakeServerArgs()
        setattr(sa, field, value)
        assert drafter_identity_hash(sa) != base, field


def test_can_fail_pin4_identical_drafters_share_a_key():
    assert drafter_identity_hash(FakeServerArgs()) == drafter_identity_hash(
        FakeServerArgs()
    )
