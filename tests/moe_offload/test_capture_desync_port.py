# SPDX-License-Identifier: Apache-2.0
"""#443 -- the V4 GGUF expert fetch, desynchronised: what is proven at the desk.

The eager offload step (``run_waves``) reads ``topk_ids.tolist()`` once per
layer per forward. That single host read is what makes a decode graph
uncapturable, which is why the DeepSeek-V4-Flash GGUF recipe has run with
``--disable-cuda-graph`` since #123. The #122 capturable path already exists
(``prepare_capturable_remap`` + a captured ``index_select`` over a UVA view of
the pinned pool); this file pins that PORTING the V4 path onto it is a
data-movement change and not a math change, and that no host read survives in
the ported step.

Four things are pinned here, all on CPU tensors under
``CUDA_VISIBLE_DEVICES=99``:

1. **Scratch bound** -- ``worst_case_unique_spill``. The capture-admission
   check in ``layer.py`` used ``topk_ids.numel()``, which refuses captures that
   are provably safe once the cold set is smaller than the bucket's routed
   slots. Under the #82 GGUF expert-dim shard that is the normal case.
2. **Sync-freedom** -- ``torch.Tensor.{item,tolist,cpu,numpy,nonzero,__bool__,
   __int__,__float__,__index__}`` are intercepted around a full
   ``prepare_capturable``, and the call list must be empty. The can-fail runs
   the EAGER path under the same probes and shows them firing.
3. **Byte equivalence** -- the captured gather lands the same expert rows in
   the same scratch slots as the eager ``_fetch``, and every remapped id
   addresses the row of the expert it was routed to. Data movement, not math.
4. **The #394 seam** -- a DELEGATED expert has no row in this rank's pool. The
   capturable remap cannot branch on that (a host read), so it counts on device
   and clamps; ``offload_capture_gate`` turns the count into a named exception
   at the CUDA-graph replay boundary, the #431 pattern.

BOOT-PENDING and deliberately not claimed here: that a real V4 decode graph
captures, replays correctly, and is faster. See
``scripts/dev/443_graph_proof/``.

Run:  python -m pytest tests/moe_offload/test_capture_desync_port.py -q
"""

import contextlib
import os
import sys
from unittest import mock

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.layers.moe import offload_capture_gate  # noqa: E402
from sglang.srt.layers.moe.expert_offload import (  # noqa: E402
    ExpertResidencyPlanner,
    MoEExpertOffloadCache,
    build_capturable_luts,
    prepare_capturable_remap,
    refuse_capturable_cold_tier,
    worst_case_unique_spill,
)

# Expert-tensor attrs and their per-expert row shapes. Two of them, with
# different dtypes and ranks, because the gather is issued once per attr and a
# port that silently handled only the first would pass a single-attr test.
ATTRS = {"w13_weight": (4, 6), "w2_weight": (6,)}
DTYPES = {"w13_weight": torch.float32, "w2_weight": torch.bfloat16}


# --------------------------------------------------------------------------
# stubs: borrow the real methods, carry only the attributes they read
# --------------------------------------------------------------------------


class _StubCache:
    """A ``MoEExpertOffloadCache`` reduced to the fetch path.

    It borrows the real methods rather than reimplementing them, so the test
    tracks the implementation instead of a paraphrase of it. On CPU the UVA
    view collapses to the pinned tensor ITSELF, which is exactly the aliasing
    ``device_view_of_pinned`` asserts on a card (it raises if
    ``torch.as_tensor`` copied instead of aliasing), so the gather under test
    is the same gather over the same storage.
    """

    _fetch = MoEExpertOffloadCache._fetch
    _build_lut = MoEExpertOffloadCache._build_lut
    _remap = staticmethod(MoEExpertOffloadCache._remap)
    _issue_fetch_capturable = MoEExpertOffloadCache._issue_fetch_capturable
    prepare_capturable = MoEExpertOffloadCache.prepare_capturable
    check_capture_breach = MoEExpertOffloadCache.check_capture_breach

    def __init__(self, *, E, R, C, resident_slot=None, spill_pool_index=None, seed=0):
        self.num_local_experts = E
        self.resident_count = R
        self.scratch = C
        self.layer = _StubLayer()
        self.planner = ExpertResidencyPlanner(
            num_local_experts=E,
            resident_count=R,
            scratch=C,
            resident_ids=(
                frozenset(resident_slot) if resident_slot is not None else None
            ),
            resident_slot=resident_slot,
        )
        self._spill_pool_index = spill_pool_index
        self._cold_tier = None
        self._remote_ids = frozenset()
        self._stream = None
        self._cap_breach = None
        self._capturable_ready = True

        g = torch.Generator().manual_seed(seed)
        # The ground truth: one row per GLOBAL expert id. Everything below is a
        # placement of these rows, so "did the port move the right bytes" is a
        # question with an answer independent of both code paths.
        self.full = {
            attr: torch.randn((E, *shape), generator=g).to(DTYPES[attr])
            for attr, shape in ATTRS.items()
        }
        resident_ids = (
            sorted(resident_slot) if resident_slot is not None else list(range(R))
        )
        cold_ids = [e for e in range(E) if e not in set(resident_ids)]
        self.cold_ids = cold_ids
        self._pinned = {}
        self._resident = {}
        for attr, shape in ATTRS.items():
            pool = torch.empty((E - R, *shape), dtype=DTYPES[attr])
            for e in cold_ids:
                row = spill_pool_index[e] if spill_pool_index is not None else e - R
                pool[row] = self.full[attr][e]
            # Page-locking is a CUDA property; on CPU the pool is the same
            # storage the UVA view would alias, which is all this path needs.
            self._pinned[attr] = pool
            buf = torch.zeros((R + C, *shape), dtype=DTYPES[attr])
            for e in resident_ids:
                slot = resident_slot[e] if resident_slot is not None else e
                buf[slot] = self.full[attr][e]
            self._resident[attr] = buf

        (
            self._cap_resident_slot_lut,
            self._cap_is_spill,
            self._cap_spill_pool_row_lut,
        ) = build_capturable_luts(E, R, resident_slot, spill_pool_index, device="cpu")
        self._cap_pool_dev = dict(self._pinned)
        self._cap_scratch_dst = {
            attr: self._resident[attr][R : R + C] for attr in ATTRS
        }

    # -- the eager step, reduced to the same two effects the ported one has --
    def prepare_eager(self, topk_ids):
        """``run_waves``' single-wave fast path: the host read, the resolve, the
        fetch, the LUT remap. Kept here so the two paths can be compared under
        one harness -- and so the interception test has a positive control that
        is the real thing rather than a stand-in."""
        ids_list = topk_ids.tolist()
        needed = sorted({e for row in ids_list for e in row if e >= 0})
        slot_of_needed, fetch_plan = self.planner.resolve(needed)
        self._fetch(fetch_plan)
        lut = self._build_lut(slot_of_needed, topk_ids.dtype, topk_ids.device)
        return self._remap(topk_ids, lut)


class _StubLayer:
    layer_id = 7


def _hotset(E, R, seed):
    """A frozen residency exactly as ``_freeze_hotset`` installs it: hot experts
    sorted into slots [0,R), cold sorted into pool rows [0,E-R)."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(E, generator=g).tolist()
    hot = sorted(perm[:R])
    cold = sorted(e for e in range(E) if e not in set(hot))
    return {e: i for i, e in enumerate(hot)}, {e: j for j, e in enumerate(cold)}


def _single_wave_ids(cache, *, T, k, seed):
    """Random routing whose distinct SPILL experts fit the scratch region."""
    g = torch.Generator().manual_seed(seed)
    resident = set(cache.planner.resident_ids or range(cache.resident_count))
    for _ in range(400):
        ids = torch.randint(0, cache.num_local_experts, (T, k), generator=g)
        ids = torch.where(
            torch.rand(T, k, generator=g) < 0.15, torch.full_like(ids, -1), ids
        )
        spill = {int(e) for row in ids.tolist() for e in row if e >= 0} - resident
        if 0 < len(spill) <= cache.scratch:
            return ids
    pytest.skip("could not sample a single-wave routing")


# --------------------------------------------------------------------------
# 1. the scratch bound
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "slots,E,R,expected",
    [
        (64, 128, 120, 8),  # cold set binds (the #82 GGUF shard's normal case)
        (4, 128, 120, 4),  # routed slots bind (tiny bucket)
        (64, 256, 0, 64),  # nothing resident: routed slots bind again
        (64, 128, 128, 0),  # fully resident: no spill is possible at all
        (0, 128, 64, 0),  # empty bucket
    ],
)
def test_bound_is_the_min_of_routed_slots_and_the_cold_set(slots, E, R, expected):
    assert worst_case_unique_spill(slots, E, R) == expected


@pytest.mark.parametrize("slots,E,R", [(-5, 128, 64), (64, 64, 200)])
def test_bound_is_never_negative(slots, E, R):
    assert worst_case_unique_spill(slots, E, R) == 0


def test_the_bound_is_tight_and_not_merely_safe():
    """Adversarial routing cannot exceed it, and can reach it.

    Both directions matter: a bound that is never reached would be a loose
    bound wearing a tight one's name, and one that can be exceeded would admit
    a capture whose gather silently drops experts.
    """
    E, R, C = 64, 56, 8
    cache = _StubCache(E=E, R=R, C=C)
    cold = cache.cold_ids
    bound = worst_case_unique_spill(4 * 16, E, R)
    assert bound == len(cold) == C

    # reachable: route every cold expert at once
    ids = torch.tensor([cold[i % len(cold)] for i in range(4 * 16)]).reshape(4, 16)
    _, _, num_spill = prepare_capturable_remap(
        ids,
        cache._cap_resident_slot_lut,
        cache._cap_is_spill,
        cache._cap_spill_pool_row_lut,
        R,
        C,
    )
    assert int(num_spill) == bound

    # not exceedable: 200 random draws over the full local table
    g = torch.Generator().manual_seed(11)
    for _ in range(200):
        ids = torch.randint(0, E, (4, 16), generator=g)
        _, _, num_spill = prepare_capturable_remap(
            ids,
            cache._cap_resident_slot_lut,
            cache._cap_is_spill,
            cache._cap_spill_pool_row_lut,
            R,
            C,
        )
        assert int(num_spill) <= bound


def test_the_loose_bound_would_have_refused_a_provably_safe_capture():
    """Can-fail for the bound: the pre-#443 expression, on the same inputs.

    ``topk_ids.numel()`` is what ``layer.py`` compared against the scratch size.
    With bs=4, top_k=16 and a cold set of 8 it demands 64 scratch slots for a
    step that can never need more than 8 -- i.e. the V4 GGUF recipe's capture
    was refused by an arithmetic overestimate, not by a real limit.
    """
    E, R, C = 64, 56, 8
    routed_slots = 4 * 16
    assert routed_slots > C  # the old check refuses
    assert worst_case_unique_spill(routed_slots, E, R) <= C  # the new one admits


# --------------------------------------------------------------------------
# 2. sync-freedom (the #440 interception technique)
# --------------------------------------------------------------------------

#: Every ``torch.Tensor`` entry point that forces the host to wait for the
#: device. ``nonzero`` is included because its output shape is data-dependent,
#: which makes it a sync even though it returns a tensor; the dunders are
#: included because ``int(t)``/``if t:`` are the syncs that do not look like
#: one at the call site.
_SYNC_METHODS = (
    "item",
    "tolist",
    "cpu",
    "numpy",
    "nonzero",
    "__bool__",
    "__int__",
    "__float__",
    "__index__",
)


@contextlib.contextmanager
def _intercept_host_reads():
    calls = []
    with contextlib.ExitStack() as stack:
        for name in _SYNC_METHODS:
            real = getattr(torch.Tensor, name)

            def probed(self, *a, _name=name, _real=real, **kw):
                calls.append((_name, tuple(self.shape)))
                return _real(self, *a, **kw)

            stack.enter_context(mock.patch.object(torch.Tensor, name, probed))
        yield calls


@pytest.mark.parametrize("hot", [False, True])
def test_the_ported_step_performs_no_host_read(hot):
    E, R, C = 64, 32, 8
    resident_slot, spill_pool_index = _hotset(E, R, 3) if hot else (None, None)
    cache = _StubCache(
        E=E, R=R, C=C, resident_slot=resident_slot, spill_pool_index=spill_pool_index
    )
    ids = _single_wave_ids(cache, T=4, k=8, seed=5)
    with _intercept_host_reads() as calls:
        remapped = cache.prepare_capturable(ids)
    assert remapped.shape == ids.shape
    assert calls == []


def test_the_seam_armed_step_performs_no_host_read_either():
    """The #394 breach counter must not become the sync it exists to avoid.

    Counting is a device ``add_``; the READ happens at the replay boundary, in
    ``check_capture_breach``. If the count were tested here instead, the seam
    would have reintroduced exactly the host dependency this port removed.
    """
    E, R, C = 64, 32, 8
    cache = _StubCache(E=E, R=R, C=C)
    cache._cap_breach = torch.zeros(1, dtype=torch.int32)
    # Delegate two cold experts: no local row, so the frozen LUT holds -1.
    for e in cache.cold_ids[:2]:
        cache._cap_spill_pool_row_lut[e] = -1
    ids = _single_wave_ids(cache, T=4, k=8, seed=9)
    with _intercept_host_reads() as calls:
        cache.prepare_capturable(ids)
    assert calls == []


def test_the_interception_can_fail():
    """Positive control: the EAGER path under the same probes.

    Without this the empty call list above would be equally consistent with a
    probe that never installed. ``run_waves``' host read is the one this port
    removes, so it is the right thing to catch.
    """
    E, R, C = 64, 32, 8
    cache = _StubCache(E=E, R=R, C=C)
    ids = _single_wave_ids(cache, T=4, k=8, seed=5)
    with _intercept_host_reads() as calls:
        cache.prepare_eager(ids)
    assert any(name == "tolist" for name, _ in calls), calls


# --------------------------------------------------------------------------
# 3. byte equivalence: the port moves data, it does not change math
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("hot", [False, True])
def test_captured_gather_lands_the_same_bytes_as_the_eager_fetch(seed, hot):
    E, R, C = 64, 32, 8
    resident_slot, spill_pool_index = _hotset(E, R, seed) if hot else (None, None)
    kwargs = dict(
        E=E, R=R, C=C, resident_slot=resident_slot, spill_pool_index=spill_pool_index
    )
    eager = _StubCache(seed=seed, **kwargs)
    ported = _StubCache(seed=seed, **kwargs)
    ids = _single_wave_ids(eager, T=4, k=8, seed=seed)

    remapped_eager = eager.prepare_eager(ids)
    remapped_ported = ported.prepare_capturable(ids)

    assert torch.equal(remapped_eager, remapped_ported)

    _, _, num_spill = prepare_capturable_remap(
        ids,
        ported._cap_resident_slot_lut,
        ported._cap_is_spill,
        ported._cap_spill_pool_row_lut,
        R,
        C,
    )
    used = R + int(num_spill)
    for attr in ATTRS:
        # The USED region is byte-identical. Beyond it the two differ by
        # construction and harmlessly: the eager fetch writes only the slots in
        # its plan, the captured gather writes all C slots (unused ones take
        # pool row 0). No remapped id addresses that tail -- asserted next.
        assert torch.equal(
            eager._resident[attr][:used], ported._resident[attr][:used]
        ), attr
    vals = remapped_ported[remapped_ported >= 0]
    assert not ((vals >= used) & (vals < R + C)).any()


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("hot", [False, True])
def test_every_remapped_id_addresses_the_row_of_the_expert_it_routed_to(seed, hot):
    """The gate that does not depend on the eager path being right.

    Equivalence to ``_fetch`` proves the port reproduces the incumbent. This
    proves the incumbent's semantics directly: after the step, the buffer row a
    routed slot points at holds the weights of the GLOBAL expert the router
    chose. A port that consistently mis-indexed both paths would pass the
    comparison above and fail here.
    """
    E, R, C = 64, 32, 8
    resident_slot, spill_pool_index = _hotset(E, R, seed) if hot else (None, None)
    cache = _StubCache(
        E=E,
        R=R,
        C=C,
        resident_slot=resident_slot,
        spill_pool_index=spill_pool_index,
        seed=seed,
    )
    ids = _single_wave_ids(cache, T=4, k=8, seed=seed)
    remapped = cache.prepare_capturable(ids)
    for t in range(ids.shape[0]):
        for j in range(ids.shape[1]):
            expert = int(ids[t, j])
            slot = int(remapped[t, j])
            if expert < 0:
                assert slot == -1
                continue
            assert 0 <= slot < R + C
            for attr in ATTRS:
                assert torch.equal(
                    cache._resident[attr][slot], cache.full[attr][expert]
                ), f"expert {expert} -> slot {slot} ({attr})"


def test_padding_only_routing_gathers_nothing_and_stays_padding():
    E, R, C = 32, 16, 8
    cache = _StubCache(E=E, R=R, C=C)
    before = {attr: cache._resident[attr].clone() for attr in ATTRS}
    ids = torch.full((2, 4), -1, dtype=torch.long)
    remapped = cache.prepare_capturable(ids)
    assert torch.equal(remapped, ids)
    for attr in ATTRS:
        # Slot 0 of the pool is gathered into every scratch slot (num_spill=0,
        # so src_row is all zeros) -- the RESIDENT region must be untouched.
        assert torch.equal(cache._resident[attr][:R], before[attr][:R])


# --------------------------------------------------------------------------
# 4. the #394 seam: count on device, raise at the replay boundary (#431)
# --------------------------------------------------------------------------


def _delegated_cache(n_delegated=2, E=64, R=32, C=8):
    cache = _StubCache(E=E, R=R, C=C)
    cache._cap_breach = torch.zeros(1, dtype=torch.int32)
    delegated = cache.cold_ids[:n_delegated]
    for e in delegated:
        cache._cap_spill_pool_row_lut[e] = -1
    return cache, delegated


def test_a_delegated_expert_is_counted_and_the_index_is_clamped():
    cache, delegated = _delegated_cache()
    ids = torch.tensor([delegated + [0, 1]], dtype=torch.long)
    _, src_row, _ = prepare_capturable_remap(
        ids,
        cache._cap_resident_slot_lut,
        cache._cap_is_spill,
        cache._cap_spill_pool_row_lut,
        cache.resident_count,
        cache.scratch,
        breach_counter=cache._cap_breach,
    )
    assert int(cache._cap_breach.item()) == len(delegated)
    assert int(src_row.min()) >= 0  # in bounds -> the gather cannot walk off


def test_without_a_counter_the_default_path_is_unchanged():
    """The seam must be invisible when it is not armed.

    Same inputs, no ``-1`` anywhere: passing a counter must not perturb the
    outputs, and the counter must stay at zero. This is the statement the
    docstring in ``prepare_capturable_remap`` makes, held to.
    """
    E, R, C = 64, 32, 8
    cache = _StubCache(E=E, R=R, C=C)
    ids = _single_wave_ids(cache, T=4, k=8, seed=13)
    args = (
        cache._cap_resident_slot_lut,
        cache._cap_is_spill,
        cache._cap_spill_pool_row_lut,
        R,
        C,
    )
    plain = prepare_capturable_remap(ids, *args)
    counter = torch.zeros(1, dtype=torch.int32)
    armed = prepare_capturable_remap(ids, *args, breach_counter=counter)
    assert torch.equal(plain[0], armed[0])
    assert torch.equal(plain[1], armed[1])
    assert torch.equal(plain[2], armed[2])
    assert int(counter.item()) == 0


def test_check_capture_breach_raises_named_and_then_resets():
    cache, delegated = _delegated_cache()
    ids = torch.tensor([delegated + [0, 1]], dtype=torch.long)
    cache.prepare_capturable(ids)
    with pytest.raises(offload_capture_gate.OffloadCaptureBreach) as excinfo:
        cache.check_capture_breach("unit")
    assert excinfo.value.layer_id == 7
    assert excinfo.value.breaches == len(delegated)
    assert "DELEGATED" in str(excinfo.value)
    # Reset: a window that catches and continues must not re-raise on the same
    # breach and read one wrong step as a permanent state.
    cache.check_capture_breach("unit")


def test_check_capture_breach_is_silent_without_the_seam():
    cache = _StubCache(E=64, R=32, C=8)
    assert cache._cap_breach is None
    cache.check_capture_breach("unit")


# --------------------------------------------------------------------------
# the gate module itself (the #431 shape)
# --------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_gate_registry():
    offload_capture_gate.reset_for_test()
    yield
    offload_capture_gate.reset_for_test()


def test_an_empty_registry_is_the_whole_default_path():
    assert offload_capture_gate.registered() == []
    offload_capture_gate.check_after_graph_replay()  # must not raise


def test_registration_is_idempotent_and_reversible():
    cache, _ = _delegated_cache()
    offload_capture_gate.register(cache)
    offload_capture_gate.register(cache)
    assert offload_capture_gate.registered() == [cache]
    offload_capture_gate.unregister(cache)
    assert offload_capture_gate.registered() == []
    offload_capture_gate.unregister(cache)  # tolerant of a double unregister


def test_the_boundary_check_reports_a_registered_cache():
    cache, delegated = _delegated_cache()
    ids = torch.tensor([delegated + [0, 1]], dtype=torch.long)
    cache.prepare_capturable(ids)
    offload_capture_gate.register(cache)
    with pytest.raises(offload_capture_gate.OffloadCaptureBreach):
        offload_capture_gate.check_after_graph_replay()


def test_the_kill_switch_suppresses_the_check():
    cache, delegated = _delegated_cache()
    ids = torch.tensor([delegated + [0, 1]], dtype=torch.long)
    cache.prepare_capturable(ids)
    offload_capture_gate.register(cache)
    with mock.patch.dict(os.environ, {offload_capture_gate.ENV_ENABLE: "0"}):
        assert not offload_capture_gate.gate_enabled()
        offload_capture_gate.check_after_graph_replay()  # suppressed
    assert offload_capture_gate.gate_enabled()  # default-on, read per call
    with pytest.raises(offload_capture_gate.OffloadCaptureBreach):
        offload_capture_gate.check_after_graph_replay()


def test_the_replay_boundary_does_not_import_the_offload_stack():
    """Pins the reason this module exists separately at all.

    ``full_cuda_graph_backend`` is imported by every launch. If the gate lived
    in ``expert_offload`` the boundary would drag 3.6k lines of staging,
    planner and NVML code into processes that never offload anything. Run in a
    subprocess because the assertion is about a COLD ``sys.modules``.
    """
    import subprocess

    probe = (
        "import sys; "
        "import sglang.srt.model_executor.runner_backend.full_cuda_graph_backend; "
        "print(int('sglang.srt.layers.moe.offload_capture_gate' in sys.modules), "
        "int('sglang.srt.layers.moe.expert_offload' in sys.modules))"
    )
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="99")
    env["PYTHONPATH"] = os.pathsep.join(
        [os.path.join(os.path.dirname(__file__), "..", "..", "python")]
        + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    out = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    gate, offload = out.stdout.strip().splitlines()[-1].split()
    assert gate == "1"  # the boundary really does reach this module
    assert offload == "0"  # and it does not reach the one it guards


def test_a_cache_without_the_hook_is_ignored():
    """The registry holds duck-typed objects; a member with no counter must be
    stepped over rather than crash the replay boundary."""
    offload_capture_gate.register(object())
    offload_capture_gate.check_after_graph_replay()


# --------------------------------------------------------------------------
# the cold-tier refusal that keeps the seam a seam
# --------------------------------------------------------------------------


def test_the_cold_tier_capture_combination_refuses_by_default():
    with envs.SGLANG_MOE_COLD_TIER_GRAPH_UNSAFE.override(False):
        with pytest.raises(RuntimeError) as excinfo:
            refuse_capturable_cold_tier(128)
    message = str(excinfo.value)
    # Both gaps must be named. A refusal that names only the pointer would send
    # the next reader looking for a hardware problem they do not have.
    assert "SGLANG_MOE_COLD_TIER_SHM" in message
    assert "delegated expert has no row" in message
    assert "cudaHostRegister" in message


def test_the_development_seam_warns_instead_of_refusing():
    with envs.SGLANG_MOE_COLD_TIER_GRAPH_UNSAFE.override(True):
        with mock.patch("logging.Logger.warning") as warn:
            refuse_capturable_cold_tier(128)
    assert warn.call_count == 1
    assert "BOOT-PENDING" in warn.call_args[0][0]


# --------------------------------------------------------------------------
# capture-branch discipline in the layer (the #431 shape, one level up)
# --------------------------------------------------------------------------
#
# Removing the sync from the ported step is only half of it. The layer still
# owns two host-only branches -- the #390 routing trace and the eager
# ``run_waves`` -- and BOTH have to be decided by capture state rather than by
# what the batch happens to look like. That is the #440 lesson: the guard that
# invalidated upstream's capture was a runtime probe where a structural test
# would have done.


class _StubMoELayer:
    """A ``FusedMoE`` reduced to the offload dispatch, borrowing the real
    methods so the branch under test is the shipped one."""

    _run_moe_core_with_offload = None  # bound below, after the import
    _run_moe_core_offload_capturable = None

    def __init__(self, cache, *, graph_mode, trace_path=""):
        self._expert_offload = cache
        self._expert_offload_fraction = 0.5
        self._expert_offload_install_failed = False
        self._moe_offload_graph_mode = graph_mode
        # #462 added a third mode to the same selection point; the
        # real FusedMoE.__init__ always sets this, so the stub must too.
        self._moe_offload_breakable = False
        self._moe_offload_arena = None
        self._moe_offload_trace_path = trace_path
        self._moe_offload_trace_step = 0
        self.moe_tp_rank = 0
        self.moe_ep_rank = 0
        self.layer_id = 7
        self.applied = []

        class _QuantMethod:
            @staticmethod
            def apply(layer, dispatch_output):
                layer.applied.append(dispatch_output)
                return dispatch_output

        self.quant_method = _QuantMethod()


def _bind_layer_methods():
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

    _StubMoELayer._run_moe_core_with_offload = FusedMoE._run_moe_core_with_offload
    _StubMoELayer._run_moe_core_offload_capturable = (
        FusedMoE._run_moe_core_offload_capturable
    )


def _dispatch_output(ids):
    from typing import NamedTuple

    from sglang.srt.layers.moe.topk import StandardTopKOutput

    class _Dispatch(NamedTuple):
        hidden_states: torch.Tensor
        topk_output: StandardTopKOutput

    weights = torch.ones(ids.shape, dtype=torch.float32)
    return _Dispatch(
        hidden_states=torch.zeros((ids.shape[0], 4)),
        topk_output=StandardTopKOutput(
            topk_weights=weights, topk_ids=ids, router_logits=weights
        ),
    )


@contextlib.contextmanager
def _capture_mode(capturing):
    with mock.patch(
        "sglang.srt.model_executor.runner_utils.capture_mode.get_is_capture_mode",
        return_value=capturing,
    ):
        yield


#: A layer-level fixture whose cold set FITS the scratch region -- i.e. the
#: capture-eligible shape. E=40/R=32 leaves 8 cold experts for 8 slots, which
#: is the configuration the V4 recipe's ``--rank-moe-resident-fraction`` and
#: ``SGLANG_MOE_SCRATCH_SLOTS`` are chosen to produce.
_CAPTURABLE_SHAPE = dict(E=40, R=32, C=8)


@pytest.mark.parametrize("capturing", [False, True])
def test_the_capturable_branch_is_taken_only_under_capture(capturing):
    _bind_layer_methods()
    cache = _StubCache(**_CAPTURABLE_SHAPE)
    layer = _StubMoELayer(cache, graph_mode=True)
    ids = _single_wave_ids(cache, T=4, k=8, seed=21)
    with _capture_mode(capturing):
        with mock.patch.object(
            MoEExpertOffloadCache, "run_waves", create=True
        ) as run_waves:
            with mock.patch.object(
                _StubCache, "prepare_capturable", wraps=cache.prepare_capturable
            ) as ported:
                cache.run_waves = run_waves
                layer._run_moe_core_with_offload(_dispatch_output(ids))
    assert bool(ported.called) is capturing
    assert bool(run_waves.called) is (not capturing)


def test_without_the_opt_in_capture_never_reaches_the_ported_step():
    """``SGLANG_MOE_OFFLOAD_CUDA_GRAPH`` off must keep the pre-#443 behaviour.

    The default launch is the one that must not change: even under capture the
    dispatch has to fall through to ``run_waves``, whose host read is what the
    boot-time ``--disable-cuda-graph`` refusal exists to protect.
    """
    _bind_layer_methods()
    cache = _StubCache(**_CAPTURABLE_SHAPE)
    layer = _StubMoELayer(cache, graph_mode=False)
    ids = _single_wave_ids(cache, T=4, k=8, seed=22)
    with _capture_mode(True):
        with mock.patch.object(
            MoEExpertOffloadCache, "run_waves", create=True
        ) as run_waves:
            cache.run_waves = run_waves
            layer._run_moe_core_with_offload(_dispatch_output(ids))
    assert run_waves.called


def test_the_routing_trace_is_structurally_off_under_capture():
    """``write_routing_trace`` copies ``topk_ids`` to the host. It is
    measurement tooling, so the guard is a capture test and not a sampling
    heuristic -- a heuristic would eventually sample during a capture."""
    _bind_layer_methods()
    cache = _StubCache(**_CAPTURABLE_SHAPE)
    ids = _single_wave_ids(cache, T=4, k=8, seed=23)
    for capturing in (False, True):
        layer = _StubMoELayer(cache, graph_mode=True, trace_path="/dev/null")
        with _capture_mode(capturing):
            with mock.patch(
                "sglang.srt.layers.moe.expert_offload.write_routing_trace"
            ) as trace:
                with mock.patch.object(MoEExpertOffloadCache, "run_waves", create=True):
                    cache.run_waves = mock.Mock()
                    layer._run_moe_core_with_offload(_dispatch_output(ids))
        assert trace.called is (not capturing)
        assert layer._moe_offload_trace_step == (0 if capturing else 1)


def test_the_admission_check_refuses_a_bucket_that_cannot_fit():
    """The §2 invariant still bites when it is genuinely violated.

    #443 tightened the bound; it did not remove it. A cold set larger than the
    scratch region can still route more distinct spill experts than there are
    slots, and that must abort at capture rather than gather a subset.
    """
    _bind_layer_methods()
    cache = _StubCache(E=64, R=8, C=4)  # cold set 56 >> scratch 4
    layer = _StubMoELayer(cache, graph_mode=True)
    ids = torch.arange(8, 40, dtype=torch.long).reshape(4, 8)
    with _capture_mode(True):
        with pytest.raises(RuntimeError, match="unique spill experts"):
            layer._run_moe_core_with_offload(_dispatch_output(ids))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
