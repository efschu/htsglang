"""The bind seam of the union arena, slice 3b (2026-09-21).

The CUDA half (pack, export, attach, rebind) is proven by the two-process
metal probe beside this file. What is pinned here is everything that decides
WHETHER and WHAT gets bound, because those decisions are the ones that can
corrupt a phase or silently keep both copies.
"""

import pytest
import torch

from sglang.srt.model_executor.weights_arena import plan_arena_layout
from sglang.srt.weg2 import union_arena as ua
from sglang.srt.weg2 import union_arena_bind as ub


def _owner_manifest(named, phase=ua.PHASE_P, card="GPU-aaaabbbbcccc"):
    layout = plan_arena_layout(dict(named))
    return layout, ua.manifest_from_layout(
        layout,
        tag="fnFL2",
        card=card,
        owner_phase=phase,
        checksums=ua.checksums_for(named),
    )


def test_the_owner_publishes_its_own_layout_not_a_union():
    """An arena holding what EITHER side might want would leave the inactive
    side's bytes resident for the whole boot."""
    named = {"a": torch.randn(4, 4), "b": torch.randn(8)}
    layout, manifest = _owner_manifest(named)
    assert manifest.total_bytes == layout.total_bytes
    assert [s.name for s in manifest.slots] == ["a", "b"]
    assert manifest.active_for(ua.PHASE_P) == ("a", "b")
    assert manifest.owner_phase == ua.PHASE_P


def test_the_peer_binds_only_what_it_can_prove_and_keeps_the_rest():
    shared = torch.randn(16, 16)
    owner = {"attn.qkv": shared, "p.only": torch.randn(4)}
    _, manifest = _owner_manifest(owner)
    peer = {
        "attn.qkv": shared.clone(),  # provably identical
        "experts.w13": torch.randn(32, 4),  # not in the manifest
    }
    binding = ua.verify_peer(manifest, ua.PHASE_D, peer)
    assert [s.name for s in binding.shared] == ["attn.qkv"]
    assert binding.private == ("experts.w13",)
    assert binding.shared_bytes == 16 * 16 * 4


def test_the_owner_may_not_bind_its_own_manifest():
    named = {"a": torch.randn(4)}
    _, manifest = _owner_manifest(named)
    with pytest.raises(ua.UnionShareError, match="IS the owner"):
        ua.verify_peer(manifest, ua.PHASE_P, named)


def test_a_same_shape_different_content_tensor_is_never_bound():
    """Expert-index sharding: rank r holds experts [r*k, (r+1)*k). Same name,
    same shape, different bytes -- binding them would corrupt a phase."""
    owner = {"experts.w13": torch.randn(8, 4)}
    _, manifest = _owner_manifest(owner)
    peer = {"experts.w13": torch.randn(8, 4)}
    with pytest.raises(ua.UnionShareError, match="different BYTES"):
        ua.verify_peer(manifest, ua.PHASE_D, peer)


def test_the_hook_is_off_unless_both_switches_are_set(monkeypatch):
    monkeypatch.delenv(ua.UNION_DIR_ENV, raising=False)
    monkeypatch.delenv(ub.UNION_MODE_ENV, raising=False)
    assert ub.maybe_union_image(object(), device=0) is None
    monkeypatch.setenv(ua.UNION_DIR_ENV, "/dev/shm/x")
    assert ub.maybe_union_image(object(), device=0) is None  # mode still off
    monkeypatch.setenv(ub.UNION_MODE_ENV, "bind")
    monkeypatch.setattr(
        "sglang.srt.managers.weg2_memory_saver.weg2_group_name", lambda: ""
    )
    assert ub.maybe_union_image(object(), device=0) is None  # not a weg2 rank


def test_an_unknown_mode_is_refused_not_ignored(monkeypatch):
    monkeypatch.setenv(ua.UNION_DIR_ENV, "/dev/shm/x")
    monkeypatch.setenv(ub.UNION_MODE_ENV, "maybe")
    with pytest.raises(ua.UnionShareError, match="must be 'own', 'bind' or 'off'"):
        ub.maybe_union_image(object(), device=0)


def test_the_boot_hook_is_not_wrapped_in_a_swallowing_except():
    """A boot that believes it is sharing and is not holds both copies and
    OOMs later; the census may be swallowed, this may not."""
    import inspect

    from sglang.srt.model_executor import model_runner

    src = inspect.getsource(model_runner)
    call = src.index("maybe_union_image(")
    tail = src[call : call + 400]
    assert "except Exception" not in tail
    assert 'role="draft" if self.is_draft_worker else "main"' in tail


def test_a_missing_owner_is_fatal_for_the_main_image_and_not_for_the_draft():
    """P has no draft model today, so D's draft worker must not wait 10
    minutes for an owner that cannot come -- and the MAIN image must never
    fall back silently, because that is the boot holding both copies."""
    import inspect

    src = inspect.getsource(ub.bind_image)
    assert "if required:\n            raise" in src
    hook = inspect.getsource(ub.maybe_union_image)
    assert 'timeout_s=600.0 if role == "main" else 20.0' in hook
    assert 'required=(role == "main")' in hook
