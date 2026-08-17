"""#545: runtime attach / resize / detach of the HiCache disk tier.

Hermetic: pure accounting over injected collaborators, no filesystem, no CUDA.
"""

import pytest
from sglang.srt.mem_cache.disk_tier_control import (
    DetachRefused,
    DiskTierController,
    DiskTierError,
    ShrinkRefused,
)

MB = 1024 * 1024


class _Tier:
    """Backend double: an LRU whose evictor removes WHOLE pages only."""

    def __init__(self, only_copy=()):
        self.pages: dict[str, int] = {}
        self.order: list[str] = []
        self.only_copy = set(only_copy)
        self.truncations = 0

    def write(self, key, nbytes):
        self.pages[key] = nbytes
        self.order.append(key)

    def evict(self, need):
        """Evict whole pages, oldest first, until `need` bytes are freed."""
        freed, out = 0, []
        while self.order and freed < need:
            key = self.order.pop(0)
            freed += self.pages.pop(key, 0)
            out.append(key)
        return out

    def is_only_copy(self, key):
        return key in self.only_copy


def _ctl(tier=None, enabled=True, **kw):
    tier = tier or _Tier()
    return (
        DiskTierController(
            enabled=enabled,
            evict_fn=tier.evict,
            size_fn=lambda k: tier.pages.get(k, 0),
            only_copy_fn=tier.is_only_copy,
            **kw,
        ),
        tier,
    )


def _fill(ctl, tier, n, each=10 * MB):
    for i in range(n):
        key = f"p{i}"
        tier.write(key, each)
        ctl.note_write(key, each)


def test_attach_while_serving_needs_no_restart():
    ctl, _ = _ctl()
    st = ctl.attach("/mnt/l3", 100 * MB)
    assert ctl.attached
    assert st.path == "/mnt/l3" and st.capacity_bytes == 100 * MB
    assert st.used_bytes == 0 and st.free_bytes == 100 * MB


def test_grow_does_not_evict_anything():
    ctl, tier = _ctl()
    ctl.attach("/mnt/l3", 100 * MB)
    _fill(ctl, tier, 5)  # 50 MB
    rep = ctl.resize(200 * MB)
    assert rep.evicted_keys == () and rep.freed_bytes == 0
    assert not rep.shrank
    assert ctl.state().used_bytes == 50 * MB


def test_SHRINK_UNDER_CONTENT_evicts_down_to_the_bound():
    """The core case: a smaller budget removes whole pages, never bytes."""
    ctl, tier = _ctl()
    ctl.attach("/mnt/l3", 100 * MB)
    _fill(ctl, tier, 9)  # 90 MB across 9 pages

    rep = ctl.resize(50 * MB)

    assert rep.shrank
    assert rep.used_after <= 50 * MB
    assert ctl.state().used_bytes <= 50 * MB
    assert len(rep.evicted_keys) == 4, "40 MB freed as four whole 10 MB pages"
    assert rep.freed_bytes == 40 * MB
    # Whole pages only: nothing was cut short.
    assert tier.truncations == 0
    assert all(v == 10 * MB for v in tier.pages.values())


def test_a_shrink_the_evictor_cannot_honour_is_REFUSED_not_partial():
    """A bound the tier cannot reach is a configuration error.

    Accepting it and quietly staying above it would make the reported capacity
    a lie, and the only way to reach it would be truncation.
    """

    class _Stubborn(_Tier):
        def evict(self, need):
            return []  # pinned pages: nothing is evictable

    ctl, tier = _ctl(_Stubborn())
    ctl.attach("/mnt/l3", 100 * MB)
    _fill(ctl, tier, 9)
    with pytest.raises(ShrinkRefused, match="truncated"):
        ctl.resize(10 * MB)
    assert ctl.state().capacity_bytes == 100 * MB, "capacity unchanged on refusal"


def test_shrink_without_an_evictor_refuses_rather_than_truncating():
    ctl = DiskTierController(enabled=True, evict_fn=None)
    ctl.attach("/mnt/l3", 100 * MB)
    ctl.note_write("p", 90 * MB)
    with pytest.raises(ShrinkRefused, match="no evictor"):
        ctl.resize(10 * MB)


def test_DETACH_IS_REFUSED_while_pages_are_only_copy_here():
    """CAN-FAIL: detaching would silently destroy the sole copy."""
    tier = _Tier(only_copy={"p2"})
    ctl, tier = _ctl(tier)
    ctl.attach("/mnt/l3", 100 * MB)
    _fill(ctl, tier, 4)

    with pytest.raises(DetachRefused, match="ONLY here"):
        ctl.detach()
    assert ctl.attached, "a refused detach must leave the tier attached"


def test_detach_succeeds_when_every_page_has_another_copy():
    ctl, tier = _ctl(_Tier(only_copy=set()))
    ctl.attach("/mnt/l3", 100 * MB)
    _fill(ctl, tier, 4)
    ctl.detach()
    assert not ctl.attached


def test_force_detach_is_available_and_loud():
    ctl, tier = _ctl(_Tier(only_copy={"p0"}))
    ctl.attach("/mnt/l3", 100 * MB)
    _fill(ctl, tier, 2)
    ctl.detach(force=True)
    assert not ctl.attached


def test_the_flag_gates_every_mutation():
    ctl, _ = _ctl(enabled=False)
    with pytest.raises(DiskTierError, match="disabled"):
        ctl.attach("/mnt/l3", 100 * MB)


def test_a_second_attach_is_refused():
    """Two directories under one controller would give the evictor two budgets
    and one bound."""
    ctl, _ = _ctl()
    ctl.attach("/mnt/l3", 100 * MB)
    with pytest.raises(DiskTierError, match="already attached"):
        ctl.attach("/mnt/other", 50 * MB)


def test_operations_on_an_unattached_tier_are_refused():
    ctl, _ = _ctl()
    with pytest.raises(DiskTierError, match="no disk tier"):
        ctl.state()
    with pytest.raises(DiskTierError, match="no disk tier"):
        ctl.resize(10 * MB)
    with pytest.raises(DiskTierError, match="no disk tier"):
        ctl.detach()


def test_a_reattach_after_detach_starts_clean():
    ctl, tier = _ctl()
    ctl.attach("/mnt/l3", 100 * MB)
    _fill(ctl, tier, 3)
    ctl.detach()
    st = ctl.attach("/mnt/l3b", 20 * MB)
    assert st.used_bytes == 0 and st.page_count == 0


def test_snapshot_reports_enough_to_act_on():
    ctl, tier = _ctl()
    assert ctl.snapshot()["disk_tier_attached"] == 0
    ctl.attach("/mnt/l3", 100 * MB)
    _fill(ctl, tier, 2)
    snap = ctl.snapshot()
    assert snap["disk_tier_path"] == "/mnt/l3"
    assert snap["disk_tier_used_bytes"] == 20 * MB
    assert snap["disk_tier_free_bytes"] == 80 * MB
    assert snap["disk_tier_pages"] == 2


def test_capacity_must_be_positive():
    ctl, _ = _ctl()
    with pytest.raises(DiskTierError, match="positive"):
        ctl.attach("/mnt/l3", 0)
    ctl.attach("/mnt/l3", 10 * MB)
    with pytest.raises(DiskTierError, match="positive"):
        ctl.resize(-1)
