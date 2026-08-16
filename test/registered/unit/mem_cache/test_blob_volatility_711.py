"""#711: a blob's volatility class must be expressible per put.

From the #423 analysis: the kept recommendation was tier CHOICE per volatility
class — reconstructible bulk to the slow tier, non-reconstructible state to the
fast durable one. That was not expressible: `BlobStore.put(session_id, blob)`
took no class, and `TieredGdnBlobStore` bound ONE tier at construction, so
every blob from a store went to the same place regardless of what losing it
would cost.

Hermetic: no CUDA beyond a CPU tensor, no server.
"""

import pytest

torch = pytest.importorskip("torch")

from sglang.srt.mem_cache.gdn_slot_executor import (
    NON_RECONSTRUCTIBLE,
    RECONSTRUCTIBLE,
    LocalGdnBlobStore,
    TieredGdnBlobStore,
)


class _Tier:
    """A #224 DestinationTier double that records what it was given."""

    def __init__(self, name, accept=True):
        self.name = name
        self.accept = accept
        self.puts = []
        self._store = {}

    def put(self, key, flat):
        if not self.accept:
            return False
        self.puts.append(key)
        self._store[key] = flat.clone()
        return True

    def get_into(self, key, buf):
        if key not in self._store:
            return False
        buf.copy_(self._store[key])
        return True


def _blob():
    return {"temporal": torch.ones(4, dtype=torch.float32)}


def _store(**kw):
    return TieredGdnBlobStore(tier=_Tier("default"), key_fn=lambda s: f"k/{s}", **kw)


def test_the_single_tier_world_CANNOT_express_the_distinction():
    """CAN-FAIL PROOF: this is the gap #711 closes.

    With one tier and no class map, a reconstructible bulk blob and a
    non-reconstructible state blob land in exactly the same place. No caller
    can say otherwise, which is what made the #423 recommendation unbuildable.
    """
    st = _store()
    st.put("a", _blob(), volatility=RECONSTRUCTIBLE)
    st.put("b", _blob(), volatility=NON_RECONSTRUCTIBLE)
    # Both classes resolved to the same destination -- the defect, pinned.
    assert st._resolve_tier(RECONSTRUCTIBLE, None) is st._tier
    # ...except that the non-reconstructible one refuses to ship at all
    # without a durable destination (see the dedicated test below).


def test_each_class_reaches_its_own_tier():
    slow = _Tier("slow-bulk")
    fast = _Tier("fast-durable")
    st = _store(
        tiers_by_class={RECONSTRUCTIBLE: slow, NON_RECONSTRUCTIBLE: fast}
    )
    st.put("bulk", _blob(), volatility=RECONSTRUCTIBLE)
    st.put("state", _blob(), volatility=NON_RECONSTRUCTIBLE)
    assert slow.puts == ["k/bulk"]
    assert fast.puts == ["k/state"]
    assert st._tier.puts == [], "the default tier saw neither"


def test_a_round_trip_reads_from_the_tier_THAT_WROTE_IT():
    """The bug this feature could easily have introduced.

    With per-class tiers the writer is no longer the default, so popping from
    the default would report a perfectly good blob as unrecoverable.
    """
    fast = _Tier("fast-durable")
    st = _store(tiers_by_class={NON_RECONSTRUCTIBLE: fast})
    st.put("s", _blob(), volatility=NON_RECONSTRUCTIBLE)
    got = st.pop("s")
    assert torch.equal(got["temporal"], torch.ones(4))


def test_no_argument_is_byte_identical_to_the_old_behaviour():
    """Backward compatibility: an unclassified put uses the default tier."""
    slow = _Tier("slow-bulk")
    st = _store(tiers_by_class={RECONSTRUCTIBLE: slow})
    st.put("legacy", _blob())
    assert st._tier.puts == ["k/legacy"]
    assert slow.puts == [], "an unclassified put must not be re-routed"


def test_non_reconstructible_stays_LOCAL_when_no_durable_tier_exists():
    """Safety rule from #423, and it is the whole point of the class.

    Losing this blob kills the request. With no durable destination
    configured, keeping it in-process is strictly safer than shipping it to a
    tier that can lose it independently: local dies only with the process,
    which is already fatal for the session.
    """
    st = _store(tiers_by_class={RECONSTRUCTIBLE: _Tier("slow-bulk")})
    st.put("s", _blob(), volatility=NON_RECONSTRUCTIBLE)
    assert st._tier.puts == [], "must NOT ship to the generic default"
    assert st.has("s")
    assert torch.equal(st.pop("s")["temporal"], torch.ones(4))


def test_the_registry_answers_when_the_explicit_map_does_not():
    class _Registry:
        def __init__(self, tier):
            self.tier = tier
            self.asked = []

        def tier_for_class(self, volatility):
            self.asked.append(volatility)
            return self.tier

    fast = _Tier("registry-durable")
    reg = _Registry(fast)
    st = _store(registry=reg)
    st.put("s", _blob(), volatility=NON_RECONSTRUCTIBLE)
    assert reg.asked == [NON_RECONSTRUCTIBLE]
    assert fast.puts == ["k/s"]


def test_the_explicit_map_outranks_the_registry():
    class _Registry:
        def tier_for_class(self, volatility):
            raise AssertionError("the registry must not be consulted")

    explicit = _Tier("explicit")
    st = _store(
        tiers_by_class={RECONSTRUCTIBLE: explicit}, registry=_Registry()
    )
    st.put("s", _blob(), volatility=RECONSTRUCTIBLE)
    assert explicit.puts == ["k/s"]


def test_a_tier_hint_outranks_everything():
    hinted = _Tier("hinted")
    st = _store(tiers_by_class={RECONSTRUCTIBLE: _Tier("slow")})
    st.put("s", _blob(), volatility=RECONSTRUCTIBLE, tier_hint=hinted)
    assert hinted.puts == ["k/s"]


def test_the_failure_fallback_is_preserved_per_class():
    """A refusing tier still means "stay local", not "lose the blob"."""
    refusing = _Tier("full", accept=False)
    st = _store(tiers_by_class={RECONSTRUCTIBLE: refusing})
    st.put("s", _blob(), volatility=RECONSTRUCTIBLE)
    assert st.has("s")
    assert torch.equal(st.pop("s")["temporal"], torch.ones(4))


def test_an_unknown_class_is_refused_loudly():
    """A typo must not silently pick the default tier -- that is the failure
    #711 exists to remove, reappearing as a spelling mistake."""
    st = _store()
    with pytest.raises(ValueError, match="unknown volatility class"):
        st.put("s", _blob(), volatility="reconstructable")


def test_the_local_store_accepts_the_new_arguments_and_ignores_them():
    """The local tier never leaves the process, so no class can be misplaced."""
    loc = LocalGdnBlobStore()
    loc.put("s", _blob(), volatility=NON_RECONSTRUCTIBLE)
    assert loc.has("s")
    loc.put("t", _blob())
    assert len(loc) == 2
