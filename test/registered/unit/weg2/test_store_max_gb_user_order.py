"""`--store-max-gb`: the store's size as the operator set it, not as it derives.

USER ORDER 2026-09-10, verbatim: "der l3 auf der platte hatte ich mal auf 150gb
groesse festgelegt, nicht 28gb".

#1236 sizes the store as `P pool bytes x --store-sidecar-factor`, where the
factor is a MEASURED census ratio (3.02, from boot weg2sb5g's own store census)
with its provenance recorded beside it. That answers "how big must the store be
to hold one prefill leg plus its sidecars" -- a FLOOR. It cannot express "the
operator gave this store 150 GB of disk", and bending the factor to hit an
absolute size would make a measured ratio into a size knob and put the store's
size at the mercy of the P cut: re-solve the cut, and the store silently moves.

So the absolute size is its own flag, in GB (10^9) -- the unit the user used and
the unit `HiCacheFile`'s `max_size` already speaks.

THE FLOOR STILL BINDS. A store below the P pool cannot hold what one prefill leg
produces; boot weg2sb5g logged that refusal 1,324 times. A user value under the
pool is therefore REFUSED BY NAME (W57) rather than clamped: an operator asking
for less than the law allows should be told, not silently overridden.
"""

import pytest

from sglang.srt.weg2.launcher import (
    STORE_SIDECAR_FACTOR,
    Weg2StoreDiskRefused,
    plan_store,
)

# The shipped P cut of the serving base: 461690 tokens, 16 attn layers,
# 2048 B/token/attn-layer -> 32768 B/token.
POOL_TOKENS = 461690
ATTN = (9, 4, 3)
KV_MIB = 2048 / (1024 * 1024)


def _plan(tmp_path, **kw):
    return plan_store("t", POOL_TOKENS, ATTN, KV_MIB, root=str(tmp_path), **kw)


def test_the_user_set_size_is_what_ships(tmp_path):
    """150 GB means 150 GB, not 150 x something."""
    plan = _plan(tmp_path, store_max_gb=150.0)
    assert plan.max_size_bytes == 150_000_000_000


def test_the_derived_size_is_unchanged_when_the_flag_is_absent(tmp_path):
    """Default 0 = derive exactly as before -- the flag adds no behaviour to a
    boot that does not pass it."""
    plan = _plan(tmp_path)
    expected = plan.p_pool_bytes * STORE_SIDECAR_FACTOR
    assert plan.max_size_bytes == pytest.approx(expected, rel=1e-9)
    assert plan.max_size_bytes < 150_000_000_000, (
        "the derived size is the small one -- 'nicht 28gb' is what the order is about"
    )


def test_a_size_below_the_p_pool_is_refused_by_name_not_clamped(tmp_path):
    """#1236's law, kept: store >= P pool."""
    with pytest.raises(Weg2StoreDiskRefused) as e:
        _plan(tmp_path, store_max_gb=1.0)
    msg = str(e.value)
    assert "W57" in msg
    assert "BELOW this boot's P pool" in msg
    assert "--store-max-gb to at least" in msg, "the refusal must name the fix"


def test_the_flag_beats_the_factor_rather_than_multiplying_it(tmp_path):
    """An absolute size must not be scaled by the sidecar factor as well."""
    plan = _plan(tmp_path, store_max_gb=150.0, sidecar_factor=9.0)
    assert plan.max_size_bytes == 150_000_000_000


def test_min_free_is_independent_of_the_size_override(tmp_path):
    plan = _plan(tmp_path, store_max_gb=150.0, min_free_gib=32.0)
    assert plan.min_free_bytes == int(round(32.0 * (1 << 30)))
    assert plan.needed_bytes == plan.max_size_bytes + plan.min_free_bytes
