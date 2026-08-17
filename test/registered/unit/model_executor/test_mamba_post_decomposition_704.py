"""#704: split the lumped mamba budget post into its three named components.

The post has been labelled "mamba state pool + speculative intermediate state +
prefill activation reserve" all along, but emitted as ONE number. That lump is
why the superset-smaller-than-its-parts gap could not be attributed: the post
nominally covers MORE than the "Mamba Cache is allocated" line (it adds the
prefill activation reserve) yet measures 0.155 / 0.111 / 0.089 GiB LESS on
PP0/PP1/PP2. A term covering more cannot legitimately measure less, but with
one number there is no way to say which of the three carries it.

This is the same missing-instrument fix for the third time in this ticket
(budget posts, then available_bytes, now the post's own components), so the
decomposition is a pure function that can be tested without a boot.

Two properties matter and are pinned separately:

* the parts SUM to the lump exactly -- a decomposition that changed the total
  would silently move the KV budget;
* the residual is named, not dropped. Whatever the two measured components do
  not explain IS the state pool, and calling it that keeps the arithmetic
  closed instead of leaving an anonymous remainder.

Hermetic: pure arithmetic, no CUDA.
"""

import pytest

from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
    MAMBA_BUDGET_POST,
    decompose_mamba_budget_post,
)


def test_the_parts_sum_to_the_lump():
    posts = decompose_mamba_budget_post(
        total_gb=0.895,
        components={
            "speculative intermediate state": 0.620,
            "prefill activation reserve": 0.050,
        },
    )
    assert sum(gb for _, gb in posts) == pytest.approx(0.895, rel=1e-12)


def test_the_residual_is_named_as_the_state_pool():
    posts = decompose_mamba_budget_post(
        total_gb=0.895,
        components={
            "speculative intermediate state": 0.620,
            "prefill activation reserve": 0.050,
        },
    )
    names = [n for n, _ in posts]
    assert names[0] == "mamba state pool"
    assert dict(posts)["mamba state pool"] == pytest.approx(0.225, rel=1e-9)
    assert "speculative intermediate state" in names
    assert "prefill activation reserve" in names


def test_no_components_falls_back_to_the_lump_unchanged():
    """Other budget branches never populate the components; they must not break."""
    posts = decompose_mamba_budget_post(total_gb=0.639, components={})
    assert posts == [(MAMBA_BUDGET_POST, 0.639)]


def test_a_negative_residual_is_reported_not_hidden():
    """If the named parts exceed the lump, that is the bug -- surface it.

    Silently clamping to zero would hide exactly the class of accounting error
    this instrument exists to find.
    """
    posts = decompose_mamba_budget_post(
        total_gb=0.100,
        components={"speculative intermediate state": 0.620},
    )
    pool = dict(posts)["mamba state pool"]
    assert pool < 0
    assert sum(gb for _, gb in posts) == pytest.approx(0.100, rel=1e-12)


def test_zero_valued_components_are_still_emitted():
    """A zero that was MEASURED is information; omitting it looks like absence."""
    posts = decompose_mamba_budget_post(
        total_gb=0.500,
        components={
            "speculative intermediate state": 0.0,
            "prefill activation reserve": 0.0,
        },
    )
    names = [n for n, _ in posts]
    assert "speculative intermediate state" in names
    assert "prefill activation reserve" in names
    assert dict(posts)["mamba state pool"] == pytest.approx(0.500, rel=1e-12)


def test_the_lump_label_still_names_all_three_parts():
    """Guards the labels from drifting apart from the decomposition."""
    for part in (
        "mamba state pool",
        "speculative intermediate state",
        "prefill activation reserve",
    ):
        assert part in MAMBA_BUDGET_POST
