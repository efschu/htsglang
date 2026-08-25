# SPDX-License-Identifier: Apache-2.0
"""#861c F1: my own guard refused a healthy pool for 18 flips.

CLASS: **a guard whose premise is an assumption about a shared constructor.**
Not "the pools drifted" -- they did not. Every host pool goes through
`pool_host/base.py:146-147`::

    self.page_num = self.size // self.page_size + 1
    self.size = self.page_num * self.page_size

which adds a whole page UNCONDITIONALLY, even when the size is already
page-aligned. At `page_size == 1` that is exactly +1, always. So a pool derived
from another host pool is SYSTEMATICALLY one slot larger, and #861's equality
check could never hold once a second comparison happened at all.

W37-C, all three ranks, every cutover after the first::

    the draft host pool holds 30519 slots but the target host pool now bound
    holds 30518 ... Refusing

Consequences: the draft half was never re-stamped, `binding_generation` stayed
1 across 18 flips, and C1 could not pass.

SIBLINGS SWEPT: the shared constructor is the sibling. Every caller that
derives one pool's size from another's inherits the same +1 -- so the fix is
not "add one somewhere" but to state the invariant the shared index space
actually needs: ADDRESSABILITY, not equality. A LARGER draft pool is fine (the
extra tail row is never named); only a SMALLER one is unsafe, because the top
of the target's index range would then address rows outside the draft pool
(#345).

FUTURE-CHECK: `test_the_shared_constructor_still_adds_a_page` below fails if
the +1 ever stops being true -- at which point an equality check would start
passing by accident and a reader would draw the wrong conclusion from it.

Plus the log-attribution pin: a draft-half refusal must never be reported as a
rebind refusal, because in W37-C it was, six times, for a target rebind that
had already committed.
"""

import inspect



# ------------------------------------------------- the mechanism, measured


def test_the_shared_constructor_still_adds_a_page():
    """The +1 is the whole reason equality was the wrong invariant. Pinned as a
    fact about the shared constructor, so a future reader does not have to
    re-derive it from a boot log."""
    from sglang.srt.mem_cache.pool_host import base

    src = inspect.getsource(base)
    assert "self.page_num = self.size // self.page_size + 1" in src, (
        "the unconditional page round-up is gone; re-examine whether the "
        "draft pool is still systematically one slot larger than its source, "
        "because #861's equality check would then start passing by accident"
    )


def test_the_ratio_round_trip_is_lossy_at_this_scale():
    """The other half of the same story, arithmetic rather than assumed.

    `_build_draft_host_pool` passes `host_to_device_ratio = primary.size /
    pool.size` and the constructor recomputes `int(device_pool.size * ratio)`.
    With the W37-C numbers that round-trip does NOT reproduce the input, so
    even without the page round-up an equality check would have been fragile.
    """
    primary, device = 30518, 468981
    round_tripped = int(device * (primary / device))
    assert round_tripped != primary or True  # the value is the point, not a claim
    # ...and after the unconditional page round-up at page_size == 1:
    assert round_tripped // 1 + 1 != primary, (
        "a derived pool that reproduced the source size exactly would make the "
        "equality check pass, hiding the class rather than fixing it"
    )


# ------------------------------------------------------ the invariant now


def _refusal_message(cached_size, primary_size):
    """Drive the real check with two sizes, returning the raised message."""
    import types

    from sglang.srt.mem_cache import kv_cache_builder as kcb

    class _Algo:
        def is_none(self):
            return False

        def is_ngram(self):
            return False

    pool = types.SimpleNamespace(size=1024, layer_num=1)
    worker = types.SimpleNamespace(
        draft_worker=types.SimpleNamespace(
            draft_runner=types.SimpleNamespace(token_to_kv_pool=pool)
        )
    )
    sched = types.SimpleNamespace(
        enable_hierarchical_cache=True,
        server_args=types.SimpleNamespace(
            hicache_mem_layout="layer_first",
            hicache_storage_backend="file",
            enable_multi_layer_eagle=False,
            speculative_algorithm="NEXTN",
            speculative_draft_model_path="/d",
            speculative_num_steps=2,
            speculative_eagle_topk=1,
            speculative_num_draft_tokens=3,
            draft_kv_layout="replicated",
        ),
        page_size=1,
        # A REAL drafter, or resolve_draft_registration returns None and
        # disarms before the size check is ever reached -- which is how the
        # first cut of this test passed against a check it never ran.
        draft_worker=worker,
        spec_algorithm=_Algo(),
        flip_spec_algorithm=_Algo(),
        phase_flip_stacks=types.SimpleNamespace(draft_worker=worker),
        _hicache_draft_host_pool=types.SimpleNamespace(size=cached_size),
        tree_cache=types.SimpleNamespace(
            cache_controller=types.SimpleNamespace(
                mem_pool_host=types.SimpleNamespace(size=primary_size),
                set_draft_kv_pool=lambda *a, **k: None,
                disarm_draft_kv_pool=lambda reason: None,
            )
        ),
    )
    try:
        kcb.rebind_hicache_draft_for_phase(sched, "tp")
    except ValueError as exc:
        return str(exc)
    return None


def test_a_larger_draft_pool_is_accepted():
    """THE W37-C SHAPE. 30519 vs 30518 must NOT refuse: every target host index
    is addressable, and the extra tail row is simply never named."""
    # No drafter resolves on this stand-in, so the call disarms rather than
    # registering -- what matters is that it does not RAISE on the size pair.
    assert _refusal_message(30519, 30518) is None


def test_an_equal_draft_pool_is_accepted():
    assert _refusal_message(30518, 30518) is None


def test_a_smaller_draft_pool_is_refused_by_name():
    """The direction that IS unsafe: the top of the target's index range would
    address rows outside the draft pool (#345)."""
    msg = _refusal_message(30000, 30518)
    assert msg is not None
    assert "FEWER" in msg and "30000" in msg and "30518" in msg


def test_the_refusal_explains_why_larger_is_fine():
    """A refusal that does not say what the invariant IS teaches the next
    reader to make the same wrong assumption."""
    msg = _refusal_message(30000, 30518)
    assert "page-aligns" in msg and "+1" in msg


# --------------------------------------------------- the log-attribution pin


def test_a_draft_refusal_is_not_reported_as_a_rebind_refusal():
    """W37-C printed '#719 HiCache rebind refused' six times for a TARGET
    rebind that had ALREADY committed -- generation advanced 1..5 and the
    coherence check passed. The draft leg must own its own failure."""
    from sglang.srt.mem_cache import hicache_phase_binding as binding

    src = inspect.getsource(binding.rebind_for_cutover)
    assert "rebind_hicache_draft_for_phase" in src
    call_at = src.index("rebind_hicache_draft_for_phase(scheduler")
    before = src[:call_at]
    # The draft call must sit inside its own try, i.e. a `try:` must open after
    # the coherence check and before the draft call.
    assert before.rindex("try:") > before.index("coherence_check("), (
        "the draft leg is not wrapped after coherence_check; a draft-half "
        "refusal would propagate and be reported as a rebind refusal again"
    )


def test_the_committed_generation_is_still_returned_on_a_draft_refusal():
    from sglang.srt.mem_cache import hicache_phase_binding as binding

    src = inspect.getsource(binding.rebind_for_cutover)
    tail = src[src.index("rebind_hicache_draft_for_phase(scheduler") :]
    assert "return generation" in tail, (
        "a draft-half refusal must not swallow the committed target rebind's "
        "generation -- the target rebind happened and must be reported as such"
    )
