"""#928 -- a radix mamba anchor must be read from the pool that HOLDS ITS
BYTES, not from the pool of whichever phase happens to be computing.

THE GARBAGE THIS CLOSES, measured on metal 2026-08-27 (boot 2g,
``/spinning/evidence-665-f1/boot_2f_698cd396ce_0827_0704.log``, and the 2g-2
length ladder): under strict phase purity every prompt -- 718 tokens as much as
9447, one prefill chunk as much as three -- came back with text that had nothing
to do with the prompt, and a second send of the SAME prompt at temperature 0
came back with DIFFERENT garbage. One send answered with a single EOS token.

WHY IT IS EVERY REQUEST. Under strict purity the prefill runs in the PP phase;
the request is then retracted at the ``pp_to_tp`` cutover and RE-ADMITTED in the
TP phase against its own just-cached prefix. The log shows the re-admission
verbatim (:3805/:3808, :4672/:4673)::

    ADMIT prefix_lens=9447 ...
    Prefill batch phase=tp, #new-seq: 1, #new-token: 1, #cached-token: 9447

So the decode's recurrent state does NOT come out of the live mamba slot. It
comes out of the radix anchor, through the deferred COW at
``model_executor/model_runner.py:4368-4373``::

    pool.mamba_pool.copy_from(src_anchor_slot, dst_request_slot)

with ``pool = self.req_to_token_pool`` (:4317) -- the pool of the runner that is
executing, i.e. the TP stack's.

WHERE THE BYTES ACTUALLY ARE. The anchor was produced during the PP prefill.
On this configuration (``--enable-mamba-extra-buffer``; the boot's
``mamba_ckpt_size=None`` rules out the int8 lineage) the donate branch is
``mem_cache/unified_cache_components/mamba_component.py:891-899``, which copies
NOTHING: it hands the tree the ping-pong slot ID whose conv/ssm bytes the
prefill forward wrote in place, into the COMPUTING STACK'S tensors. The anchor
is a bare slot id with no record of which stack backs it.

AND NOTHING MOVES THEM. ``gdn_flip_mover`` moves the mamba state of RESIDENT
slots at a cutover; the retracted request has none, and the log agrees --
``PHASE-FLIP DONE tp_to_pp ... 0 live slots, sent 0 cells / 0.00 MiB``. Cached
anchors are never moved.

The two stacks' pools are distinct objects; ``gdn_flip_mover.py:848-851`` names
both::

    primary_pool = scheduler.req_to_token_pool.mamba_pool
    tp_pool = scheduler.phase_flip_stacks.tp_worker.model_runner
                       .req_to_token_pool.mamba_pool

So the resume reads slot N of the TP pool while the bytes sit at slot N of the
PP pool. Slot N of the TP pool holds whatever the TP stack last left there --
which is why three identical sends at temperature 0 produced three different
answers, and why the failure needs no chunk boundary to appear.

THIS IS #767'S UNFIXED HALF, AND THE FIX SAYS SO ITSELF.
``install_phase_aware_mamba_state_pool`` (gdn_flip_mover.py:838-846) scopes
itself in its own docstring to "only the byte copies (checkpoint donation copy,
int8 store)" -- the WRITE side. Every one of the six call sites of
``active_mamba_state_pool`` is a donate/store (mamba_component.py:701, 880, 916;
mamba_radix_cache.py:899, 973, 1712). There is not one resume site. The module
header of ``mem_cache/mamba_state_pool.py`` describes this exact failure from
the other direction, measured 2026-08-19: "checkpoint slots filled with a
previous PP-phase occupant's state (a kite prompt answered with a foreign river
essay)".

THE RULE THIS PINS, and it is why the resolver as written cannot be the whole
answer: ``active_mamba_state_pool`` resolves by WHO IS COMPUTING NOW. An
anchor's bytes are fixed at the moment it was donated. The two agree only while
the phase does not change between donate and resume -- which under strict
purity is never. State-byte access for a TREE-HELD anchor must resolve by
PROVENANCE (the stack that wrote it), not by current phase.

THE SILENTLY-WRONG SITE, for the record, is
``MambaComponent.finalize_match_result`` (mamba_component.py:310-347): it hands
back ``req.mamba_cow_src_index = mamba_value`` and clears
``req.mamba_needs_clear`` without any check that the anchor's bytes are
reachable from the phase that will consume them, and with no ``else`` for
``mamba_value is None``. Thirty lines below, the same function states the
correct rule for a different failure (:340-344): "Reusing the KV prefix without
the matching mamba state would be silently wrong, so the whole match is zeroed."
That is the refusal this seam needs and does not have -- a refusal costs a
re-prefill, the silence costs a wrong answer.

CORROBORATED FROM THE OTHER SIDE (#929 analysis,
/spinning/evidence-665-f1/ANALYSE_929_mamba_slot_deficit.md): at the re-admit
insert the tree answers ``mamba_exist=True`` with ``len(key) == 0``
(unified_radix_cache.py:1569/:1668) -- the whole key, full prefix, IS in the
tree and DOES carry a mamba value -- and the decode is garbage anyway. That
removes "no anchor" as the explanation and leaves only "the anchor was read
from the wrong place", which is what this test pins. Anchor DEPTH is not the
remaining suspect either: the donate pairs key and state at one position by
construction (``cache_len = req.mamba_last_track_seqlen``,
mamba_component.py:721, and mamba_ckpt_utils' "Never floor" rule). Provenance
is the term that is unheld.

NOT THE SAME BUG AS #929'S ORPHANED SLOT, and the counter-check is cheap. On
the plain-finished path with ``mamba_exist=True`` the freshly allocated donate
slot (mamba_component.py:902-903) is dropped without being freed:
``cleanup_after_caching_req``'s ``enable_mamba_extra_buffer`` branch (:947-956)
frees only ``req``'s own slots, and the ``insert_params.mamba_value`` free
exists only on the int8 branch (:942-943) and the unfinished branch (:961-963).
That slot leaks -- but it leaks precisely BECAUSE it was never attached to a
node, and the resume's COW source is ``last_node.component_data[ct].value``,
i.e. a slot the TREE holds. Orphaned-and-unreachable and tree-held are disjoint
sets, so the resume can never read the orphan. #929 explains the rank-0 +1 slot
divergence as one insert-side booking; it is not a second root of this garbage.

THE TEST drives the real ``install_phase_aware_mamba_state_pool`` and the real
``active_mamba_state_pool`` over two distinct pool objects and asks the seam the
question the resume asks it: with the anchor donated while PP computed, which
pool does the seam name once TP is computing? No source-text inspection; the
assertion is on the object the shipped resolver returns.
"""

import unittest

from sglang.srt.managers.gdn_flip_mover import install_phase_aware_mamba_state_pool
from sglang.srt.managers.phase_flip_runtime import PHASE_PP, PHASE_TP
from sglang.srt.mem_cache.mamba_state_pool import active_mamba_state_pool


class _Pool:
    """Stands in for a stack's mamba STATE pool. Identity is the whole point:
    the two stacks own different tensors at the same slot ids."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:  # pragma: no cover - failure message only
        return f"<mamba pool {self.name}>"


class _ReqToTokenPool:
    def __init__(self, mamba_pool: _Pool) -> None:
        self.mamba_pool = mamba_pool


class _ModelRunner:
    def __init__(self, pool: _ReqToTokenPool) -> None:
        self.req_to_token_pool = pool


class _Worker:
    def __init__(self, runner: _ModelRunner) -> None:
        self.model_runner = runner


class _Stacks:
    def __init__(self, tp_worker: _Worker) -> None:
        self.tp_worker = tp_worker


class _TreeCache:
    """The radix cache: bound once to the PRIMARY stack's pool, never rebound
    (mem_cache/mamba_state_pool.py module header)."""

    def __init__(self, bound: _ReqToTokenPool) -> None:
        self.req_to_token_pool = bound


class _Scheduler:
    def __init__(self) -> None:
        self.req_to_token_pool = _ReqToTokenPool(_Pool("pp-primary"))
        self.phase_flip_stacks = _Stacks(
            _Worker(_ModelRunner(_ReqToTokenPool(_Pool("tp"))))
        )
        self.tree_cache = _TreeCache(self.req_to_token_pool)
        # Unset until the first cutover: the primary stack computes.
        self.phase_flip_active_stack = PHASE_PP


class TestAnchorBytesFollowProvenance928(unittest.TestCase):
    def test_anchor_donated_in_pp_is_still_resolved_to_pp_bytes_in_tp(self):
        sched = _Scheduler()
        install_phase_aware_mamba_state_pool(sched)
        cache = sched.tree_cache

        # PREFILL PHASE. The anchor's conv/ssm bytes are written here, into
        # whichever pool is computing. This is the pool that backs the slot id
        # the tree stores.
        donated_from = active_mamba_state_pool(cache)
        self.assertIs(
            donated_from,
            sched.req_to_token_pool.mamba_pool,
            "precondition: the PP stack computes the prefill",
        )

        # THE CUTOVER. Nothing moves a cached anchor: gdn_flip_mover moves
        # resident slots only, and the retracted request has none
        # ("0 live slots, sent 0 cells / 0.00 MiB").
        sched.phase_flip_active_stack = PHASE_TP

        # DECODE PHASE. The request is re-admitted against its own cached
        # prefix and the deferred COW reads the anchor's bytes.
        resumed_from = active_mamba_state_pool(cache)

        self.assertIs(
            resumed_from,
            donated_from,
            "the anchor's bytes were written into "
            f"{donated_from} during the PP prefill, but the seam names "
            f"{resumed_from} for the TP resume: the resume reads slot N of a "
            "pool that never held this request's recurrent state. "
            "active_mamba_state_pool resolves by WHO COMPUTES NOW; a "
            "tree-held anchor must resolve by WHO WROTE IT. Under strict "
            "phase purity every request donates in PP and resumes in TP, so "
            "every request reads a foreign stack's bytes",
        )


if __name__ == "__main__":
    unittest.main()
