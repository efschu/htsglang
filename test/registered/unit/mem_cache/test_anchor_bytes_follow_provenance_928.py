"""#928 -- a radix mamba anchor must be read from the pool that HOLDS ITS
BYTES, and refused when that pool is not the one computing.

THE GARBAGE THIS CLOSES, measured on metal 2026-08-27 (boot 2g,
``/spinning/evidence-665-f1/boot_2f_698cd396ce_0827_0704.log``, and the 2g-2
length ladder): under strict phase purity every prompt -- 718 tokens as much as
9447, one prefill chunk as much as three -- came back with text that had nothing
to do with the prompt, and a second send of the SAME prompt at temperature 0
came back with DIFFERENT garbage. One send answered with a single EOS token.

WHY IT IS EVERY REQUEST. Strict purity prefills in PP, retracts the request at
the ``pp_to_tp`` cutover and RE-ADMITS it in TP against its own just-cached
prefix (:3805/:3808 and :4672/:4673 -- ``ADMIT prefix_lens=9447`` then
``Prefill batch phase=tp, #new-seq: 1, #new-token: 1, #cached-token: 9447``).
The decode's recurrent state therefore does not come from the live mamba slot;
it comes from the radix anchor, through the deferred COW at
``model_executor/model_runner.py:4368-4373`` with ``pool =
self.req_to_token_pool`` (:4317) -- the EXECUTING runner's pool.

WHERE THE BYTES ARE. The anchor was produced by the PP prefill. This rig runs
``mamba_radix_cache_strategy='no_buffer'``, and ``enable_mamba_extra_buffer`` is
True only for the two ``extra_buffer`` strategies (server_args.py:19472-19476),
so the branch that actually executes is the plain one at
``mem_cache/unified_cache_components/mamba_component.py:901-918``. It DOES copy
-- and it copies into ``active_mamba_state_pool(self.cache)``, i.e. the pool of
the phase computing AT DONATE TIME. (``mamba_ckpt_size=None`` rules out the int8
lineage.) So the anchor's bytes are deliberately placed in the PP pool, and the
anchor itself is a bare slot id with no record of that. Nothing moves them
afterwards either -- ``gdn_flip_mover`` moves RESIDENT slots, and the log reads
``PHASE-FLIP DONE tp_to_pp ... 0 live slots, sent 0 cells / 0.00 MiB``.

THE ASYMMETRY IN TWO LINES. The WRITE resolves by phase
(``active_mamba_state_pool``, mamba_state_pool.py:32-35, via
``phase_active_mamba_pool``). The READ does not: ``model_runner.py:4368-4373``
uses ``pool.mamba_pool`` unconditionally, and ``model_runner.py`` contains no
occurrence of the phase-aware accessor at all. Across a flip the writer and the
reader therefore resolve to different pool objects.

AND THE ACCESSOR ALONE CANNOT FIX THE READ SIDE, which is why this seam is a
ledger and not one more call to it: ``active_mamba_state_pool`` answers "who
computes NOW", and at a TP resume the answer is TP while the anchor sits in PP.
The read side needs the anchor's PROVENANCE -- what the donate wrote down --
not the current phase.

WHY THE HOOK IS AT THE TREE INSERT AND NOT IN A DONATE BRANCH. There are four
donate branches (int8 extra-buffer, int8 plain, extra-buffer ping-pong,
no_buffer plain) and they disagree about whether they copy at all. Teaching
provenance to one of them would have been inert on any rig running another --
the enablement-gap class. ``commit_insert_component_data`` is the single place
an anchor becomes a node's value, so recording there covers every branch by
construction, and ``test_provenance_is_recorded_at_the_tree_insert_for_any_branch``
pins that it is the insert, not a branch, that carries the rule.

The two pools are distinct objects, named together at
``gdn_flip_mover.py:848-851``. So the resume read slot N of the TP pool while
the state sat at slot N of the PP pool, and slot N of the TP pool holds whatever
the TP stack last left there -- which is why three identical sends produced
three different answers, and why no chunk boundary is needed to see it.

CORROBORATED FROM THE OTHER SIDE (#929 analysis,
/spinning/evidence-665-f1/ANALYSE_929_mamba_slot_deficit.md): at the re-admit
insert the tree answers ``mamba_exist=True`` with ``len(key) == 0``
(unified_radix_cache.py:1569/:1668) -- the whole key IS in the tree and DOES
carry a mamba value -- and the decode is garbage anyway. "No anchor" is
therefore not the explanation; "the anchor was read from the wrong place" is.
Anchor DEPTH is not the open term either: the donate pairs key and state at one
position by construction (``cache_len = req.mamba_last_track_seqlen``,
mamba_component.py:721, and mamba_ckpt_utils' "Never floor" rule).

NOT #929'S ORPHANED SLOT, and the counter-check is cheap enough to state. On the
plain-finished path with ``mamba_exist=True`` the freshly allocated donate slot
(mamba_component.py:902-903) is dropped without a free -- but it leaks precisely
BECAUSE it was never attached to a node, and the resume's COW source is
``last_node.component_data[ct].value``, a slot the tree HOLDS.
Orphaned-unreachable and tree-held are disjoint sets, so no resume can read the
orphan. #929's rank-0 +1 slot is an insert-side booking, not a second root here.

#767'S UNFIXED HALF, AND THE FIX SAYS SO. ``install_phase_aware_mamba_state_pool``
(gdn_flip_mover.py:838-846) scopes itself in its own docstring to "only the byte
copies (checkpoint donation copy, int8 store)". All six call sites of
``active_mamba_state_pool`` are donate/store (mamba_component.py:701, 880, 916;
mamba_radix_cache.py:899, 973, 1712). Not one resume site.
``mem_cache/mamba_state_pool.py``'s header describes the same failure from the
write side, measured 2026-08-19: "a kite prompt answered with a foreign river
essay".

THE RULE. ``active_mamba_state_pool`` resolves by WHO COMPUTES NOW. An anchor's
bytes are fixed when it is donated. The two agree only if the phase does not
change between donate and resume, which under strict purity is never. A
tree-held anchor's byte access must resolve by PROVENANCE, and where the bytes
are unreachable from the consuming layout the match must be REFUSED, not read
anyway -- the rule ``finalize_match_result`` already states for its neighbour
case at :340-344.

WHY REFUSE RATHER THAN FETCH. The PP pool is layer-axis sharded and the TP pool
head-axis sharded; translating between them is what ``gdn_flip_mover`` does, on
a plan and a collective. A per-request copy at match time cannot. Carrying
cached anchors across the cutover the way resident slots already are is the
CAPACITY fix and is a separate posten; it restores the hit, it is not what makes
the answer right.

WHAT EACH TEST HOLDS DOWN
  1. foreign provenance is detected and refused  -- the defect itself;
  2. same-phase provenance is NOT refused        -- the mutant guard: a fix
     that refuses everything is not a fix, it is an outage;
  3. single-pool boots are never refused         -- non-flip builds unchanged;
  4. an unrecorded anchor on a split boot is refused -- the conservative
     direction, because an anchor this seam cannot vouch for costs a
     re-prefill if refused and a wrong answer if trusted;
  5. the ledger follows a REUSED slot            -- the danger direction for a
     ledger: a stale entry from the slot's previous tenant must not be able to
     vouch for the new one;
  6. provenance is recorded by the TREE INSERT, not by a donate branch -- the
     enablement guard, so the rule cannot be inert on the strategy this rig
     actually runs.
"""

import unittest

from sglang.srt.managers.gdn_flip_mover import install_phase_aware_mamba_state_pool
from sglang.srt.managers.phase_flip_runtime import PHASE_PP, PHASE_TP
from sglang.srt.mem_cache.mamba_state_pool import (
    active_mamba_state_pool,
    anchor_bytes_reachable,
    anchor_provenance_verdict,
    note_anchor_bytes,
)


class _Pool:
    """A stack's mamba STATE pool. Identity is the whole point: the two stacks
    own different tensors at the same slot ids."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:  # pragma: no cover - failure messages only
        return f"<mamba pool {self.name}>"


class _ReqToTokenPool:
    def __init__(self, mamba_pool: _Pool) -> None:
        self.mamba_pool = mamba_pool
        self.mamba_ckpt_pool = None  # int8 lineage off, as on the shipping boot


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


def _anchor(slot: int):
    """An anchor value as the tree stores it: a slot id, nothing else."""
    import torch

    return torch.tensor([slot], dtype=torch.int64)


class TestAnchorBytesFollowProvenance928(unittest.TestCase):
    def _flip_boot(self):
        sched = _Scheduler()
        install_phase_aware_mamba_state_pool(sched)
        return sched, sched.tree_cache

    def test_anchor_donated_in_pp_is_refused_when_tp_computes(self):
        """The defect: PP writes the bytes, TP reads the same slot id."""
        sched, cache = self._flip_boot()

        # PREFILL PHASE: the anchor's bytes are written into the computing
        # stack's tensors and the tree is handed the slot id.
        anchor = _anchor(7)
        note_anchor_bytes(cache, anchor)
        self.assertIs(
            active_mamba_state_pool(cache),
            sched.req_to_token_pool.mamba_pool,
            "precondition: the PP stack computes the prefill",
        )
        self.assertEqual(anchor_provenance_verdict(cache, anchor), "same")

        # THE CUTOVER. Nothing moves a cached anchor: the mover moves resident
        # slots only, and the retracted request has none.
        sched.phase_flip_active_stack = PHASE_TP

        # DECODE PHASE: the request is re-admitted against its own cached
        # prefix and the deferred COW would read slot 7 of the TP pool.
        self.assertEqual(
            anchor_provenance_verdict(cache, anchor),
            "foreign",
            "the seam must name the anchor's bytes as belonging to the other "
            "stack once the phase has changed",
        )
        self.assertFalse(
            anchor_bytes_reachable(cache, anchor),
            "the resume would read slot 7 of a pool that never held this "
            "request's recurrent state; the match must be refused, not read",
        )

    def test_same_phase_resume_is_not_refused(self):
        """MUTANT GUARD. Refusing every resume is an outage, not a fix."""
        sched, cache = self._flip_boot()
        sched.phase_flip_active_stack = PHASE_TP
        anchor = _anchor(7)
        note_anchor_bytes(cache, anchor)  # donated while TP computes

        self.assertTrue(
            anchor_bytes_reachable(cache, anchor),
            "an anchor donated and resumed inside the same phase is readable "
            "and must keep its cache hit",
        )
        self.assertEqual(anchor_provenance_verdict(cache, anchor), "same")

    def test_single_pool_boot_is_never_refused(self):
        """MUTANT GUARD. Non-flip builds have one pool and must be unchanged."""
        sched = _Scheduler()  # no resolver installed: one pool, no phases
        cache = sched.tree_cache
        anchor = _anchor(7)
        note_anchor_bytes(cache, anchor)

        self.assertTrue(anchor_bytes_reachable(cache, anchor))
        self.assertEqual(anchor_provenance_verdict(cache, anchor), "single")

    def test_unrecorded_anchor_on_a_split_boot_is_refused(self):
        """The conservative direction for an anchor nothing vouched for."""
        _sched, cache = self._flip_boot()
        never_recorded = _anchor(11)

        self.assertEqual(
            anchor_provenance_verdict(cache, never_recorded), "unknown"
        )
        self.assertFalse(
            anchor_bytes_reachable(cache, never_recorded),
            "refusing an unvouched anchor costs a re-prefill; trusting one "
            "costs the wrong answer",
        )

    def test_a_reused_slot_does_not_inherit_the_previous_tenants_pool(self):
        """DANGER DIRECTION FOR A LEDGER. Slot ids are recycled."""
        sched, cache = self._flip_boot()
        slot = _anchor(3)

        note_anchor_bytes(cache, slot)  # tenant 1, donated under PP
        sched.phase_flip_active_stack = PHASE_TP
        note_anchor_bytes(cache, slot)  # freed, re-donated under TP

        self.assertTrue(
            anchor_bytes_reachable(cache, slot),
            "the re-donate overwrote the record, so the slot is vouched for "
            "by its CURRENT tenant",
        )
        sched.phase_flip_active_stack = PHASE_PP
        self.assertFalse(
            anchor_bytes_reachable(cache, slot),
            "and the stale PP record must not survive to vouch for it again",
        )


class _Lru:
    def insert_mru(self, node) -> None:
        pass


class _NodeData:
    def __init__(self) -> None:
        self.value = None
        self.host_value = None


class _Node:
    def __init__(self, ct) -> None:
        self.component_data = {ct: _NodeData()}


class _InsertParams:
    def __init__(self, mamba_value, key_len: int) -> None:
        self.mamba_value = mamba_value
        self.key = [0] * key_len


class _InsertResult:
    def __init__(self) -> None:
        self.mamba_exist = False


class TestProvenanceIsRecordedAtTheTreeInsert928(unittest.TestCase):
    """ENABLEMENT GUARD. This rig runs mamba_radix_cache_strategy='no_buffer',
    so ``enable_mamba_extra_buffer`` is False (server_args.py:19472-19476) and
    the ping-pong donate branch is never entered. A rule taught to one donate
    branch would be inert here. The rule lives at the tree insert instead, and
    this drives the REAL ``commit_insert_component_data`` to say so.
    """

    def _component(self, cache, ct):
        from sglang.srt.mem_cache.unified_cache_components.mamba_component import (
            MambaComponent,
        )

        # The real method body, without the boot-sized __init__: this test is
        # about what commit_insert_component_data DOES, and every attribute it
        # touches is set here explicitly.
        comp = object.__new__(MambaComponent)
        comp.cache = cache
        comp.component_type = ct
        comp.mamba_checkpoint_interval = None  # grid off, as on this boot
        comp._off_grid_insert_refusals = 0
        return comp

    def test_provenance_is_recorded_at_the_tree_insert_for_any_branch(self):
        ct = "mamba"
        sched = _Scheduler()
        install_phase_aware_mamba_state_pool(sched)
        cache = sched.tree_cache
        cache.is_eagle = False
        cache.lru_lists = {ct: _Lru()}
        cache.host_lru_lists = {ct: _Lru()}
        cache.component_evictable_size_ = {ct: 0}

        comp = self._component(cache, ct)
        node = _Node(ct)
        anchor = _anchor(5)

        # PP computes: whatever donate branch produced this value -- the
        # no_buffer branch copies into the active pool, the ping-pong branch
        # copies nothing -- the anchor lands in the tree HERE.
        comp.commit_insert_component_data(
            node, True, _InsertParams(anchor, 4096), _InsertResult()
        )

        self.assertIs(
            node.component_data[ct].value,
            anchor,
            "precondition: the insert planted the anchor",
        )
        self.assertEqual(
            anchor_provenance_verdict(cache, anchor),
            "same",
            "the tree insert must record which pool holds the anchor's bytes, "
            "whichever donate branch produced it -- teaching this to a branch "
            "instead would be inert on a no_buffer rig",
        )

        sched.phase_flip_active_stack = PHASE_TP
        self.assertFalse(
            anchor_bytes_reachable(cache, anchor),
            "and after the flip that same anchor must read as foreign",
        )


if __name__ == "__main__":
    unittest.main()
