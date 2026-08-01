"""#404 round 2: does a CORRUPTED PROPOSAL reach committed content?

The pool-axis rollback has survived every elimination this window ran. Residue
volume does not reach it (18 arms, 738 rejected candidate rows, a flat dose
response, eager AND captured). The captured static buffers do not carry it
(#399: the clones are a latent-hazard fix, and the committed trajectory is
invariant to a written-through ``_hidden``). The ``_kv_len`` read side does not
carry it (the mixed-rung arms are clean). What is left is the hypothesis that
the leak needs corrupted proposal CONTENT -- and until now nothing had produced
any on purpose.

This file produces it, deterministically, through the shipped
``_propose`` -> ``_verify`` -> ``_rollback_draft`` sequence, and asks the one
question that decides the hypothesis desk-side:

    with exact rollback, is the COMMITTED CONTENT invariant to a corrupted
    proposal?

The invariance is what the greedy accept rule promises. A rejected proposal is
not a token: it is a guess that cost a KV slot and a recurrent step, and the
target's own prediction decides what gets committed. So corrupting proposal
``i`` of round ``r`` may move the accept length and the round count, and may
move nothing else. Every committed position must hold what it held in the
unperturbed run -- the tokens, and the probe's digests of the pool rows and the
recurrent state behind them.

WHAT MAKES THE ANSWER MEAN ANYTHING is the target mock, and it is the one thing
this file does not inherit unchanged from ``test_lane_pool_checksum_404.py``.
There the target predicts from its input token, so residue in a committed row
could not reach a committed token even in principle and the invariance would be
true by construction. Here the target READS ITS CONTEXT out of the pool --
``req_to_token[idx, :pos+1]`` and the KV rows those slots point at -- and its
prediction carries the DIFFERENCE between the context it read and the context
it should have read. Any residue in a committed row therefore changes a
committed token. That is a maximally sensitive stand-in for attention: it does
not model what attention computes, it models what attention READS, which is the
only property the leak hypothesis is about.

The can-fail is built on the same amplifier. ``_LeakyAllocator`` hands a verify
round a slot that a COMMITTED position still points at -- the use-after-free
shape of exactly the leak being hunted -- and with it the invariance breaks,
the probe's append-only reading names the round, and the cross-job reading
names the position. An instrument that has never been shown to fire on the
defect it exists for has not been calibrated.

Hermetic: CPU tensors, no card, no server.
"""

import importlib.util
import os
import sys
import unittest

import torch

from sglang.srt.model_executor.dual_group_lane import DualGroupLane
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=25, suite="base-a-test-cpu")


_HERE = os.path.dirname(os.path.abspath(__file__))
_PERTURB = "SGLANG_LANE_PROPOSAL_PERTURB"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


#: The checksum file's vehicle, loaded under a private name so pytest collects
#: its tests once (from its own file) and this file only borrows the harness.
_VEHICLE = _load(
    os.path.join(_HERE, "test_lane_pool_checksum_404.py"), "_lane_checksum_vehicle_404"
)


class _LeakyAllocator(_VEHICLE._Allocator):
    """An allocator that hands out a slot a committed position still holds.

    The defect class under investigation, planted at the only place it could
    physically come from: a slot that the rollback gave back (or never took)
    while the request's ``req_to_token`` still points at it. The verify then
    writes a CANDIDATE's KV into a row the lane had already committed, and
    every later forward reads the corrupted context.

    ``leak_at`` is the allocation call it fires on, counted from the first, so
    the plant lands on one verify round and not on the prompt.
    ``leak_candidate`` is WHICH candidate of that round lands on the committed
    row, and it is a parameter rather than "the last one" for a reason the
    can-fail depends on: the head's chain is not perturbed, so only the
    candidate the falsifier actually corrupted carries a value that differs
    between the two arms. A leak that lands on any other candidate writes the
    same residue in both and is invisible to the comparison -- which is itself
    worth knowing about this class of defect.
    """

    def __init__(self, kvcache, first=1, leak_slot=None, leak_at=1, leak_candidate=-1):
        super().__init__(kvcache, first=first)
        self.leak_slot = leak_slot
        self.leak_at = leak_at
        self.leak_candidate = leak_candidate
        self.verify_allocs = 0

    def alloc(self, n):
        out = super().alloc(n)
        if n > 1:
            self.verify_allocs += 1
            if self.verify_allocs == self.leak_at and self.leak_slot is not None:
                out[self.leak_candidate] = int(self.leak_slot)
        return out


class _DriftHarness(_VEHICLE._Harness):
    """The vehicle, with a target that reads the pool instead of its input.

    Row ``i`` of the verify is conditioned on committed positions
    ``0 .. n_cached-1`` plus candidates ``0 .. i``; the mock reads exactly those
    positions out of ``req_to_token`` and the KV pool, and adds the difference
    between what it read and what should have been there to its prediction. A
    clean round has a difference of zero and the mock is then byte-for-byte the
    counting target the sibling files use -- so the two arms of every test below
    differ in the perturbation and in nothing else.
    """

    def _head_forward(self, batch_d, hidden_states):
        """The counting head, made consistent for a chain longer than one.

        The head's two inputs are SHIFTED against each other at the first draft
        step -- ``_hidden`` is the target's state of the position BEFORE
        ``_next`` -- and level with each other at every step after it, because
        from then on the head is reading its own output. Continuing the count
        therefore means skipping the token in between exactly once. The sibling
        files pin ``+2`` because they run K = 1 and never reach the second
        step; a K = 3 chain that keeps adding two proposes tokens the target
        never predicts, and every round would stop at accept 1 for a reason
        that has nothing to do with what is under test.

        The proposal is still read out of the HIDDEN state, which is what keeps
        a corrupted ``_hidden`` observable at all.
        """
        hidden = int(hidden_states[0, 0].item())
        token = int(batch_d.input_ids[0].item())
        proposal = hidden + 1 + int(hidden < token)
        logits = torch.zeros(1, _VEHICLE.VOCAB)
        logits[0, proposal % _VEHICLE.VOCAB] = 10.0
        return _VEHICLE._Out(torch.full((1, _VEHICLE.HIDDEN), float(proposal)), logits)

    def _context_drift(self, tokens):
        n_cached = len(self.committed)
        expected = list(self.committed) + list(tokens)
        drift = []
        for i in range(len(tokens)):
            actual = 0.0
            for pos in range(n_cached + i + 1):
                slot = int(self.pool.req_to_token[0, pos].item())
                actual += float(self.kv.k[0][slot][0, 0].item())
            drift.append(actual - float(sum(expected[: n_cached + i + 1])))
        return drift

    def _target_forward(self, batch, capture_mode=None):
        tokens = [int(t) for t in batch.input_ids.tolist()]
        loc = batch.out_cache_loc
        if loc is not None:
            self.kv.write([int(s) for s in loc.tolist()], tokens)
            for i in range(len(tokens)):
                self.parked[i] = self._state_for(self.committed + tokens[: i + 1])
        drift = self._context_drift(tokens)
        rows = [int(t + d) for t, d in zip(tokens, drift)]
        self.drift_seen = getattr(self, "drift_seen", [])
        self.drift_seen.append(list(drift))
        return self.graph.replay(len(tokens), rows), 1.0


def _run(
    rounds=6,
    rung=3,
    perturb=None,
    leak_slot=None,
    leak_at=1,
    leak_candidate=-1,
    lane_cls=None,
    intervening_replay=False,
):
    """One job, N speculative rounds, with the probe recording every round.

    Returns the harness. ``perturb`` is the env value the hook parses, set for
    the duration of the run and removed afterwards -- the hook is off unless a
    test asks for it, in this file as in a boot.
    """
    _VEHICLE._on(SGLANG_LANE_POOL_CHECKSUM_PER_POS=1)
    saved = os.environ.pop(_PERTURB, None)
    if perturb is not None:
        os.environ[_PERTURB] = perturb
    original_cls = _VEHICLE.DualGroupLane
    if lane_cls is not None:
        _VEHICLE.DualGroupLane = lane_cls
    try:
        h = _DriftHarness(rung=rung)
        if leak_slot is not None:
            leaky = _LeakyAllocator(
                h.kv,
                first=h.allocator.next,
                leak_slot=int(h.pool.req_to_token[0, leak_slot].item()),
                leak_at=leak_at,
                leak_candidate=leak_candidate,
            )
            h.allocator = leaky
            h.job["_batch"].token_to_kv_pool_allocator = leaky
            h.lane.runner.token_to_kv_pool_allocator = leaky
        h.accepts = []
        for _ in range(rounds):
            _, n_accept = h.spec_round()
            h.accepts.append(n_accept)
            if intervening_replay:
                # The #399 hazard: a further replay of the verify shape between
                # this round's verify and the next round's _propose.
                h.graph.replay(rung + 1, [777] * (rung + 1))
        return h
    finally:
        _VEHICLE.DualGroupLane = original_cls
        os.environ.pop(_PERTURB, None)
        if saved is not None:
            os.environ[_PERTURB] = saved


def _preclone_lane_class():
    """``DualGroupLane`` with the four #399 clone sites reverted.

    A scratch copy of the shipped module -- read, four exact substitutions,
    loaded under its own name. Nothing on disk in the tree is touched and the
    running process keeps the shipped class; what comes back is the pre-#399
    assignment, so the arm below is the real defect and not a re-implementation
    of it.
    """
    path = os.path.join(
        _HERE,
        "..",
        "..",
        "..",
        "..",
        "python",
        "sglang",
        "srt",
        "model_executor",
        "dual_group_lane.py",
    )
    with open(path) as handle:
        source = handle.read()
    sites = (
        'job["_hidden"] = out.hidden_states[n_accept : n_accept + 1].clone()',
        'job["_verify_rows"] = out.hidden_states[: d - 1].clone()',
        'job["_verify_tokens"] = verify_input.draft_token[:d].clone()',
    )
    replaced = 0
    for site in sites:
        count = source.count(site)
        replaced += count
        source = source.replace(site, site.replace(".clone()", ""))
    if replaced < 4:
        raise unittest.SkipTest(
            f"the #399 clone sites moved ({replaced} of 4 found); the "
            "pre-clone arm needs updating before it means anything"
        )
    scratch = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "_preclone_scratch_404.py"
    )
    with open(scratch, "w") as handle:
        handle.write(source)
    try:
        mod = _load(scratch, "_dual_group_lane_preclone_404")
    finally:
        os.remove(scratch)
    return mod.DualGroupLane


def _committed(h):
    return list(h.job["output_ids"])


def _by_len(h):
    return {int(r["committed_len"]): r for r in h.records()}


def _surface_differences(a, b, surfaces=("kv", "conv", "ssm")):
    """Every shared committed length where two runs' digests disagree."""
    ref = _by_len(a)
    out = []
    for length, rec in _by_len(b).items():
        other = ref.get(length)
        if other is None:
            continue
        for surface in surfaces:
            if rec.get(surface) != other.get(surface):
                out.append(
                    {
                        "committed_len": length,
                        "surface": surface,
                        "round": rec["round"],
                    }
                )
    return out


class _Base(CustomTestCase):
    def setUp(self):
        for var in _VEHICLE._ENV + (_PERTURB,):
            os.environ.pop(var, None)

    def tearDown(self):
        for var in _VEHICLE._ENV + (_PERTURB,):
            os.environ.pop(var, None)


class TestTheHookIsOffAndStaysOff(_Base):
    """A falsifier that can fire without being asked for is a defect."""

    def test_no_env_no_perturbation(self):
        self.assertIsNone(DualGroupLane._proposal_perturbation())
        h = _run(rounds=3)
        self.assertNotIn("_perturbed", h.job)

    def test_the_parsed_form_is_round_index_delta(self):
        os.environ[_PERTURB] = "2:1:5"
        self.assertEqual(DualGroupLane._proposal_perturbation(), (2, 1, 5))

    def test_a_malformed_value_raises_instead_of_quietly_doing_nothing(self):
        for bad in ("2:1", "2:1:5:9", "round:1:5", "-1:0:5", "2:-1:5"):
            os.environ[_PERTURB] = bad
            with self.assertRaises(ValueError, msg=bad):
                DualGroupLane._proposal_perturbation()

    def test_the_hook_fires_once_at_the_named_round_and_index(self):
        h = _run(rounds=6, perturb="2:1:5")
        events = h.job["_perturbed"]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["round"], 2)
        self.assertEqual(events[0]["index"], 1)
        self.assertEqual(events[0]["to"] - events[0]["from"], 5)

    def test_an_index_past_the_chain_is_a_no_op_and_not_an_error(self):
        h = _run(rounds=4, rung=1, perturb="1:7:5")
        self.assertNotIn("_perturbed", h.job)


class TestThePerturbationActuallyDoesSomething(_Base):
    """Without this, every invariance below would be vacuous."""

    def test_the_clean_chain_accepts_everything(self):
        h = _run(rounds=5)
        self.assertEqual(h.accepts, [3, 3, 3, 3, 3])

    def test_the_corrupted_proposal_is_rejected_at_its_own_position(self):
        h = _run(rounds=5, perturb="2:1:5")
        self.assertEqual(h.accepts, [3, 3, 1, 3, 3])

    def test_the_corrupted_token_reached_the_verify_input(self):
        """It is not merely dropped on the way: the pool holds it.

        The rejected candidate's KV row is written by the verify forward before
        the accept rule has run, which is the whole reason its residue is a
        question at all.
        """
        h = _run(rounds=5, perturb="2:1:5")
        self.assertTrue(
            any(
                int(h.kv.k[0][slot][0, 0].item()) == h.job["_perturbed"][0]["to"]
                for slot in range(1, h.allocator.next)
            )
        )


class TestCommittedContentIsInvariantToACorruptedProposal(_Base):
    """THE question of the round, asked of the shipped code.

    The two runs differ in exactly one thing -- one proposal token at one
    round -- and are compared on everything a leak could move: the committed
    tokens, and the probe's digests of the KV rows and the recurrent state at
    every committed length the two runs share.
    """

    def test_the_committed_tokens_are_identical_on_the_shared_prefix(self):
        clean = _run(rounds=6)
        dirty = _run(rounds=6, perturb="2:1:5")
        shared = min(len(_committed(clean)), len(_committed(dirty)))
        self.assertGreaterEqual(shared, 8)
        self.assertEqual(_committed(clean)[:shared], _committed(dirty)[:shared])

    def test_the_pool_digests_agree_at_every_shared_committed_length(self):
        clean = _run(rounds=6)
        dirty = _run(rounds=6, perturb="2:1:5")
        self.assertEqual(_surface_differences(clean, dirty), [])

    def test_every_committed_position_holds_the_same_row_in_both_runs(self):
        """The reading that survives the round boundaries moving apart.

        A perturbed round accepts less, so from there on the two runs stand at
        different committed LENGTHS and the join on ``committed_len`` can only
        reach the rounds before the perturbation. The per-position digests do
        not have that limitation: position ``p`` is position ``p`` in both
        runs whatever round committed it, so this compares the whole shared
        prefix -- including everything committed AFTER the corruption, which is
        where a leak would have to show.
        """
        clean = _run(rounds=6)
        dirty = _run(rounds=6, perturb="2:1:5")
        a = clean.records()[-1]["kv_pos"]
        b = dirty.records()[-1]["kv_pos"]
        shared = min(len(a), len(b))
        self.assertGreater(shared, 20)
        self.assertEqual(a[:shared], b[:shared])

    def test_the_numeric_fingerprints_agree_position_by_position(self):
        clean = _run(rounds=6)
        dirty = _run(rounds=6, perturb="2:1:5")
        a = clean.records()[-1]["kv_num"]
        b = dirty.records()[-1]["kv_num"]
        shared = min(len(a), len(b))
        self.assertEqual(a[:shared], b[:shared])

    def test_the_state_surfaces_agree_where_the_two_runs_meet(self):
        """conv/ssm are whole-state digests, so they join on length or not at
        all. The rounds before the perturbation are where they meet."""
        clean = _run(rounds=6)
        dirty = _run(rounds=6, perturb="2:1:5")
        ref = _by_len(clean)
        compared = 0
        for length, rec in _by_len(dirty).items():
            other = ref.get(length)
            if other is None:
                continue
            compared += 1
            self.assertEqual(rec["conv_num"], other["conv_num"], f"conv at {length}")
            self.assertEqual(rec["ssm_num"], other["ssm_num"], f"ssm at {length}")
        self.assertGreaterEqual(compared, 2)

    def test_the_within_job_append_only_reading_is_clean_on_both_arms(self):
        for perturb in (None, "2:1:5", "1:0:7", "3:2:11"):
            h = _run(rounds=6, perturb=perturb)
            for rec in h.records():
                prev = [r for r in h.records() if r["round"] == rec["round"] - 1]
                if not prev or rec["prev_committed_len"] != prev[0]["committed_len"]:
                    continue
                self.assertEqual(
                    rec["kv_stable"], prev[0]["kv"], f"{perturb} round {rec['round']}"
                )

    def test_the_target_mock_would_have_told_us_otherwise(self):
        """The amplifier is live: a hand-planted residue moves a token.

        Without this the invariance above could be a property of the mock. The
        same run, with one committed row overwritten by hand between two
        rounds, commits different tokens from there on.
        """
        h = _run(rounds=3)
        before = list(_committed(h))
        slot = int(h.pool.req_to_token[0, 2].item())
        h.kv.k[0][slot][0, 0] += 13.0
        h.spec_round()
        self.assertNotEqual(_committed(h)[len(before) :], [])
        clean = _run(rounds=4)
        self.assertNotEqual(_committed(h), _committed(clean)[: len(_committed(h))])


class TestCanFailAPlantedResidueBreaksTheInvariance(_Base):
    """The instrument, shown firing on the defect class it exists for.

    ``_LeakyAllocator`` gives one verify round a slot that a committed position
    still points at. Nothing else changes: same code, same mock, same rounds.
    """

    LEAK_POS = 5
    #: Candidate 2 is proposal 1 -- the one the falsifier corrupts.
    LEAK_CANDIDATE = 2

    def test_the_perturbation_now_changes_committed_content(self):
        clean = _run(
            rounds=6,
            leak_slot=self.LEAK_POS,
            leak_at=3,
            leak_candidate=self.LEAK_CANDIDATE,
        )
        dirty = _run(
            rounds=6,
            leak_slot=self.LEAK_POS,
            leak_at=3,
            leak_candidate=self.LEAK_CANDIDATE,
            perturb="2:1:5",
        )
        shared = min(len(_committed(clean)), len(_committed(dirty)))
        self.assertNotEqual(_committed(clean)[:shared], _committed(dirty)[:shared])

    def test_the_per_position_digests_separate_the_two_arms_as_well(self):
        """The same statement on the instrument rather than on the tokens."""
        clean = _run(
            rounds=6,
            leak_slot=self.LEAK_POS,
            leak_at=3,
            leak_candidate=self.LEAK_CANDIDATE,
        )
        dirty = _run(
            rounds=6,
            leak_slot=self.LEAK_POS,
            leak_at=3,
            leak_candidate=self.LEAK_CANDIDATE,
            perturb="2:1:5",
        )
        a = clean.records()[-1]["kv_pos"]
        b = dirty.records()[-1]["kv_pos"]
        shared = min(len(a), len(b))
        moved = [i for i in range(shared) if a[i] != b[i]]
        self.assertIn(self.LEAK_POS, moved)

    def test_the_append_only_reading_names_the_round_it_happened(self):
        h = _run(rounds=6, leak_slot=self.LEAK_POS, leak_at=3)
        broken = [
            rec
            for rec in h.records()
            if any(
                rec["kv_stable"] != prev["kv"]
                for prev in h.records()
                if prev["round"] == rec["round"] - 1
                and rec["prev_committed_len"] == prev["committed_len"]
            )
        ]
        self.assertTrue(broken, "the planted residue was not caught at any round")
        self.assertEqual(broken[0]["round"], 2)

    def test_the_per_position_digests_name_the_position(self):
        h = _run(rounds=6, leak_slot=self.LEAK_POS, leak_at=3)
        recs = {r["round"]: r for r in h.records()}
        before, after = recs[1]["kv_pos"], recs[2]["kv_pos"]
        moved = [i for i, (a, b) in enumerate(zip(before, after)) if a != b]
        self.assertIn(self.LEAK_POS, moved)

    def test_the_cross_job_reading_separates_the_two_runs(self):
        clean = _run(rounds=6)
        leaked = _run(rounds=6, leak_slot=self.LEAK_POS, leak_at=3)
        diffs = _surface_differences(clean, leaked)
        self.assertTrue(diffs)
        self.assertEqual(diffs[0]["surface"], "kv")


class TestTheRevertedCloneArm(_Base):
    """#399's clones taken back out, on a scratch copy of the module.

    The briefed expectation was that the perturbation would then change
    committed content. It does not, and that is a result rather than a
    disappointment: #399 already established by construction that ``_hidden``
    is read by ``_draft_forward`` and by nothing else, so a written-through row
    moves the PROPOSALS and the accept length and cannot reach a committed
    token. The arm is kept because it is the executable form of that argument
    and because it pins what the clone is worth -- accept length, not content.
    """

    def setUp(self):
        super().setUp()
        self.preclone = _preclone_lane_class()

    def test_the_scratch_copy_really_lost_the_clone(self):
        h = _run(rounds=1, lane_cls=self.preclone)
        stored = h.job["_hidden"]
        produced = h.graph.hidden[h.rung + 1]
        self.assertEqual(
            stored.untyped_storage().data_ptr(), produced.untyped_storage().data_ptr()
        )

    def test_the_written_through_row_costs_accept_length(self):
        clean = _run(rounds=5, lane_cls=self.preclone, intervening_replay=False)
        dirty = _run(rounds=5, lane_cls=self.preclone, intervening_replay=True)
        self.assertEqual(clean.accepts, [3, 3, 3, 3, 3])
        self.assertNotEqual(dirty.accepts, clean.accepts)

    def test_and_still_does_not_change_committed_content(self):
        clean = _run(rounds=5, lane_cls=self.preclone, intervening_replay=False)
        dirty = _run(rounds=5, lane_cls=self.preclone, intervening_replay=True)
        shared = min(len(_committed(clean)), len(_committed(dirty)))
        self.assertGreaterEqual(shared, 5)
        self.assertEqual(_committed(clean)[:shared], _committed(dirty)[:shared])

    def test_the_perturbation_is_still_content_invariant_without_the_clones(self):
        clean = _run(rounds=6, lane_cls=self.preclone)
        dirty = _run(rounds=6, lane_cls=self.preclone, perturb="2:1:5")
        shared = min(len(_committed(clean)), len(_committed(dirty)))
        self.assertEqual(_committed(clean)[:shared], _committed(dirty)[:shared])
        self.assertEqual(_surface_differences(clean, dirty), [])


if __name__ == "__main__":
    unittest.main()
