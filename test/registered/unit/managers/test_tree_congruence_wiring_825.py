"""#825 wiring -- the digest actually rides the consensus payload.

The sibling file pins the pure arithmetic. This one pins the WIRING, which is
where this change can go wrong in ways arithmetic cannot see:

  * the payload and the ``fields`` list are two parallel lists whose order
    must match. The tp_vector is VARIABLE LENGTH, so a field appended in the
    wrong place shifts every index after it and ``config_fp`` silently starts
    reading a vector element -- a desync check that checks the wrong thing.
  * the digest must NOT join ``eq_checked``, which RAISES. If it did, the
    first cutover of every boot would take the instance down, because a
    divergent tree is the EXPECTED state this field exists to observe.
  * the reconcile must be driven by the GROUP verdict and must fire only on
    pp_to_tp.

Desk-written-never-executed is the failure this file exists to prevent: the
pure module could be perfect and the feature still dead.
"""

import threading
from typing import List

import torch

from sglang.srt.managers import tree_congruence as tc
from sglang.srt.managers.phase_flip_runtime import PHASE_PP, PhaseFlipRuntime
from sglang.srt.managers.phase_policy import PP_TO_TP, TP_TO_PP
from sglang.srt.managers.kv_reshard import KvPoolView

N_RANKS = 2
VEC = (7, 9)
# One full-attention ordinal per stage: every ordinal exactly once, which is
# what validate_layer_map requires (phase_flip_plan.py:52-64).
LAYER_MAP = ((0,), (1,))
N_LAYERS = 2


class _Key:
    def __init__(self, token_ids, extra_key=None):
        self.token_ids = token_ids
        self.extra_key = extra_key


class _Node:
    def __init__(self, token_ids=None, children=None):
        self.key = _Key(token_ids) if token_ids is not None else None
        self.children = children or {}


class _Tree:
    def __init__(self, root):
        self.root_node = root
        self.resets = 0

    def reset(self):
        self.resets += 1


class _Scheduler:
    def __init__(self, tree):
        self.tree_cache = tree


def _tree_with(keys):
    return _Tree(_Node((), children={i: _Node(k) for i, k in enumerate(keys)}))


class _CapturingChannel:
    """A REAL barrier reduction: element-wise MIN over ALL ranks' payloads.

    The first version of this harness reduced over "the payloads submitted so
    far", so rank 0's call saw only its own payload and read agreement -- the
    divergence cases passed vacuously on a broken channel. That is the same
    mistake as an indicator that cannot report the state it claims to measure,
    so the channel now blocks until every rank has submitted, exactly as the
    real gloo collective does.
    """

    def __init__(self):
        self.payloads: List[List[int]] = []
        self._barrier = threading.Barrier(N_RANKS)
        self._lock = threading.Lock()
        self._round: List[List[int]] = []
        self._reduced = None

    def channel_for(self, rank):
        def _chan(payload, timeout_s=None):
            with self._lock:
                self._round.append(list(payload))
                self.payloads.append(list(payload))
            i = self._barrier.wait()
            if i == 0:
                pool = self._round
                self._reduced = [min(p[k] for p in pool) for k in range(len(pool[0]))]
            self._barrier.wait()
            out = list(self._reduced)
            self._barrier.wait()
            if i == 0:
                self._round = []
            self._barrier.wait()
            return out

        return _chan


def _round_all(rts):
    """Drive on_round on every rank concurrently -- the reduction is a real
    barrier, so calling them sequentially would deadlock."""
    errors = []

    def _go(rt):
        try:
            rt.on_round()
        except BaseException as e:  # noqa: BLE001 - surfaced below
            errors.append(e)

    threads = [threading.Thread(target=_go, args=(rt,)) for rt in rts]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not any(t.is_alive() for t in threads), "on_round deadlocked"
    if errors:
        raise errors[0]


def _view(n_layers):
    """Smallest pool view the runtime accepts. The PP view must cover exactly
    this stage's ordinals; the TP view covers all of them. This file exercises
    the consensus payload, not the mover, so the buffers are never read."""
    return KvPoolView(
        [torch.zeros(4, 8) for _ in range(n_layers)],
        [torch.zeros(4, 8) for _ in range(n_layers)],
    )


def _runtime(rank, channel, tree, boot_phase=PHASE_PP):
    return PhaseFlipRuntime(
        n_ranks=N_RANKS,
        rank=rank,
        layer_map=LAYER_MAP,
        n_layers=N_LAYERS,
        tp_vector=VEC,
        boot_phase=boot_phase,
        consensus_interval=1,
        collective_min=channel.channel_for(rank),
        exchange=lambda *a, **k: None,
        pp_pool_view=_view(len(LAYER_MAP[rank])),
        tp_pool_view=_view(N_LAYERS),
        live_slots_fn=lambda: [],
        ready_fn=lambda: True,
        cutover_fn=lambda d: None,
    )


# --------------------------------------------------------------------------
# Payload shape and index alignment.
# --------------------------------------------------------------------------


def test_payload_grew_by_exactly_one_field_and_indices_still_align():
    """THE DANGEROUS ONE.

    ``fields`` and the payload are parallel. If the digest were inserted
    before the variable-length vector, ``config_fp`` would decode a vector
    element and the DESYNC check would compare the wrong quantity -- silently,
    on every boot.
    """
    ch = _CapturingChannel()
    tree = _tree_with([(1, 2), (3,)])
    rts = [_runtime(r, ch, tree) for r in range(N_RANKS)]
    for rt in rts:
        rt._census_scheduler = _Scheduler(tree)
    _round_all(rts)

    assert ch.payloads, "on_round never reduced -- the wiring test is inert"
    payload = ch.payloads[-1]
    # 6 scalars + n vector entries + 1 tree digest, each encoded as (x, -x).
    assert len(payload) == 2 * (6 + N_RANKS + 1)

    # Decode with the same rule on_round uses and check the KNOWN fields land
    # where they belong. The vector is the anchor: if the digest had been
    # inserted before it, these would shift.
    decoded = [payload[2 * i] for i in range(len(payload) // 2)]
    assert tuple(decoded[6 : 6 + N_RANKS]) == VEC
    # The last field is the tree digest, and it is a real digest of the tree
    # this scheduler holds -- not a placeholder.
    assert decoded[-1] == tc.tree_digest_of(tree)
    assert decoded[-1] != tc.ABSENT_TREE_DIGEST


def test_a_scheduler_without_a_tree_still_contributes_uniform_width():
    """Payload width must never become a per-rank capability -- the argument
    ``_update_uniform_pool_budget`` makes for its host and mamba pairs."""
    ch = _CapturingChannel()
    rts = [_runtime(r, ch, None) for r in range(N_RANKS)]
    # deliberately leave _census_scheduler as None on both
    _round_all(rts)
    payload = ch.payloads[-1]
    assert len(payload) == 2 * (6 + N_RANKS + 1)
    assert payload[-2] == tc.ABSENT_TREE_DIGEST


# --------------------------------------------------------------------------
# The verdict, and that it is NOT fatal.
# --------------------------------------------------------------------------


def test_congruent_trees_produce_no_onset_and_no_reconcile():
    ch = _CapturingChannel()
    tree = _tree_with([(1, 2), (3,)])
    rts = [_runtime(r, ch, tree) for r in range(N_RANKS)]
    for rt in rts:
        rt._census_scheduler = _Scheduler(tree)
    _round_all(rts)
    for rt in rts:
        assert rt._tree_congruence is not None
        assert rt._tree_congruence.congruent is True
        assert rt.tree_divergence_onsets == 0
        rt._reconcile_trees_if_diverged(PP_TO_TP)
        assert rt.tree_reconciles == 0


def test_divergent_trees_are_counted_and_do_not_raise():
    """A divergent tree must NOT join the ``eq_checked`` family: that family
    raises KvReshardError, and divergence here is the EXPECTED measured state
    (the PP phase runs with the floors off). Raising would take the instance
    down at the first cutover of every boot."""
    ch = _CapturingChannel()
    tree_a = _tree_with([(1, 2), (3,)])
    tree_b = _tree_with([(1, 2)])  # peer evicted a node
    rts = [_runtime(r, ch, None) for r in range(N_RANKS)]
    rts[0]._census_scheduler = _Scheduler(tree_a)
    rts[1]._census_scheduler = _Scheduler(tree_b)
    _round_all(rts)  # must not raise
    for rt in rts:
        assert rt._tree_congruence.congruent is False
        assert rt.tree_divergence_onsets == 1
        assert rt.tree_divergence_rounds == 1


def test_every_rank_reaches_the_same_verdict():
    """The repair must not itself become a divergence."""
    ch = _CapturingChannel()
    rts = [_runtime(r, ch, None) for r in range(N_RANKS)]
    rts[0]._census_scheduler = _Scheduler(_tree_with([(1, 2), (3,)]))
    rts[1]._census_scheduler = _Scheduler(_tree_with([(1, 2)]))
    _round_all(rts)
    verdicts = {rt._tree_congruence.must_reconcile for rt in rts}
    assert verdicts == {True}, "ranks disagreed about whether to reconcile"


# --------------------------------------------------------------------------
# The reconcile action.
# --------------------------------------------------------------------------


def _diverged_pair():
    ch = _CapturingChannel()
    rts = [_runtime(r, ch, None) for r in range(N_RANKS)]
    trees = [_tree_with([(1, 2), (3,)]), _tree_with([(1, 2)])]
    for rt, t in zip(rts, trees):
        rt._census_scheduler = _Scheduler(t)
    _round_all(rts)
    return rts, trees


def test_reconcile_fires_on_pp_to_tp_and_resets_every_rank():
    rts, trees = _diverged_pair()
    for rt in rts:
        rt._reconcile_trees_if_diverged(PP_TO_TP)
    assert [t.resets for t in trees] == [1, 1]
    assert [rt.tree_reconciles for rt in rts] == [1, 1]


def test_reconcile_does_not_fire_on_tp_to_pp():
    """The PP phase does not require identical trees -- #791 says each PP rank
    re-derives its own verdict from its own radix state. Paying the capacity
    cost there would be a cost for nothing."""
    rts, trees = _diverged_pair()
    for rt in rts:
        rt._reconcile_trees_if_diverged(TP_TO_PP)
    assert [t.resets for t in trees] == [0, 0]
    assert [rt.tree_reconciles for rt in rts] == [0, 0]


def test_reconcile_never_raises_when_the_tree_cannot_be_reset():
    """This runs with requests parked between the movers and the cutover. A
    raise here takes the instance down for a cache-capacity repair."""

    class _Unresettable:
        root_node = None

    rts, _ = _diverged_pair()
    rts[0]._census_scheduler = _Scheduler(_Unresettable())
    rts[0]._reconcile_trees_if_diverged(PP_TO_TP)  # must not raise
    assert rts[0].tree_reconciles == 0

    class _Exploding:
        root_node = None

        def reset(self):
            raise RuntimeError("boom")

    rts[1]._census_scheduler = _Scheduler(_Exploding())
    rts[1]._reconcile_trees_if_diverged(PP_TO_TP)  # must not raise
    assert rts[1].tree_reconciles == 0


def test_recovery_edge_is_reported_not_only_the_onset():
    """#823's lesson applied at build time: a divergence reported only at
    onset can say THAT the trees parted and never that they healed."""
    ch = _CapturingChannel()
    rts = [_runtime(r, ch, None) for r in range(N_RANKS)]
    diverged = [_tree_with([(1, 2), (3,)]), _tree_with([(1, 2)])]
    for rt, t in zip(rts, diverged):
        rt._census_scheduler = _Scheduler(t)
    _round_all(rts)
    assert all(rt.tree_divergence_onsets == 1 for rt in rts)

    same = _tree_with([(1, 2)])
    for rt in rts:
        rt._census_scheduler = _Scheduler(same)
    _round_all(rts)
    for rt in rts:
        assert rt.tree_congruence_recoveries == 1
        assert rt._tree_divergence_open is False
