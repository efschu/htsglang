# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""#622/#649 falsifier: a prep sync must never be ordered behind a collective.

WHAT IS BEING TESTED, AND WHY IT IS NOT A CALL-SITE TEST
--------------------------------------------------------
Five production hangs on 2026-08-07 landed on three DIFFERENT host lines
(``dcp/owner.py:566`` twice, the draft ``kv_indptr[...].cpu()`` twice, a
host-path abort once), always with all three ranks on the identical line and
never a divergence. A test built around any one of those lines would pass
while the class stayed open -- and worse, #623 already demonstrated the
failure mode of per-site fixes: it removed the ``.item()`` at
``owner.py:548`` and the next specimen wedged at ``owner.py:566``, five lines
further down the same function.

So what is pinned here is the ordering property the whole family shares:

    P: a blocking host sync issued in the out-of-graph metadata-prep phase
       must not be transitively ordered behind a barlink collective.

P is a property of stream placement, not of any call site. It is decided from
``barlink_stream_policy`` by building the ordering graph a production step
would produce and asking whether any collective is reachable backwards from a
prep sync. That needs no GPU, which is the point: these tests run on a host
with no CUDA device.

WHY P IS THE RIGHT PROPERTY
---------------------------
CUDA streams are in-order, and ``cudaStreamSynchronize`` waits for every node
enqueued on the stream, ``cudaStreamWaitEvent`` nodes included. When P is
violated, a collective that stalls for D blocks the host thread for D. The
measured D for this rig is not hypothetical: ANALYSE_622 section 1.2 clocks
three ranks entering the abort path 104 s, 156 s and 159 s after a common spin
start, those being the same cycle deadline on a 5090 and two 3080s. A host
thread blocked for 104 s cannot enqueue, cannot service the abort gate and
cannot be the rank that unwedges its peers -- so a bounded stall on one rank
becomes an unbounded hang on all of them. P is what breaks that amplification.

P is deliberately silent about WHY a collective stalls. That question is open
(ANALYSE_622 section 6 leaves "flag written but not observed" as the leading
structural candidate) and it is not this seam's business. P holds or fails
regardless of the answer, and a fix that makes stalls rarer is not a
substitute for one that stops them taking the host thread down.

THE CAN-FAIL PROOF
------------------
``test_legacy_policy_violates_the_property`` and
``test_collective_stream_alone_violates_the_property`` assert that the current
placement and the first-proposed fix both FAIL P. They are the evidence that
the passing assertions below are load-bearing rather than vacuous. The second
one is the more valuable of the two: forking collectives onto their own stream
is the obvious fix and it is NOT sufficient, because the join back onto the
compute stream is itself a compute-stream node that a later sync waits for.
That result is recorded as an executable assertion so it cannot be forgotten
and re-proposed.
"""

import itertools
import unittest

from sglang.srt.distributed.device_communicators.barlink_stream_policy import (
    COLLECTIVE_STREAM,
    ISOLATED_PREP,
    LEGACY,
    OpKind,
    Placement,
    StreamPolicy,
    StreamRole,
)


class OrderingGraph:
    """CUDA stream ordering, modelled exactly as far as P needs and no further.

    Three rules, which are the whole of the semantics that decide P:

    1. A stream is in-order: each op is ordered after the previous op on the
       same stream.
    2. ``waits_on`` adds an edge from the current tail of another stream.
    3. ``joined_by`` inserts a join node on the other stream, so that
       everything subsequently enqueued there is ordered after this op. The
       join is a real node because in CUDA it is a real node, and that fact is
       precisely what defeats the collective-stream-only policy.

    A host sync on an op waits for that op's transitive predecessors. Nothing
    here models duration, concurrency or the device scheduler: P is a
    reachability question, so the model is a DAG and the answer is exact
    rather than sampled.
    """

    def __init__(self, policy: StreamPolicy):
        self.policy = policy
        self.kinds: list[OpKind | None] = []
        self.labels: list[str] = []
        self.preds: list[set[int]] = []
        self.tails: dict[StreamRole, int | None] = {
            role: None for role in StreamRole
        }

    def _add(self, kind: OpKind | None, label: str, preds: set[int]) -> int:
        nid = len(self.kinds)
        self.kinds.append(kind)
        self.labels.append(label)
        self.preds.append(preds)
        return nid

    def enqueue(self, kind: OpKind, label: str) -> int:
        placement: Placement = self.policy.place(kind)
        preds: set[int] = set()
        tail = self.tails[placement.role]
        if tail is not None:
            preds.add(tail)
        for role in placement.waits_on:
            other = self.tails[role]
            if other is not None:
                preds.add(other)
        nid = self._add(kind, label, preds)
        self.tails[placement.role] = nid
        for role in placement.joined_by:
            join_preds = {nid}
            other = self.tails[role]
            if other is not None:
                join_preds.add(other)
            self.tails[role] = self._add(
                None, f"join({label} -> {role.value})", join_preds
            )
        return nid

    def ancestors(self, nid: int) -> set[int]:
        seen: set[int] = set()
        stack = list(self.preds[nid])
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(self.preds[cur])
        return seen

    def collectives_awaited_by(self, nid: int) -> list[str]:
        return [
            self.labels[a]
            for a in sorted(self.ancestors(nid))
            if self.kinds[a] is OpKind.COLLECTIVE
        ]


#: Step shapes taken from the production configuration the specimens came from:
#: Qwen3.6-27B-INT8-W8A8, tp=3, uneven weighted DCP, NEXTN with
#: ``num_steps=3 topk=1 num_draft_tokens=4``, flashinfer, decode and prefill
#: graphs on. Each entry is (name, number of graph replays per step, number of
#: collectives per replay). Several shapes rather than one because P must hold
#: for the class, not for the shape that happened to be captured.
STEP_SHAPES = (
    ("decode", 1, 2),
    ("verify", 1, 4),
    ("draft-chain", 3, 2),
    ("draft-chain-wide", 3, 6),
    ("prefill", 2, 3),
)


def build_trace(
    policy: StreamPolicy,
    replays: int,
    collectives_per_replay: int,
    steps: int = 3,
    device_input: bool = False,
) -> tuple[OrderingGraph, list[int]]:
    """One production-shaped run: ``steps`` iterations of prep-then-replay.

    The order inside a step is the order production uses and is the reason the
    family exists: ``init_forward_metadata_out_graph`` -- which contains the
    blocking readback -- runs BEFORE ``_replay_graph``
    (``eagle_draft_cuda_graph_runner.py:646-659``). So the sync at step N is
    ordered behind the collectives of step N-1, which is where a stalled
    collective from the previous step reaches forward and takes the host
    thread.

    Returns the graph and the node ids of every prep sync in it.
    """
    graph = OrderingGraph(policy)
    syncs: list[int] = []
    for step in range(steps):
        graph.enqueue(OpKind.HOST_INPUT, f"s{step}.host_input")
        if device_input:
            graph.enqueue(OpKind.DEVICE_INPUT, f"s{step}.device_input")
        graph.enqueue(OpKind.PREP_KERNEL, f"s{step}.prep_kernel")
        syncs.append(graph.enqueue(OpKind.PREP_SYNC, f"s{step}.prep_sync"))
        for r in range(replays):
            graph.enqueue(OpKind.GRAPH_COMPUTE, f"s{step}.r{r}.pre")
            for c in range(collectives_per_replay):
                graph.enqueue(OpKind.COLLECTIVE, f"s{step}.r{r}.coll{c}")
                graph.enqueue(OpKind.GRAPH_COMPUTE, f"s{step}.r{r}.post{c}")
    return graph, syncs


class TestPropertyIsDecidable(unittest.TestCase):
    """The model itself behaves, before it is used to judge anything."""

    def test_first_prep_sync_awaits_nothing_on_any_policy(self):
        # Nothing precedes the very first prep sync, so every policy must be
        # clean there. A policy that fails this is reporting a dependency the
        # trace does not contain, i.e. the model is broken rather than the
        # policy.
        for policy in (LEGACY, COLLECTIVE_STREAM, ISOLATED_PREP):
            for name, replays, per_replay in STEP_SHAPES:
                graph, syncs = build_trace(policy, replays, per_replay)
                with self.subTest(policy=policy.name, shape=name):
                    self.assertEqual([], graph.collectives_awaited_by(syncs[0]))

    def test_join_is_a_node_on_the_joined_stream(self):
        # The one modelling decision P turns on. If a join were modelled as
        # free, the collective-stream policy would look sufficient and the
        # falsifier would endorse a fix that does not work.
        graph = OrderingGraph(COLLECTIVE_STREAM)
        coll = graph.enqueue(OpKind.COLLECTIVE, "coll")
        later = graph.enqueue(OpKind.GRAPH_COMPUTE, "later")
        self.assertIn(coll, graph.ancestors(later))


class TestCanFail(unittest.TestCase):
    """Proof that the passing assertions below are load-bearing.

    Both of these assert a VIOLATION. If either ever starts reporting a clean
    prep sync, the model has stopped modelling the hazard and every other test
    in this file has quietly become vacuous.
    """

    def test_legacy_policy_violates_the_property(self):
        # This is the tree as it ships today, and it is the #622/#649 shape:
        # the prep sync of every step after the first waits for the previous
        # step's collectives.
        for name, replays, per_replay in STEP_SHAPES:
            graph, syncs = build_trace(LEGACY, replays, per_replay)
            with self.subTest(shape=name):
                awaited = graph.collectives_awaited_by(syncs[1])
                self.assertEqual(replays * per_replay, len(awaited))
                self.assertTrue(all(a.startswith("s0.") for a in awaited))

    def test_collective_stream_alone_violates_the_property(self):
        # The first-proposed fix. Forking the collectives onto their own
        # stream does NOT clear the prep sync, because the collective's result
        # is consumed by the model and so must be joined back onto the compute
        # stream -- and the join is a compute-stream node that the next step's
        # sync waits for. Recorded as an assertion so the limitation survives
        # in executable form.
        for name, replays, per_replay in STEP_SHAPES:
            graph, syncs = build_trace(COLLECTIVE_STREAM, replays, per_replay)
            with self.subTest(shape=name):
                self.assertNotEqual([], graph.collectives_awaited_by(syncs[1]))


class TestIsolatedPrepSatisfiesTheProperty(unittest.TestCase):
    def test_no_prep_sync_awaits_a_collective_in_any_shape(self):
        for name, replays, per_replay in STEP_SHAPES:
            graph, syncs = build_trace(ISOLATED_PREP, replays, per_replay)
            for i, sync in enumerate(syncs):
                with self.subTest(shape=name, step=i):
                    self.assertEqual([], graph.collectives_awaited_by(sync))

    def test_holds_for_arbitrary_step_shapes(self):
        # The class is not the shapes that were captured. If P only held for
        # STEP_SHAPES it would be an enumeration wearing a property's clothes,
        # so it is checked across the whole small-shape product.
        for replays, per_replay, steps in itertools.product(
            range(1, 5), range(1, 5), range(2, 5)
        ):
            graph, syncs = build_trace(
                ISOLATED_PREP, replays, per_replay, steps=steps
            )
            for i, sync in enumerate(syncs):
                with self.subTest(
                    replays=replays, per_replay=per_replay, steps=steps, step=i
                ):
                    self.assertEqual([], graph.collectives_awaited_by(sync))

    def test_compute_stream_still_consumes_the_collective(self):
        # P must not be satisfied by cheating -- a policy that simply never
        # joined the collective back would pass P and compute wrong numbers.
        # The model output has to remain ordered after the collective.
        graph, _ = build_trace(ISOLATED_PREP, replays=1, collectives_per_replay=1)
        post = graph.labels.index("s1.r0.post0")
        awaited = graph.collectives_awaited_by(post)
        self.assertIn("s1.r0.coll0", awaited)


class TestIsolationObligation(unittest.TestCase):
    """The prep stream's soundness condition is enforced, not just documented."""

    def test_a_device_produced_prep_input_voids_the_isolation(self):
        # ISOLATED_PREP is sound only while the prep kernels' inputs are
        # host-written. Declaring a device-produced input must break P loudly
        # here rather than quietly in production, which is the entire reason
        # DEVICE_INPUT exists as a separate op kind.
        graph, syncs = build_trace(
            ISOLATED_PREP, replays=1, collectives_per_replay=2, device_input=True
        )
        self.assertNotEqual([], graph.collectives_awaited_by(syncs[1]))

    def test_host_input_alone_keeps_the_isolation(self):
        # The control for the test above: same shape, host-written inputs, and
        # the property holds. Without this pair the test above would also pass
        # if ISOLATED_PREP were broken outright.
        graph, syncs = build_trace(
            ISOLATED_PREP, replays=1, collectives_per_replay=2, device_input=False
        )
        self.assertEqual([], graph.collectives_awaited_by(syncs[1]))


class TestPolicyHygiene(unittest.TestCase):
    def test_every_policy_places_every_op_kind(self):
        # An unplaced op kind must raise rather than default. Defaulting new
        # work onto the compute stream is how this family was built in the
        # first place.
        for policy in (LEGACY, COLLECTIVE_STREAM, ISOLATED_PREP):
            for op in OpKind:
                with self.subTest(policy=policy.name, op=op.value):
                    self.assertIsInstance(policy.place(op), Placement)

    def test_unknown_op_kind_raises_with_a_reason(self):
        empty = StreamPolicy(name="empty", placements={})
        with self.assertRaises(KeyError) as caught:
            empty.place(OpKind.COLLECTIVE)
        self.assertIn("deliberately", str(caught.exception))

    def test_active_policy_is_legacy_until_a_gpu_window_says_otherwise(self):
        # Nothing in this branch has run on a GPU. The default must stay on the
        # shipped placement so that merging the seam cannot change production
        # numerics or performance; flipping it is a separate, evidenced change.
        from sglang.srt.distributed.device_communicators import (
            barlink_stream_policy,
        )

        self.assertIs(barlink_stream_policy.active_policy(), LEGACY)


if __name__ == "__main__":
    unittest.main()
