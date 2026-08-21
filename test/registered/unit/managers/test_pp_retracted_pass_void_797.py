"""#797: the retraction must not NARROW the pass -- it must switch it off.

THE FREQUENCY DATA IS THE ARGUMENT, not the crash. Three boots of the gapped
TP=1/PP=3 shape died on one width mismatch each, and each had logged thousands
of `#791 PP-ADMISSION unhonourable prefix` retractions first:

    boot        unhonourable-prefix events       fatal width mismatches
    instr15               661                              1
    instr16              1651                              1
    instr17              1718                              1

instr17 narrowed 1718 passes and exactly ONE of them produced a row count
different enough for `model_runner.forward`'s `_hs.shape[0] != _want` to see
(126 rows for a 22-token batch, /spinning/evidence-665-f1/boot_instr17.log:
62997-63056). The other ~1717 were SAME-WIDTH divergences: the upstream's
hidden states for one request set, paired with a downstream's metadata for a
DIFFERENT request set of equal token count, computed without a word. Chunked
prefill is what makes equal widths ordinary -- every chunk is capped at
`chunked_prefill_size`, so dropping one request from a batch does not shrink
the batch, it hands the freed budget to the request that is left. So every
"survival" in that series was computed on corrupt pairings, and the uptime
numbers measured on them say nothing.

WHAT #791c DID AND DID NOT DO. It reads the receiver's OWN retraction and
refuses the proxy at the stage boundary, naming rid/told/local instead of
dying thirty layers deep. That is detection. It does not stop the divergence
being created, and a boot with it dies at the FIRST retraction instead of the
1718th -- instr18, 1m44s.

THE PREVENTION UNDER TEST, both halves, because either alone is worse than
neither:

  (a) THE PASS IS VOIDED, NOT NARROWED. There are three membership outcomes
      and two are physically unavailable. The rank cannot ADMIT the retracted
      rid: it has no KV for the prefix its upstream reused, and the upstream
      sent hidden states only for the extend tokens. The upstream cannot be
      AMENDED: it sent its decision and launched its batch earlier in this
      same pass. What is left is to run the pass NOWHERE, which is the only
      direction in which uniform membership can still be restored.
      `_pp_void_retracted_pass` empties `effective` AND marks every surviving
      entry `admitted=False` (`void_pp_admission_decision`), and
      `_PP_PASS_VOIDED_KEY` carries the fact to every rank after this one --
      which the entries alone cannot do, because a rank with nothing to
      prefill falls through to its running DECODE batch and would then pair
      that with the upstream's prefill batch, the same mispairing one forward
      further on.

  (b) THE FLOOR COMES BACK, so the void does not repeat. The upstream's own
      request state is resynced by #791b's void output, which empties PP0's
      slot and re-queues its requests, and rides the chain-reconciled decision
      home so `PPAdmissionCongruenceGuard.record_return_trip` learns
      `observed_local` as a prefix floor. `prefix_len_for` clamps the next
      offer for that rid to it, and a strictly decreasing sequence of
      non-negative integers cannot cycle. #797 additionally puts that same
      payload on a SUCCESSFUL output, which is what lets a floor be CLEARED
      again -- #796 removed the only feeder for that when it deleted the ring
      wraparound, and #791b restored only the learning half.

  THE FIRST-OFFER COST, stated: one wasted upstream forward, the first time a
  given prefix is offered that a downstream cannot honour. It is not once per
  pass, because of (b). It cannot be zero: the upstream has no way to know a
  downstream's cache state before it offers.

THE ASSERTIONS ARE ON CONTENT, NOT ON WIDTH, and that is the point of the
file. A test that only distinguishes 126 rows from 22 reproduces instr17's
ONE easy case and misses the ~1717 real ones. So the two request sets here
have EQUAL row counts and the hidden states are TAGGED per row with the
request they belong to; the red arm proves that what was delivered contains
64 rows belonging to a request the victim's batch does not contain at all.

THE ARMS, and each blinding is a single return-value rebind through
`scheduler_pp_mixin`'s own module globals, in the CHILD (the spawn trap #796
paid for: a patch applied in a test method reaches no spawned child):

  test_the_same_width_mispair_is_delivered_without_either_fix   RED. Blinds
      `entries_retracted_by_rank`, which is the pre-#791c reader ("this rank
      narrowed nothing"). Every body still runs, the reconciliation still
      retracts, and the victim accepts 128 rows for its own 128-token batch --
      64 of them another request's. The specimen, in its silent form.
  test_the_pass_is_voided_and_nothing_is_computed                GREEN.
  test_detection_without_prevention_still_kills_the_pass         Blinds
      `pp_pass_should_void` ONLY, so #791c's guard is untouched and fires:
      detection alone converts silent corruption into a dead boot.
  test_the_void_reaches_the_rank_that_retracted_nothing          The third
      process. It retracted nothing of its own and must still withhold.
  test_the_upstream_proxy_is_drained_by_the_voided_rank          The corpse.
  test_an_undrained_voided_pass_strands_the_upstream_proxy       Why the
      per-hop `launched` key exists at all.
  test_a_congruent_pass_is_untouched                             Default path.

CPU-only, three live spawned processes, real gloo. Every line of logic under
test is the shipped code bound to a holder: `pp_admission_decision_to_wire` /
`_from_wire`, `reconcile_pp_admission_decision`, `void_pp_admission_decision`,
`_pp_void_retracted_pass`, `_pp_send_admission_decision`,
`_pp_recv_admission_decision`, `_pp_note_output_expectation`,
`_pp_drain_voided_proxy`, `_pp_recv_typed_dict` and the #631/#795/#791c
receive guard. The transport adapter is the one from
test_pp_proxy_retracted_pass_mispair_791c.py.
"""

import json
import os
import pickle
import tempfile
import types
import unittest
from collections import defaultdict, deque

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=120)

WORLD = 3
UPSTREAM, VICTIM, DOWNSTREAM = 0, 1, 2

LIVE_MB = 1
FLIP_EPOCH = 66
RID_HONOURABLE = "51a294650b8b464495eda568e42530d7"
RID_UNHONOURABLE = "5e744c29f8de41fe96cb2c673b8582e5"
TOLD_PREFIX = 16896

#: THE SAME-WIDTH CONSTRUCTION, and it is the ordinary one rather than a
#: contrived one. `chunked_prefill_size` caps the WHOLE batch, so the upstream
#: splits its 128-token budget between the two requests it admitted, and a
#: victim that drops one of them hands the entire budget to the one that is
#: left. Two different request sets, 128 rows each.
CHUNK = 128
UPSTREAM_ROWS_HONOURABLE = 64
UPSTREAM_ROWS_UNHONOURABLE = 64
VICTIM_TOKENS = CHUNK

#: Row tags. The hidden state of a row belongs to exactly one request, and
#: that is the fact the width comparison throws away.
TAG = {RID_HONOURABLE: 1.0, RID_UNHONOURABLE: 2.0}

PROXY_SEQ = 4181
SENTINEL_TOKEN = 7777


class _GlooWire:
    """A real point-to-point tensor-dict wire over gloo.

    Only the TRANSPORT is adapted -- pickle to bytes, bytes over gloo. From
    test_pp_proxy_retracted_pass_mispair_791c.py, with `send_tensor_dict`'s
    signature widened to the keyword form `_pp_send_dict_to_next_stage`
    actually calls (`tensor_dict=`, `async_send=`), because this file drives
    the shipped send rather than hand-rolling the message.
    """

    def __init__(self, rank: int, src: int, dst: int):
        self.rank = rank
        self.rank_in_group = rank
        self.world_size = WORLD
        self.src = src
        self.dst = dst
        self.is_first_rank = rank == 0
        self.is_last_rank = rank == WORLD - 1

    def send_tensor_dict(self, tensor_dict=None, all_gather_group=None, **kwargs):
        buf = pickle.dumps(tensor_dict)
        size = torch.tensor([len(buf)], dtype=torch.long)
        dist.send(size, dst=self.dst)
        dist.send(torch.frombuffer(bytearray(buf), dtype=torch.uint8), dst=self.dst)
        return []

    def recv_tensor_dict(self, src=None, all_gather_group=None):
        size = torch.zeros(1, dtype=torch.long)
        dist.recv(size, src=self.src)
        buf = torch.zeros(int(size.item()), dtype=torch.uint8)
        dist.recv(buf, src=self.src)
        return pickle.loads(bytes(buf.numpy()))


def _pp0_decision():
    """PP0's verdict, in instr17's own shape: two requests, one of them resting
    on a 16896-token prefix PP0 has and the victim does not."""
    from sglang.srt.managers.pp_admission_congruence import (
        PPAdmissionDecision,
        PPAdmissionEntry,
    )

    return PPAdmissionDecision(
        mb_id=LIVE_MB,
        entries=(
            PPAdmissionEntry(
                rid=RID_HONOURABLE,
                prefix_len=0,
                extend_len=UPSTREAM_ROWS_HONOURABLE,
            ),
            PPAdmissionEntry(
                rid=RID_UNHONOURABLE,
                prefix_len=TOLD_PREFIX,
                extend_len=UPSTREAM_ROWS_UNHONOURABLE,
            ),
        ),
    )


def _tagged_hidden_states():
    """The upstream's real hidden states, ROW-TAGGED BY REQUEST.

    Rows 0..63 are the honourable request's, rows 64..127 the unhonourable
    one's. The total is exactly the victim's own token count, so the shipped
    width check would pass this pair into compute without a word -- which is
    what the ~1717 silent events in instr15/16/17 were.
    """
    rows = UPSTREAM_ROWS_HONOURABLE + UPSTREAM_ROWS_UNHONOURABLE
    hs = torch.zeros(rows, 4)
    hs[:UPSTREAM_ROWS_HONOURABLE] = TAG[RID_HONOURABLE]
    hs[UPSTREAM_ROWS_HONOURABLE:] = TAG[RID_UNHONOURABLE]
    return hs


def _proxy():
    """A stamp CORRECT in every element: this pass's slot, this rank's next
    sequence number, the true row count, this flip epoch. The hazard is that
    all of that is true and the message is still not the victim's to compute
    on."""
    hs = _tagged_hidden_states()
    return {
        "__msg_type__": "proxy",
        "__stamp__": (LIVE_MB, PROXY_SEQ, int(hs.shape[0]), FLIP_EPOCH),
        "hidden_states": hs,
    }


def _sentinel():
    """One ordinary message behind the proxy, so the FIFO's alignment is
    observable: a proxy nobody took is still on the wire when this arrives and
    lands in the typed inbox's stash."""
    return {
        "__msg_type__": "output",
        "next_token_ids": torch.tensor([SENTINEL_TOKEN], dtype=torch.long),
    }


def _holder(wire, pp_rank):
    """The SHIPPED mixin methods, bound to a holder (the 630/757/795 pattern)."""
    from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin

    h = types.SimpleNamespace(
        pp_group=wire,
        require_attn_tp_allgather=False,
        attn_tp_group=None,
        pp_flip_counters=None,
        _pp_tensor_dict_inbox=defaultdict(deque),
        _pp_proxy_drops=0,
        _pp_admission_send_work=[],
        _pp_gapped_wire=False,
        pp_loop_size=3,
        phase_flip_runtime=types.SimpleNamespace(epoch=FLIP_EPOCH),
        ps=types.SimpleNamespace(pp_rank=pp_rank, pp_size=WORLD),
    )
    h._pp_boundary_stats = lambda: None
    h._pp_flip_bump_consumed = lambda chan: None
    h._pp_flip_bump_attempted = lambda chan: None
    h._pp_flip_bump_sent = lambda chan: None
    h._pp_flip_upstream = lambda: pp_rank - 1
    for name in (
        "_pp_recv_typed_dict",
        "_pp_recv_proxy_tensors",
        "_pp_send_dict_to_next_stage",
        "_pp_send_admission_decision",
        "_pp_recv_admission_decision",
        "_pp_reconcile_incoming_admission",
        "_pp_void_retracted_pass",
        "_pp_drain_voided_proxy",
        "_pp_flip_epoch",
        "_pp_note_output_expectation",
        "_pp_pass_retraction_reason",
        # #789 interface drift: a no-op here because pp_flip_counters is None.
        "_pp_wait_for_proxy_readiness",
    ):
        setattr(h, name, types.MethodType(getattr(SchedulerPPMixin, name), h))
    return h


def _local_matches(case):
    """This rank's radix-cache match per rid -- the only stubbed input.

    `congruent` is a cache that holds everything PP0 named. Every other case is
    instr17's: PP1 had nothing at all for the rid PP0 offered at 16896, because
    the cutover cold-started its radix cache.
    """
    if case == "congruent":
        return {RID_HONOURABLE: 0, RID_UNHONOURABLE: TOLD_PREFIX}
    return {RID_HONOURABLE: 0, RID_UNHONOURABLE: 0}


def _victim_pass(h, decision, case, res):
    """The victim's top-of-pass block, on the SHIPPED functions.

    `_event_loop_pp_body`'s own sequence: reconcile the received decision
    against this rank's local match lengths, apply #797's void decision, record
    the amendment for the slot, then either receive the proxy (because a batch
    was built) or drain it (because none was). The one thing stubbed out is the
    tree-cache lookup that PRODUCES the local match lengths.
    """
    from sglang.srt.distributed.pp_typed_channel import typed_inbox
    from sglang.srt.managers.pp_admission_congruence import (
        reconcile_pp_admission_decision,
    )

    effective, amended = reconcile_pp_admission_decision(
        decision, _local_matches(case), rank=VICTIM, pp_size=WORLD
    )
    effective, amended = h._pp_void_retracted_pass(effective, amended)
    h._pp_note_output_expectation(LIVE_MB, False, amended)
    res["effective"] = sorted(effective)
    res["voided"] = bool(h._pp_admission_pass_voided)
    res["forwarded_admitted"] = sorted(
        e.rid for e in amended.entries if e.admitted and not e.retracted
    )
    res["forwarded_retracted"] = sorted(e.rid for e in amended.entries if e.retracted)

    # The shipped forward: the amended decision, the void flag ORed along the
    # chain, and this rank's own slot state as the next hop's `launched`.
    h._pp_send_admission_decision(
        amended,
        expects_output=False,
        pass_voided=h._pp_admission_pass_voided,
        launched=not h._pp_admission_pass_voided,
    )

    # `_event_loop_pp_body`'s `if cur_batch: ... else: drain`. A voided pass
    # builds no batch (scheduler.py's `get_next_batch_to_run` withholds the
    # whole round), so `cur_batch` is None exactly when the pass was voided.
    if h._pp_admission_pass_voided:
        res["drained"] = bool(h._pp_drain_voided_proxy(LIVE_MB))
        res["rows"] = None
        res["foreign_rows"] = None
    else:
        got = h._pp_recv_proxy_tensors(LIVE_MB)
        hs = got["hidden_states"]
        res["rows"] = int(hs.shape[0])
        # THE CONTENT ASSERTION'S RAW MATERIAL. This rank's batch is exactly
        # the rids in `effective`; a delivered row tagged with any other rid
        # is one request's hidden state on another request's metadata.
        mine = {TAG[rid] for rid in effective}
        res["foreign_rows"] = int(
            sum(1 for v in hs[:, 0].tolist() if float(v) not in mine)
        )
        res["drained"] = False

    # One ordinary message behind the proxy: whatever is still on the wire
    # gets stashed by this receive, so the stash size is the corpse count.
    tail = h._pp_recv_typed_dict(expected_kind="output")
    res["sentinel"] = int(tail["next_token_ids"][0].item())
    # #753: the inbox lives on the GROUP, not on the scheduler, because the
    # crossing wire is a second consumer of the same channel. Counted there,
    # or an undrained message would be invisible to this assertion.
    res["stashed"] = sum(len(q) for q in typed_inbox(h.pp_group).values())


def _worker(rank, init_file, out_dir, case):
    res = {"rank": rank, "ok": False, "error": None}
    try:
        dist.init_process_group(
            "gloo", init_method=f"file://{init_file}", rank=rank, world_size=WORLD
        )
        if rank == UPSTREAM:
            from sglang.srt.managers.scheduler_pp_mixin import (
                pp_admission_decision_to_wire,
            )

            wire = _GlooWire(rank, src=DOWNSTREAM, dst=VICTIM)
            # THE DECISION FIRST, THEN THE PROXY -- the shipped order.
            # `_pp_send_admission_decision` runs right after
            # `get_next_batch_to_run`, the proxy send at the end of the same
            # pass, and both ride the one FIFO tensor-dict wire.
            msg = pp_admission_decision_to_wire(_pp0_decision())
            msg["__msg_type__"] = "admission_decision"
            msg["__pp_output_expected__"] = True
            msg["__pp_pass_voided__"] = False
            # PP0 admitted two requests, so PP0 built a batch and its proxy is
            # coming. That is the fact the victim cannot derive for itself.
            msg["__pp_upstream_launched__"] = True
            wire.send_tensor_dict(tensor_dict=msg)
            wire.send_tensor_dict(tensor_dict=_proxy())
            wire.send_tensor_dict(tensor_dict=_sentinel())
        elif rank == VICTIM:
            h = _holder(_GlooWire(rank, src=UPSTREAM, dst=DOWNSTREAM), VICTIM)
            # THE SHIPPED READER, not a hand-rolled one: it is what resolves
            # every wire key through this module's globals, which is also what
            # makes `_blind_launched_key` a real neuter rather than a no-op.
            decision = h._pp_recv_admission_decision()
            assert decision.mb_id == LIVE_MB, f"wrong slot on the wire: {decision}"
            _victim_pass(h, decision, case, res)
        else:
            # THE RANK THAT RETRACTED NOTHING. It receives the victim's
            # forwarded decision through the SHIPPED receive, so the two wire
            # keys are read by the shipped reader, and applies the shipped void
            # decision to it. Its own retraction set is empty by construction.
            h = _holder(_GlooWire(rank, src=VICTIM, dst=UPSTREAM), DOWNSTREAM)
            from sglang.srt.managers.pp_admission_congruence import (
                reconcile_pp_admission_decision,
            )

            incoming = h._pp_recv_admission_decision()
            res["saw_voided"] = bool(h._pp_pass_voided_incoming)
            res["saw_launched"] = bool(h._pp_upstream_launched_incoming)
            effective, amended = reconcile_pp_admission_decision(
                incoming, _local_matches("congruent"), rank=DOWNSTREAM, pp_size=WORLD
            )
            effective, amended = h._pp_void_retracted_pass(effective, amended)
            res["effective"] = sorted(effective)
            res["voided"] = bool(h._pp_admission_pass_voided)
            res["retracted_here"] = sorted(
                e.rid for e in amended.entries if e.retracted_by_rank == DOWNSTREAM
            )
        res["ok"] = True
    except BaseException as exc:  # noqa: BLE001 - the error IS the result here
        res["error"] = f"{type(exc).__name__}: {exc}"[:900]
    finally:
        try:
            with open(os.path.join(out_dir, f"r{rank}.json"), "w") as f:
                json.dump(res, f)
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()


def _blind_retraction_lookup(rank, init_file, out_dir, case):
    """PRE-#791c AND PRE-#797, and nothing else, IN THE CHILD.

    THE SPAWN TRAP, restated because #796 paid for it: `_run` uses the "spawn"
    start method, so a patch applied inside a test METHOD reaches no child.

    `entries_retracted_by_rank` returning an empty tuple is exactly the reader
    that shipped before #791c: "this rank narrowed nothing". The function still
    EXISTS with its signature intact and is still looked up through
    `scheduler_pp_mixin`'s own module globals at call time; `pp_pass_should_
    void`, `_pp_void_retracted_pass`, `void_pp_admission_decision`,
    `pp_proxy_pass_retraction_reason` and the receive guard all still run their
    own bodies, and `reconcile_pp_admission_decision` still performs the
    retraction. So nothing here can raise an AttributeError, and a green result
    would mean the harness never depended on the fix rather than that the fix
    is present. A wholesale revert would prove neither.
    """
    from sglang.srt.managers import scheduler_pp_mixin as m

    m.entries_retracted_by_rank = lambda decision, rank: ()
    return _worker(rank, init_file, out_dir, case)


def _blind_void_only(rank, init_file, out_dir, case):
    """#791c PRESENT, #797 ABSENT: detection without prevention.

    Blinds ONLY `pp_pass_should_void`'s return value, so
    `entries_retracted_by_rank` is untouched and #791c's receive guard sees
    exactly what it sees on the shipped path. `_pp_void_retracted_pass` still
    runs its whole body and still calls the predicate -- it simply gets False,
    which is the pre-#797 answer.
    """
    from sglang.srt.managers import scheduler_pp_mixin as m

    m.pp_pass_should_void = lambda amended, rank, incoming_voided: False
    return _worker(rank, init_file, out_dir, case)


def _blind_launched_key(rank, init_file, out_dir, case):
    """#797's VOID PRESENT, its per-hop `launched` key ABSENT.

    Rebinds the key name to one no sender writes, so
    `_pp_drain_voided_proxy`'s "is a proxy actually coming" gate reads False.
    Everything else -- the void, the withheld round, the forwarded decision --
    is the shipped path. What is left behind is the bounded-recv corpse, and
    the sentinel receive is where it becomes visible.
    """
    from sglang.srt.managers import scheduler_pp_mixin as m

    m._PP_UPSTREAM_LAUNCHED_KEY = "__pp_upstream_launched_absent__"
    return _worker(rank, init_file, out_dir, case)


def _run(target, case):
    with tempfile.TemporaryDirectory() as tmp:
        init_file = os.path.join(tmp, "pg_init")
        mp.spawn(target, args=(init_file, tmp, case), nprocs=WORLD, join=True)
        out = {}
        for r in range(WORLD):
            p = os.path.join(tmp, f"r{r}.json")
            if os.path.exists(p):
                with open(p) as f:
                    out[r] = json.load(f)
        return out


class PPRetractedPassVoid797(unittest.TestCase):
    def test_the_same_width_mispair_is_delivered_without_either_fix(self):
        """RED, and the reason this file asserts on CONTENT.

        With the retraction lookup blinded the victim narrows its pass, as
        instr15/16/17 did 4030 times between them, and accepts the upstream's
        proxy. The widths MATCH -- 128 rows for a 128-token batch -- so
        `model_runner.forward`'s check would pass it into compute silently.
        The tags show what it actually got: 64 of those rows belong to the
        request the victim retracted and does not have in its batch at all.
        """
        res = _run(_blind_retraction_lookup, "retracted")
        v = res.get(VICTIM, {})
        self.assertIsNone(v.get("error"), f"the blinded receive must accept: {v}")
        self.assertFalse(v.get("voided"), f"the blinded child must not void: {v}")
        # The narrowing really happened: only the honourable rid survived.
        self.assertEqual(v.get("effective"), [RID_HONOURABLE], f"{v}")
        # WIDTH SAYS NOTHING. This is the ~1717 case, not the 1.
        self.assertEqual(
            v.get("rows"),
            VICTIM_TOKENS,
            f"the harness must present EQUAL widths or it tests the easy case: {v}",
        )
        # CONTENT SAYS EVERYTHING.
        self.assertEqual(
            v.get("foreign_rows"),
            UPSTREAM_ROWS_UNHONOURABLE,
            f"the delivered hidden states must belong to the wrong request "
            f"set for this to be the specimen: {v}",
        )

    def test_the_pass_is_voided_and_nothing_is_computed(self):
        """GREEN. The same wire, the same decision, the shipped code.

        No batch is built, so no proxy is paired with one -- the divergence is
        not detected, it is not created. The retracted rid keeps its
        `retracted` marks (that is what teaches the floor on the return trip);
        the collaterally dropped one is neither admitted nor retracted, which
        is the third state `record_return_trip` deliberately leaves alone.
        """
        res = _run(_worker, "retracted")
        v = res.get(VICTIM, {})
        self.assertIsNone(v.get("error"), f"victim failed: {v.get('error')}")
        self.assertTrue(v.get("voided"), f"the pass was not voided: {v}")
        self.assertEqual(v.get("effective"), [], f"a voided pass admits none: {v}")
        self.assertIsNone(v.get("rows"), f"nothing may be paired on a void: {v}")
        self.assertEqual(v.get("forwarded_admitted"), [], f"{v}")
        self.assertEqual(v.get("forwarded_retracted"), [RID_UNHONOURABLE], f"{v}")

    def test_detection_without_prevention_still_kills_the_pass(self):
        """#791c ALONE, pinned as the thing #797 replaces.

        Blind only the void decision and the receive guard -- untouched, with
        its own lookup intact -- refuses the proxy and raises. That is strictly
        better than computing on it and it is still a dead pass: boot instr18
        died on the first retraction, 1m44s in. Detection is not prevention.
        """
        res = _run(_blind_void_only, "retracted")
        v = res.get(VICTIM, {})
        self.assertIsNotNone(v.get("error"), f"the guard must still fire: {v}")
        self.assertIn("#791c PROXY BATCH DIVERGED", v["error"])
        self.assertIn(RID_UNHONOURABLE, v["error"])

    def test_the_void_reaches_the_rank_that_retracted_nothing(self):
        """THE THIRD PROCESS, and the reason the fact rides the wire.

        The last rank's own cache can honour everything, so its retraction set
        is empty and every rank-local test it could make says "run normally".
        It must withhold anyway: the pass is off, and a rank that restarted it
        would be the only one running -- with a decode batch, against the
        upstream's prefill, which is the same mispair one forward on.
        """
        res = _run(_worker, "retracted")
        d = res.get(DOWNSTREAM, {})
        self.assertIsNone(d.get("error"), f"downstream failed: {d.get('error')}")
        self.assertTrue(d.get("saw_voided"), f"the void did not travel: {d}")
        self.assertEqual(d.get("retracted_here"), [], f"it retracts nothing: {d}")
        self.assertTrue(d.get("voided"), f"it must withhold anyway: {d}")
        self.assertEqual(d.get("effective"), [], f"{d}")

    def test_the_upstream_proxy_is_drained_by_the_voided_rank(self):
        """THE CORPSE THE VOID WOULD OTHERWISE LEAVE.

        The upstream posted its proxy isend before anything downstream could
        tell it the pass was off, and a voided rank has no batch, so the
        ordinary receive never runs. The debt is the upstream's problem, not
        this rank's: `_pp_commit_comm_work(self.send_proxy_work)` on its next
        pass is a blocking wait on that message being taken. It is taken here,
        and the sentinel behind it arrives with an EMPTY stash -- which is the
        observable form of "the FIFO is still aligned".
        """
        res = _run(_worker, "retracted")
        v = res.get(VICTIM, {})
        self.assertIsNone(v.get("error"), f"victim failed: {v.get('error')}")
        self.assertTrue(v.get("drained"), f"the proxy was not drained: {v}")
        self.assertEqual(v.get("sentinel"), SENTINEL_TOKEN, f"{v}")
        self.assertEqual(v.get("stashed"), 0, f"a message was left behind: {v}")

    def test_an_undrained_voided_pass_strands_the_upstream_proxy(self):
        """CAN-FAIL FOR THE DRAIN, and the reason `launched` is per-hop.

        Blind only the key the sender writes its own slot state into and the
        drain declines, exactly as it must when no proxy is coming. Here one
        IS coming, so it is still on the wire when the sentinel is received and
        the typed channel stashes it: one unmatched message per voided pass,
        which is the bounded-recv corpse and wedges the UPSTREAM.
        """
        res = _run(_blind_launched_key, "retracted")
        v = res.get(VICTIM, {})
        self.assertIsNone(v.get("error"), f"victim failed: {v.get('error')}")
        self.assertTrue(v.get("voided"), f"the void itself must be unaffected: {v}")
        self.assertFalse(v.get("drained"), f"the drain must have declined: {v}")
        self.assertEqual(v.get("sentinel"), SENTINEL_TOKEN, f"{v}")
        self.assertEqual(
            v.get("stashed"), 1, f"the stranded proxy must be observable: {v}"
        )

    def test_a_congruent_pass_is_untouched(self):
        """DEFAULT PATH. A decision every rank can honour retracts nothing, so
        nothing is voided, nothing is drained, and the proxy is delivered whole
        -- no new refusal and no new withholding on a healthy pass."""
        res = _run(_worker, "congruent")
        v = res.get(VICTIM, {})
        self.assertIsNone(v.get("error"), f"victim failed: {v.get('error')}")
        self.assertFalse(v.get("voided"), f"a congruent pass must not void: {v}")
        self.assertEqual(
            v.get("effective"), sorted([RID_HONOURABLE, RID_UNHONOURABLE]), f"{v}"
        )
        self.assertEqual(v.get("rows"), VICTIM_TOKENS, f"proxy not delivered: {v}")
        self.assertEqual(v.get("foreign_rows"), 0, f"{v}")
        d = res.get(DOWNSTREAM, {})
        self.assertFalse(d.get("saw_voided"), f"{d}")
        self.assertFalse(d.get("voided"), f"{d}")


class PPVoidForwardRule797(unittest.TestCase):
    """WHERE THE VOID STOPS, and why #797 would otherwise trade a silent
    mispair for a wedge.

    #791b's void keeps the output ring matched for the FIRST rank, because
    that is where boot instr11 died. When the retraction happens on rank r,
    every rank in 1..r-1 ALSO holds a launched batch for that slot and also
    has an output receive posted for it -- and PP0, having absorbed the void,
    forwards nothing. Before #797 that shape produced a mispair (the
    retracting rank narrowed rather than emptied, so the last rank still sent
    a real output); with #797 it would produce a block. So the void travels
    the whole way the real output would have, and stops at r-1, because rank r
    and everything past it have empty slots and receives that early-return --
    one hop further is the bounded-recv corpse.

    Pure: the rule is a property of the decision the void already carries, so
    it needs no processes.
    """

    def _holder(self, pp_rank, is_last):
        return types.SimpleNamespace(
            pp_group=types.SimpleNamespace(is_last_rank=is_last),
            ps=types.SimpleNamespace(pp_rank=pp_rank, pp_size=WORLD),
        )

    def _void(self, retracted_by):
        from dataclasses import replace

        from sglang.srt.managers.pp_admission_congruence import PPAdmissionDecision
        from sglang.srt.managers.scheduler_pp_mixin import (
            _PP_VOID_OUTPUT_KEY,
            pp_admission_decision_to_wire,
        )

        entries = tuple(
            replace(
                e,
                admitted=False,
                retracted=True,
                retracted_by_rank=retracted_by,
                observed_local=0,
            )
            if e.rid == RID_UNHONOURABLE
            else replace(e, admitted=False)
            for e in _pp0_decision().entries
        )
        payload = {_PP_VOID_OUTPUT_KEY: True}
        payload.update(
            pp_admission_decision_to_wire(
                PPAdmissionDecision(mb_id=LIVE_MB, entries=entries)
            )
        )
        return payload

    def test_the_retracting_rank_is_read_off_the_carried_decision(self):
        from sglang.srt.managers.scheduler_pp_mixin import (
            pp_admission_decision_from_wire,
            pp_first_retracting_rank,
        )

        for r in (1, 2):
            self.assertEqual(
                pp_first_retracting_rank(
                    pp_admission_decision_from_wire(self._void(r))
                ),
                r,
            )
        self.assertIsNone(pp_first_retracting_rank(_pp0_decision()))

    def test_a_retraction_on_the_first_downstream_rank_stops_at_rank_zero(self):
        """r=1: no rank between PP0 and the retraction, so PP0 absorbs and
        forwards nothing. Rank 1's slot is empty and its receive
        early-returns; a hop here would be an unmatched message."""
        from sglang.srt.managers.scheduler_pp_mixin import pp_void_forward_payload

        self.assertIsNone(
            pp_void_forward_payload(self._holder(0, False), self._void(1))
        )

    def test_a_retraction_on_a_later_rank_reaches_every_rank_that_launched(self):
        """r=2: rank 1 ran the pass and is waiting for an output. PP0 forwards
        to it; rank 1 absorbs and stops."""
        from sglang.srt.managers.scheduler_pp_mixin import (
            _PP_VOID_OUTPUT_KEY,
            pp_void_forward_payload,
        )

        fwd = pp_void_forward_payload(self._holder(0, False), self._void(2))
        self.assertIsNotNone(fwd, "rank 1 would block without this hop")
        self.assertTrue(fwd[_PP_VOID_OUTPUT_KEY])
        # Verbatim, decision included, so the next hop applies the same rule.
        self.assertIsNone(pp_void_forward_payload(self._holder(1, False), fwd))

    def test_the_last_rank_never_forwards_and_an_ordinary_output_never_does(self):
        from sglang.srt.managers.scheduler_pp_mixin import (
            _PP_VOID_OUTPUT_KEY,
            pp_void_forward_payload,
        )

        self.assertIsNone(
            pp_void_forward_payload(self._holder(WORLD - 1, True), self._void(2))
        )
        # A message with no decision on it -- an ordinary output -- is not a
        # void and must never be forwarded as one.
        self.assertIsNone(
            pp_void_forward_payload(self._holder(0, False), {_PP_VOID_OUTPUT_KEY: True})
        )


if __name__ == "__main__":
    unittest.main()
