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

    #801 NARROWED WHAT "STOPS" MEANS HERE. The stop above is unchanged and is
    still #796's law. What changed is the case this class could not name: a
    void whose carried decision names NO retracting rank, or that carries no
    decision at all. #797 stopped on those too, and stopping on them is the
    intermediate-hop wedge -- see `test_a_void_with_no_decision_on_it_is_
    relayed_not_swallowed` below and `pp_void_relay_stop_rank`.
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
            (
                replace(
                    e,
                    admitted=False,
                    retracted=True,
                    retracted_by_rank=retracted_by,
                    observed_local=0,
                )
                if e.rid == RID_UNHONOURABLE
                else replace(e, admitted=False)
            )
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

    def test_the_last_rank_never_forwards(self):
        from sglang.srt.managers.scheduler_pp_mixin import pp_void_forward_payload

        self.assertIsNone(
            pp_void_forward_payload(self._holder(WORLD - 1, True), self._void(2))
        )

    def test_a_void_with_no_decision_on_it_is_relayed_not_swallowed(self):
        """#801 CORRECTION, and the mislabel is the whole story.

        This assertion used to read `assertIsNone` and to justify itself with
        "a message with no decision on it -- an ordinary output -- is not a
        void". The payload it passes is `{_PP_VOID_OUTPUT_KEY: True}`, which
        is not an ordinary output and cannot be one: `_pp_absorb_void_output`
        reaches `pp_void_forward_payload` only AFTER popping that key, so
        this function is never called for an ordinary output at all. What the
        old assertion actually pinned was the pre-#801 DEFAULT -- a void whose
        stop position cannot be derived is swallowed -- and swallowing it
        parks every rank between here and the void's source in an
        unconditional output receive (specimen wedge_802f_1712).

        See `pp_void_relay_stop_rank` and
        test_pp_void_send_contract_801.py for the contract that replaces it.
        """
        from sglang.srt.managers.scheduler_pp_mixin import (
            _PP_VOID_OUTPUT_KEY,
            pp_void_forward_payload,
        )

        fwd = pp_void_forward_payload(
            self._holder(0, False), {_PP_VOID_OUTPUT_KEY: True}
        )
        self.assertIsNotNone(fwd)
        self.assertTrue(fwd[_PP_VOID_OUTPUT_KEY])
        # Still one hop short of the source, which is the rank that has no
        # slot for this generation and issues no receive.
        self.assertIsNone(pp_void_forward_payload(self._holder(WORLD - 2, False), fwd))


CHUNK_SIZE = 512
PREFIX_DONE = 4096
RID_CHUNKED = "c0ffee00000000000000000000000000"


class _StubChunkedReq:
    """The scheduler's `self.chunked_req`, in the state boot instr19 had it.

    ~17000-token prompt, `--chunked-prefill-size 512`: 4096 tokens already
    computed and stashed, chunk 9 prepared this round. Only the fields the
    two shipped functions under test read are present.
    """

    def __init__(self, pool_idx=0):
        from sglang.srt.utils.common import Range

        self.rid = RID_CHUNKED
        self.req_pool_idx = pool_idx
        self.prefix_indices = torch.arange(PREFIX_DONE, dtype=torch.int64)
        self.extend_range = Range(PREFIX_DONE, PREFIX_DONE + CHUNK_SIZE)
        self.inflight_middle_chunks = 1
        self.retracted = False

    def reset_for_retract(self):
        # The two lines of the shipped `Req.reset_for_retract` that matter
        # here (schedule_batch.py): `extend_range` becomes None -- which is
        # what the next round dereferences -- and the tree handles for every
        # chunk already stashed are thrown away.
        self.extend_range = None
        self.prefix_indices = torch.empty((0,), dtype=torch.int64)
        self.retracted = True


class _StubPool:
    def __init__(self, rows=1, cols=PREFIX_DONE + CHUNK_SIZE):
        self.req_to_token = torch.arange(rows * cols, dtype=torch.int64).view(
            rows, cols
        )
        self.freed_req = []

    def free(self, req):
        self.freed_req.append(req)


class _StubAllocator:
    def __init__(self):
        self.freed = []

    def free(self, indices):
        self.freed.append(indices.clone())


class PPVoidChunkedRequest797b(unittest.TestCase):
    """#797b: what a voided pass owes `self.chunked_req`.

    THE SPECIMEN, boot instr19 (the first metal run carrying #797): health
    08:46:48, DEAD 08:47:41 -- 53 seconds. Retractions 3, passes voided 3,
    `PROXY BATCH DIVERGED` 0, proxy/batch mismatch 0, so the prevention did
    exactly what it claimed and then the bookkeeping killed the boot:

        AttributeError: 'NoneType' object has no attribute 'end'
          scheduler.py  get_next_batch_to_run
            if self.chunked_req.extend_range.end > len(...prefix_indices):

    `self.chunked_req` is SCHEDULER state that outlives the round, and it is
    ALSO a member of the batch the round builds -- so #797's void handed it to
    the disposal loop like any other admitted request, and
    `reset_for_retract` set `extend_range = None`. With
    `--chunked-prefill-size 512` against ~17000-token prompts a chunked
    request is in flight essentially always, so the first void hit it.

    THE ARMS DRIVE THE SHIPPED `get_next_batch_to_run`, not a copy of its
    dereference, so the red arm reproduces the traceback's own line and a
    later edit to that line cannot leave this test passing against nothing.
    The method is bounded by #797's own withheld-round gate, which is the
    first `return` after the block that crashed.
    """

    def _scheduler(self, chunked_req, batch_reqs):
        from sglang.srt.managers.scheduler import Scheduler
        from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin

        pool = _StubPool()
        h = types.SimpleNamespace(
            # -- what `_pp_absorb_void_output` reads
            ps=types.SimpleNamespace(pp_rank=0, pp_size=WORLD),
            pp_group=types.SimpleNamespace(is_last_rank=False, is_first_rank=True),
            chunked_req=chunked_req,
            waiting_queue=[],
            running_mbs=[None] * 3,
            req_to_token_pool=pool,
            token_to_kv_pool_allocator=_StubAllocator(),
            _pp_admission_guard=None,
            _pp_chunked_req_before_by_slot=[None] * 3,
            # -- what `get_next_batch_to_run`'s prologue reads, all inert
            enable_hierarchical_cache=False,
            enable_fpm=False,
            enable_hisparse=False,
            enable_lora=False,
            dllm_config=None,
            kv_session_offload=None,
            _regime_observer_mode="off",
            regime_observer=None,
            kv_capacity_runtime=None,
            _pp_admission_pass_voided=True,
            _round_built_nothing=True,
            stashed=[],
            server_args=types.SimpleNamespace(
                kv_reshard_vectors=None,
                enable_phase_flip=False,
                enable_vram_dial=False,
                kv_pressure_ladder=None,
                gdn_resident_state_slots=None,
            ),
        )
        h.process_pending_chunked_abort = lambda: None
        h._abort_on_waiting_timeout = lambda: None
        h._abort_on_running_timeout = lambda rb: None
        h._update_uniform_pool_budget = lambda: None
        h._census_tick = lambda: None
        h._corridor_trace_tick = lambda: None
        h._flight_serving_tick = lambda: None
        h.stash_chunked_request = lambda req: h.stashed.append(req)
        h._pp_absorb_void_output = types.MethodType(
            SchedulerPPMixin._pp_absorb_void_output, h
        )
        h._pp_note_chunked_req_before_admission = types.MethodType(
            SchedulerPPMixin._pp_note_chunked_req_before_admission, h
        )
        h.get_next_batch_to_run = types.MethodType(Scheduler.get_next_batch_to_run, h)
        h._batch = types.SimpleNamespace(reqs=list(batch_reqs))
        return h

    def _void_pass(self, h, mb_id=0):
        """One voided pass over the SHIPPED code: note the pre-admission
        chunked request (what `_event_loop_pp_body` does immediately before
        `get_next_batch_to_run`), let the round admit it, then absorb the
        void."""
        from sglang.srt.managers.scheduler_pp_mixin import (
            _PP_VOID_OUTPUT_KEY,
            pp_admission_decision_to_wire,
        )

        h._pp_note_chunked_req_before_admission(mb_id)
        mbs = [None] * 3
        mbs[mb_id] = h._batch
        mb_metadata = [None] * 3
        payload = {_PP_VOID_OUTPUT_KEY: True}
        payload.update(pp_admission_decision_to_wire(_pp0_decision()))
        return h._pp_absorb_void_output(mb_id, payload, mbs, mb_metadata)

    def _next_round(self, h):
        """The pass AFTER the void, on the shipped method. Returns nothing --
        the point is whether it raises."""
        running = types.SimpleNamespace(
            is_prefill_only=False, reqs=[], batch_is_full=False
        )
        return h.get_next_batch_to_run(running_batch=running, last_batch=None)

    def test_instr19s_raise_returns_when_the_pre_admission_value_is_blinded(self):
        """RED, and the can-fail proof for every green below.

        ONE RETURN VALUE IS NEUTERED, through `scheduler_pp_mixin`'s own
        module globals: `pp_void_keeps_request` returning False is exactly the
        disposal that shipped before it existed -- "every batch member is the
        round's to hand back". Everything else still runs its own body: the
        pre-admission value is still recorded, `self.chunked_req` is still put
        back, `_park_chunked_prefill_chunk` still parks the chunk. The loop
        then retracts the chunked request anyway, `reset_for_retract` sets
        `extend_range = None`, and the SHIPPED `get_next_batch_to_run` raises
        instr19's own AttributeError on instr19's own line. A wholesale revert
        would yield an AttributeError from the harness and prove nothing.
        """
        from sglang.srt.managers import scheduler_pp_mixin as m

        chunked = _StubChunkedReq()
        h = self._scheduler(chunked, [chunked])
        keeps = m.pp_void_keeps_request
        m.pp_void_keeps_request = lambda req, resident, chunked_before: False
        try:
            self._void_pass(h)
        finally:
            m.pp_void_keeps_request = keeps
        # The retraction really happened in the blinded run -- that is what
        # makes the raise below the specimen rather than a broken harness.
        self.assertTrue(chunked.retracted)
        self.assertIsNone(chunked.extend_range)
        with self.assertRaises(AttributeError) as caught:
            self._next_round(h)
        self.assertIn("'NoneType' object has no attribute 'end'", str(caught.exception))

    def test_a_carried_chunk_survives_the_void_and_the_next_round_runs(self):
        """GREEN, and the arm that reproduces instr19 when the fix is absent.

        The chunked request is put back, parked, and NOT retracted, so the
        next round's `self.chunked_req.extend_range.end` is a number rather
        than an attribute of None.
        """
        chunked = _StubChunkedReq()
        h = self._scheduler(chunked, [chunked])
        self.assertTrue(self._void_pass(h))
        self.assertIs(h.chunked_req, chunked, "the chunked request was dropped")
        self.assertIsNotNone(
            chunked.extend_range,
            "reset_for_retract ran on the chunked request -- this is instr19",
        )
        self.assertFalse(chunked.retracted)
        self.assertNotIn(chunked, h.waiting_queue, "re-queued AND still chunked")
        try:
            self._next_round(h)
        except AttributeError as exc:  # pragma: no cover - this IS the defect
            self.fail(f"instr19's raise is back: {exc}")

    def test_the_parked_chunk_is_not_stashed_next_round(self):
        """THE HALF A DEFINED extend_range ALONE WOULD NOT FIX.

        `get_next_batch_to_run` stashes the chunk whenever `extend_range.end >
        len(prefix_indices)`. The chunk a voided pass prepared was computed by
        no rank downstream of the retraction, so stashing it would put a node
        in THIS rank's radix tree that they lack -- and the next offer for
        that prefix is then unhonourable, which is a void per chunk, i.e.
        #630's livelock. The park makes the stash a no-op, which is the state
        `add_chunked_req`'s own zero-budget park already produces.
        """
        chunked = _StubChunkedReq()
        h = self._scheduler(chunked, [chunked])
        self._void_pass(h)
        self.assertEqual(
            chunked.extend_range.end,
            len(chunked.prefix_indices),
            "a parked chunk must leave end == len(prefix_indices)",
        )
        self._next_round(h)
        self.assertEqual(h.stashed, [], "an uncomputed chunk was stashed")

    def test_only_this_rounds_allocation_is_freed_never_the_tree_s_prefix(self):
        """THE DOUBLE FREE THE OBVIOUS FIX WOULD HAVE MADE.

        `_release_dynamic_chunk_probe` frees `[:extend_range.end]`, which is
        right for the synthetic dynamic-chunk probe it was written for and
        wrong here: `[:len(prefix_indices)]` belongs to the RADIX TREE, held
        under a lock ref by every chunk already stashed. Only
        `[len(prefix_indices):end]` was allocated by the round being undone.
        """
        chunked = _StubChunkedReq()
        h = self._scheduler(chunked, [chunked])
        self._void_pass(h)
        self.assertEqual(len(h.token_to_kv_pool_allocator.freed), 1)
        freed = h.token_to_kv_pool_allocator.freed[0]
        self.assertEqual(len(freed), CHUNK_SIZE, "wrong number of pages returned")
        expected = h.req_to_token_pool.req_to_token[
            0, PREFIX_DONE : PREFIX_DONE + CHUNK_SIZE
        ]
        self.assertTrue(torch.equal(freed, expected))
        self.assertEqual(
            h.req_to_token_pool.freed_req, [], "the req-pool row was handed back"
        )

    def test_the_admissions_inflight_increment_is_given_back(self):
        """`inflight_middle_chunks` is incremented at admission and
        decremented in `process_batch_result_prefill`, which never runs for a
        voided pass. It gates `req.finished()`, so a leaked increment is a
        request that can never report finished."""
        chunked = _StubChunkedReq()
        h = self._scheduler(chunked, [chunked])
        self._void_pass(h)
        self.assertEqual(chunked.inflight_middle_chunks, 0)

    def test_a_chunk_started_this_round_is_un_started_not_kept(self):
        """THE OTHER DIRECTION, and it is why the pre-admission value is
        recorded rather than `self.chunked_req` simply being spared.

        A rank downstream of the retraction never ran this round, so it has NO
        chunked request. A rank that started one this round and kept it would
        be the only rank mid-chunk -- a membership divergence, which is the
        defect #797 exists to prevent. Restoring the pre-admission value (here
        None) un-starts it, and the request is then an ordinary batch member
        that the disposal loop retracts and re-queues.
        """
        started = _StubChunkedReq()
        h = self._scheduler(None, [started])
        h._pp_note_chunked_req_before_admission(0)
        h.chunked_req = started  # what `adder.new_chunked_req` did this round
        mbs, mb_metadata = [None] * 3, [None] * 3
        mbs[0] = h._batch
        from sglang.srt.managers.scheduler_pp_mixin import _PP_VOID_OUTPUT_KEY

        h._pp_absorb_void_output(0, {_PP_VOID_OUTPUT_KEY: True}, mbs, mb_metadata)
        self.assertIsNone(h.chunked_req, "a chunk started this round was kept")
        self.assertTrue(started.retracted, "it must be retracted like any member")
        self.assertIn(started, h.waiting_queue)

    def test_the_crash_line_is_still_the_line_this_pins(self):
        """DRIFT GUARD. The arms above bound the shipped method with #797's
        own gate; if the dereference this file exists for ever moves out of
        `get_next_batch_to_run`, they would pass against nothing."""
        import inspect

        from sglang.srt.managers.scheduler import Scheduler

        src = inspect.getsource(Scheduler.get_next_batch_to_run)
        self.assertIn("self.chunked_req.extend_range.end", src)
        self.assertLess(
            src.index("self.chunked_req.extend_range.end"),
            src.index("_pp_admission_pass_voided"),
            "the crash line must stay AHEAD of the gate that bounds these arms",
        )


PROMPT_TOKENS = 4096


class _ChunkingReq:
    """One request being chunk-prefilled, advanced the way the scheduler
    advances it: `prepare_for_extend` sets `extend_range` to the chunk about
    to run, and a COMPLETED round leaves `extend_range.end` at the absolute
    index computed so far (`Req.get_fill_ids` is
    `full_untruncated_fill_ids[:extend_range.end]`)."""

    def __init__(self, rid, total=PROMPT_TOKENS, chunk=CHUNK_SIZE):
        from sglang.srt.utils.common import Range

        self.rid = rid
        self.total = total
        self.chunk = chunk
        self.prefix_indices = torch.empty((0,), dtype=torch.int64)
        self.extend_range = Range(0, 0)
        self.inflight_middle_chunks = 0
        self.req_pool_idx = None

    @property
    def computed(self):
        return self.extend_range.end

    def stash_and_rematch(self):
        """The top of a round: the completed chunk goes into the radix tree
        and `init_next_round_input` matches it straight back, so
        `prefix_indices` catches up with what was computed."""
        self.prefix_indices = torch.arange(self.computed, dtype=torch.int64)

    def prepare_chunk(self):
        """`add_chunked_req` + `prepare_for_extend`."""
        from sglang.srt.utils.common import Range

        start = len(self.prefix_indices)
        self.extend_range = Range(start, min(self.total, start + self.chunk))
        self.inflight_middle_chunks += 1

    def done(self):
        return self.computed >= self.total


class PPChunkedOfferLivelock797c(unittest.TestCase):
    """#797c: `local` was never measured for the chunked request.

    THE SPECIMEN, boot instr19, one rid, three retractions in one second,
    `told` GROWING by exactly one chunk while `local` stayed 0:

        08:47:26 PP1 unhonourable prefix: rid=d4f59edf... told=512  local=0
        08:47:27 PP1 unhonourable prefix: rid=d4f59edf... told=1024 local=0
        08:47:27 PP1 unhonourable prefix: rid=d4f59edf... told=1536 local=0
        -> 3 voided passes, 1 distinct rid

    and one second earlier, in the same log, all three ranks had admitted that
    request's FIRST chunk congruently (`prefix_lens=0`). PP1 had the KV. What
    it did not have was a way to say so: `_pp_reconcile_incoming_admission`
    looks a rid up in `self.waiting_queue`, and scheduler.py drops every
    admitted request out of that queue, so a chunked request -- which then
    lives in `self.chunked_req` -- misses the lookup on every round but its
    first and the miss DEFAULTS to 0.

    So the retraction was spurious, on essentially every large request
    (`--chunked-prefill-size 512`, ~17000-token prompts), on every round but
    the first. Before #797 that was a narrowed pass and a silent same-width
    mispair; after #797 it is a voided pass, and the boot's own amended gate
    (voided passes <= distinct retracted rids) fails it 3 to 1.

    THE TERMINATION ARGUMENT THESE ARMS PROVE, rather than assert:
      * `told` is PP0's count of the tokens it has computed for the request;
        `local` is the same count on the downstream (`extend_range.end`).
      * Every rank advances the request through the same chunk sequence, so
        after every COMPLETED round the two counts are EQUAL and `local >=
        told` holds -- no retraction, hence no void.
      * A round that does NOT complete is parked on the ranks that prepared it
        (#797b), which restores the same count -- so `told` is NON-INCREASING
        across a void and strictly increasing only across a completed round.
      * A strictly increasing count bounded by the prompt length terminates.
    """

    def _reconciler(self, chunked_req):
        from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin

        h = types.SimpleNamespace(
            ps=types.SimpleNamespace(pp_rank=VICTIM, pp_size=WORLD),
            waiting_queue=[],
            chunked_req=chunked_req,
            tree_cache=None,
        )
        h._pp_reconcile_incoming_admission = types.MethodType(
            SchedulerPPMixin._pp_reconcile_incoming_admission, h
        )
        return h

    def _offer(self, pp0_req):
        """PP0's decision for this round, through the SHIPPED builder."""
        from sglang.srt.managers.pp_admission_congruence import (
            build_pp_admission_decision,
        )

        return build_pp_admission_decision(
            LIVE_MB, [_PP0View(pp0_req)], pp_size=WORLD, guard=None
        )

    def _drive(self, rounds=12, blind=False, park=True):
        """Run ONE rid through repeated offers across chunk boundaries, on the
        shipped reconcile. Returns (told sequence, retraction count, done)."""
        from sglang.srt.managers import scheduler_pp_mixin as m

        pp0 = _ChunkingReq("d4f59edff89e46758eedf3d56227fe72")
        pp1 = _ChunkingReq("d4f59edff89e46758eedf3d56227fe72")
        h = self._reconciler(pp1)
        told_seq, retractions = [], 0
        real = m.pp_chunked_local_match
        if blind:
            # The pre-#797c reader: "not in the waiting queue, so nothing
            # known". Every body still runs -- the reconcile, the builder, the
            # retraction test -- only this one lookup answers as it did.
            m.pp_chunked_local_match = lambda req: None
        try:
            for _ in range(rounds):
                if pp0.done():
                    break
                # Top of PP0's round: stash the completed chunk, re-match it,
                # prepare the next one, then publish the offer.
                pp0.stash_and_rematch()
                pp0.prepare_chunk()
                decision = self._offer(pp0)
                told_seq.append(decision.entries[0].prefix_len)
                _, amended = h._pp_reconcile_incoming_admission(decision)
                if any(e.retracted for e in amended.entries):
                    retractions += 1
                    if park:
                        # #797b: the pass is voided, so PP0 parks the chunk it
                        # prepared -- its count does NOT advance.
                        from sglang.srt.utils.common import Range

                        pp0.extend_range = Range(
                            len(pp0.prefix_indices), len(pp0.prefix_indices)
                        )
                    # park=False is instr19's tree: the prepared chunk is left
                    # in place, so the next round stashes a chunk no rank
                    # downstream computed and the offer climbs.
                    continue
                # The round completed on every rank.
                pp1.stash_and_rematch()
                pp1.prepare_chunk()
        finally:
            m.pp_chunked_local_match = real
        # DONE IS THE DOWNSTREAM'S, not PP0's. PP0 racing ahead of a stage
        # that never computes the request is the livelock, not the cure --
        # under park=False PP0 reaches the end of the prompt alone.
        return told_seq, retractions, pp1.done()

    def test_repeated_offers_of_one_rid_never_retract_and_the_request_finishes(self):
        """GREEN, and the termination proof.

        `told` is non-decreasing, advances by exactly one chunk per COMPLETED
        round, never repeats a value (which is what a livelock looks like),
        and the request reaches the end of its prompt.
        """
        told, retractions, done = self._drive()
        self.assertEqual(retractions, 0, f"spurious retractions: told={told}")
        self.assertTrue(done, f"the request never finished: told={told}")
        self.assertEqual(told, sorted(told), "told must never go backwards")
        self.assertEqual(len(told), len(set(told)), f"told repeated: {told}")
        self.assertEqual(
            told[: len(told)],
            [i * CHUNK_SIZE for i in range(len(told))],
            "told must advance by exactly one chunk per completed round",
        )

    def test_the_blinded_lookup_reproduces_instr19s_growing_offer(self):
        """RED #1: instr19's tree exactly -- #797c blinded AND no park.

        The first round is congruent at told=0 and RUNS (log line 1737, all
        three ranks `prefix_lens=0`). From the second round on the lookup
        misses, `local` reads 0 for a request this rank has computed, every
        offer retracts, and because the prepared chunk is never parked PP0
        stashes it anyway and re-offers one chunk more:

            told=512 local=0 / told=1024 local=0 / told=1536 local=0

        3 voided passes, 1 distinct rid -- the amended gate's failing ratio.
        The offer grows without bound, so this cannot converge: it is a
        livelock, not a slow clamp.
        """
        told, retractions, done = self._drive(rounds=12, blind=True, park=False)
        self.assertFalse(done, "the downstream never computed the request")
        self.assertEqual(
            retractions,
            len(told) - 1,
            f"every round after the first must retract: {told}",
        )
        self.assertEqual(
            told[:4],
            [0, CHUNK_SIZE, 2 * CHUNK_SIZE, 3 * CHUNK_SIZE],
            "instr19's measured sequence (0 then 512/1024/1536) must come back",
        )
        # THE SHAPE OF THE LIVELOCK, stated as a number: PP0 walks the whole
        # prompt on its own -- one chunk per voided round -- while the
        # downstream computes nothing at all. The offer is bounded here only
        # by the prompt length, and every one of those rounds is a void.
        self.assertEqual(told[-1] + CHUNK_SIZE, PROMPT_TOKENS)

    def test_the_park_alone_leaves_a_flat_livelock(self):
        """RED #2, and the reason #797b is NOT sufficient on its own.

        With the park in place and only #797c blinded, the offer stops
        growing -- and the request still never finishes, because the retraction
        that voids the round is spurious in the first place. Fixing the
        runaway offer without fixing the false negative converts an
        accelerating livelock into a stationary one, which the amended gate
        fails just as hard.
        """
        told, retractions, done = self._drive(rounds=12, blind=True, park=True)
        self.assertFalse(done, "the blinded run was supposed to livelock")
        self.assertGreaterEqual(retractions, 11)
        self.assertEqual(
            set(told[1:]), {CHUNK_SIZE}, f"the park must hold the offer flat: {told}"
        )

    def test_told_does_not_advance_across_a_voided_round(self):
        """#797b's half of the termination argument, isolated.

        The park is what makes `told` non-increasing across a void. Without
        it, PP0 stashes a chunk no downstream rank computed and re-offers
        told + one chunk -- which is the growth instr19 measured, and it is
        unbounded.
        """
        pp0 = _ChunkingReq("rid-park")
        pp0.stash_and_rematch()
        pp0.prepare_chunk()
        before = self._offer(pp0).entries[0].prefix_len
        from sglang.srt.utils.common import Range

        pp0.extend_range = Range(len(pp0.prefix_indices), len(pp0.prefix_indices))
        pp0.stash_and_rematch()
        pp0.prepare_chunk()
        self.assertEqual(
            self._offer(pp0).entries[0].prefix_len,
            before,
            "a voided round must not advance the offer",
        )

    def test_a_rid_that_is_genuinely_unknown_is_reported_as_unresolved(self):
        """REGRESSION, RENAMED WITH THE CONTRACT IT PINS (#944).

        It was `..._still_reads_as_zero`, and the name outlived its meaning by
        one ticket: the BEHAVIOUR it guards is that a rid this rank cannot
        locate is never admitted this pass, and that is unchanged. What
        changed is how the rank SAYS SO. "Nothing known" and "nothing there"
        were the same word, and being the same word is the defect (#797c,
        #798, #944 -- three instances). A name that still said "zero" would
        keep teaching the next reader the sentence that cost three boots."""
        from sglang.srt.managers.pp_admission_congruence import (
            PPAdmissionDecision,
            PPAdmissionEntry,
        )

        h = self._reconciler(_ChunkingReq("some-other-rid"))
        decision = PPAdmissionDecision(
            mb_id=LIVE_MB,
            entries=(
                PPAdmissionEntry(
                    rid=RID_UNHONOURABLE, prefix_len=TOLD_PREFIX, extend_len=64
                ),
            ),
        )
        effective, amended = h._pp_reconcile_incoming_admission(decision)
        self.assertEqual(effective, {})
        self.assertTrue(amended.entries[0].retracted)
        # #944 CONTRACT INVERTED, NOT RELAXED. This pinned the miss as the
        # specimen's exact 0 -- correct while a lookup miss was SPELLED AS A
        # MEASUREMENT, which is the class #944 removes (#797c, #798 and #944
        # are three instances of it). The BEHAVIOUR reproduced here is
        # unchanged: the entry is still retracted and still absent from
        # `effective`. Only the number changes, from a 0 indistinguishable
        # from "no prefix" to a named sentinel. Inverted deliberately and
        # never deleted -- this file is the proof the class fix landed.
        self.assertIsNone(amended.entries[0].observed_local)
        self.assertTrue(
            amended.entries[0].unresolved,
            "and the miss must SAY it is a miss, on the wire, not by being a "
            "number that happens to look like one",
        )


class _PP0View:
    """What `build_pp_admission_decision` reads off a request."""

    def __init__(self, req):
        self.rid = req.rid
        self.prefix_indices = req.prefix_indices
        self.extend_input_len = req.extend_range.length


class _StubBatch:
    """A minimal `self.mbs[mb_id]`-shaped object: only `.reqs` is read by
    `_pp_void_own_batch`."""

    def __init__(self, reqs):
        self.reqs = list(reqs)


class _StubRoundReq:
    """A plain, round-owned batch member: not resident, not the chunked
    request. `req_pool_idx=None` makes `_release_dynamic_chunk_probe` a
    no-op on it (its own `getattr(req, "req_pool_idx", None) is not None`
    guard), which is correct here -- this stub never held real KV rows."""

    def __init__(self, rid):
        self.rid = rid
        self.req_pool_idx = None
        self.extend_range = None
        self.retracted = False

    def reset_for_retract(self):
        self.retracted = True


class PPVoidOwnBatch797d(unittest.TestCase):
    """#797d: the void a rank decides against ITS OWN pass must reach the
    BATCH `get_next_batch_to_run` handed back, not only the admission dict.

    THE SPECIMEN, SPECIMEN_wedge_19-02.txt (PP0/PP1/PP2, pp_size=3, tp_size=1,
    chunked prefill in flight, rid=429872ab). PP1 retracts and voids
    (`#791 unhonourable prefix`, `#797 pass voided on rank 1`); PP0 absorbs
    correctly (`#791b void output`). PP2's OWN frame, at the exact pass in
    question, shows both halves of the contradiction at once:
    `effective: {}` (`_pp_void_retracted_pass` ran and voided) AND
    `cur_batch: <ScheduleBatch ...>` (NOT None) in `_event_loop_pp_body`'s
    own locals, py-spy stack `_pp_recv_proxy_tensors <- _event_loop_pp_body:
    1503`. PP2 then blocks forever in a proxy receive for a message the
    voided PP1 (correctly, via `_pp_drain_voided_proxy`) never sends.

    `_pp_void_own_batch` is the gate closing this: called right after
    `get_next_batch_to_run`, before the admission-decision send and before
    `cur_batch = self.mbs[mb_id]` is read, so a rank that decided
    `_pp_admission_pass_voided = True` cannot carry a non-empty
    `self.mbs[mb_id]` into either of those two reads regardless of what
    `get_next_batch_to_run` produced.

    THE ARMS DRIVE THE SHIPPED METHOD DIRECTLY (`types.MethodType`, the same
    pattern `PPVoidChunkedRequest797b` uses for `_pp_absorb_void_output`),
    not a copy of it, so a later edit to the gate cannot leave these tests
    passing against nothing.
    """

    def _holder(self, chunked_req=None, chunked_before=None, mb_id=1, size=3):
        from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin

        h = types.SimpleNamespace(
            mbs=[None] * size,
            mb_metadata=[None] * size,
            chunked_req=chunked_req,
            waiting_queue=[],
            running_mbs=[None] * size,
            req_to_token_pool=_StubPool(),
            token_to_kv_pool_allocator=_StubAllocator(),
            _pp_chunked_req_before_by_slot=[None] * size,
        )
        h._pp_chunked_req_before_by_slot[mb_id] = chunked_before
        h._pp_void_own_batch = types.MethodType(SchedulerPPMixin._pp_void_own_batch, h)
        return h

    def test_the_voided_ranks_batch_is_cleared_even_when_get_next_batch_to_run_disagreed(
        self,
    ):
        """RED, and the can-fail proof for every other test below.

        Reproduces the specimen's own contradiction directly rather than
        depending on `get_next_batch_to_run`'s internal void guard to
        (dis)agree: `self.mbs[mb_id]` is set to a non-None batch, exactly as
        PP2's own frame shows it, on a rank that has already decided
        `_pp_admission_pass_voided = True`. Before this change
        `_pp_void_own_batch` does not exist at all, so this is
        `AttributeError`; after it, `self.mbs[mb_id]` must be None -- the
        gate `_event_loop_pp_body` applies is `if self.
        _pp_admission_pass_voided: self._pp_void_own_batch(mb_id)`.
        """
        h = self._holder()
        h.mbs[1] = _StubBatch([_StubRoundReq("round-owned")])
        h._pp_admission_pass_voided = True

        if h._pp_admission_pass_voided:
            h._pp_void_own_batch(1)

        self.assertIsNone(
            h.mbs[1],
            "the specimen's own contradiction: a voided pass still carrying a batch",
        )

    def test_a_carried_chunk_is_restored_not_consumed_by_the_own_void(self):
        """The chunked request must survive as `_pp_absorb_void_output`
        leaves it: parked (`extend_range.end == len(prefix_indices)`), not
        retracted, still `self.chunked_req`, and never appended to
        `waiting_queue` (it re-admits from `self.chunked_req` directly, so
        appending it would duplicate it)."""
        chunked = _StubChunkedReq()
        h = self._holder(chunked_req=chunked, chunked_before=chunked, mb_id=1)
        h.mbs[1] = _StubBatch([chunked, _StubRoundReq("round-owned")])
        h._pp_admission_pass_voided = True

        h._pp_void_own_batch(1)

        self.assertIs(h.chunked_req, chunked, "the chunked request was dropped")
        self.assertIsNotNone(
            chunked.extend_range,
            "reset_for_retract ran on the chunked request -- this is instr19",
        )
        self.assertEqual(chunked.extend_range.end, len(chunked.prefix_indices))
        self.assertFalse(chunked.retracted)
        self.assertNotIn(chunked, h.waiting_queue, "re-queued AND still chunked")
        # The round-owned member is not the chunked request and is not
        # resident: it must still be released and re-queued like any other
        # ordinary batch member of a voided pass.
        self.assertEqual(len(h.waiting_queue), 1)
        self.assertEqual(h.waiting_queue[0].rid, "round-owned")

    def test_launched_is_false_for_the_voided_rank_after_the_gate(self):
        """`_event_loop_pp_body` forwards `launched=self.mbs[mb_id] is not
        None` at both send sites (PP0's own decision and the non-first-rank
        forward). After the gate runs, that expression must read False for
        a voided slot -- this is what stops the upstream ever believing a
        voided downstream rank launched something."""
        h = self._holder()
        h.mbs[1] = _StubBatch([_StubRoundReq("r1")])
        h._pp_admission_pass_voided = True

        h._pp_void_own_batch(1)
        launched = h.mbs[1] is not None

        self.assertFalse(launched)

    def test_a_non_voided_pass_is_untouched_by_the_gate(self):
        """REGRESSION. The gate in `_event_loop_pp_body` is `if self.
        _pp_admission_pass_voided: self._pp_void_own_batch(mb_id)` -- a pass
        that never voided must not even reach `_pp_void_own_batch`, so its
        batch, chunked_req and waiting_queue are byte-identical to before."""
        chunked = _StubChunkedReq()
        h = self._holder(chunked_req=chunked, chunked_before=chunked, mb_id=1)
        batch = _StubBatch([chunked, _StubRoundReq("r1")])
        h.mbs[1] = batch
        h._pp_admission_pass_voided = False

        if h._pp_admission_pass_voided:
            h._pp_void_own_batch(1)

        self.assertIs(h.mbs[1], batch)
        self.assertIs(h.chunked_req, chunked)
        self.assertEqual(h.waiting_queue, [])
        self.assertFalse(chunked.retracted)

    def test_an_already_none_slot_is_a_safe_no_op(self):
        """DEFENSE-IN-DEPTH IDEMPOTENCE. When `get_next_batch_to_run`
        already honoured the void on its own (scheduler.py's own
        `_pp_admission_pass_voided` guard), `self.mbs[mb_id]` is already
        None and `_pp_void_own_batch` must do nothing and report it did
        nothing -- this is what makes stacking the gate on top of that guard
        safe rather than a second, competing source of truth."""
        h = self._holder()
        h._pp_admission_pass_voided = True

        result = h._pp_void_own_batch(1)

        self.assertFalse(result)
        self.assertIsNone(h.mbs[1])
        self.assertEqual(h.waiting_queue, [])

    def test_an_already_reset_chunked_req_does_not_crash_the_next_pass(self):
        """THE INSTR19 STATE, ARRIVING PRE-BROKEN. `chunked_before` is
        constructed already in the post-`reset_for_retract` shape
        (`extend_range=None`) -- the state `_park_chunked_prefill_chunk`
        cannot repair (its own docstring: "already parked, already reset,
        or never prepared -- nothing to give back", and it leaves
        `extend_range` untouched in that branch). Before the defensive
        check this added, `_pp_void_own_batch` would still assign
        `self.chunked_req = chunked_before` and leave it there, and the
        NEXT pass's `get_next_batch_to_run` dereferences
        `self.chunked_req.extend_range.end` unconditionally the instant
        `self.chunked_req is not None` (scheduler.py) --
        `AttributeError: 'NoneType' object has no attribute 'end'`, boot
        instr19's own crash shape. This asserts the gate clears
        `self.chunked_req` to None instead, and then runs the next pass's
        own guard directly (not a copy of it) to prove it does not raise.
        """
        broken = _StubChunkedReq()
        broken.reset_for_retract()  # extend_range -> None, exactly instr19
        h = self._holder(chunked_req=broken, chunked_before=broken, mb_id=1)
        h.mbs[1] = _StubBatch([broken])
        h._pp_admission_pass_voided = True

        h._pp_void_own_batch(1)

        self.assertIsNone(
            h.chunked_req,
            "extend_range was already None and unrepairable -- carrying "
            "the request forward as self.chunked_req is the instr19 crash",
        )
        # #968b UPDATED, AND THE OLD ASSERTION IS THE THING THAT WAS WRONG.
        # This line used to read `assertNotIn(broken, h.waiting_queue)` with
        # the rationale "it is dropped, not retracted". That rationale is the
        # same sentence the twin site carried beside its own clear ("the
        # request ... is re-admitted from the waiting queue by the ordinary
        # path"), and it is false for a chunked continuation, which is never
        # in the waiting queue by `_park_chunked_prefill_chunk` :798-802.
        # Boot 4 of window-flip-0828 measured the cost: the clear dropped
        # `4077b704`, and the refusals of the following passes reported that
        # same rid MISSING. Clearing the FIELD (asserted above, unchanged,
        # and still the instr19 fix) and DROPPING THE REQUEST are two acts;
        # only the first was ever necessary.
        self.assertIn(
            broken,
            h.waiting_queue,
            "a reset-shape carry that nothing else can reach must be queued, "
            "not discarded: `add_chunked_req` re-admits only from "
            "`self.chunked_req`, which has just been cleared, so a request "
            "dropped here is in none of the four `pp_request_locations` "
            "places for the rest of its life",
        )
        self.assertEqual(
            h.waiting_queue.count(broken),
            1,
            "queued exactly once -- an unconditional append would duplicate "
            "the request the disposal loop already re-queued",
        )
        # THE NEXT PASS'S OWN GUARD (scheduler.py get_next_batch_to_run),
        # run directly against this holder's post-gate state rather than
        # re-implemented: must not raise.
        if h.chunked_req is not None:
            _ = h.chunked_req.extend_range.end


class _StopAtDrain(Exception):
    """Raised by the rigged `_pp_drain_voided_proxy` stub the instant it is
    reached, to stop `_event_loop_pp_body`'s infinite loop exactly where this
    test's assertions are complete -- everything past the drain call (proxy
    send, `_pp_launch_batch`, output processing, the chain flush, phase-flip)
    is untouched by #797d and would need its own stubs for no additional
    coverage."""


class _StoppedAtRecvProxy(Exception):
    """Raised by the rigged `_pp_recv_proxy_tensors` stub. Reaching it means
    `cur_batch` was still truthy when `_event_loop_pp_body` read it -- i.e.
    the #797d gate did not run, did not run in time, or was bypassed. This is
    the exact deadlock site in SPECIMEN_wedge_19-02.txt
    (`_pp_recv_proxy_tensors <- _event_loop_pp_body:1503`), turned from an
    infinite block into an immediate, loud failure."""


class _LoopStubBatch:
    """The non-empty batch `get_next_batch_to_run` hands back despite the
    pass already being voided -- the specimen's own contradiction
    (`effective: {}` but `cur_batch: <ScheduleBatch ...>`), constructed
    directly rather than relying on scheduler.py's own void guard to
    (dis)agree."""

    def __init__(self, reqs=()):
        self.reqs = list(reqs)


class PPVoidOwnBatchWiring797d(unittest.TestCase):
    """WIRING, not the gate's own logic (`PPVoidOwnBatch797d` above already
    covers that in isolation). This class drives the SHIPPED
    `_event_loop_pp_body` itself -- the real bound method, not a
    hand-replicated copy of its sequence -- so it goes RED if the two-line
    call site

        if self._pp_admission_pass_voided:
            self._pp_void_own_batch(mb_id)

    is removed from `_event_loop_pp_body`, or moved to after the
    admission-decision send, or moved to after `cur_batch = self.mbs[mb_id]`
    is read. `PPVoidOwnBatch797d` alone cannot detect any of those three
    regressions: it calls `_pp_void_own_batch` directly and would stay green
    even if `_event_loop_pp_body` never called it at all.

    THE STOPPING TRICK. Fully driving one pass of `_event_loop_pp_body` to
    completion would require stubbing `_pp_launch_batch`, real forward
    compute, output post-processing, the request-chain flush and phase-flip
    -- none of which #797d touches. Instead, `_pp_recv_proxy_tensors` and
    `_pp_drain_voided_proxy` are rigged to raise the instant either is
    reached, which is also exactly the branch point the gate decides between
    (`if cur_batch: ... recv ... else: ... drain ...`) -- so the raised
    exception's TYPE is itself the assertion: `_StopAtDrain` means the gate
    ran and cleared the slot before `cur_batch` was read; `_StoppedAtRecvProxy`
    means it did not, and the loop took the branch that would have blocked
    forever on a proxy nobody sends (the specimen's own deadlock, made loud
    instead of silent).

    VERIFIED TO GO RED: with the call site in `_event_loop_pp_body` changed
    to ``if False and self._pp_admission_pass_voided:`` (disabling it while
    leaving `_pp_void_own_batch` itself intact), `test_the_real_event_loop_
    clears_the_slot_and_never_touches_the_proxy_receive` fails with
    `_StoppedAtRecvProxy` instead of the expected `_StopAtDrain`; see the
    task report for the captured output. Restoring the call site returns it
    to green.
    """

    def _holder(self):
        from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin

        calls = {"recv_proxy": 0, "drain": 0, "sent_decisions": []}
        stub_batch = _LoopStubBatch(reqs=())

        h = types.SimpleNamespace(
            ps=types.SimpleNamespace(pp_rank=1, pp_size=WORLD),
            pp_group=types.SimpleNamespace(is_first_rank=False, is_last_rank=False),
            pp_loop_size=WORLD,
            running_mbs=[None] * WORLD,
            last_mbs=[None] * WORLD,
            mbs=[None] * WORLD,
            mb_metadata=[None] * WORLD,
            chunked_req=None,
            request_receiver=types.SimpleNamespace(recv_requests=lambda: []),
            server_args=types.SimpleNamespace(
                pp_async_batch_depth=0, enable_phase_flip=False
            ),
            _pp_output_expected_incoming=False,
            _pp_admission_pass_voided=False,
        )

        h._pp_flip_pass_tick = lambda mb_id: None
        h._pp_forward_and_process_input_requests = lambda recv_reqs: None
        h._pp_recv_admission_decision = lambda: object()
        h._pp_reconcile_incoming_admission = lambda incoming: ({}, object())
        # `_pp_forwarded_schedule_from` is a real mixin method that expects a
        # real `PPAdmissionDecision`'s `.entries`; the `amended` this rig
        # forwards is a bare `object()` (its content is irrelevant to
        # #797d), so this is stubbed rather than bound.
        h._pp_forwarded_schedule_from = lambda amended: None

        def _void_retracted_pass(effective, amended):
            # THE PRECONDITION #797d ACTS ON: the pass is already decided
            # voided before `get_next_batch_to_run` runs, exactly as
            # `_pp_void_retracted_pass` (the real one) leaves it.
            h._pp_admission_pass_voided = True
            return {}, amended

        h._pp_void_retracted_pass = _void_retracted_pass

        class _Plan:
            def __init__(self, batch_to_run, running_batch):
                self.batch_to_run = batch_to_run
                self.running_batch = running_batch

        def _get_next_batch_to_run(running_batch=None, last_batch=None):
            # THE CONTRADICTION ITSELF: void already decided (True, set by
            # the stub above), yet a non-empty batch comes back anyway.
            return _Plan(batch_to_run=stub_batch, running_batch=running_batch)

        h.get_next_batch_to_run = _get_next_batch_to_run

        def _recv_proxy_tensors(mb_id):
            calls["recv_proxy"] += 1
            raise _StoppedAtRecvProxy(
                f"cur_batch was truthy for mb_id={mb_id} on a voided pass -- "
                "the #797d gate did not clear self.mbs[mb_id] in time"
            )

        h._pp_recv_proxy_tensors = _recv_proxy_tensors

        def _drain_voided_proxy(mb_id):
            calls["drain"] += 1
            raise _StopAtDrain()

        h._pp_drain_voided_proxy = _drain_voided_proxy

        def _send_admission_decision(
            amended,
            expects_output=None,
            pass_voided=None,
            launched=None,
            # #978 HARNESS REPAIR (interface drift, no assertion touched):
            # the event loop now also forwards the per-generation launched
            # chain; this recorder ignores it exactly as it ignores
            # `expects_output`.
            launched_chain=(),
        ):
            calls["sent_decisions"].append(
                {"pass_voided": pass_voided, "launched": launched}
            )

        h._pp_send_admission_decision = _send_admission_decision

        # THE METHODS UNDER TEST, bound for real -- not stubbed, not
        # reimplemented. If `_event_loop_pp_body` stops calling
        # `_pp_void_own_batch` at the #797d call site, nothing in this rig
        # papers over that; the real bound methods are the only things that
        # can clear `h.mbs[0]` or record a call.
        # #798 added a second void gate at this same call site, immediately
        # after `_pp_void_own_batch`. It is bound FOR REAL here rather than
        # stubbed away, on this rig's own stated principle: a stub would let a
        # future change to that gate silently alter what this test observes.
        # It must be inert in this rig, and that is itself worth pinning --
        # the #797d void has already cleared `h.mbs[mb_id]` by the time it
        # runs, and its first act is to return False on an empty slot. If it
        # ever stops being inert here, the `_StopAtDrain` expectation below
        # changes and this test says so.
        for name in (
            "_event_loop_pp_body",
            "_pp_void_own_batch",
            "_pp_void_pass_without_upstream_launch",
            "_pp_note_output_expectation",
            "_pp_note_chunked_req_before_admission",
        ):
            setattr(h, name, types.MethodType(getattr(SchedulerPPMixin, name), h))

        return h, calls

    def test_the_real_event_loop_clears_the_slot_and_never_touches_the_proxy_receive(
        self,
    ):
        h, calls = self._holder()

        with self.assertRaises(_StopAtDrain):
            h._event_loop_pp_body()

        # The drain branch ran, not the receive branch -- the receive is
        # rigged to raise a DIFFERENT exception type, so reaching it instead
        # would have failed this `assertRaises` with `_StoppedAtRecvProxy`
        # rather than silently passing.
        self.assertEqual(calls["drain"], 1)
        self.assertEqual(calls["recv_proxy"], 0)
        # The gate reached `self.mbs[0]` (mb_id starts at 0) before the
        # pass's own admission-decision send and before the drain/receive
        # branch read it.
        self.assertIsNone(h.mbs[0])
        self.assertEqual(len(calls["sent_decisions"]), 1)
        self.assertFalse(
            calls["sent_decisions"][0]["launched"],
            "the forwarded decision must carry launched=False for a voided "
            "slot the gate cleared",
        )
        self.assertTrue(calls["sent_decisions"][0]["pass_voided"])


if __name__ == "__main__":
    unittest.main()
