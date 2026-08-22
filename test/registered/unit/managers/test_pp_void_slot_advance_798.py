"""#798: the proxy channel's one-message-per-pass debt across a void.

WHAT THIS FILE PINS, and why it is not the file the register asked for.

The #798 register recorded a first reading of two metal specimens: that a
voided pass is "pure spin" in the sense `_pp_flip_hold_slot`'s docstring
(scheduler_pp_mixin.py:2499-2509) gives that phrase, so a voiding rank runs
extra slot iterations and its `mb_id` runs AHEAD of its peers'. Two
measurements taken while building this file contradict that reading, and the
tests below are written to let either reading fail:

  (1) DIRECTION. In both specimens the voiding rank is BEHIND, not ahead.
      Specimen 2 (boot_seed797b, 22:52:20Z): PP1 sits on `mb_id=0` and the
      arriving proxy is stamped `mb_id=1`. A rank whose slot index had run
      ahead would read a stamp BELOW its own index. Reading one ABOVE it
      means the opposite: this rank has consumed more of the proxy channel
      than its own slot progression accounts for.

  (2) PACING. A voided pass keeps every per-hop rendezvous that paces the
      loop -- the chain receive (:1263), the admission-decision receive
      (:1299), the decision-send reap (:1607) and the chain flush (:1609)
      all run unconditionally on a voided pass. Only the proxy receive
      (:1522) and `_pp_launch_batch` (:1542) are skipped. So a void makes a
      pass CHEAPER, not unpaced, and cannot by itself desynchronise the slot
      ring. `test_genuine_void_keeps_the_channel_paired` is the falsifier
      for that claim and is expected to pass BEFORE any fix.

THE DEFECT THIS FILE ACTUALLY REPRODUCES. Whether a proxy message exists for
a given pass is decided independently on the two ends of the wire:

    sender    (scheduler_pp_mixin.py:1567-1580)  sends iff `self.mbs[mb_id]`
    receiver  (scheduler_pp_mixin.py:1518-1529)  receives iff `self.mbs[mb_id]`,
                                                 else drains iff voided AND
                                                 `_pp_upstream_launched_incoming`

Three of the four combinations are reconciled. `_pp_drain_voided_proxy`
(:4610) was given an explicit gate on the sender's own statement, and its
docstring states the rule this file is named after: "GATED ON THE SENDER'S
OWN STATEMENT, never on an inference from this rank's state ... 'did my
upstream launch' is not derivable here: its batch can be empty for capacity
reasons that never appear in the decision, and a blocking receive for a
message nobody sent is the deadlock family this whole feature is a list of."

The RECEIVE branch never got that gate. `_pp_upstream_launched_incoming` is
read in exactly one place in the whole module -- :4636, inside the drain. So
the fourth combination, THIS RANK HAS A BATCH AND ITS UPSTREAM DID NOT
LAUNCH, enters a blocking receive for a message that was never posted,
consumes the NEXT pass's proxy instead, and every receive after it is
permanently one message ahead. The first one whose stamp is compared then
reads exactly `+1` slot in the same flip epoch, which is both specimens'
invariant.

WHY THAT COMBINATION OCCURS AFTER A RETRACTION, which is what ties this to
#798's specimens rather than making it a theoretical fourth case. The void
is deliberately asymmetric in what it preserves: `_pp_void_own_batch`
(:4701-4710) does NOT touch `running_mbs`, because "the pass simply did not
run, and it decodes again next pass from the state it still holds", while
PP0's `#791b` void output RELEASES and re-queues rank 0's requests
(`_pp_absorb_void_output`; both specimens log "1 of rank 0's 2 request(s)
have been released and re-queued"). So immediately after a void the upstream
can have nothing to launch for a slot on which a downstream rank still holds
resident work -- the exact fourth combination, arriving one to two passes
after the void, which is where both specimens died.

METHOD. Three real gloo processes, spawned, no stubs standing in for
anything that decides message flow. The chain, decision and proxy channels
are all driven through the SHIPPED methods bound onto a holder, following
the harness established by test_pp_admission_chain_flush_deadlock_795.py and
test_pp_chain_flush_deadlock_788.py. What is modelled rather than shipped is
BATCH DERIVATION only -- `get_next_batch_to_run` cannot be run without a
real Scheduler -- and it is modelled as the shipped code documents it:
`self.mbs[mb_id]` non-empty from local continuation independently of the
admission decision (`_pp_void_own_batch`'s docstring, :4661-4672), and
cleared on a void (:4722). Every predicate that decides whether a message is
sent, received, drained or refused is the shipped one.
"""

import json
import os
import pickle
import tempfile
import time
import types
import unittest
from collections import deque

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD = 3
PP0, PP1, PP2 = 0, 1, 2
LAST = WORLD - 1

#: The microbatch slot ring. Production derives this from pp_size /
#: pp_max_micro_batch_size; the defect is ring-size independent, and three
#: slots is the rig's own value.
RING = 3

#: Passes to run. Enough to lap the ring twice after the scripted event, so
#: a standing one-message offset has room to be detected by the stamp rather
#: than merely to exist.
N_PASSES = 9

#: The pass on which the scripted event happens (1-based, so slot
#: `(p - 1) % RING`). Chosen mid-run: past the first lap, with two full laps
#: left afterwards.
EVENT_PASS = 5

#: Same payload-size reasoning as test_pp_admission_chain_flush_deadlock_795.py:
#: large enough that this environment's gloo backend genuinely blocks rather
#: than completing the send for free.
PAYLOAD = [b"x" * 65536]

#: Rows in the modelled hidden-states tensor, so `_pp_proxy_stamp` records a
#: real row count rather than its -1 fallback.
PROXY_ROWS = 8

GREEN_JOIN_TIMEOUT_S = 60.0
RED_JOIN_TIMEOUT_S = 60.0

RID = "rid-798"

#: PP0's offered prefix, and the shorter match a retracting downstream rank
#: reports. `local < told` is what `reconcile_pp_admission_decision`
#: (pp_admission_congruence.py:597-626) turns into a genuine retraction.
TOLD_PREFIX = 4096
LOCAL_PREFIX = 2048
EXTEND_LEN = 512


class _RingWire:
    """Real point-to-point tensor-dict transport over gloo, ring-default.

    Copied from test_pp_admission_chain_flush_deadlock_795.py, including its
    fidelity fix: `async_send=True` does NOT wait, matching
    `GroupCoordinator.send_tensor_dict`, with the handles kept alive in
    `self._inflight` rather than going out of scope.
    """

    def __init__(self, rank: int):
        self.rank = rank
        self.rank_in_group = rank
        self.world_size = WORLD
        self._inflight = []

    def send_tensor_dict(
        self, tensor_dict, dst=None, all_gather_group=None, async_send=False
    ):
        if dst is None:
            dst = (self.rank_in_group + 1) % self.world_size
        buf = pickle.dumps(tensor_dict)
        size = torch.tensor([len(buf)], dtype=torch.long)
        payload = torch.frombuffer(bytearray(buf), dtype=torch.uint8)
        if async_send:
            w1 = dist.isend(size, dst=dst)
            w2 = dist.isend(payload, dst=dst)
            self._inflight.append((w1, size))
            self._inflight.append((w2, payload))
        else:
            dist.send(size, dst=dst)
            dist.send(payload, dst=dst)
        return []

    def recv_tensor_dict(self, src=None, all_gather_group=None):
        if src is None:
            src = (self.rank_in_group - 1) % self.world_size
        size = torch.zeros(1, dtype=torch.long)
        dist.recv(size, src=src)
        buf = torch.zeros(int(size.item()), dtype=torch.uint8)
        dist.recv(buf, src=src)
        return pickle.loads(bytes(buf.numpy()))


def _make_holder(rank: int, wire: _RingWire, chain_group):
    """Bind the SHIPPED chain, decision AND proxy methods onto one holder.

    Extends test_pp_admission_chain_flush_deadlock_795.py's holder with the
    proxy-channel set -- `_pp_recv_proxy_tensors`, `_pp_drain_voided_proxy`,
    `_pp_proxy_stamp` -- and the void decision itself,
    `_pp_void_retracted_pass`. Those four are the functions under test; none
    of them is stubbed.
    """
    from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin

    ps = types.SimpleNamespace(
        pp_rank=rank,
        pp_size=WORLD,
        tp_size=1,
        attn_tp_rank=0,
        attn_cp_rank=0,
        attn_dp_rank=0,
        attn_cp_size=1,
        attn_tp_size=1,
    )
    wire.is_first_rank = rank == PP0
    wire.is_last_rank = rank == LAST

    h = types.SimpleNamespace(
        ps=ps,
        pp_group=wire,
        world_group=types.SimpleNamespace(cpu_group=chain_group),
        send_req_work=[],
        send_proxy_work=[],
        process_input_requests=lambda recv_reqs: None,
        pp_phase_flip_armed=lambda: False,
        pp_flip_service=lambda: None,
        require_attn_tp_allgather=False,
        attn_tp_group=None,
        waiting_queue=[],
        tree_cache=None,
        # The slot ring. `mbs` is what BOTH ends of the proxy wire read to
        # decide whether a message exists for this pass (:1518 receiver,
        # :1567 sender), which is the whole subject of this file.
        mbs=[None] * RING,
        mb_metadata=[None] * RING,
        # #789's readiness gate is a documented no-op when there are no flip
        # counters, which is the pre-flip behaviour this harness models.
        pp_flip_counters=None,
        # #797's per-pass facts, reset per pass in the driver exactly as
        # :1289-1296 resets them.
        _pp_pass_voided_incoming=False,
        _pp_upstream_launched_incoming=False,
        _pp_admission_pass_voided=False,
        _pp_output_expected_incoming=False,
        _pp_admission_send_work=[],
        _pp_admission_amended_to_forward=None,
        _pp_admission_incoming_effective=None,
        _pp_admission_incoming_schedule=None,
        # `_pp_void_own_batch`'s restore idiom reads all three of these; with
        # no chunked request and no resident batch it releases nothing, which
        # is the state this harness models.
        chunked_req=None,
        running_mbs=[None] * RING,
        _pp_chunked_req_before_by_slot=[None] * RING,
        _pp_gapped_wire=False,
    )
    h._pp_flip_bump_sent = lambda chan: None
    h._pp_flip_bump_attempted = lambda chan: None
    h._pp_flip_bump_consumed = lambda chan: None
    h._pp_boundary_stats = lambda: None
    for name in (
        # chain channel
        "_pp_forward_and_process_input_requests",
        "_pp_commit_comm_work",
        "_pp_send_pyobj_to_next_stage",
        "_pp_recv_pyobj_from_prev_stage",
        "_pp_commit_pending_req_work",
        # decision channel
        "_pp_send_admission_decision",
        "_pp_recv_admission_decision",
        "_pp_try_recv_admission_decision",
        "_pp_commit_admission_send_work",
        "_pp_send_dict_to_next_stage",
        "_pp_recv_typed_dict",
        # the functions under test
        "_pp_void_retracted_pass",
        "_pp_void_own_batch",
        "_pp_void_pass_without_upstream_launch",
        "_pp_forwarded_schedule_from",
        "_pp_recv_proxy_tensors",
        "_pp_drain_voided_proxy",
        "_pp_proxy_stamp",
        "_pp_wait_for_proxy_readiness",
    ):
        setattr(h, name, types.MethodType(getattr(SchedulerPPMixin, name), h))
    return h


def _progress(out_dir, rank, msg):
    path = os.path.join(out_dir, f"progress_r{rank}.txt")
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        f.write(msg)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _read_progress(out_dir, rank):
    path = os.path.join(out_dir, f"progress_r{rank}.txt")
    if not os.path.exists(path):
        return "<no progress recorded>"
    with open(path) as f:
        return f.read()


def _proxy_result():
    """A stand-in forward result carrying a real hidden-states tensor, so
    `_pp_proxy_stamp` records a genuine row count."""
    return types.SimpleNamespace(
        pp_hidden_states_proxy_tensors=types.SimpleNamespace(
            tensors={"hidden_states": torch.zeros(PROXY_ROWS, 4)}
        )
    )


def _fresh_decision(mb_id):
    from sglang.srt.managers.pp_admission_congruence import (
        PPAdmissionDecision,
        PPAdmissionEntry,
    )

    return PPAdmissionDecision(
        mb_id=mb_id,
        entries=(
            PPAdmissionEntry(
                rid=RID,
                prefix_len=TOLD_PREFIX,
                extend_len=EXTEND_LEN,
                admitted=True,
            ),
        ),
    )


def _worker(rank, init_file, out_dir, variant, n_passes):
    """One rank's per-pass driver, mirroring `_event_loop_pp_body`'s slot
    loop (scheduler_pp_mixin.py:1246-1658) in the order production runs it.

    `variant`:
      "aligned"        -- every rank launches every pass. Control.
      "genuine_void"   -- PP1 retracts on EVENT_PASS through the shipped
                          `reconcile_pp_admission_decision`, so the shipped
                          `_pp_void_retracted_pass` genuinely voids the pass
                          on PP1 and, by forwarding, on PP2. This is the
                          register's stated mechanism, isolated.
      "upstream_idle"  -- PP0 does not launch on EVENT_PASS while the
                          downstream ranks still hold resident work, i.e.
                          the fourth, unreconciled combination.
    """
    from sglang.srt.managers.pp_admission_congruence import (
        PPAdmissionDecision,
        reconcile_pp_admission_decision,
    )

    res = {"rank": rank, "ok": False, "refusals": []}
    try:
        dist.init_process_group(
            "gloo", init_method=f"file://{init_file}", rank=rank, world_size=WORLD
        )
        chain_group = dist.new_group(ranks=list(range(WORLD)))
        wire = _RingWire(rank)
        h = _make_holder(rank, wire, chain_group)
        pending_sends = deque()

        mb_id = 0
        p = 0
        while p < n_passes:
            p += 1
            if mb_id >= RING:
                mb_id = 0

            # ---- chain channel (:1263-1264) -- paces the loop per hop.
            _progress(out_dir, rank, f"pass{p}_slot{mb_id}_chain_recv")
            if rank == PP0:
                recv_reqs = list(PAYLOAD)
            else:
                recv_reqs = h._pp_recv_pyobj_from_prev_stage()
            h._pp_forward_and_process_input_requests(recv_reqs)

            # ---- per-pass resets (:1289-1296), verbatim in order.
            h._pp_output_expected_incoming = False
            h._pp_pass_voided_incoming = False
            h._pp_upstream_launched_incoming = False
            h._pp_admission_pass_voided = False

            # ---- decision channel (:1297-1334) on every non-first rank.
            amended = None
            effective = {}
            if rank != PP0:
                _progress(out_dir, rank, f"pass{p}_slot{mb_id}_decision_recv")
                incoming = h._pp_recv_admission_decision()
                if incoming is None:
                    incoming = PPAdmissionDecision(mb_id=mb_id, entries=())
                # The rank-local reconciliation, SHIPPED. A retraction is
                # produced only by `local < told`, never fabricated.
                retracts = variant == "genuine_void" and rank == PP1 and p == EVENT_PASS
                local_match = {RID: LOCAL_PREFIX if retracts else TOLD_PREFIX}
                effective, amended = reconcile_pp_admission_decision(
                    incoming,
                    local_match,
                    rank=rank,
                    pp_size=WORLD,
                )
                # #797's void decision, SHIPPED. Sets
                # `_pp_admission_pass_voided` and empties both halves.
                effective, amended = h._pp_void_retracted_pass(effective, amended)

            # ---- batch derivation. THE ONLY MODELLED STEP, and modelled as
            # the shipped code documents it. `_pp_void_own_batch`'s docstring
            # (:4661-4672) states that `get_next_batch_to_run`'s local
            # continuation -- chunked_req and the resident running batch --
            # can leave `self.mbs[mb_id]` non-empty independently of the
            # admission decision. So resident work is the default here, and
            # only the two documented clearings apply: the void (:4722) and,
            # for PP0 in the "upstream_idle" variant, having nothing to run.
            if rank == PP0:
                has_batch = not (
                    variant.startswith("upstream_idle") and p == EVENT_PASS
                )
            else:
                has_batch = True
            if h._pp_admission_pass_voided:
                has_batch = False
            h.mbs[mb_id] = _proxy_result() if has_batch else None

            # ---- #798's guard (:1368-1386 call site), SHIPPED, at exactly
            # its production position: after `_pp_void_own_batch`, strictly
            # before the decision is forwarded, so a void taken here is
            # already reflected in the `launched=` this rank sends. The
            # "unguarded" variant omits the call and nothing else -- that is
            # the pre-fix loop, kept so the proof can still fail.
            h._pp_admission_amended_to_forward = amended
            if variant != "upstream_idle_unguarded":
                h._pp_void_pass_without_upstream_launch(mb_id)
                amended = h._pp_admission_amended_to_forward

            # ---- forward this pass's decision (:1371-1516), strictly after
            # the batch is known and strictly before the proxy exchange.
            _progress(out_dir, rank, f"pass{p}_slot{mb_id}_decision_send")
            if rank == PP0:
                decision = _fresh_decision(mb_id)
                h._pp_send_admission_decision(
                    decision,
                    expects_output=h.mbs[mb_id] is not None,
                    pass_voided=False,
                    launched=h.mbs[mb_id] is not None,
                )
                pending_sends.append(p)
            else:
                fwd = (
                    amended
                    if amended is not None
                    else PPAdmissionDecision(mb_id=mb_id, entries=())
                )
                h._pp_send_admission_decision(
                    fwd,
                    expects_output=h._pp_output_expected_incoming,
                    pass_voided=h._pp_admission_pass_voided,
                    launched=h.mbs[mb_id] is not None,
                )

            # ---- proxy channel (:1518-1540), in production's order:
            # receive-or-drain, then commit the PREVIOUS pass's send.
            cur_batch = h.mbs[mb_id]
            if cur_batch:
                _progress(out_dir, rank, f"pass{p}_slot{mb_id}_proxy_recv")
                h._pp_recv_proxy_tensors(mb_id)
            else:
                _progress(out_dir, rank, f"pass{p}_slot{mb_id}_proxy_drain")
                h._pp_drain_voided_proxy(mb_id)

            _progress(out_dir, rank, f"pass{p}_slot{mb_id}_proxy_commit")
            h._pp_commit_comm_work(h.send_proxy_work)

            # ---- proxy send (:1567-1580), after the launch, non-last only.
            if rank != LAST and cur_batch:
                _progress(out_dir, rank, f"pass{p}_slot{mb_id}_proxy_send")
                h.send_proxy_work = h._pp_send_dict_to_next_stage(
                    cur_batch.pp_hidden_states_proxy_tensors.tensors,
                    async_send=True,
                    msg_type="proxy",
                    stamp=h._pp_proxy_stamp(mb_id, cur_batch),
                )

            # ---- end-of-iteration commits (:1607-1609).
            h._pp_commit_admission_send_work()
            if rank != LAST:
                _progress(out_dir, rank, f"pass{p}_slot{mb_id}_chain_flush")
                h._pp_commit_pending_req_work()

            mb_id += 1

        res["ok"] = True
    except BaseException as exc:  # noqa: BLE001 - the error IS the result here
        res["error"] = f"{type(exc).__name__}: {exc}"[:1200]
    finally:
        res["voids"] = getattr(h, "_pp_pass_voids", 0) if "h" in dir() else 0
        res["drains"] = getattr(h, "_pp_voided_proxy_drains", 0) if "h" in dir() else 0
        with open(os.path.join(out_dir, f"result_r{rank}.json"), "w") as f:
            json.dump(res, f)
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except BaseException:  # noqa: BLE001 - best-effort teardown only
                pass


def _run(variant, n_passes=N_PASSES, join_timeout=GREEN_JOIN_TIMEOUT_S):
    ctx = mp.get_context("spawn")
    with tempfile.TemporaryDirectory() as tmp:
        init_file = os.path.join(tmp, "pg_init")
        procs = [
            ctx.Process(target=_worker, args=(r, init_file, tmp, variant, n_passes))
            for r in range(WORLD)
        ]
        for proc in procs:
            proc.start()
        deadline = time.time() + join_timeout
        for proc in procs:
            proc.join(timeout=max(0.1, deadline - time.time()))
        stuck_ranks = [r for r, proc in enumerate(procs) if proc.is_alive()]
        stall_report = {r: _read_progress(tmp, r) for r in stuck_ranks}
        for proc in procs:
            if proc.is_alive():
                proc.terminate()
        for proc in procs:
            proc.join(timeout=5)

        def _load(path, default):
            if not os.path.exists(path):
                return default
            with open(path) as f:
                return json.load(f)

        out = {"stuck_ranks": stuck_ranks, "stall_report": stall_report}
        for r in range(WORLD):
            out[f"result_{r}"] = _load(os.path.join(tmp, f"result_r{r}.json"), None)
        return out


class PPVoidSlotAdvance798(unittest.TestCase):
    def test_aligned_passes_complete(self):
        """CONTROL. With both ends of the proxy wire launching every pass,
        the channel stays paired for two full laps of the slot ring."""
        out = _run("aligned")
        self.assertEqual(out["stuck_ranks"], [], f"ranks stalled: {out}")
        for r in range(WORLD):
            self.assertIsNotNone(out[f"result_{r}"], f"no result from rank {r}: {out}")
            self.assertTrue(
                out[f"result_{r}"]["ok"],
                f"rank {r} did not complete: {out[f'result_{r}']}",
            )

    def test_genuine_void_keeps_the_channel_paired(self):
        """THE FALSIFIER for #798's first reading.

        A genuine retraction on PP1 -- produced by the shipped
        `reconcile_pp_admission_decision` from a short local match, voided by
        the shipped `_pp_void_retracted_pass` -- and NOTHING ELSE. If a void
        by itself desynchronised the slot ring, as the register's first
        reading held, this would refuse a proxy. It does not: the retracting
        rank drains the message its upstream had already posted, and the rank
        below it neither receives nor drains because its own upstream
        truthfully reported `launched=False`. The debt balances on every hop.

        This test passing is not the absence of a defect. It is the evidence
        that the defect is NOT "a void advances the slot", and it is what
        stops a fix being aimed at the slot cursor.
        """
        out = _run("genuine_void")
        self.assertEqual(out["stuck_ranks"], [], f"ranks stalled: {out}")
        for r in range(WORLD):
            self.assertIsNotNone(out[f"result_{r}"], f"no result from rank {r}: {out}")
            self.assertTrue(
                out[f"result_{r}"]["ok"],
                f"rank {r} did not complete after a genuine void: {out[f'result_{r}']}",
            )
        self.assertEqual(
            out["result_1"]["voids"],
            1,
            f"PP1 was supposed to void exactly one pass: {out['result_1']}",
        )
        self.assertEqual(
            out["result_1"]["drains"],
            1,
            f"PP1 was supposed to drain exactly one orphaned proxy: {out['result_1']}",
        )

    def test_upstream_idle_pass_stays_paired(self):
        """THE FIX. The same upstream-idle pass, with the shipped guard in.

        `_pp_void_pass_without_upstream_launch` runs the pass nowhere on PP1
        instead of receiving a proxy nobody posted, and forwards the void so
        PP2 takes the same decision off the same per-hop fact. The channel
        stays paired and all three ranks finish both remaining laps of the
        ring.
        """
        out = _run("upstream_idle")
        self.assertEqual(out["stuck_ranks"], [], f"ranks stalled: {out}")
        for r in range(WORLD):
            self.assertIsNotNone(out[f"result_{r}"], f"no result from rank {r}: {out}")
            self.assertTrue(
                out[f"result_{r}"]["ok"],
                f"rank {r} did not complete with the #798 guard in place: "
                f"{out[f'result_{r}']}",
            )

    def test_unguarded_loop_desyncs_the_proxy_channel(self):
        """THE CAN-FAIL PROOF: the same run with the guard call removed.

        Without this, the test above proves nothing -- a channel that never
        desynced would pass it just as well. Removing ONLY the
        `_pp_void_pass_without_upstream_launch` call reproduces the metal
        specimens exactly, including their invariant.

        THE SPECIMEN, reproduced: upstream did not launch, this rank did.

        PP0 runs nothing for one slot -- the state both specimens are in one
        pass after a void, because `#791b`'s void output released and
        re-queued its requests while the downstream rank's resident work was
        deliberately preserved (`_pp_void_own_batch`, :4701-4710). PP0
        therefore posts no proxy for that slot and says so on the decision
        message (`launched=False`, :1445). PP1 still holds resident work, so
        it takes the `if cur_batch:` branch at :1520 and enters
        `_pp_recv_proxy_tensors` -- which never consults
        `_pp_upstream_launched_incoming`, the fact its sibling
        `_pp_drain_voided_proxy` was given an explicit gate on.

        PP1 consequently consumes the NEXT pass's proxy, and the stamp names
        a slot exactly one ahead of the slot PP1 is on, in the same flip
        epoch. That is both metal specimens' invariant.
        """
        out = _run("upstream_idle_unguarded", join_timeout=RED_JOIN_TIMEOUT_S)
        r1 = out["result_1"]
        self.assertIsNotNone(r1, f"no result from PP1: {out}")
        self.assertFalse(
            r1["ok"],
            "PP1 completed cleanly -- the upstream-idle pass did NOT desync "
            f"the proxy channel, so this file's reading of #798 is wrong: {out}",
        )
        err = r1.get("error", "")
        self.assertIn(
            "PROXY LEFTOVER REFUSED",
            err,
            f"PP1 failed, but not with the specimens' signature: {err}",
        )
        # The specimens' exact invariant: stamp slot is the receiver's slot
        # plus one, same epoch. EVENT_PASS is 1-based, so PP1 is on slot
        # (EVENT_PASS - 1) % RING and reads the following slot's stamp.
        expected_on = (EVENT_PASS - 1) % RING
        expected_stamp = (expected_on + 1) % RING
        self.assertIn(
            f"mb_id={expected_stamp}",
            err,
            f"expected a stamp naming slot {expected_stamp}: {err}",
        )
        self.assertIn(
            f"this rank is on mb_id={expected_on}",
            err,
            f"expected PP1 to be on slot {expected_on}: {err}",
        )


if __name__ == "__main__":
    unittest.main()
