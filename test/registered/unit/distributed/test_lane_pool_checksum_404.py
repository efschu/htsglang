"""#404: the per-round pool checksum probe, and the proof that it LOCALIZES.

The #404 bracket window falsified the volume story for the pool-axis rollback
(18 arms, 738 rejected candidate rows, a flat dose response) and left the
instrument problem untouched: "the tokens diverged somewhere downstream" names
neither a surface nor a round. The probe added in ``dual_group_lane.py`` hashes
the three surfaces a rejected candidate can leave residue in, after every
committed round. This file is the probe's own calibration.

A detector that has never been shown to miss is not calibrated, and one that
has only been shown to FIRE is not localized. So every planted perturbation
here is checked three ways:

* it is CAUGHT -- the probe reports a mismatch;
* it is caught at the PLANTED SURFACE and the PLANTED ROUND -- not merely
  somewhere, and not merely eventually;
* it is MISSED under two deliberate mis-settings: with the probe off, and with
  the probe pointed at the FREED TAIL instead of the committed prefix
  (``SGLANG_LANE_POOL_CHECKSUM_TAIL``). The second is the sharper of the two,
  because the freed tail is exactly where a careless version of this
  instrument would have looked.

Hermetic: CPU tensors, the two model forwards mocked, and the real
``_propose`` -> ``_verify`` -> ``_rollback_draft`` -> ``_record_pool_checksum``
sequence in between. The mock is faithful in the properties the probe reads:
the verify writes KV rows for every candidate and parks one recurrent state
per draft step, and the commit moves the ACCEPTED step's state into the
persistent buffers -- which is what makes "the committed prefix" a thing that
can be hashed and a thing that can be corrupted.
"""

import json
import os
import tempfile
import unittest

import torch

from sglang.srt.model_executor.dual_group_lane import DualGroupLane
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


HIDDEN = 4
VOCAB = 512
POOL_SLOTS = 512
MAX_POS = 128
LAYERS = 3
KV_HEADS = 2
HEAD_DIM = 4
CONV_WIDTH = 3
SSM_DIM = 5
DRAFT_STEPS = 4

_ENV = (
    "SGLANG_LANE_POOL_CHECKSUM",
    "SGLANG_LANE_POOL_CHECKSUM_TAIL",
    "SGLANG_LANE_POOL_CHECKSUM_PER_POS",
    "SGLANG_LANE_POOL_CHECKSUM_PATH",
    "SGLANG_LANE_MARGIN_PROBE",
    "SGLANG_LANE_SPEC_DEBUG",
    "SGLANG_LANE_SPEC_ROW_ORACLE",
    "SGLANG_LANE_SPEC_TV_MAX_ACCEPT",
    "SGLANG_LANE_SPEC_VERIFY",
)


class _Out:
    def __init__(self, hidden_states, next_token_logits):
        self.hidden_states = hidden_states
        self.next_token_logits = next_token_logits


class _StaticGraphOutputs:
    """One retained buffer per captured shape, overwritten by every replay.

    The same contract ``test_lane_hidden_view_399.py`` pins, reused so the two
    files describe one vehicle rather than two.
    """

    def __init__(self):
        self.hidden = {}
        self.logits = {}
        self.replays = 0

    def replay(self, rows, tokens):
        self.replays += 1
        h = self.hidden.get(rows)
        if h is None:
            h = self.hidden[rows] = torch.zeros(rows, HIDDEN)
        lg = self.logits.get(rows)
        if lg is None:
            lg = self.logits[rows] = torch.zeros(rows, VOCAB)
        lg.zero_()
        for i, tok in enumerate(tokens):
            h[i].fill_(float(tok))
            lg[i, (int(tok) + 1) % VOCAB] = 10.0
        return _Out(h[:rows], lg[:rows])


class _KVPool:
    """A KV cache with the three methods the probe calls.

    Rows are written from the TOKEN that occupies the position, so two jobs
    that committed the same tokens hold byte-identical rows at the same
    committed position however differently the allocator numbered their slots.
    That is the property the cross-job reading of the probe rests on, and
    making the mock have it is what lets the test check the reading rather than
    the mock.
    """

    def __init__(self):
        self.k = [torch.zeros(POOL_SLOTS, KV_HEADS, HEAD_DIM) for _ in range(LAYERS)]
        self.v = [torch.zeros(POOL_SLOTS, KV_HEADS, HEAD_DIM) for _ in range(LAYERS)]
        self.full_attention_layer_id_mapping = {i: i for i in range(LAYERS)}

    def get_key_buffer(self, layer_id):
        return self.k[layer_id]

    def get_value_buffer(self, layer_id):
        return self.v[layer_id]

    def write(self, slots, tokens):
        for slot, tok in zip(slots, tokens):
            for layer in range(LAYERS):
                self.k[layer][int(slot)].fill_(float(tok) + layer)
                self.v[layer][int(slot)].fill_(float(tok) - layer)


def _mamba_cache():
    """A real ``MambaPool.SpeculativeState``, because the lane checks the type.

    ``_verify_state_buffers`` refuses a pool that is not one -- deliberately,
    since a TARGET_VERIFY over a plain state pool would drag the recurrence
    over rejected candidates. Building the real dataclass rather than a
    look-alike keeps this test on the same side of that check as the boot is.
    ``conv`` is a per-layer list and ``temporal`` one tensor; both are indexed
    ``[:, slot]``, which is the layout the probe reads.
    """
    from sglang.srt.mem_cache.memory_pool import MambaPool

    return MambaPool.SpeculativeState(
        conv=[torch.zeros(2, POOL_SLOTS, CONV_WIDTH) for _ in range(LAYERS)],
        temporal=torch.zeros(LAYERS, POOL_SLOTS, SSM_DIM),
        intermediate_ssm=torch.zeros(LAYERS, POOL_SLOTS, DRAFT_STEPS, SSM_DIM),
        intermediate_conv_window=[
            torch.zeros(2, POOL_SLOTS, DRAFT_STEPS, CONV_WIDTH) for _ in range(LAYERS)
        ],
    )


class _MambaPool:
    def __init__(self):
        self.mamba_cache = _mamba_cache()


class _Allocator:
    def __init__(self, kvcache, first=1):
        self.next = first
        self.freed = []
        self._kvcache = kvcache

    def alloc(self, n):
        out = torch.arange(self.next, self.next + n, dtype=torch.int64)
        self.next += n
        return out

    def free(self, slots):
        self.freed.append(slots.clone() if torch.is_tensor(slots) else slots)

    def get_kvcache(self):
        return self._kvcache


class _ReqPool:
    def __init__(self, mamba_pool=None):
        self.req_to_token = torch.zeros(2, MAX_POS, dtype=torch.int32)
        self.mamba_pool = mamba_pool


class _Req:
    def __init__(self, idx, n_cached, mamba_slot=None):
        self.req_pool_idx = idx
        self.decode_batch_idx = n_cached
        self.kv_committed_len = n_cached
        self.kv_allocated_len = n_cached
        self.mamba_pool_idx = mamba_slot


class _Batch:
    def __init__(self, idx, n_cached, pool, allocator, mamba_slot=None):
        self.device = torch.device("cpu")
        self.req_to_token_pool = pool
        self.token_to_kv_pool_allocator = allocator
        self.reqs = [_Req(idx, n_cached, mamba_slot)]
        self.seq_lens = torch.tensor([n_cached], dtype=torch.int64)
        self.seq_lens_cpu = torch.tensor([n_cached], dtype=torch.int64)
        self.orig_seq_lens = torch.tensor([n_cached], dtype=torch.int64)
        self.seq_lens_sum = n_cached
        self.input_ids = None
        self.out_cache_loc = None
        self.spec_info = None
        self.forward_mode = None
        self.capture_hidden_mode = None
        self._idx = idx
        self.last_decode_slot = None

    def prepare_for_decode(self):
        pos = int(self.seq_lens[0].item())
        slot = self.token_to_kv_pool_allocator.alloc(1)
        self.req_to_token_pool.req_to_token[self._idx, pos] = slot.to(torch.int32)[0]
        self.seq_lens.fill_(pos + 1)
        self.seq_lens_cpu.fill_(pos + 1)
        self.orig_seq_lens.fill_(pos + 1)
        self.reqs[0].decode_batch_idx = pos + 1
        self.last_decode_slot = int(slot[0].item())


class _ServerArgs:
    page_size = 1


class _AttnBackend:
    """Only ``update_mamba_state_after_mtp_verify`` matters here.

    The lane calls it with the accepted step index and it moves that step's
    parked state into the persistent conv/ssm buffers -- which is the whole
    reason a rejected candidate's recurrent state is not supposed to survive a
    round, and therefore the reason the state surfaces are worth hashing.
    """

    def __init__(self, harness):
        self.h = harness

    def update_mamba_state_after_mtp_verify(
        self,
        last_correct_step_indices,
        mamba_track_indices,
        mamba_steps_to_track,
        model,
    ):
        step = int(last_correct_step_indices[0].item())
        self.h.commit_state(self.h.parked[step])


class _Runner:
    def __init__(self, pool, allocator, attn_backend):
        self.server_args = _ServerArgs()
        self.req_to_token_pool = pool
        self.token_to_kv_pool_allocator = allocator
        self.attn_backend = attn_backend
        self.decode_cuda_graph_runner = None
        self.model = None
        self.tp_rank = 0


class _Harness:
    """The real lane round over mocked forwards, with real pool surfaces."""

    MAMBA_SLOT = 1

    def __init__(self, first_token=100, n_cached=8, rung=1, slot_base=1):
        self.graph = _StaticGraphOutputs()
        self.kv = _KVPool()
        self.mamba = _MambaPool()
        self.pool = _ReqPool(self.mamba)
        self.draft_pool = _ReqPool()
        self.allocator = _Allocator(self.kv, first=slot_base)
        self.draft_allocator = _Allocator(None, first=slot_base)
        self.rung = rung
        self.parked = {}

        lane = DualGroupLane.__new__(DualGroupLane)
        lane.lane_id = 0
        lane.stream = None
        lane.runner = _Runner(self.pool, self.allocator, _AttnBackend(self))
        lane.draft_runner = object()
        lane.spec_steps = rung
        lane.work_total = {"prefill_tokens": 0, "decode_tokens": 0}
        lane._timed_forward_raw = self._target_forward
        lane._draft_forward = self._head_forward
        lane._timed_forward = self._decode_forward
        lane._last_margin = None
        self.lane = lane

        batch = _Batch(0, n_cached, self.pool, self.allocator, self.MAMBA_SLOT)
        batch_d = _Batch(0, n_cached, self.draft_pool, self.draft_allocator)
        prompt = [first_token - n_cached + i for i in range(n_cached)]
        for pos, tok in enumerate(prompt):
            slot = self.allocator.alloc(1)
            self.pool.req_to_token[0, pos] = slot.to(torch.int32)[0]
            self.kv.write([int(slot[0].item())], [tok])
            self.draft_pool.req_to_token[0, pos] = self.draft_allocator.alloc(1).to(
                torch.int32
            )[0]
        self.committed = list(prompt)
        self.commit_state(self._state_for(self.committed))
        self.job = {
            "input_ids": prompt,
            "_batch": batch,
            "_batch_d": batch_d,
            "_req": batch.reqs[0],
            "_req_pool_idx": 0,
            "_kv_len": n_cached,
            "_next": torch.tensor([first_token], dtype=torch.int64),
            "_rung": rung,
            "output_ids": [],
            "decode_ms": [],
        }
        self.job["_hidden"] = self.graph.replay(
            1, [first_token - 1]
        ).hidden_states.clone()

    # -- the fake model's state ------------------------------------------
    @staticmethod
    def _state_for(tokens):
        """A recurrent state that is a pure function of the committed tokens.

        Pure, so that two jobs standing at the same committed position hold the
        same state -- the cross-job invariant the probe is asked to check.
        """
        acc = 0
        for tok in tokens:
            acc = (acc * 31 + int(tok)) % 9973
        return float(acc)

    def commit_state(self, value):
        cache = self.mamba.mamba_cache
        for layer in range(LAYERS):
            cache.conv[layer][:, self.MAMBA_SLOT].fill_(value + layer)
        cache.temporal[:, self.MAMBA_SLOT].fill_(value)

    # -- the two mocked forwards -----------------------------------------
    def _target_forward(self, batch, capture_mode=None):
        tokens = [int(t) for t in batch.input_ids.tolist()]
        loc = batch.out_cache_loc
        if loc is not None:
            self.kv.write([int(s) for s in loc.tolist()], tokens)
            # One parked state per DRAFT STEP, exactly as the GDN backend does
            # under TARGET_VERIFY: step i is the state after candidates 0..i.
            # The commit then moves step ``n_accept`` -- and only that one --
            # into the persistent buffers, which is the behaviour whose absence
            # would be the round-1 defect.
            for i in range(len(tokens)):
                self.parked[i] = self._state_for(self.committed + tokens[: i + 1])
        return self.graph.replay(len(tokens), tokens), 1.0

    def _decode_forward(self, batch):
        token = int(self.job["_next"][0].item())
        nxt = token + 1
        if batch.last_decode_slot is not None:
            self.kv.write([batch.last_decode_slot], [token])
        self.committed = self.committed + [token]
        self.commit_state(self._state_for(self.committed))
        return torch.tensor([nxt], dtype=torch.int64), 1.0

    def _head_forward(self, batch_d, hidden_states):
        proposal = int(hidden_states[0, 0].item()) + 2
        logits = torch.zeros(1, VOCAB)
        logits[0, proposal % VOCAB] = 10.0
        return _Out(torch.full((1, HIDDEN), float(proposal)), logits)

    # -- the real round ---------------------------------------------------
    def spec_round(self, plant=None):
        """One speculative round; ``plant`` fires where a leak would land.

        Between the commit and the probe: that is the window a rejected
        candidate's residue occupies, so a perturbation planted anywhere else
        is testing something the round does not do.
        """
        job = self.job
        proposals = self.lane._propose(job)
        n_cached = int(job["_kv_len"])
        emitted, n_accept, _ = self.lane._verify(job, proposals)
        self.lane._rollback_draft(job, n_accept)
        job["output_ids"].extend(emitted)
        job["decode_ms"].append(1.0)
        self.committed = self.committed + [
            int(self.pool_token(n_cached + i)) for i in range(len(emitted))
        ]
        if plant is not None:
            plant(self)
        self.lane._record_pool_checksum(job, path="spec", n_accept=n_accept)
        return emitted, n_accept

    def pool_token(self, pos):
        """The token whose KV occupies committed position ``pos``."""
        slot = int(self.pool.req_to_token[0, pos].item())
        return int(self.kv.k[0][slot][0, 0].item())

    def decode_round(self):
        self.lane._decode_step(self.job)

    def records(self):
        return self.job.get("_pool_checksums") or []


def _on(**extra):
    os.environ["SGLANG_LANE_POOL_CHECKSUM"] = "1"
    for key, value in extra.items():
        os.environ[key] = str(value)


class _Base(CustomTestCase):
    def setUp(self):
        for var in _ENV:
            os.environ.pop(var, None)

    def tearDown(self):
        for var in _ENV:
            os.environ.pop(var, None)


class TestTheProbeIsOffUnlessAskedFor(_Base):
    """Default-unchanged, asserted rather than assumed."""

    def test_no_records_and_no_result_field_without_the_env(self):
        h = _Harness()
        for _ in range(4):
            h.spec_round()
        self.assertEqual(h.records(), [])
        self.assertNotIn("_pool_checksums", h.job)

    def test_the_env_produces_one_record_per_committed_round(self):
        _on()
        h = _Harness()
        for _ in range(4):
            h.spec_round()
        recs = h.records()
        self.assertEqual(len(recs), 4)
        self.assertEqual([r["round"] for r in recs], [0, 1, 2, 3])
        for rec in recs:
            for surface in ("map", "kv", "conv", "ssm"):
                self.assertIsInstance(rec[surface], str, surface)
        self.assertEqual({r["region"] for r in recs}, {"committed_prefix"})

    def test_the_committed_length_is_the_join_key_on_both_paths(self):
        _on()
        h = _Harness()
        h.spec_round()
        h.decode_round()
        recs = h.records()
        self.assertEqual([r["path"] for r in recs], ["spec", "decode"])
        self.assertEqual(
            [r["committed_len"] for r in recs],
            [r["kv_len"] for r in recs],
            "the fallback arithmetic and _kv_len disagree on a spec job",
        )
        self.assertLess(recs[0]["committed_len"], recs[1]["committed_len"])

    def test_records_reach_the_jsonl_one_line_per_round(self):
        with tempfile.TemporaryDirectory() as tmp:
            prefix = os.path.join(tmp, "ck")
            _on(SGLANG_LANE_POOL_CHECKSUM_PATH=prefix)
            h = _Harness()
            for _ in range(3):
                h.spec_round()
            path = f"{prefix}.lane0.rank0.jsonl"
            self.assertTrue(os.path.exists(path))
            with open(path) as handle:
                lines = [json.loads(x) for x in handle if x.strip()]
            self.assertEqual([r["round"] for r in lines], [0, 1, 2])
            self.assertEqual(lines, h.records())


class TestTheProbeLocalizesAPlantedKvPerturbation(_Base):
    """One byte of one committed KV row, flipped at a named round."""

    PLANT_ROUND = 2

    def _run(self, plant=True, rounds=5):
        h = _Harness()
        for i in range(rounds):
            h.spec_round()
            if plant and i == self.PLANT_ROUND:
                # A position the lane committed SEVERAL rounds ago -- the shape
                # a residue leak has, not a mis-write of the row just added.
                slot = int(h.pool.req_to_token[0, 1].item())
                h.kv.k[1][slot][0, 0] += 1.0
        return h

    def test_the_planted_flip_is_caught_at_the_next_round_on_the_kv_surface(self):
        _on()
        h = self._run()
        recs = h.records()
        broken = [
            r["round"]
            for r in recs
            if r["kv_stable"] is not None
            and r["kv_stable"] != recs[r["round"] - 1]["kv"]
        ]
        self.assertEqual(
            broken,
            [self.PLANT_ROUND + 1],
            "the append-only KV digest did not name the round after the plant",
        )

    def test_no_other_surface_moves(self):
        """Localisation, not detection: map and state must stay clean."""
        _on()
        h = self._run()
        recs = h.records()
        for rec in recs[1:]:
            self.assertEqual(
                rec["map_stable"],
                recs[rec["round"] - 1]["map"],
                f"the slot mapping moved at round {rec['round']}",
            )

    def test_the_clean_run_breaks_nothing(self):
        _on()
        h = self._run(plant=False)
        recs = h.records()
        for rec in recs[1:]:
            self.assertEqual(rec["kv_stable"], recs[rec["round"] - 1]["kv"])
            self.assertEqual(rec["map_stable"], recs[rec["round"] - 1]["map"])

    def test_can_fail_the_probe_off_misses_it_entirely(self):
        h = self._run()
        self.assertEqual(h.records(), [])

    def test_can_fail_hashing_the_freed_tail_misses_it(self):
        """The mis-setting a careless version of this instrument would ship.

        The rejected candidates' slot pointers are still in the row past
        ``committed_len``; hashing THERE reads a region the plant never
        touched, and the leak walks straight past the probe.
        """
        _on(SGLANG_LANE_POOL_CHECKSUM_TAIL=4)
        h = self._run()
        recs = h.records()
        self.assertEqual({r["region"] for r in recs}, {"freed_tail"})
        self.assertTrue(all(r["kv_stable"] is None for r in recs))
        self.assertTrue(all(r["map_stable"] is None for r in recs))

    def test_the_freed_tail_arm_is_alive_and_not_merely_silent(self):
        """The can-fail arm must be a MISS, not a crash or an empty run.

        The record stream is complete, well formed -- and BYTE-IDENTICAL with
        and without the plant. That is what "the instrument was pointed at the
        wrong region" looks like when it is demonstrated rather than argued.
        """
        _on(SGLANG_LANE_POOL_CHECKSUM_TAIL=4)
        planted = self._run(plant=True)
        clean = self._run(plant=False)
        recs = planted.records()
        self.assertEqual(len(recs), 5)
        self.assertTrue(all(isinstance(r["kv"], str) for r in recs))
        self.assertEqual(
            [r["kv"] for r in recs],
            [r["kv"] for r in clean.records()],
            "the freed-tail arm saw the plant after all",
        )

    def test_the_same_plant_under_the_committed_prefix_is_not_missed(self):
        """The other half of the can-fail: the setting that DOES see it.

        Two runs of one plant differing only in ``..._TAIL`` -- one blind, one
        not. Without this pair the miss above would be indistinguishable from
        an instrument that never sees anything.
        """
        _on()
        planted = self._run(plant=True)
        clean = self._run(plant=False)
        self.assertNotEqual(
            [r["kv"] for r in planted.records()],
            [r["kv"] for r in clean.records()],
        )


class TestTheProbeLocalizesAPlantedStatePerturbation(_Base):
    """One element of the committed conv window, at a named round."""

    PLANT_ROUND = 1

    def _run(self, rounds=4, surface="conv"):
        _on()

        def _plant(h):
            cache = h.mamba.mamba_cache
            if surface == "conv":
                cache.conv[2][0, h.MAMBA_SLOT, 1] += 0.5
            else:
                cache.temporal[1, h.MAMBA_SLOT, 2] += 0.5

        clean = _Harness(slot_base=1)
        dirty = _Harness(slot_base=1)
        for i in range(rounds):
            clean.spec_round()
            dirty.spec_round(plant=_plant if i == self.PLANT_ROUND else None)
        return clean, dirty

    def _first_divergence(self, clean, dirty, surface):
        by_len = {r["committed_len"]: r for r in clean.records()}
        for rec in dirty.records():
            ref = by_len.get(rec["committed_len"])
            if ref is None:
                continue
            if ref[surface] != rec[surface]:
                return rec["round"], rec["committed_len"]
        return None, None

    def test_a_conv_flip_is_caught_at_the_planted_round_on_the_conv_surface(self):
        clean, dirty = self._run(surface="conv")
        round_, _ = self._first_divergence(clean, dirty, "conv")
        self.assertEqual(round_, self.PLANT_ROUND)

    def test_a_conv_flip_does_not_move_the_kv_or_ssm_surfaces(self):
        clean, dirty = self._run(surface="conv")
        self.assertIsNone(self._first_divergence(clean, dirty, "kv")[0])
        self.assertIsNone(self._first_divergence(clean, dirty, "ssm")[0])

    def test_an_ssm_flip_is_caught_on_the_ssm_surface_and_not_on_conv(self):
        clean, dirty = self._run(surface="ssm")
        self.assertEqual(
            self._first_divergence(clean, dirty, "ssm")[0], self.PLANT_ROUND
        )
        self.assertIsNone(self._first_divergence(clean, dirty, "conv")[0])

    def test_can_fail_two_clean_jobs_agree_on_every_shared_position(self):
        """The reference direction: without a plant there is nothing to find.

        Also the cross-job invariant itself -- two independently allocated jobs
        hold different SLOT ids and identical CONTENT, which is why ``kv`` and
        the state surfaces join across jobs and ``map`` does not.
        """
        _on()
        a = _Harness(slot_base=1)
        b = _Harness(slot_base=257)
        for _ in range(4):
            a.spec_round()
            b.spec_round()
        by_len = {r["committed_len"]: r for r in a.records()}
        compared = 0
        for rec in b.records():
            ref = by_len.get(rec["committed_len"])
            if ref is None:
                continue
            compared += 1
            for surface in ("kv", "conv", "ssm"):
                self.assertEqual(ref[surface], rec[surface], surface)
            self.assertNotEqual(
                ref["map"], rec["map"], "two jobs drew the same physical slots"
            )
        self.assertGreaterEqual(compared, 3)


class TestTheSpeculativeAndPlainPathsAgreeOnTheCommittedPrefix(_Base):
    """The comparison the window will actually run, in miniature.

    A K = 1 speculative job and a K = 0 job of the same content must hash the
    same committed prefix at every position they share. This is the assertion
    the GPU window makes against a no-spec reference; making it here first is
    what stops a card boot from being the place a harness bug is discovered.
    """

    def test_a_spec_job_and_a_k0_job_agree_at_every_shared_position(self):
        _on()
        spec = _Harness(slot_base=1)
        for _ in range(5):
            spec.spec_round()
        plain = _Harness(slot_base=257)
        for _ in range(8):
            plain.decode_round()
        by_len = {r["committed_len"]: r for r in plain.records()}
        compared = 0
        for rec in spec.records():
            ref = by_len.get(rec["committed_len"])
            if ref is None:
                continue
            compared += 1
            self.assertEqual(ref["kv"], rec["kv"], f"kv at {rec['committed_len']}")
            self.assertEqual(ref["conv"], rec["conv"])
            self.assertEqual(ref["ssm"], rec["ssm"])
        self.assertGreaterEqual(compared, 3, "the two paths never met at a position")


class TestPerPositionDigestsNarrowTheLocalisationToAToken(_Base):
    def test_the_per_position_list_names_the_position_that_moved(self):
        _on(SGLANG_LANE_POOL_CHECKSUM_PER_POS=1)
        h = _Harness()
        h.spec_round()
        before = list(h.records()[0]["kv_pos"])
        slot = int(h.pool.req_to_token[0, 3].item())
        h.kv.v[0][slot][1, 2] += 1.0
        h.spec_round()
        after = h.records()[1]["kv_pos"]
        moved = [i for i, (x, y) in enumerate(zip(before, after)) if x != y]
        self.assertEqual(moved, [3])

    def test_the_list_is_absent_unless_asked_for(self):
        _on()
        h = _Harness()
        h.spec_round()
        self.assertNotIn("kv_pos", h.records()[0])

    def test_the_aggregate_is_the_digest_of_the_positions_in_logical_order(self):
        """Map-independence as a construction, not as an assurance.

        The aggregate ``kv`` is derived from the per-position digests, which
        are addressed by ``req_to_token[idx, p]``. Pinning the identity here is
        what makes "two jobs that drew different slots hash the same" a thing
        the code cannot stop doing rather than a thing it happens to do.
        """
        import hashlib

        _on(SGLANG_LANE_POOL_CHECKSUM_PER_POS=1)
        h = _Harness()
        h.spec_round()
        rec = h.records()[0]
        want = hashlib.blake2b(
            "|".join(rec["kv_pos"]).encode(), digest_size=16
        ).hexdigest()
        self.assertEqual(rec["kv"], want)


class TestTheNumericCrossJobFingerprints(_Base):
    """The fields that join two jobs on a stack that is not bitwise stable.

    The 2026-08-02 window measured a 100 % byte-dirty floor between two no-spec
    reference jobs. A reading with no tolerance cannot survive that, so the
    probe records a float32 sum and absmax beside every digest and the reader
    compares them against a MEASURED floor. What is checked here is the
    separation the whole scheme rests on: a last-bit difference and a leaked
    row are indistinguishable to the digest and an order of magnitude apart to
    the number.
    """

    def _deviation(self, a, b):
        scale = max(abs(a[0]), abs(b[0]), abs(a[1]), abs(b[1]), 1e-9)
        return max(abs(x - y) for x, y in zip(a, b)) / scale

    def test_each_surface_carries_a_numeric_fingerprint(self):
        _on()
        h = _Harness()
        h.spec_round()
        rec = h.records()[0]
        self.assertEqual(len(rec["kv_num"]), rec["committed_len"])
        self.assertEqual(len(rec["kv_num"][0]), 2)
        self.assertEqual(len(rec["conv_num"]), 2)
        self.assertEqual(len(rec["ssm_num"]), 2)

    def test_two_jobs_with_different_slots_agree_numerically(self):
        _on()
        a = _Harness(slot_base=1)
        b = _Harness(slot_base=257)
        a.spec_round()
        b.spec_round()
        self.assertEqual(a.records()[0]["kv_num"], b.records()[0]["kv_num"])
        self.assertEqual(a.records()[0]["conv_num"], b.records()[0]["conv_num"])

    def test_a_last_bit_difference_breaks_the_digest_and_not_the_number(self):
        """The defect class the cross-job reading was drowning in."""
        _on(SGLANG_LANE_POOL_CHECKSUM_PER_POS=1)
        h = _Harness()
        h.spec_round()
        before = h.records()[0]
        slot = int(h.pool.req_to_token[0, 3].item())
        value = h.kv.k[0][slot][1, 2]
        value += float(value.item()) * 1e-6
        h.spec_round()
        after = h.records()[1]
        self.assertNotEqual(before["kv_pos"][3], after["kv_pos"][3])
        self.assertLess(self._deviation(before["kv_num"][3], after["kv_num"][3]), 1e-3)

    def test_a_leaked_row_moves_the_number_by_an_order_of_magnitude_more(self):
        _on(SGLANG_LANE_POOL_CHECKSUM_PER_POS=1)
        h = _Harness()
        h.spec_round()
        before = h.records()[0]
        slot = int(h.pool.req_to_token[0, 3].item())
        h.kv.k[0][slot].fill_(999.0)
        h.spec_round()
        after = h.records()[1]
        self.assertGreater(
            self._deviation(before["kv_num"][3], after["kv_num"][3]), 0.1
        )


class TestTheRungSchedule(_Base):
    """``spec_steps`` as a LIST: the only shape that is mixed-rung by design.

    A scalar pin is constant for the whole job, so the READ side of the
    ``_kv_len`` advance -- a verify round taking ``n_cached`` right after a
    K = 0 round -- is unreachable by any pinned recipe and was reachable only
    by whatever the adaptive policy happened to decide. A cycled schedule makes
    it a property of the request.
    """

    def _lane(self, rungs, pin):
        from sglang.srt.model_executor.lane_spec_policy import LaneSpecPolicy

        lane = DualGroupLane.__new__(DualGroupLane)
        lane.spec_steps = max(rungs)
        lane.spec_policy = LaneSpecPolicy(rungs)
        job = {"spec_steps": pin, "_rungs": []}
        return lane, job

    def _walk(self, rungs, pin, rounds):
        lane, job = self._lane(rungs, pin)
        out = []
        for _ in range(rounds):
            rung = lane._choose_rung(job)
            job["_rungs"].append(rung)
            out.append(rung)
        return out

    def test_a_list_pin_cycles_through_the_schedule(self):
        self.assertEqual(self._walk((0, 1, 3), [0, 1], 6), [0, 1, 0, 1, 0, 1])

    def test_a_three_element_schedule_cycles_too(self):
        self.assertEqual(self._walk((0, 1, 3), [0, 3, 1], 6), [0, 3, 1, 0, 3, 1])

    def test_a_scalar_pin_is_unchanged(self):
        self.assertEqual(self._walk((0, 1, 3), 3, 4), [3, 3, 3, 3])

    def test_no_pin_is_unchanged(self):
        """The default path: the policy answers, the job says nothing."""
        self.assertEqual(self._walk((1,), None, 3), [1, 1, 1])

    def test_an_empty_schedule_falls_back_to_the_policy(self):
        self.assertEqual(self._walk((1,), [], 3), [1, 1, 1])

    def test_an_off_ladder_rung_in_a_schedule_is_still_honoured(self):
        """Same rule a scalar pin gets: honoured, and the verify falls back to
        eager, which the result row reports. A schedule must not become a
        second, quieter policy."""
        self.assertEqual(self._walk((0, 1), [0, 3], 4), [0, 3, 0, 3])


class TestTheProbeNeverTakesTheLaneDown(_Base):
    def test_a_broken_pool_disables_the_probe_and_the_round_continues(self):
        _on()
        h = _Harness()
        h.spec_round()

        def _boom(layer_id):
            raise RuntimeError("planted pool failure")

        h.kv.get_key_buffer = _boom
        h.spec_round()
        h.spec_round()
        self.assertEqual(len(h.records()), 1, "the probe kept recording after a fault")
        self.assertEqual(len(h.job["output_ids"]), 6, "the lane stopped emitting")


if __name__ == "__main__":
    unittest.main()
