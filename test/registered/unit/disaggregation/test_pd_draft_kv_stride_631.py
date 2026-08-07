# SPDX-License-Identifier: Apache-2.0
"""The draft KV pool's treatment at the PD transfer boundary (#631 / #646).

WHAT THIS FILE WAS, AND WHAT IT IS NOW
--------------------------------------
It started as a falsifier for DESIGN_631b section 0a, which settled "is the
draft pool's treatment at the transfer boundary genuinely identical to the
target's?" BY CONSTRUCTION: under ``--draft-kv-layout dcp`` the draft pool
carries ``get_total_num_kv_heads()`` exactly as the target does, so per-rank
item lengths agree. That reasoning is correct about HEAD COUNTS and says
nothing about the two places the boundary actually reads. Run against the
unmodified code it found a real defect, #646, in the third of those places.

Measured on the pre-fix code, 48 target layers over three PP stages with a
draft pool on both arms: 18 of 34 buffers mispaired per stage, 54 in total,
with NO exception raised -- source ``V0`` written into destination ``K16``.
Control arm, identical geometry with the draft pool removed: 0 of 32 on every
stage. That control is what made the number evidence rather than an artefact
of the oracle, and it is retained below for the same reason.

The three reads, and their state now:

1. ``mooncake/conn.py`` -- the decode arm advertised ONE scalar item length
   for its whole registration, ``kv_args.kv_item_lens[0]``, i.e. target layer
   0. FIXED: it now also announces the full ``kv_item_lens`` list and the size
   of its draft section (wire frames 16/17, appended so older peers are
   unaffected).
2. ``mooncake/conn.py`` -- the only stride guard compared that scalar against
   the prefill arm's ``kv_item_lens[0]``. Also index 0, so a divergence
   confined to the appended draft layers passed it. FIXED: the comparison now
   runs section by section via
   ``CommonKVManager.describe_kv_geometry_mismatch``.
3. ``common/conn.py get_mha_kv_ptrs_with_pp`` -- split the FLAT registration
   list in half to recover K and V pointers. The target pool registers
   ``k_buffer + v_buffer``, so the halves are exact; ``prefill.py`` /
   ``decode.py`` then APPEND the draft pool to that same flat list and the
   half-split has no notion that they are there. FIXED: the split now uses the
   section boundary the registration site declares, and returns the draft pool
   as its own section.

The tests below are therefore regression guards, not falsifiers. Each guard
that can refuse has a CAN-FAIL arm next to it, because a refusal never
observed refusing is not a validated refusal.
"""

import contextlib
import struct
import unittest
from types import SimpleNamespace

from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.disaggregation.mooncake.conn import (
    KVArgsRegisterInfo,
    MooncakeKVReceiver,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


# --------------------------------------------------------------------------
# Labelled registrations: a "pointer" is an int, and LABELS maps it back to a
# human-readable role so a mispairing names what got crossed with what.
# --------------------------------------------------------------------------


class Registration:
    """One arm's flat (ptrs, item_lens) exactly as the disagg code builds it."""

    def __init__(self):
        self.ptrs = []
        self.item_lens = []
        self.labels = {}

    def _add(self, label, item_len):
        ptr = 0x1000 + len(self.ptrs) * 0x100
        # Pointers must be unique across an arm for the oracle to be able to
        # tell buffers apart; the offset above guarantees it.
        self.ptrs.append(ptr)
        self.item_lens.append(item_len)
        self.labels[ptr] = label
        return ptr

    @classmethod
    def build(cls, layer_ids, target_item_len, draft_item_len=None, num_draft=1):
        """``k_buffer + v_buffer``, then the draft pool appended as extra
        layers (``prefill.py`` / ``decode.py``)."""
        r = cls()
        for lid in layer_ids:
            r._add(f"K{lid}", target_item_len)
        for lid in layer_ids:
            r._add(f"V{lid}", target_item_len)
        if draft_item_len is not None:
            for d in range(num_draft):
                r._add(f"Kd{d}", draft_item_len)
            for d in range(num_draft):
                r._add(f"Vd{d}", draft_item_len)
        return r

    @property
    def num_draft_buffers(self):
        return sum(1 for lbl in self.labels.values() if lbl.startswith(("Kd", "Vd")))


def _manager_for(start_layer, end_layer=None, num_target_kv_buffers=None):
    """The real class with no ``__init__``: the split reads ``kv_args`` only."""
    mgr = CommonKVManager.__new__(CommonKVManager)
    mgr.kv_args = SimpleNamespace(
        prefill_start_layer=start_layer,
        prefill_end_layer=end_layer,
        num_target_kv_buffers=num_target_kv_buffers,
    )
    return mgr


def pair_roles(src_reg, dst_reg, prefill_start_layer, num_target_layers=None):
    """Run the REAL split and return the (src_label, dst_label) pairs it forms.

    Mirrors ``_send_kvcache_generic``'s ``layers_params`` construction: K pairs
    take ``item_lens[i]``, V pairs take ``item_lens[stage + i]``, and draft
    pairs take the source index the pairing hands back.
    """
    if num_target_layers is None:
        num_target_layers = sum(
            1 for lbl in src_reg.labels.values() if lbl.startswith("K")
        ) - sum(1 for lbl in src_reg.labels.values() if lbl.startswith("Kd"))
    mgr = _manager_for(
        prefill_start_layer,
        prefill_start_layer + num_target_layers,
        2 * num_target_layers,
    )
    # Deliberately tolerant of the PRE-FIX signature and return shape, so this
    # same file can be run against the base commit and produce a BEHAVIOURAL
    # red (mispaired labels) rather than a TypeError. A regression guard whose
    # failure at the base is only "that argument did not exist" has not shown
    # that the behaviour it guards was ever wrong.
    try:
        out = mgr.get_mha_kv_ptrs_with_pp(
            src_reg.ptrs, dst_reg.ptrs, dst_reg.num_draft_buffers
        )
    except TypeError:
        out = mgr.get_mha_kv_ptrs_with_pp(src_reg.ptrs, dst_reg.ptrs)
    src_k, src_v, dst_k, dst_v, stage = out[0], out[1], out[2], out[3], out[4]
    draft_pairs = out[5] if len(out) > 5 else []
    pairs = []
    for i in range(stage):
        pairs.append(
            (
                src_reg.labels.get(src_k[i], f"<oob {i}>"),
                dst_reg.labels.get(dst_k[i], f"<oob {i}>"),
                src_reg.item_lens[i],
            )
        )
    for i in range(stage):
        pairs.append(
            (
                src_reg.labels.get(src_v[i], f"<oob {i}>"),
                dst_reg.labels.get(dst_v[i], f"<oob {i}>"),
                src_reg.item_lens[stage + i],
            )
        )
    for src_ptr, dst_ptr, src_item_idx in draft_pairs:
        pairs.append(
            (
                src_reg.labels.get(src_ptr, "<oob draft>"),
                dst_reg.labels.get(dst_ptr, "<oob draft>"),
                src_reg.item_lens[src_item_idx],
            )
        )
    return pairs


def split_tolerantly(mgr, src_ptrs, dst_ptrs, dst_num_draft_buffers):
    """Call the split, accepting the pre-fix signature. See ``pair_roles``."""
    try:
        return mgr.get_mha_kv_ptrs_with_pp(src_ptrs, dst_ptrs, dst_num_draft_buffers)
    except TypeError:
        return mgr.get_mha_kv_ptrs_with_pp(src_ptrs, dst_ptrs)


def mispairings(pairs):
    return [(s, d) for s, d, _ in pairs if s != d]


def wrong_item_lens(pairs, expected_by_label):
    return [(s, ln) for s, _d, ln in pairs if ln != expected_by_label(s)]


TARGET_ITEM = 4096
DRAFT_ITEM = 4096

# A 48-layer target model over a 3-stage pipeline.
ALL_LAYERS = list(range(48))
STAGES = [(0, list(range(0, 16))), (16, list(range(16, 32))), (32, list(range(32, 48)))]


class ControlArm(CustomTestCase):
    """The oracle must be able to say PASS, or its failures mean nothing."""

    def test_monolithic_no_draft_pairs_correctly(self):
        src = Registration.build(ALL_LAYERS, TARGET_ITEM)
        dst = Registration.build(ALL_LAYERS, TARGET_ITEM)
        self.assertEqual(mispairings(pair_roles(src, dst, 0)), [])

    def test_pp_prefill_without_draft_pairs_correctly(self):
        """CONTROL: the exact PP geometry of the #646 case, draft removed.

        This is arm 1 of the #631 boot recipe -- PP prefill group, speculation
        still refused. It paired perfectly before the fix and must keep doing
        so after it; it is what licenses reading the draft-pool arm as a real
        defect rather than a broken oracle.
        """
        dst = Registration.build(ALL_LAYERS, TARGET_ITEM)
        for start, layers in STAGES:
            with self.subTest(stage_start=start):
                src = Registration.build(layers, TARGET_ITEM)
                self.assertEqual(mispairings(pair_roles(src, dst, start)), [])

    def test_monolithic_with_draft_on_both_arms_pairs_correctly(self):
        """Both arms carry a draft pool and the flat lists are equal length.

        Before the fix the half-split mislabelled V0 as a K pointer on BOTH
        sides identically, so the pairing survived -- which is why the defect
        was invisible on a non-PP pair. It must still pair after the fix, now
        for the right reason.
        """
        src = Registration.build(ALL_LAYERS, TARGET_ITEM, DRAFT_ITEM)
        dst = Registration.build(ALL_LAYERS, TARGET_ITEM, DRAFT_ITEM)
        self.assertEqual(mispairings(pair_roles(src, dst, 0)), [])


class PipelinedDraftPoolPairsCorrectly(CustomTestCase):
    """REGRESSION GUARD for #646: PP prefill arm that also registers a draft pool.

    W1 of DESIGN_631b REQUIRES this combination -- "the prefill arm must LOAD
    the draft layer's weights even though it never drafts", and ``prefill.py``
    appends the draft pool on every stage whenever layer-sharding is off
    (which PP does not turn on). So the #631 topology walks straight into it.

    Pre-fix measurement: 18 of 34 mispaired on each of the three stages.
    """

    def test_pp_prefill_with_draft_pairs_correctly(self):
        dst = Registration.build(ALL_LAYERS, TARGET_ITEM, DRAFT_ITEM)
        for start, layers in STAGES:
            with self.subTest(stage_start=start):
                src = Registration.build(layers, TARGET_ITEM, DRAFT_ITEM)
                self.assertEqual(mispairings(pair_roles(src, dst, start)), [])

    def test_every_pair_carries_its_own_item_length(self):
        """Right slot is not enough; the copy length must match too.

        A draft pool whose rows are a different size from the target's is the
        case the old ``item_lens[i]`` / ``item_lens[stage + i]`` formula could
        not express, because the draft entries sit past the K/V halves.
        """
        dst = Registration.build(ALL_LAYERS, TARGET_ITEM, DRAFT_ITEM * 3)
        expected = lambda lbl: (
            DRAFT_ITEM * 3 if lbl.startswith(("Kd", "Vd")) else TARGET_ITEM
        )  # noqa: E731
        for start, layers in STAGES:
            with self.subTest(stage_start=start):
                src = Registration.build(layers, TARGET_ITEM, DRAFT_ITEM * 3)
                pairs = pair_roles(src, dst, start)
                self.assertEqual(mispairings(pairs), [])
                self.assertEqual(wrong_item_lens(pairs, expected), [])

    def test_draft_section_is_shipped_once_by_the_last_stage(self):
        """Every stage shipping it would write the same rows pp_size times.

        The draft model sits behind the final target layer, so the last stage
        owns it. On pp=1 the only stage IS the last stage, which is why the
        monolithic path is unchanged.
        """
        dst = Registration.build(ALL_LAYERS, TARGET_ITEM, DRAFT_ITEM)
        shipped = {}
        for start, layers in STAGES:
            src = Registration.build(layers, TARGET_ITEM, DRAFT_ITEM)
            pairs = pair_roles(src, dst, start)
            shipped[start] = sum(1 for s, _d, _l in pairs if s.startswith(("Kd", "Vd")))
        self.assertEqual(shipped, {0: 0, 16: 0, 32: 2})

    def test_a_draft_layer_count_mismatch_refuses(self):
        """CAN-FAIL ARM: the split refuses rather than pairing across sections."""
        src = Registration.build(STAGES[2][1], TARGET_ITEM, DRAFT_ITEM, num_draft=1)
        dst = Registration.build(ALL_LAYERS, TARGET_ITEM, DRAFT_ITEM, num_draft=2)
        mgr = _manager_for(STAGES[2][0], STAGES[2][0] + 16, 32)
        with self.assertRaisesRegex(RuntimeError, "draft KV pool geometry mismatch"):
            mgr.get_mha_kv_ptrs_with_pp(src.ptrs, dst.ptrs, dst.num_draft_buffers)

    def test_a_stage_that_does_not_fit_the_peer_refuses(self):
        """CAN-FAIL ARM: an out-of-range PP stage refuses instead of slicing short."""
        src = Registration.build(list(range(16)), TARGET_ITEM, DRAFT_ITEM)
        dst = Registration.build(list(range(20)), TARGET_ITEM, DRAFT_ITEM)
        mgr = _manager_for(16, 32, 32)
        with self.assertRaisesRegex(RuntimeError, "does not fit inside"):
            mgr.get_mha_kv_ptrs_with_pp(src.ptrs, dst.ptrs, dst.num_draft_buffers)


class NoDraftPathIsByteForByteUnchanged(CustomTestCase):
    """BEHAVIOUR PIN: the production path today has no draft pool anywhere.

    #631a refuses PD together with speculation, so neither arm allocates a
    draft pool and every live transfer goes through the branches below. The
    reference implementation is the pre-fix function copied verbatim; the fix
    is only allowed to differ from it where a draft section exists.
    """

    @staticmethod
    def legacy_reference(start_layer, src_kv_ptrs, dst_kv_ptrs):
        """``get_mha_kv_ptrs_with_pp`` exactly as it stood before #646."""
        num_kv_layers = len(src_kv_ptrs) // 2
        end_layer = start_layer + num_kv_layers
        dst_num_total_layers = len(dst_kv_ptrs) // 2
        src_k_ptrs = src_kv_ptrs[:num_kv_layers]
        src_v_ptrs = src_kv_ptrs[num_kv_layers:]
        if num_kv_layers == dst_num_total_layers:
            dst_k_ptrs = dst_kv_ptrs[:dst_num_total_layers]
            dst_v_ptrs = dst_kv_ptrs[dst_num_total_layers:]
        elif (
            num_kv_layers < dst_num_total_layers
            and dst_num_total_layers % num_kv_layers != 0
        ):
            multiplier_ratio = dst_num_total_layers // num_kv_layers
            dst_k_ptrs = dst_kv_ptrs[start_layer:end_layer]
            v_ptr_offset = num_kv_layers * multiplier_ratio
            dst_v_ptrs = dst_kv_ptrs[
                v_ptr_offset + start_layer : v_ptr_offset + end_layer
            ]
        else:
            dst_k_ptrs = dst_kv_ptrs[start_layer:end_layer]
            dst_v_ptrs = dst_kv_ptrs[
                dst_num_total_layers + start_layer : dst_num_total_layers + end_layer
            ]
        return src_k_ptrs, src_v_ptrs, dst_k_ptrs, dst_v_ptrs, len(src_k_ptrs)

    def test_identical_to_the_pre_fix_function_wherever_no_draft_pool_exists(self):
        # Total layers x PP stage counts, including a stage count that does
        # not divide the total (which is the branch the old elif reached).
        matrix = [(48, 1), (48, 2), (48, 3), (48, 4), (20, 3), (7, 2), (48, 48)]
        for total, stages in matrix:
            per_stage = total // stages
            dst = Registration.build(list(range(total)), TARGET_ITEM)
            for s in range(stages):
                start = s * per_stage
                layers = list(
                    range(start, total if s == stages - 1 else start + per_stage)
                )
                with self.subTest(total=total, stages=stages, start=start):
                    src = Registration.build(layers, TARGET_ITEM)
                    mgr = _manager_for(start, start + len(layers), 2 * len(layers))
                    got = split_tolerantly(mgr, src.ptrs, dst.ptrs, 0)
                    want = self.legacy_reference(start, src.ptrs, dst.ptrs)
                    self.assertEqual(tuple(got)[:5], want)
                    self.assertEqual(list(got[5]) if len(got) > 5 else [], [])

    def test_unannounced_peer_sections_also_take_the_legacy_path(self):
        """A peer predating frames 16/17 announces nothing; nothing changes."""
        dst = Registration.build(ALL_LAYERS, TARGET_ITEM)
        for start, layers in STAGES:
            with self.subTest(stage_start=start):
                src = Registration.build(layers, TARGET_ITEM)
                mgr = _manager_for(start, start + len(layers), 2 * len(layers))
                got = split_tolerantly(mgr, src.ptrs, dst.ptrs, None)
                want = self.legacy_reference(start, src.ptrs, dst.ptrs)
                self.assertEqual(tuple(got)[:5], want)


class GeometryGateComparesRowsNotLayoutNames(CustomTestCase):
    """The redesigned L10 gate (DESIGN_631c section 2).

    L10 as written -- "both arms run the SAME --draft-kv-layout" -- is
    unsatisfiable on the #631 topology: a pp>1/tp=1 prefill group cannot take
    ``dcp`` (the #108 gate wants ``dcp_size == tp_size > 1``) and a
    token-sharded decode group cannot take ``replicated`` (#642). It also
    cannot sit at parse time, because parse time does not see the peer. What
    must agree is the GEOMETRY the layout produces, compared at the handshake.
    """

    @staticmethod
    def manager_with(item_lens, num_target, start_layer=0):
        mgr = CommonKVManager.__new__(CommonKVManager)
        mgr.kv_args = SimpleNamespace(
            kv_item_lens=item_lens,
            num_target_kv_buffers=num_target,
            prefill_start_layer=start_layer,
        )
        return mgr

    def test_matching_geometry_is_accepted_though_nothing_names_a_layout(self):
        """PASS arm. Both arms hold 48 target layers and one draft layer."""
        lens = [TARGET_ITEM] * 96 + [DRAFT_ITEM] * 2
        mgr = self.manager_with(lens, 96)
        self.assertIsNone(mgr.describe_kv_geometry_mismatch(lens, 2))

    def test_draft_only_divergence_is_caught(self):
        """CAN-FAIL ARM, and the exact hole #646 opened.

        Identical target strides, divergent DRAFT strides -- the case the old
        index-0 comparison was structurally unable to see. Not hypothetical:
        DESIGN_631c section 1 established that on the #631 topology the two
        arms are FORCED onto different --draft-kv-layout values, and the
        layout is what decides the draft pool's geometry.
        """
        local = [TARGET_ITEM] * 96 + [DRAFT_ITEM] * 2
        peer = [TARGET_ITEM] * 96 + [DRAFT_ITEM * 3] * 2
        mgr = self.manager_with(local, 96)
        reason = mgr.describe_kv_geometry_mismatch(peer, 2)
        self.assertIsNotNone(reason)
        self.assertIn("draft KV pool row geometry differs", reason)

    def test_a_draft_section_of_a_different_size_is_caught(self):
        local = [TARGET_ITEM] * 96 + [DRAFT_ITEM] * 2
        peer = [TARGET_ITEM] * 96 + [DRAFT_ITEM] * 4
        mgr = self.manager_with(local, 96)
        self.assertIsNotNone(mgr.describe_kv_geometry_mismatch(peer, 4))

    def test_target_divergence_is_caught_beyond_layer_zero(self):
        """CAN-FAIL ARM: the widened target comparison sees a non-zero layer.

        The old guard compared ``kv_item_lens[0]`` on both sides, so a skew
        anywhere else was invisible.
        """
        local = [TARGET_ITEM] * 96
        peer = [TARGET_ITEM] * 96
        peer[17] = TARGET_ITEM * 2
        mgr = self.manager_with(local, 96)
        self.assertIsNone(mgr.describe_kv_geometry_mismatch(peer, 0))
        reason = mgr.describe_kv_geometry_mismatch(peer, 0, compare_target_rows=True)
        self.assertIsNotNone(reason)
        self.assertIn("layer 17", reason)

    def test_layer_zero_divergence_still_caught_the_old_way(self):
        """The legacy scalar channel keeps working for a peer that predates 16/17."""
        local = [TARGET_ITEM] * 96
        mgr = self.manager_with(local, 96)
        self.assertIsNotNone(
            mgr.describe_kv_geometry_mismatch(
                [], None, peer_row_bytes=TARGET_ITEM * 2, compare_target_rows=True
            )
        )
        self.assertIsNone(
            mgr.describe_kv_geometry_mismatch(
                [], None, peer_row_bytes=TARGET_ITEM, compare_target_rows=True
            )
        )

    def test_a_pp_stage_compares_only_the_layers_it_covers(self):
        """PASS arm under PP: unequal list lengths are expected, not a mismatch.

        The prefill stage registers 16 of the peer's 48 target layers. A
        whole-list comparison would refuse this pair; the per-layer one must
        not.
        """
        peer = [TARGET_ITEM] * 96 + [DRAFT_ITEM] * 2
        local = [TARGET_ITEM] * 32 + [DRAFT_ITEM] * 2
        mgr = self.manager_with(local, 32, start_layer=16)
        self.assertIsNone(
            mgr.describe_kv_geometry_mismatch(peer, 2, compare_target_rows=True)
        )

    def test_a_pp_stage_outside_the_peers_range_is_caught(self):
        """CAN-FAIL ARM for the same comparison."""
        peer = [TARGET_ITEM] * 40
        local = [TARGET_ITEM] * 32
        mgr = self.manager_with(local, 32, start_layer=16)
        reason = mgr.describe_kv_geometry_mismatch(peer, 0, compare_target_rows=True)
        self.assertIsNotNone(reason)
        self.assertIn("registered only", reason)

    def test_silence_when_the_peer_announced_nothing(self):
        """Both-sides-present rule, as guard 1 established it.

        Refusing on absence would break a mixed-version rollout over a check
        that cannot be performed anyway.
        """
        local = [TARGET_ITEM] * 96 + [DRAFT_ITEM] * 2
        mgr = self.manager_with(local, 96)
        self.assertIsNone(mgr.describe_kv_geometry_mismatch([], None))


class RegistrationWireCarriesTheSections(CustomTestCase):
    """Frames 16/17 must actually BIND, not merely be declared.

    The decode arm's registration used to carry one scalar item length. The
    two new frames carry the full list and the draft section size; a field
    that is packed but never parsed back would leave the widened guard
    comparing ``None`` forever and looking green.
    """

    @staticmethod
    def base_frames():
        """Frames 0-15 as they stood before #646, with plausible contents."""
        return [
            b"None",  # 0 room
            b"127.0.0.1",  # 1 endpoint
            b"31243",  # 2 dst_port
            b"sess-1",  # 3 mooncake session id
            struct.pack("2Q", 0x1000, 0x2000),  # 4 dst_kv_ptrs
            struct.pack("1Q", 0x3000),  # 5 dst_aux_ptrs
            b"",  # 6 state data ptrs
            b"0",  # 7 dst_tp_rank
            b"1",  # 8 dst_attn_tp_size
            b"4096",  # 9 dst_kv_item_len (scalar, target layer 0)
            b"",  # 10 state item lens
            b"",  # 11 state dim per tensor
            b"",  # 12 state dim offsets
            b"",  # 13 staging base ptr
            b"",  # 14 staging total size
            b"",  # 15 state conv segments
        ]

    def test_frames_16_and_17_round_trip(self):
        msg = self.base_frames() + [
            struct.pack("4Q", 4096, 4096, 8192, 8192),  # 16 full item lens
            b"2",  # 17 draft buffer count
        ]
        info = KVArgsRegisterInfo.from_zmq(msg)
        self.assertEqual(info.dst_kv_item_lens, [4096, 4096, 8192, 8192])
        self.assertEqual(info.dst_num_draft_buffers, 2)
        # The pre-existing scalar must not have moved.
        self.assertEqual(info.dst_kv_item_len, 4096)

    def test_a_peer_predating_the_frames_parses_to_the_absent_defaults(self):
        """Mixed-version rollout: 16 frames, and the guard stays silent."""
        info = KVArgsRegisterInfo.from_zmq(self.base_frames())
        self.assertEqual(info.dst_kv_item_lens, [])
        self.assertIsNone(info.dst_num_draft_buffers)
        self.assertIsNone(info.geometry_error)

    def test_the_real_sender_emits_the_frames_the_parser_reads(self):
        """Both ends of the wire, executed -- not one end plus a reading.

        The parser tests above would stay green if the decode arm never
        actually appended frames 16/17, because they hand-build the message.
        This one runs the REAL ``MooncakeKVReceiver._register_kv_args`` against
        a captured socket and feeds what it emitted straight into the parser.
        """
        captured = []

        class FakeSock:
            def send_multipart(self, frames):
                captured.append(frames)

        rcv = MooncakeKVReceiver.__new__(MooncakeKVReceiver)
        rcv.bootstrap_infos = [{"is_dummy": False}]
        rcv.session_id = "sess-1"
        rcv.kv_mgr = SimpleNamespace(
            kv_args=SimpleNamespace(
                kv_data_ptrs=[0x1000, 0x2000, 0x3000, 0x4000],
                aux_data_ptrs=[0x9000],
                state_data_ptrs=[],
                state_item_lens=[],
                engine_rank=0,
                # 1 target layer (K+V) plus 1 draft layer (K+V).
                kv_item_lens=[4096, 4096, 8192, 8192],
                num_target_kv_buffers=2,
                num_draft_kv_buffers=2,
            ),
            attn_tp_size=1,
            local_ip="127.0.0.1",
            rank_port=31243,
            enable_staging=False,
        )
        rcv._connect_to_bootstrap_server = lambda _info: (
            FakeSock(),
            contextlib.nullcontext(),
        )

        rcv._register_kv_args()

        self.assertEqual(len(captured), 1)
        info = KVArgsRegisterInfo.from_zmq(captured[0])
        self.assertEqual(info.dst_kv_item_lens, [4096, 4096, 8192, 8192])
        self.assertEqual(info.dst_num_draft_buffers, 2)
        self.assertEqual(info.dst_kv_item_len, 4096)
        self.assertEqual(info.dst_kv_ptrs, [0x1000, 0x2000, 0x3000, 0x4000])

    def test_the_parsed_sections_drive_the_guard(self):
        """End to end over the wire representation: divergence -> refusal."""
        msg = self.base_frames() + [
            struct.pack("4Q", 4096, 4096, 8192, 8192),
            b"2",
        ]
        info = KVArgsRegisterInfo.from_zmq(msg)
        mgr = CommonKVManager.__new__(CommonKVManager)
        mgr.kv_args = SimpleNamespace(
            kv_item_lens=[4096, 4096, 2048, 2048],
            num_target_kv_buffers=2,
            prefill_start_layer=0,
        )
        reason = mgr.describe_kv_geometry_mismatch(
            info.dst_kv_item_lens, info.dst_num_draft_buffers
        )
        self.assertIsNotNone(reason)
        self.assertIn("draft KV pool row geometry differs", reason)


if __name__ == "__main__":
    unittest.main()
