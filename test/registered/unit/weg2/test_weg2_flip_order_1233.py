# SPDX-License-Identifier: Apache-2.0
"""Weg 2 draft KV across the flip (#1233, fix 4): the interleave's PAUSE ORDER
and the carrier bound's POPULATION -- the two defects boot weg2dk4 measured.

weg2dk4 (2026-09-07, tip 53bd804e2e) died in the flip back with
``[torch_memory_saver.cpp] CUresult error: 2 (out of memory) func=cu_mem_create``
on card nvml2 -- group D fatal (W4 Weg2WakeRefused, wake(D, weights_5)).  The
root is a geometry mismatch the tag index hides: the interleave pauses the
source's ``weights_k`` against the destination's ``weights_k``, which is a fair
trade per card ONLY when both sides mean the same card by ``k``.  Group P is
PP (a chunk tag is a layer band on ONE stage's card), group D is TP (every card
holds a shard of every layer), so on P's last-stage card the pauses of
``weights_0..5`` released nothing while D's resumes took ~813 MiB each.

Hermetic: CUDA_VISIBLE_DEVICES="" and no server.
"""

import asyncio
import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers.weg2_memory_saver import chunk_tag_cards, weights_family_tags
from sglang.srt.weg2 import front as front_mod
from sglang.srt.weg2.front import Front, interleave_pause_order
from sglang.srt.weg2 import launcher as launcher_mod
from sglang.srt.weg2.launcher import carrier_bound_from_lines
from sglang.test.test_utils import CustomTestCase

# ---------------------------------------------------------------- MEASURED
#: Boot weg2dk4's own geometry: 64 layers, ``--pp-stage-ratio 32,18,14``, 8
#: chunk tags of 8 layers (launcher log 20:18:27Z "WEG2-WEIGHT-CHUNKS N=8 tags
#: (layers per chunk 8 of 64)"), CUDA ordinal -> NVML index [1, 0, 2]
#: (launcher log 20:19:13Z "budget D ordinal=0 nvml_idx=1 ... 5090").
N_LAYERS = 64
LAYERS_PER_CHUNK = 8
CHUNK_COUNT = 8
NVML_OF_STAGE = [1, 0, 2]

#: The reference checkpoint's layer families, written out rather than read from
#: disk so this file stays hermetic: 64 layers, ``full_attention_interval`` 4,
#: so every 4th layer (index 3, 7, 11, ...) is full attention -- 16 of 64,
#: exactly what ``Qwen3.8-27B-INT8-gdncov-vocabembed/config.json``'s
#: ``text_config.layer_types`` lists.
LAYER_KINDS = [(i % 4) == 3 for i in range(N_LAYERS)]


def launcher_stage_layers(scores=None, attn_scores=None, kinds=None):
    """The vector the launcher hands ``chunk_tag_cards`` as ``stage_layers``.

    #1233 fix 5.  BOTH trees are asked through this one seam so the red/green
    is behavioural rather than an ImportError: before the fix the launcher had
    only ``P_PP_STAGE_RATIO`` and handed THAT vector to the map, with a comment
    calling it "the pipeline layer split"; it is the per-stage capability SCORE
    vector, and the split is whatever ``derive_pp_layer_split`` makes of it.
    """
    from unittest import mock

    if hasattr(launcher_mod, "p_stage_layers"):
        with mock.patch.object(
            launcher_mod, "P_PP_STAGE_RATIO_SCORES",
            tuple(scores if scores is not None else (32, 18, 14)),
        ), mock.patch.object(
            launcher_mod, "P_PP_ATTN_STAGE_RATIO_SCORES",
            tuple(attn_scores if attn_scores is not None else (8, 4, 4)),
        ):
            return list(launcher_mod.p_stage_layers(kinds or LAYER_KINDS))
    return list(scores if scores is not None else launcher_mod.P_PP_STAGE_RATIO)


#: P's real layer split under the flags ``argv_p`` passes.  It equals the score
#: vector for exactly this (checkpoint, 32/18/14, 8/4/4) triple -- which is why
#: the old map was right by coincidence -- so every number below is unchanged.
P_STAGE_LAYERS = launcher_stage_layers()

#: The driver_free instants per card at the start of the epoch-1 interleave,
#: MiB (D.log #1028c BOUND driver_free, 20:24:17Z: TP0=nvml1 14252.8,
#: TP1=nvml0 9291.8, TP2=nvml2 4239.8).
FREE_AT_INTERLEAVE_START = {1: 14252.8, 0: 9291.8, 2: 4239.8}

#: What ONE of D's weight chunk resumes costs on ONE card, MiB.  MEASURED as
#: the steady step of the same series on nvml2: 4511.8 -> 3529.8 -> 2717.8 ->
#: 1903.8 -> 1089.8 -> 277.8, i.e. 982 / 812 / 814 / 814 / 812.
D_CHUNK_MIB = 813.0

#: What ONE layer of the model weighs at FULL width, MiB -- derived from the
#: line above, not chosen: D is TP=3, so its chunk holds 1/3 of each of the 8
#: layers in the band, hence 813 * 3 / 8.  This is what P (PP, full width)
#: gives back when a band of its own stage is paused.
P_LAYER_MIB = D_CHUNK_MIB * 3 / LAYERS_PER_CHUNK


def _p_release_mib(tag_cards, tag, layers_of_tag_on_card):
    return layers_of_tag_on_card.get((tag, tag_cards), 0)


def _layers_per_tag_per_card():
    """{(tag, nvml): layers} for group P -- the same walk chunk_tag_cards does,
    written out independently here so the test does not grade the code under
    test with the code under test."""
    out = {}
    layer = 0
    for stage, n in enumerate(P_STAGE_LAYERS):
        for _ in range(n):
            tag = f"weights_{min(layer // LAYERS_PER_CHUNK, CHUNK_COUNT - 1)}"
            out[(tag, NVML_OF_STAGE[stage])] = out.get((tag, NVML_OF_STAGE[stage]), 0) + 1
            layer += 1
    return out


def simulate_interleave(pause_order, resume_tags):
    """Per-card driver_free through the interleave, MiB.

    Step i pauses the source's ``pause_order[i]`` (releases that band's layers
    on the card(s) that hold them) and resumes the destination's
    ``resume_tags[i]`` (costs D_CHUNK_MIB on EVERY card, because D is TP).
    Returns {nvml: minimum free reached}.  The base weights tag is out of
    scope: only the eight chunk steps were measured.
    """
    per = _layers_per_tag_per_card()
    free = dict(FREE_AT_INTERLEAVE_START)
    low = dict(free)
    chunk_steps = [(s, d) for s, d in zip(pause_order, resume_tags) if s.startswith("weights_")]
    for s_tag, _d_tag in chunk_steps:
        for nvml in free:
            free[nvml] += per.get((s_tag, nvml), 0) * P_LAYER_MIB
        for nvml in free:
            free[nvml] -= D_CHUNK_MIB
            low[nvml] = min(low[nvml], free[nvml])
    return low


class TestChunkTagCards(CustomTestCase):
    def test_pp_bands_land_on_one_stage_and_the_last_chunks_on_the_last_stage(self):
        m = chunk_tag_cards(P_STAGE_LAYERS, LAYERS_PER_CHUNK, CHUNK_COUNT, card_of_stage=NVML_OF_STAGE)
        self.assertEqual(
            m,
            {
                "weights_0": (1,), "weights_1": (1,), "weights_2": (1,), "weights_3": (1,),
                "weights_4": (0,), "weights_5": (0,),
                "weights_6": (0, 2),  # layers 48,49 on stage 1 and 50-55 on stage 2
                "weights_7": (2,),
            },
        )
        # THE DEFECT, stated as a fact about the map: the card that OOM'd holds
        # P bytes in the LAST two tags only, so the first six pauses of the
        # natural order are free of charge there.
        self.assertEqual([t for t, cards in m.items() if 2 in cards], ["weights_6", "weights_7"])
        self.assertEqual(sum(P_STAGE_LAYERS), N_LAYERS)

    def test_no_map_when_chunking_is_off_or_the_group_has_no_layer_split(self):
        self.assertEqual(chunk_tag_cards(P_STAGE_LAYERS, 0, CHUNK_COUNT), {})
        self.assertEqual(chunk_tag_cards(P_STAGE_LAYERS, LAYERS_PER_CHUNK, 0), {})
        self.assertEqual(chunk_tag_cards([], LAYERS_PER_CHUNK, CHUNK_COUNT), {})

    def test_a_layer_beyond_the_family_clamps_to_the_last_chunk(self):
        m = chunk_tag_cards([4], 2, 1)
        self.assertEqual(m, {"weights_0": (0,)})

    def test_card_list_that_does_not_match_the_stages_is_refused(self):
        with self.assertRaises(ValueError):
            chunk_tag_cards(P_STAGE_LAYERS, LAYERS_PER_CHUNK, CHUNK_COUNT, card_of_stage=[0, 1])


class TestTheMapIsBuiltFromTheDerivedSplit(CustomTestCase):
    """#1233 fix 5: the map's ``stage_layers`` must be the DERIVED layer split.

    ``--pp-stage-ratio`` takes per-stage capability SCORES;
    ``server_args._handle_pp_stage_ratio`` feeds them to
    ``derive_pp_layer_split(scores, is_full_attention=kinds,
    attn_scores=...)`` and its hybrid snap decides the real per-stage layer
    counts.  Restating the score vector as the split is a second bookkeeping of
    a value the tree already derives, and it is UNGUARDED by construction:
    ``interleave_pause_order`` refuses only an INCOMPLETE map (missing tag,
    card absent from the NVML sample), so a map that is complete but wrong
    produces a confident ``why="tightest-card-first"`` order against the wrong
    card -- the wrong-per-card-accounting class that killed weg2dk4, now
    wearing an instrument that says it is fine.
    """

    #: MEASURED hermetically (CUDA_VISIBLE_DEVICES="", the checkpoint's own
    #: layer_types): (scores, attn_scores) -> derived split.  The first row is
    #: the boot form, where flag and truth agree; the other three are its
    #: NEIGHBOURS, where they do not.
    CASES = [
        ((32, 18, 14), (8, 4, 4), [32, 18, 14]),
        ((32, 18, 14), (7, 5, 4), [31, 19, 14]),
        ((31, 17, 16), (8, 4, 4), [32, 16, 16]),
        ((30, 20, 14), (8, 4, 4), [32, 18, 14]),
    ]

    def test_the_launcher_hands_the_map_the_derived_split_for_every_neighbour(self):
        for scores, attn, want in self.CASES:
            with self.subTest(scores=scores, attn=attn):
                self.assertEqual(launcher_stage_layers(scores, attn), want)

    def test_the_boot_form_is_unchanged_so_this_fix_moves_no_boot_number(self):
        scores, attn, want = self.CASES[0]
        self.assertEqual(launcher_stage_layers(scores, attn), want)
        self.assertEqual(
            chunk_tag_cards(want, LAYERS_PER_CHUNK, CHUNK_COUNT, card_of_stage=NVML_OF_STAGE),
            chunk_tag_cards(list(scores), LAYERS_PER_CHUNK, CHUNK_COUNT, card_of_stage=NVML_OF_STAGE),
        )

    def test_a_neighbour_puts_a_tag_on_the_wrong_card_when_the_flag_is_restated(self):
        # weights_3 is the discriminating tag in all three diverging rows: the
        # restated flag and the derived truth disagree about which card holds it,
        # and the pause order is built from exactly that.
        for scores, attn, want in self.CASES[1:]:
            with self.subTest(scores=scores, attn=attn):
                flag = chunk_tag_cards(list(scores), LAYERS_PER_CHUNK, CHUNK_COUNT,
                                       card_of_stage=NVML_OF_STAGE)
                truth = chunk_tag_cards(want, LAYERS_PER_CHUNK, CHUNK_COUNT,
                                        card_of_stage=NVML_OF_STAGE)
                self.assertNotEqual(flag["weights_3"], truth["weights_3"])
                self.assertEqual(launcher_stage_layers(scores, attn), want)

    def test_the_two_score_vectors_have_one_definition_each(self):
        # argv_p and the map must read the SAME two constants; a second literal
        # is how "8,4,4" and the map drifted apart in the first place.
        argv = launcher_mod.argv_p("py", "/m", [1, 2, 3], 1, 600, 1.0, [])
        i = argv.index("--pp-stage-ratio")
        self.assertEqual(
            argv[i + 1], ",".join(str(n) for n in launcher_mod.P_PP_STAGE_RATIO_SCORES)
        )
        j = argv.index("--pp-attn-stage-ratio")
        self.assertEqual(
            argv[j + 1],
            ",".join(str(n) for n in launcher_mod.P_PP_ATTN_STAGE_RATIO_SCORES),
        )

    def test_layer_kinds_are_read_from_the_checkpoint_and_refused_by_name(self):
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "config.json"), "w") as f:
                json.dump({"text_config": {"num_hidden_layers": 4,
                                           "layer_types": ["linear_attention"] * 3
                                           + ["full_attention"]}}, f)
            self.assertEqual(
                launcher_mod.model_layer_kinds(d), [False, False, False, True]
            )
        # fix 6: a config with NO layer markers is no longer a refusal here.
        # It was STRICTER than the authority this function claims to mirror
        # (server_args.declared_layer_kinds_from_config), which reads such a
        # checkpoint as homogeneous all-attention -- and a launcher refusal
        # meant NO chunk->card map, i.e. the identity pause order that killed
        # weg2dk4.  The two now share one derivation.
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "config.json"), "w") as f:
                json.dump({"text_config": {"num_hidden_layers": 4}}, f)
            self.assertEqual(launcher_mod.model_layer_kinds(d), [True] * 4)
        # What IS still a refusal: a config whose depth cannot be read at all,
        # because there is no authority to defer to for that.
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "config.json"), "w") as f:
                json.dump({"text_config": {"hidden_size": 5120}}, f)
            with self.assertRaises(launcher_mod.Weg2LaunchRefused) as ei:
                launcher_mod.model_layer_kinds(d)
            self.assertIn("num_hidden_layers", str(ei.exception))


class TestInterleavePauseOrder(CustomTestCase):
    def setUp(self):
        self.tags = weights_family_tags(CHUNK_COUNT)
        self.map = {
            t: list(v)
            for t, v in chunk_tag_cards(
                P_STAGE_LAYERS, LAYERS_PER_CHUNK, CHUNK_COUNT, card_of_stage=NVML_OF_STAGE
            ).items()
        }

    def test_tightest_card_first_on_the_weg2dk4_geometry(self):
        order, why = interleave_pause_order(self.tags, self.map, FREE_AT_INTERLEAVE_START)
        self.assertEqual(why, "tightest-card-first")
        self.assertEqual(
            order,
            ["weights_6", "weights_7", "weights_4", "weights_5",
             "weights_0", "weights_1", "weights_2", "weights_3", "weights"],
        )

    def test_the_base_tag_still_closes_the_sleep(self):
        order, _ = interleave_pause_order(self.tags, self.map, FREE_AT_INTERLEAVE_START)
        self.assertEqual(order[-1], "weights")

    def test_the_order_is_always_a_permutation_of_the_family(self):
        for free in ({1: 1, 0: 2, 2: 3}, {1: 3, 0: 2, 2: 1}, {1: 5, 0: 5, 2: 5}):
            order, _ = interleave_pause_order(self.tags, self.map, free)
            self.assertEqual(sorted(order), sorted(self.tags))

    def test_a_uniform_source_keeps_the_natural_order(self):
        order, why = interleave_pause_order(self.tags, {}, FREE_AT_INTERLEAVE_START)
        self.assertEqual(order, self.tags)
        self.assertIn("no chunk->card map", why)

    def test_no_free_sample_keeps_the_natural_order(self):
        order, why = interleave_pause_order(self.tags, self.map, {})
        self.assertEqual(order, self.tags)
        self.assertIn("no NVML free sample", why)

    def test_an_incomplete_map_is_a_named_refusal_to_reorder(self):
        partial = dict(self.map)
        partial.pop("weights_7")
        order, why = interleave_pause_order(self.tags, partial, FREE_AT_INTERLEAVE_START)
        self.assertEqual(order, self.tags)
        self.assertIn("REFUSED", why)
        self.assertIn("weights_7", why)

    def test_a_card_missing_from_the_free_sample_is_a_named_refusal(self):
        order, why = interleave_pause_order(self.tags, self.map, {1: 100, 0: 100})
        self.assertEqual(order, self.tags)
        self.assertIn("REFUSED", why)
        self.assertIn("[2]", why)

    def test_ties_keep_the_natural_index(self):
        order, _ = interleave_pause_order(self.tags, self.map, {0: 7, 1: 7, 2: 7})
        self.assertEqual(order, self.tags)


class TestTheMeasuredPeak(CustomTestCase):
    """The red/green pin: the natural order runs card nvml2 out of memory at
    exactly the step weg2dk4 died at, the new order does not."""

    def setUp(self):
        self.tags = weights_family_tags(CHUNK_COUNT)
        self.map = {
            t: list(v)
            for t, v in chunk_tag_cards(
                P_STAGE_LAYERS, LAYERS_PER_CHUNK, CHUNK_COUNT, card_of_stage=NVML_OF_STAGE
            ).items()
        }

    def test_the_natural_order_reproduces_the_death_on_the_last_stage_card(self):
        low = simulate_interleave(self.tags, self.tags)
        self.assertLess(low[2], 0.0, low)
        # and it is the SIXTH chunk resume that goes under, as measured
        # (front.log: weights_0..4 done, W4 on wake(D, weights_5)).
        free = FREE_AT_INTERLEAVE_START[2]
        steps = []
        for k in range(CHUNK_COUNT):
            free -= D_CHUNK_MIB  # tags 0..5 hold no stage-2 layer: nothing given back
            steps.append(free)
            if free < 0:
                break
        self.assertEqual(len(steps), 6)

    def test_the_new_order_keeps_every_card_positive(self):
        order, why = interleave_pause_order(self.tags, self.map, FREE_AT_INTERLEAVE_START)
        self.assertEqual(why, "tightest-card-first")
        low = simulate_interleave(order, self.tags)
        for nvml, mib in low.items():
            self.assertGreater(mib, 0.0, f"nvml{nvml} went negative: {low}")
        # the tight card keeps GiB-scale margin where it had 277.8 MiB
        self.assertGreater(low[2], 1024.0, low)

    def test_the_endpoint_is_unchanged_only_the_path_is(self):
        order, _ = interleave_pause_order(self.tags, self.map, FREE_AT_INTERLEAVE_START)
        self.assertEqual(sorted(order), sorted(self.tags))
        end_natural = simulate_interleave(self.tags, self.tags)
        end_new = simulate_interleave(order, self.tags)
        self.assertEqual(set(end_natural), set(end_new))


class _Rec:
    """A Group stub good enough for flip(): no outstanding, a session id of 0."""


class TestFlipUsesTheOrder(CustomTestCase):
    """C: the front actually pauses in that order and resumes in the natural
    one -- the wiring, not just the helper."""

    def _front(self, src_chunk_cards):
        f = Front("http://p", "http://d", "P", "t", "", 0, 0, {}, 45.0,
                  weight_chunks=CHUNK_COUNT, src_chunk_cards=src_chunk_cards)
        f.calls = []

        async def rpc(g, path, body, timeout):
            f.calls.append((g.name, path, tuple(body.get("tags", ()))))
            return 200, "{}"

        async def quiesce(g):
            return True, "idle"

        f.rpc = rpc
        f.quiesce = quiesce
        return f

    def _run(self, f):
        old = front_mod._nvml_free
        front_mod._nvml_free = lambda: [(i, f"uuid{i}", int(v)) for i, v in FREE_AT_INTERLEAVE_START.items()]
        try:
            asyncio.run(f.flip("P", "D"))
        finally:
            front_mod._nvml_free = old

    def test_p_as_source_pauses_tightest_card_first_and_d_resumes_naturally(self):
        m = {t: list(v) for t, v in chunk_tag_cards(
            P_STAGE_LAYERS, LAYERS_PER_CHUNK, CHUNK_COUNT, card_of_stage=NVML_OF_STAGE).items()}
        f = self._front({"P": m})
        self._run(f)
        self.assertIsNone(f.stop, f.stop)
        # ON THE RING (C9): the family travels as ONE gathered leg per group,
        # so the pause ORDER is the ORDER OF THAT LEG'S TAG LIST, not a sequence
        # of per-tag RPCs.  The asserted permutation is UNCHANGED -- only where
        # it is read from.  (This used to take t[0] of each call, which was the
        # whole tag when every call carried exactly one.)
        paused = [list(t) for g, p, t in f.calls if g == "P" and p == "/release_memory_occupation"]
        resumed = [list(t) for g, p, t in f.calls if g == "D" and p == "/resume_memory_occupation"]
        self.assertEqual(paused[0], ["kv_cache"])
        self.assertEqual(len(paused), 2, "C9: kv, then ONE gathered family leg")
        self.assertEqual(
            paused[1],
            ["weights_6", "weights_7", "weights_4", "weights_5",
             "weights_0", "weights_1", "weights_2", "weights_3", "weights"],
        )
        # D resumes the family in the NATURAL order, then its kv.
        self.assertEqual(resumed, [weights_family_tags(CHUNK_COUNT), ["kv_cache"]])
        self.assertEqual(f.epoch, 1)
        self.assertEqual(f.awake, "D")

    def test_an_order_that_is_not_a_permutation_stops_before_any_weight_moves(self):
        f = self._front({})
        old_fn = front_mod.interleave_pause_order
        front_mod.interleave_pause_order = lambda tags, cards, free: (["weights_0"] * len(tags), "bogus")
        try:
            self._run(f)
        finally:
            front_mod.interleave_pause_order = old_fn
        self.assertIsNotNone(f.stop)
        self.assertEqual(f.state, "STOP")
        weights_calls = [c for c in f.calls if c[2] and c[2][0].startswith("weights")]
        self.assertEqual(weights_calls, [])
        self.assertEqual(f.epoch, 0)

    def test_without_a_map_the_order_is_the_natural_one(self):
        # Renamed on the ring rebase: there is no per-tag "sequence" left to be
        # byte-for-byte with.  The invariant that survives is the one that
        # mattered -- with no chunk->card map the gathered leg carries the
        # family in its natural order, so a source the launcher could not map
        # is never reordered on a guess.
        f = self._front({})
        self._run(f)
        self.assertIsNone(f.stop, f.stop)
        paused = [list(t) for g, p, t in f.calls if g == "P" and p == "/release_memory_occupation"]
        self.assertEqual(paused, [["kv_cache"], weights_family_tags(CHUNK_COUNT)])


class TestCarrierBoundPopulation(CustomTestCase):
    """The second weg2dk4 finding: three pools, one sentence, one min -> 17."""

    KV = ("[2026-09-07 20:19:57 TP{r}] HiCache host KV pool [MHATokenToKVPoolHost] (30518 tokens) is smaller "
          "than the device pool (368781 tokens);L2 cache effectiveness is reduced.\n")
    MAMBA = ("[2026-09-07 20:19:57 TP{r}] HiCache Mamba anchor host pool (19 slots) is smaller than the device "
             "pool (20 slots);L2 anchor coverage is reduced.\n")

    def test_the_mamba_anchor_pool_is_not_in_the_population(self):
        lines = [self.KV.format(r=r) for r in range(3)] + [self.MAMBA.format(r=r) for r in range(3)]
        bound, pools = carrier_bound_from_lines(lines)
        self.assertEqual(pools, {"MHATokenToKVPoolHost": [30518, 30518, 30518]})
        self.assertEqual(bound, 27466)  # 0.9 x 30518, the spec's own cap

    def test_a_second_labelled_kv_pool_binds_when_it_is_smaller(self):
        lines = [self.KV.format(r=0),
                 "TP0 HiCache host KV pool [DraftTokenToKVPoolHost] (4096 tokens) is smaller than the device "
                 "pool (9999 tokens);\n"]
        bound, pools = carrier_bound_from_lines(lines)
        self.assertEqual(sorted(pools), ["DraftTokenToKVPoolHost", "MHATokenToKVPoolHost"])
        self.assertEqual(bound, 3686)

    def test_an_unlabelled_line_yields_no_bound_rather_than_a_wrong_one(self):
        old = "TP0 HiCache host KV pool (30518 tokens) is smaller than the device pool (368781 tokens);\n"
        bound, pools = carrier_bound_from_lines([old])
        self.assertEqual((bound, pools), (0, {}))

    #: The two line shapes boot weg2dk4 actually wrote (D.log:889-891 and
    #: 899/904/905), before either emitter named its pool.
    DK4_KV_OLD = ("[2026-09-07 20:19:57 TP{r}] HiCache host KV pool (30518 tokens) is smaller than the device "
                  "pool (368781 tokens);L2 cache effectiveness is reduced.\n")
    DK4_MAMBA_OLD = ("[2026-09-07 20:19:57 TP{r}] HiCache host KV pool (19 tokens) is smaller than the device "
                     "pool (20 tokens);L2 cache effectiveness is reduced.\n")

    def test_the_dk4_log_shape_can_no_longer_produce_17(self):
        lines = ([self.DK4_KV_OLD.format(r=r) for r in range(3)]
                 + [self.DK4_MAMBA_OLD.format(r=r) for r in range(3)])
        bound, pools = carrier_bound_from_lines(lines)
        self.assertEqual(pools, {})
        self.assertEqual(bound, 0)  # route disabled AND logged, never a wrong number
        self.assertNotEqual(bound, 17)


if __name__ == "__main__":
    unittest.main()
