"""DUAL-TP3PP3 (F26): the per-card byte plan of the double layout.

Pinned without a GPU:
  * the three parts add up in both directions -- shared + pp_only = P's stage,
    shared + tp_only = D's shard, the P stages sum to the model, the D shards
    sum to the model plus the replicated norms (memory planer-ergebnis-gegen-
    physik-pruefen-0928: every number checked both ways);
  * direction: moving D weight TOWARDS a card grows that card's shared part
    and shrinks its pp_only (whoever gives must win);
  * a card whose P stage holds NO layers shares nothing but embed/lm_head;
  * the equal-context solver returns a cut whose context shares track the
    capacity shares better than a naive even cut;
  * bad input is refused by name, never silently accepted.
"""

from __future__ import annotations

import dataclasses
import unittest

from flliper.srt.pdflip import dual_layout_plan as dp

MiB = dp.MIB


def _model(n=64, embed=2425, lm_head=1213, draft=1478, vision=879):
    fam = {}
    for i in range(n):
        f = {"mlp": int(143.4 * MiB) if i < 56 else int(255 * MiB), "input_layernorm": 10240,
             "post_attention_layernorm": 10240}
        if i % 4 == 3:
            f["self_attn"] = int(100 * MiB)
        else:
            f["linear_attn"] = int(111 * MiB)
        fam[i] = f
    return dp.model_bytes_from_families(fam, embed=embed * MiB, lm_head=lm_head * MiB,
                                        draft=draft * MiB, vision=vision * MiB)


CAP = {0: 20055 * MiB, 1: 32088 * MiB, 2: 20055 * MiB}
OV = {0: 3000 * MiB, 1: 5000 * MiB, 2: 3000 * MiB}
VR = (20055, 32088, 20055)


def _lay(**kw):
    base = dict(d_card=(0, 1, 2), p_card=(0, 1, 2), p_cut=(10, 42, 12), mixer=VR, mlp=VR,
                vocab=VR, draft_stage=-1, vision_in_p=True)
    base.update(kw)
    return dp.DualLayout(**base)


class TestDualLayoutPlan(unittest.TestCase):
    def test_parts_add_up_both_ways(self):
        m = _model()
        for lay in (_lay(), _lay(p_card=(1, 0, 2), p_cut=(49, 8, 7)),
                    _lay(mlp=(19, 98, 19), mixer=(1, 2, 1), draft_stage=None, vision_in_p=False)):
            rows = dp.plan(m, lay, CAP, OV)
            self.assertEqual(dp.check_physics(rows, m, lay), [], lay)
            self.assertEqual(sum(r.p_total for r in rows) // MiB,
                             (m.layer_total + m.embed + m.lm_head
                              + (m.draft if lay.draft_stage is not None else 0)
                              + (m.vision if lay.vision_in_p else 0)) // MiB)

    def test_direction_whoever_gives_wins(self):
        m = _model()
        a = {r.card: r for r in dp.plan(m, _lay(), CAP, OV)}
        heavier = (20055, 3 * 32088, 20055)
        b = {r.card: r for r in dp.plan(m, _lay(mixer=heavier, mlp=heavier), CAP, OV)}
        # D moved weight onto card 1: card 1 shares more, P needs less extra there,
        # D needs more there in total.
        self.assertGreater(b[1].shared, a[1].shared)
        self.assertLess(b[1].pp_only, a[1].pp_only)
        self.assertGreater(b[1].d_total, a[1].d_total)
        # ...and the cards that gave shrink their D total.
        self.assertLess(b[0].d_total, a[0].d_total)

    def test_empty_stage_shares_only_vocab(self):
        m = _model()
        rows = {r.card: r for r in dp.plan(m, _lay(p_cut=(0, 64, 0)), CAP, OV)}
        # card 0 is P stage 0 with no layers: it keeps embed (+vision) whole,
        # and shares only D's vocab shard of embed.
        self.assertAlmostEqual(rows[0].shared, m.embed * VR[0] / sum(VR), delta=2)
        self.assertEqual(rows[0].p_total, m.embed + m.vision)

    def test_solver_tracks_capacity(self):
        m = _model()
        cut, rows = dp.solve_cut_equal_context(m, _lay(), CAP, OV)
        self.assertEqual(sum(cut), 64)
        tot = sum(r.context for r in rows)
        cap = sum(CAP.values())
        dev = max(abs(r.context / tot - CAP[r.card] / cap) for r in rows)
        naive = dp.plan(m, _lay(p_cut=(21, 22, 21)), CAP, OV)
        ntot = sum(r.context for r in naive)
        ndev = max(abs(r.context / ntot - CAP[r.card] / cap) for r in naive)
        self.assertLess(dev, ndev)
        self.assertLess(dev, 0.05)

    def test_tokens(self):
        m = _model()
        rows = dp.plan(m, _lay(), CAP, OV)
        per_tok = 16 * m.kv_bytes_per_token_layer
        self.assertEqual(per_tok, 32 * 1024)
        self.assertEqual(dp.d_tokens(rows, m), sum(r.context for r in rows) // per_tok)
        self.assertGreater(dp.p_tokens(rows, m), 0)

    def test_stage_1a_pays_the_shared_part_twice(self):
        m = _model()
        a = {r.card: r for r in dp.plan(m, _lay(), CAP, OV)}
        b = {r.card: r for r in dp.plan(m, _lay(), CAP, OV, share=False)}
        for c in CAP:
            self.assertEqual(a[c].context - b[c].context, a[c].shared)
            self.assertAlmostEqual(b[c].weights, a[c].d_total + a[c].p_total, delta=2)

    def test_refusals(self):
        m = _model()
        with self.assertRaises(dp.DualPlanError):
            dp.plan(m, _lay(p_cut=(10, 42, 11)), CAP, OV)
        with self.assertRaises(dp.DualPlanError):
            dp.plan(m, _lay(p_card=(0, 1, 1)), CAP, OV)
        with self.assertRaises(dp.DualPlanError):
            dp.plan(m, _lay(mlp=(1, 1)), CAP, OV)
        with self.assertRaises(dp.DualPlanError):
            dp.plan(m, _lay(vocab=(0, 0, 0)), CAP, OV)


if __name__ == "__main__":
    unittest.main()


class TestReplicatedEmbed(unittest.TestCase):
    def test_replicated_embed_is_shared_whole_on_stage0(self):
        m = _model()
        a = {r.card: r for r in dp.plan(m, _lay(), CAP, OV)}
        b_lay = _lay(d_embed_replicated=True)
        rows = dp.plan(m, b_lay, CAP, OV)
        self.assertEqual(dp.check_physics(rows, m, b_lay), [])
        b = {r.card: r for r in rows}
        # stage 0 lives on card 0: shared grows by the rest of the embed, P's diff shrinks by it
        grow = m.embed - m.embed * VR[0] / sum(VR)
        self.assertAlmostEqual(b[0].shared - a[0].shared, grow, delta=2)
        self.assertAlmostEqual(a[0].pp_only - b[0].pp_only, grow, delta=2)
        # every D rank holds the whole embed
        for c in CAP:
            self.assertGreater(b[c].d_total, a[c].d_total - 1)


class TestContextSplit(unittest.TestCase):
    def test_p_prompt_and_d_rest_are_consistent(self):
        m = _model()
        rows = dp.plan(m, _lay(p_card=(1, 0, 2), p_cut=(49, 8, 7)), CAP, OV)
        t = dp.p_max_prompt(rows, m, {})
        need = dp.p_kv_need(rows, m, t)
        # at the bound, the binding card is (almost) exactly full, none is over
        self.assertTrue(all(need[r.card] <= r.context for r in rows))
        self.assertTrue(any(r.context - need[r.card] < 12 * m.kv_bytes_per_token_layer * 1 + 1 << 20
                            for r in rows))
        # more P prompt -> less D
        self.assertGreater(dp.d_tokens_after_p(rows, m, 1000), dp.d_tokens_after_p(rows, m, t))
        # D's floor lowers P's bound
        self.assertLess(dp.p_max_prompt(rows, m, {r.card: 1 << 30 for r in rows}), t)
