"""NFBLOCK 1022: NF on 2, 4, 5, 6 and 7 identical RTX 3090 was refused by the launcher's D-EIGENTUM solve (#239 S3f) as D-BUDGET, though memory was never the limit.

Two launcher defects decide it (the launcher dry runs of every N are pinned by ``test_ncards_dryruns_1021``):

* N = 2: the guard of the owned solve held the Env wave cap 2 (``FLLIPER_OPT_MOE_POOL_OVERFLOW_WAVES``), although the D group runs the waves its form needs when the waves
  are the launcher's own default (OVERFLOW-WAVES, ``derive_waves``) -- Form A at the stated vector already needs 6.  When no cut form holds the Env cap, the solve now runs
  once more at Form A's own waves (never above them).
* N = 4, 5, 6, 7: Form A keeps every worker expert resident (miss time 0.00 ms); every token cut takes worker rows for KV, x1 refuses it, and the solve refused the BEST
  form ("keine tragbare Form") instead of keeping it.  Form A is now the answer when no cut form carries and keeps x1, judged by the same checks as every uncut D (W122 /
  W130 / pool step), with its own KV stage form on the host; the S3h flip riegel names it.

A latent third one, not hit by the 3090 boards (their workers form two classes, middle and last, by the borrowed role reference): a class of 5 or 6 IDENTICAL workers had
no KV share vector at all (64 grid units do not split into equal multiples of 4) -- 0 candidates.  The class shares now split a class total as evenly as integers allow
where the stepped enumeration is empty.

And the time of a dry run (question 3): the seat table (1..6 seats) ran its OWN ownership search per seat count whenever the owned solve had not published a form (NF 4 x 3090:
655 s); it now prices the decided form.  The checkpoint header read (26 times per NF dry run) is memoized per process.

CPU only, no GPU, no launcher run.
"""

import json
import os
import struct
import tempfile
import types
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import inspect  # noqa: E402

from flliper.srt.planner import expert_residency as E  # noqa: E402
from flliper.srt.planner import pp_cut as P  # noqa: E402


class ClassShares(unittest.TestCase):
    def test_five_and_six_identical_workers_get_share_vectors(self):
        for n in (6, 7):
            groups = [[0], list(range(1, n))]
            sh = E.owned_share_vectors_classes(n, 0, groups, E.KV_TOKEN_SHARE_GRID, E.OWNED_SHARE_STEP)
            self.assertTrue(sh, n)                                                   # was [] -> 0 candidates -> D-BUDGET
            for v in sh:
                self.assertEqual(sum(v), E.KV_TOKEN_SHARE_GRID)
                self.assertEqual(v[0], 0)                                            # the host holds no KV under the cut
                self.assertLessEqual(max(v[1:]) - min(v[1:]), 1, v)                  # one class: as even as integers allow

    def test_a_divisible_class_keeps_the_stepped_vectors_bit_for_bit(self):
        self.assertEqual(E.owned_share_vectors_classes(5, 0, [[0], [1, 2, 3, 4]], 64, 4), [(0, 16, 16, 16, 16)])
        mixed = E.owned_share_vectors_classes(6, 0, [[0], [1, 2, 3, 4], [5]], 64, 4)
        self.assertTrue(mixed)
        for v in mixed:
            self.assertEqual(len({v[1], v[2], v[3], v[4]}), 1)                      # the old enumeration answered: unchanged

    def test_four_ranks_run_over_classes_three_stay_exact(self):
        self.assertEqual(E.OWNED_CLASS_MIN_RANKS, 3)


# a 2-rank instance in the shape of NF on 2 x 3090: the host's KV goes to the worker under the cut, ~100 rows per card for ~257 experts per rank
CAP = (100, 100)
KV_ROWS = 20
SCRATCH = (20, 20)


def _fit(rank, ratio, cut):
    E_ = int(ratio) + 1
    kv = (KV_ROWS if rank == 0 else 0) if cut is None else KV_ROWS * cut[rank] / float(sum(cut))
    ceiling = int(CAP[rank] - kv)
    rows = min(ceiling - SCRATCH[rank], E_ - 2)
    return types.SimpleNamespace(rank=rank, local_experts=E_, ceiling_max_rows=ceiling, scratch_rows=SCRATCH[rank],
                                 ceiling_fraction=(rows / float(E_)) if rows >= 1 else None)


def _owned(waves):
    return E.solve_owned_cut(lambda rat, sh: tuple(_fit(r, rat[r], sh) for r in range(2)), (256, 256), 0, num_experts=512, n_layers=48,
                             ids_per_step=40, x1_scope="round", seats_cap=6, waves_cap=waves)


class WaveCap(unittest.TestCase):
    def test_the_env_cap_refuses_what_the_form_a_waves_carry(self):
        at_env = _owned(2)
        self.assertEqual(at_env.feasible, 0)
        self.assertEqual(at_env.guard_refused, at_env.candidates)                   # every candidate at the hard limits: 240 ids, 2 waves, ~100 rows
        at_form_a = _owned(6)
        self.assertGreater(at_form_a.feasible, 0)
        self.assertEqual(dict(at_form_a.guard)["waves"], 6)

    def test_the_plan_retries_at_form_a_waves_only_when_the_env_cap_finds_nothing(self):
        src = inspect.getsource(E.plan_d_residency)
        i = src.index("D-EIGENTUM WELLEN (nfblock 1022)")
        head = src[:i]
        self.assertIn("not sol.feasible and _forced_shares is None and derive_waves", head)   # a cut that holds the Env cap is never re-searched (rig: bit for bit)
        self.assertIn("_form_a_waves > int(_lim.get(\"waves_cap\", 1))", head)                # never above Form A's own waves
        self.assertIn("pool_step_waves_needed(", head)                                        # the same formula as OVERFLOW-WAVES


class FormAKept(unittest.TestCase):
    def test_no_cut_form_means_form_a_not_a_refusal(self):
        src = inspect.getsource(E.plan_d_residency)
        i = src.index("FORM A BLEIBT (nfblock 1022)")
        block = src[src.rindex("if _forced_shares is None:", 0, i):i + 900]
        self.assertIn("owner_refusal = None", block)
        self.assertIn('("form_a_kept", True)', block)
        # a forced cut (the measurement arm M1b) keeps the refusal: the guard is "_forced_shares is None"

    def test_the_seat_table_prices_form_a_once(self):
        from flliper.srt.pdflip import launcher as L

        failed = types.SimpleNamespace(solved_owner_ratio=(), kv_token_cut=(), solved_fractions=())
        kw, fr = L.d_seat_table_form(failed, dict(kv_token_shares="owned", kv_dtype_bytes=1), [0.9, 0.98])
        self.assertEqual(kw, {})                                                    # no 'owned' handed on: no ownership search per seat count
        self.assertEqual(fr, [0.9, 0.98])

    def test_the_flip_riegel_names_form_a_instead_of_refusing(self):
        from flliper.srt.pdflip import launcher as L

        ns = types.SimpleNamespace(d_kv_token_cut="owned", d_only=False, dry_run=False, env_d="",
                                   _d_kv_stage_written={"host": 0, "rows": 33, "cut_workers": [], "worker_rows": {}})
        form = types.SimpleNamespace(kv="qsa_forma_dcp")
        lines = []
        msg = L.refuse_flip_stage_form_without_trim_cells(ns, form, lines.append)
        self.assertIn("STUFEN ERLAUBT (Form A, nfblock 1022)", msg)
        self.assertEqual(lines, [msg])
        # an unsolved cut still refuses by name
        ns2 = types.SimpleNamespace(d_kv_token_cut="owned", d_only=False, dry_run=True, env_d="",
                                    _d_kv_stage_written={"host": 0, "rows": 33, "cut_workers": [-1], "worker_rows": {}})
        self.assertIn("ungeloest", L.refuse_flip_stage_form_without_trim_cells(ns2, form, lambda _l: None))


def _write_shard(path, tensors):
    header = {name: {"dtype": "BF16", "shape": shape, "data_offsets": [0, 0]} for name, shape in tensors.items()}
    raw = json.dumps(header).encode()
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(raw)))
        fh.write(raw)


class WeightTermsMemo(unittest.TestCase):
    def test_read_once_per_unchanged_checkpoint_and_again_after_a_change(self):
        with tempfile.TemporaryDirectory() as d:
            shard = os.path.join(d, "model-00001.safetensors")
            _write_shard(shard, {"model.layers.0.self_attn.q_proj.weight": [8, 8], "model.layers.1.linear_attn.x.weight": [8, 8],
                                 "model.embed_tokens.weight": [4, 8], "lm_head.weight": [4, 8]})
            calls = []
            real = P._checkpoint_weight_terms_read

            def counting(path):
                calls.append(path)
                return real(path)

            P._checkpoint_weight_terms_read = counting
            try:
                a = P.checkpoint_weight_terms(d)
                b = P.checkpoint_weight_terms(d)
                self.assertEqual(len(calls), 1)
                self.assertEqual(a, b)
                b.replicated_breakdown["x"] = 1.0                                     # a caller's change never reaches the memo
                self.assertNotIn("x", P.checkpoint_weight_terms(d).replicated_breakdown)
                _write_shard(shard, {"model.layers.0.self_attn.q_proj.weight": [16, 8], "model.layers.1.linear_attn.x.weight": [8, 8],
                                     "model.embed_tokens.weight": [4, 8], "lm_head.weight": [4, 8], "model.layers.2.linear_attn.x.weight": [8, 8]})
                c = P.checkpoint_weight_terms(d)
                self.assertEqual(len(calls), 2)                                        # size changed: read again
                self.assertEqual(c.n_layers, 3)
            finally:
                P._checkpoint_weight_terms_read = real

    def test_a_missing_checkpoint_still_raises(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(P.DraftResidencyUnavailable):
                P.checkpoint_weight_terms(d)


if __name__ == "__main__":
    unittest.main()
