"""NCARDS 1021 REVIEW (independent review of desk/flliper-planer-ncards-1021 @ 3efdb36783): the tests the author's round lacks.

Two defects found in review, each pinned here so that the FIX turns the test red ("unexpected success") instead of staying silent:

A1  NF on N != 3 identical cards is NOT checked by the launcher's D residency budget: the D FRACTION-SOLVE of the launcher says
    ``FRACTION-SOLVE D(Karte, Erwartung) ENTFAELLT: die eingebaute Referenz (fnFL2x151 + fnFL2x158) gilt fuer ... 1,0,0 auf 3 Raengen,
    dieser Boot faehrt ... 1,0 auf 2`` (dry run of nf-2x3090 / nf-4x3090), so "N=2 und N=4 bestehen dieselbe Pruefung" (author report
    section 6) is a skipped check, not a passed one.  The launcher's OWN formula (``expert_residency.buffer_rows``: rows = min(R + S, E),
    48 layers x 2.417 MiB per row, plus the fixed / Mamba / spec / activation / KV posts of the role) reproduces the N=3 refusal exactly
    (-5622 / -5426 / -4996 MiB) and puts nf-2x3090 ~12 GB over on the host and nf-4x3090 54 MiB over on the host -- profiles that carry
    PROFILE_STATUS=hochrechnung (bootable with --force).  Root cause on the planner side: ``propose_rules.form_a_d`` /
    ``expert_residency.d_rank_fraction_caps`` size the resident share with ``scratch_rows=0`` while the runtime holds R + S rows, and the
    host keeps the 5090's FLLIPER_MOE_SCRATCH_SLOTS 100 (11.6 GB of rows) by ROLE on a 24 GB card.

A2  ``launcher.dual_alloc_cache_book_env`` reads the record through the inventory view only when the LENGTH differs: on 3 x RTX 3090
    (same count, foreign inventory) the reference rig's positional vector 1318,0,0 (the 5090's growth on ordinal 0) is booked as is,
    while 2 and 4 x RTX 3090 get the class borrow 0,...,0.  Same count is not same inventory (``inventory_view.same_n_foreign``,
    PLANER-3090 1021).

CPU only, no GPU, no boot.  Light: imports the launcher (no model load).
"""

import os
import pathlib
import re
import types
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import form as F  # noqa: E402
from flliper.srt.pdflip import inventory_view as IV  # noqa: E402
from flliper.srt.pdflip import launcher as L  # noqa: E402
from flliper.srt.planner import expert_residency as ER  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
TREE = HERE.parents[3]
PROFILES = TREE / "docker" / "flliper" / "profiles_release"
GOLDEN = HERE / "fixtures" / "ncards_1021" / "golden"

#: one expert row of NF on one D rank: 48 layers x 2.417 MiB (launcher line "48 Layer x 2.417 MiB/Zeile")
ROW_MIB = 48 * 2.417
#: the fixed posts per ROLE as the launcher prices them in the nf-3x3090 dry run (D(Karte, Erwartung), 6 seats, 262144 tokens KV):
#: host = rank 0 (fest 7481 + mamba 2134 + spec 134 + Aktivierung 1104 + KV 3536), middle worker (1075 + 1024), last worker (953 + 1024).
#: Records D_FIXED_MIB / D_ACTIVATION_MIB are NF records without policy -> ROLE borrow (inventory_view.borrow_vector), the same on every N.
POSTS = {"host": 7481 + 2134 + 134 + 1104 + 3536, "middle": 1075 + 1024, "last": 953 + 1024}
NF_RIG_SCRATCH = (100, 48, 48)
FOREIGN = "RTX3090/24576MiB/sm86"


def _role(j, n):
    return "host" if j == 0 else ("last" if j == n - 1 else "middle")


def _profile_value(text, kind, key):
    """The value the NCARDS block sets for ``key`` (``_p3090_extra d KEY V`` / ``_p3090_env d KEY V``), else None."""
    m = re.search(r"^_p3090_%s d %s (\S+)$" % (kind, re.escape(key)), text, re.M)
    return m.group(1) if m else None


def _d_rests(n):
    """Per D rank of nf-<n>x3090: budget - (buffer rows x row + role posts), the launcher's W122 balance."""
    text = (PROFILES / ("nf-%dx3090.env" % n)).read_text(encoding="utf-8")
    ratio = [int(x) for x in _profile_value(text, "extra", "--rank-moe-ratio").split(",")]
    frac = [float(x) for x in _profile_value(text, "extra", "--rank-moe-resident-fraction").split(",")]
    sv = _profile_value(text, "env", "FLLIPER_MOE_SCRATCH_SLOTS")
    scratch = [int(x) for x in sv.split(",")] if sv else list(NF_RIG_SCRATCH)
    import json

    gold = json.loads((GOLDEN / ("dryrun_nf-%dx3090.json" % n)).read_text(encoding="utf-8"))
    budget = {o: mib for p, o, mib in gold["budgets"] if p == "D(Karte, Erwartung)"}
    out = []
    for j in range(n):
        rows = ER.buffer_rows(local_experts=ratio[j] + 1, fraction=frac[j], scratch_rows=scratch[j])   # + 1 pad row (#82)
        out.append(round(budget[j] - (rows * ROW_MIB + POSTS[_role(j, n)])))
    return out, text


class NfDResidencyOnNIdenticalCards(unittest.TestCase):
    """A1: the launcher's W122 balance, recomputed per profile."""

    def test_the_formula_reproduces_the_launchers_n3_refusal(self):
        rests, text = _d_rests(3)
        self.assertEqual(rests, [-5622, -5426, -4996])            # launcher: "5622 / 5426 / 4996 MiB zuviel"
        self.assertIn("PROFILE_STATUS=blockiert-d-budget", text)

    @unittest.expectedFailure
    def test_nf_2x3090_marked_bootable_fits_the_d_balance(self):
        """DEFECT A1 (review 1021): -12003 MiB on the host, -5460 on the worker; the dry run passes only because the D FRACTION-SOLVE
        ENTFAELLT for 2 ranks.  Fix = planner counts scratch rows / host scratch by class, launcher checks N != 3 (role-borrowed
        reference) or names the skipped check as a forced code; then regenerate and drop this decorator."""
        rests, text = _d_rests(2)
        self.assertIn("PROFILE_STATUS=hochrechnung", text)
        self.assertTrue(all(r >= 0 for r in rests), rests)

    @unittest.expectedFailure
    def test_nf_4x3090_marked_bootable_fits_the_d_balance(self):
        """DEFECT A1 (review 1021): host -54 MiB (rows 42 = whole share, f 0.952), skipped check as above."""
        rests, text = _d_rests(4)
        self.assertIn("PROFILE_STATUS=hochrechnung", text)
        self.assertTrue(all(r >= 0 for r in rests), rests)


class DualAllocCacheBookOnForeignInventory(unittest.TestCase):
    """A2: same count is not same inventory."""

    def _model(self):
        import json
        import tempfile

        d = tempfile.mkdtemp(prefix="m27b")
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump({"architectures": ["Qwen3_5ForConditionalGeneration"], "quantization_config": {"quant_method": "modelopt"}}, f)
        return d

    def _env(self, n):
        cal, _ = L.records_inventory(F.PROFILE_QWEN27B)
        live = (FOREIGN,) * n
        IV.set_active(F.PROFILE_QWEN27B, cal, live, borrow=["D_DUAL_ALLOC_CACHE_BOOK_MIB"], twins={FOREIGN: "RTX3080"})
        try:
            cards = [L.Card(nvml_index=i, uuid="GPU-%d" % i, name="NVIDIA GeForce RTX 3090", total_mib=24576) for i in range(n)]
            ns = types.SimpleNamespace(dual_share=True, dual_unified_kv="on", model=self._model(), profile=F.PROFILE_QWEN27B)
            return L.dual_alloc_cache_book_env(ns, cards, lambda _l: None)
        finally:
            IV.clear_active()

    def test_two_and_four_foreign_cards_take_the_class_borrow(self):
        env2, env4 = self._env(2), self._env(4)
        self.assertEqual(list(env2.values()), ["0,0"])
        self.assertEqual(list(env4.values()), ["0,0,0,0"])

    @unittest.expectedFailure
    def test_three_foreign_cards_take_the_same_borrow_not_the_rig_ordinals(self):
        """DEFECT A2 (review 1021): returns 1318,0,0 (the 5090's ordinal-0 growth on a 3090); fix at launcher.py:15292
        (``len(vals) != len(cards)`` -> also when the active view's inventory is not the live one)."""
        self.assertEqual(list(self._env(3).values()), ["0,0,0"])


if __name__ == "__main__":
    unittest.main()
