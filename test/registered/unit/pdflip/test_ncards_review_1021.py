"""NCARDS 1021 REVIEW (independent review of desk/flliper-planer-ncards-1021 @ 3efdb36783): the tests the author's round lacks.

Two defects found in review (round 2 of the branch: BOTH FIXED, the expectedFailure decorators are gone and the tests are plain regressions):

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

FIX of round 2: A1 -- ``propose_rules.form_a_d`` books the pad row and the scratch rows (``min(R + S, E)`` rows against the rank's rows), the host scratch
is capped by the rank's budget (and by the worker value on a host card of another class than the profile's); ``expert_residency.role_borrow_reference``
takes the 3-rank reference to N ranks by role so the launcher's D check RUNS for N != 3 (dry-run golden ``d_check``).  A2 -- ``dual_alloc_cache_book_env``
reads the record through the inventory view on every count.  What stays honest: the planner's posts (boot fn8ah) and the launcher's role records differ by
about 3 rows on the host, booked as a NAMED host post (``propose_rules.launcher_host_gap_mib``, listed in ``unverified``); the remaining difference is under one row
and the launcher's own ``d_scratch_cap`` trims it (``HOST_EDGE_ROWS`` below) -- the planner is not given a hidden reserve.

CPU only, no GPU, no boot.  Light: imports the launcher (no model load).
"""

import math
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


#: the launcher's consumer-follows-the-booked-budget rule (``launcher.d_scratch_cap``, rc12c) lowers a rank's scratch to the edge of its budget.  The planner's
#: posts (boot fn8ah, with the named host calibration ``propose_rules.launcher_host_gap_mib``) and the launcher's role records differ by less than one row
#: (116 MiB) on every rank, e.g. the middle worker of N = 3 (corridor 1768 against 1576 of the last worker, the planner books one worker post).  A profile
#: may start at most this many scratch rows above the edge; more would be the planner's blind spot, not the launcher's trim.
HOST_EDGE_ROWS = 1


def _d_rests(n, edge=False):
    """Per D rank of nf-<n>x3090: budget - (buffer rows x row + role posts), the launcher's W122 balance.  ``edge``: apply the launcher's own scratch
    trim first (lower the scratch of a rank that does not fit by the rows missing, never below 2) and return ``(rests, rows_trimmed)``."""
    text = (PROFILES / ("nf-%dx3090.env" % n)).read_text(encoding="utf-8")
    ratio = [int(x) for x in _profile_value(text, "extra", "--rank-moe-ratio").split(",")]
    frac = [float(x) for x in _profile_value(text, "extra", "--rank-moe-resident-fraction").split(",")]
    sv = _profile_value(text, "env", "FLLIPER_MOE_SCRATCH_SLOTS")
    scratch = [int(x) for x in sv.split(",")] if sv else list(NF_RIG_SCRATCH)
    import json

    gold = json.loads((GOLDEN / ("dryrun_nf-%dx3090.json" % n)).read_text(encoding="utf-8"))
    budget = {o: mib for p, o, mib in gold["budgets"] if p == "D(Karte, Erwartung)"}
    out, trimmed = [], []
    for j in range(n):
        rows = ER.buffer_rows(local_experts=ratio[j] + 1, fraction=frac[j], scratch_rows=scratch[j])   # + 1 pad row (#82)
        rest = round(budget[j] - (rows * ROW_MIB + POSTS[_role(j, n)]))
        cut = 0
        if edge and rest < 0:
            cut = min(int(math.ceil(-rest / ROW_MIB)), scratch[j] - 2)
            rows = ER.buffer_rows(local_experts=ratio[j] + 1, fraction=frac[j], scratch_rows=scratch[j] - cut)
            rest = round(budget[j] - (rows * ROW_MIB + POSTS[_role(j, n)]))
        out.append(rest)
        trimmed.append(cut)
    return ((out, trimmed) if edge else out), text


class NfDResidencyOnNIdenticalCards(unittest.TestCase):
    """A1 (fixed): the launcher's W122 balance, recomputed per profile from the profile file."""

    def test_the_formula_reproduces_the_launchers_old_n3_refusal(self):
        """The formula is the launcher's own: with the pre-fix planner values of nf-3x3090 (host scratch 100, ownership 89/212/211, resident 0.494/0.787/0.791)
        it gives the refusal the launcher printed (-5622 / -5426 / -4996 MiB), so the profile tests below measure the launcher, not a model of it."""
        ratio, frac, scratch = [89, 212, 211], [0.494, 0.787, 0.791], [100, 48, 48]
        import json

        gold = json.loads((GOLDEN / "dryrun_nf-3x3090.json").read_text(encoding="utf-8"))
        budget = {o: mib for p, o, mib in gold["budgets"] if p == "D(Karte, Erwartung)"}
        rests = [round(budget[j] - (ER.buffer_rows(local_experts=ratio[j] + 1, fraction=frac[j], scratch_rows=scratch[j]) * ROW_MIB + POSTS[_role(j, 3)]))
                 for j in range(3)]
        self.assertEqual(rests, [-5622, -5426, -4996])

    def _fits(self, n):
        (rests, trimmed), text = _d_rests(n, edge=True)
        # the W122 balance (what the review recomputed) fits on every N; whether the PROFILE is bootable is decided by the launcher's D-EIGENTUM solve as well
        # (dry-run golden, test_ncards_dryruns_1021): bootable on N = 3, blockiert-d-budget on N = 2 / 4 (the status is named, never hidden)
        self.assertIn("PROFILE_STATUS=" + ("hochrechnung" if n == 3 else "blockiert-d-budget"), text)
        self.assertTrue(all(r >= 0 for r in rests), (rests, trimmed))
        self.assertLessEqual(max(trimmed), HOST_EDGE_ROWS, (rests, trimmed))   # the planner is within one row of the launcher on every rank

    def test_nf_2x3090_fits_the_d_balance(self):
        """was the xfail A1 (-12003 / -5460 MiB): the W122 balance fits now."""
        self._fits(2)

    def test_nf_3x3090_fits_the_d_balance(self):
        self._fits(3)

    def test_nf_4x3090_fits_the_d_balance(self):
        """was the xfail A1 (host -54 MiB): the W122 balance fits now."""
        self._fits(4)

    def test_host_scratch_is_not_the_rig_5090_role_value(self):
        for n in (2, 3, 4):
            _, text = _d_rests(n)
            sv = _profile_value(text, "env", "FLLIPER_MOE_SCRATCH_SLOTS")
            self.assertIsNotNone(sv)
            scratch = [int(x) for x in sv.split(",")]
            self.assertEqual(len(scratch), n)
            self.assertLessEqual(scratch[0], 48, scratch)                       # the 5090's 100 rows (11.6 GB) do not fit a 24 GB host

    def test_the_launcher_check_runs_for_n_not_3_by_role_borrow(self):
        """The 3-rank reference (fnFL2) is taken to N ranks by ROLE and named; the shapes must match (host + zero-tp workers) and N must differ."""
        ref = ER.D_RESIDENCY_REFERENCE_FNFL2
        for n in (2, 4):
            b = ER.role_borrow_reference(ref, n, ",".join(["1"] + ["0"] * (n - 1)))
            self.assertIsNotNone(b)
            self.assertEqual(len(b.fixed_mib), n)
            self.assertEqual((b.fixed_mib[0], b.fixed_mib[-1]), (ref.fixed_mib[0], ref.fixed_mib[-1]))   # host = host, last = last
            self.assertEqual(set(b.kv_cell_bytes[1:]), {ref.kv_cell_bytes[1]})                        # workers carry the worker cell
            self.assertIn("BORROWED BY RANK ROLE to %d ranks" % n, b.source)
        self.assertIsNone(ER.role_borrow_reference(ref, 3, "1,0,0"))             # the reference's own count: no borrow
        self.assertIsNone(ER.role_borrow_reference(ref, 4, "1,1,0,0"))           # another shape (two hosts): not borrowed, the check stays named-skipped
        self.assertIsNone(ER.role_borrow_reference(ref, 1, "1"))


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

    def test_three_foreign_cards_take_the_same_borrow_not_the_rig_ordinals(self):
        """A2 (fixed): was 1318,0,0 (the 5090's ordinal-0 growth on a 3090); the record is read through the inventory view on every count."""
        self.assertEqual(list(self._env(3).values()), ["0,0,0"])


if __name__ == "__main__":
    unittest.main()
