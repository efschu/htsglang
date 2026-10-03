"""#239 S0 im Planer: ein Form-A-Worker bucht keine QSA-Schluessel.

Die Referenzen (fnFL2x98-100, x151/x158) sind vor S0 (e56d78a4ab) gemessen
und bepreisen einen Worker mit 768 B/Token (12 x 64 B QSA-Index). Seit S0
baut der Worker keinen Index; rc12z30w (0929_082743, Schnitt [0,20,44]) legte
auf TP1/TP2 genau EINEN Pool an: 'KV pool sizing: cell_size=12288',
'#tokens: 81925' bzw. '180235'. Der Planer buchte 262144 x 768 B = 192 MiB je
Worker darueber -- Platz, der den Expertenzeilen gehoert
(freier-kv-platz-gehoert-experten-0929).
"""

import unittest

import msgspec

from sglang.srt.planner import expert_residency as er

MIB = 1024 * 1024
FA_CELL = 12288  # 12 FA-Layer x K+V x 2 kv-Koepfe x 256 x fp8
TOKENS = 262144


class ReferenceCellsS0(unittest.TestCase):
    def test_form_a_workers_cost_nothing_the_host_keeps_its_measurement(self):
        for ref in (er.D_RESIDENCY_REFERENCE_FNFL2, er.D_RESIDENCY_REFERENCE_FNFL2_H39):
            self.assertEqual(ref.kv_cell_bytes, (14143, 768, 768))
            self.assertEqual(er.reference_kv_cells_s0(ref), (14143, 0, 0))

    def test_a_reference_without_one_attention_host_is_unchanged(self):
        ref = msgspec.structs.replace(er.D_RESIDENCY_REFERENCE_FNFL2_H39, rank_tp_ratio="1,1,1")
        self.assertEqual(er.reference_kv_cells_s0(ref), (14143, 768, 768))


class CutCellsMatchTheMetal(unittest.TestCase):
    def test_worker_cell_is_its_fa_share_only(self):
        # rc12z30w: Schnitt [0,20,44]/64 -> TP1 81925, TP2 180235 Token a 12288 B.
        cells = er.kv_token_cut_cells(er.D_RESIDENCY_REFERENCE_FNFL2_H39, (0, 20, 44), FA_CELL)
        self.assertEqual(cells[0], 14143 - FA_CELL)
        self.assertEqual(cells[1], 20 / 64 * FA_CELL)
        self.assertEqual(cells[2], 44 / 64 * FA_CELL)
        for r, metal_tokens in ((1, 81925), (2, 180235)):
            booked_mib = TOKENS * cells[r] / MIB
            metal_mib = metal_tokens * FA_CELL / MIB
            self.assertLess(abs(booked_mib - metal_mib), 1.0, (r, booked_mib, metal_mib))


class SolveBooksNoPhantomKeys(unittest.TestCase):
    def _solve(self, **kw):
        return er.solve_d_rank_residency(
            budgets_mib=(26312.0, 17640.0, 17640.0),
            fractions=(0.127, 0.5, 0.45),
            ratios=(215.0, 121.0, 152.0),
            scratch_rows=(100, 48, 48),
            staging_rows=12,
            num_experts=512,
            pad_rows=1,
            n_layers=48,
            slot_bytes=2.417 * MIB,
            reference=er.D_RESIDENCY_REFERENCE_FNFL2_H39,
            vocab_mib=2425.0,
            share_embed=True,
            kv_tokens=TOKENS,
            **kw,
        )

    def test_form_a_without_cut_books_no_worker_kv(self):
        fits = self._solve()
        self.assertEqual([f.kv_cell_bytes for f in fits], [14143, 0, 0])
        self.assertEqual([round(f.kv_mib) for f in fits[1:]], [0, 0])

    def test_the_cut_books_the_fa_share_only(self):
        fits = self._solve(kv_token_shares=(0, 32, 32), kv_dcp_cell_bytes=FA_CELL)
        for f in fits[1:]:
            self.assertEqual(f.kv_cell_bytes, FA_CELL // 2)
            self.assertAlmostEqual(f.kv_mib, TOKENS * FA_CELL / 2 / MIB, places=3)
        # vorher 768 + 6144 = 6912 B/Token: 1728 MiB je Worker, 192 MiB davon Phantom
        self.assertEqual(round(TOKENS * 768 / MIB), 192)


if __name__ == "__main__":
    unittest.main()
