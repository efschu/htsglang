# SPDX-License-Identifier: Apache-2.0
"""#242b: the P-card reference of the multi-sequence form (H91) measured from
the H55 chunk windows, not from the GRAPH-POOL high-water marks.

THE BEFUND (28.09., boots rc12z14..rc12z19 of the H91 Dauerform). The reference
took its growth from ``last = max(samples, key=(chunk_index, peak))`` minus the
chunk-0 point. A GRAPH-POOL line is a high-water mark, its chunk index is the
first request's start, and under H91 one 16k chunk carries two to five
sequences. Boot dkrnfh91dprsavisbar1dauer09280831 at 08:45:23: a bs4 chunk at
chunk index 1, peak +1942 MiB over the chunk-0 point -> PP0 growth 1942 MiB per
16k (Kopfraum -10646 at 262144 tokens), which is the multi-sequence TRANSIENT
(P_ACTIVATION_MIB carries it), not growth with depth.

THE MEASUREMENT. ``allocated`` after a chunk (the base, not the peak) grows with
depth and saturates: PP0 +1211, PP1 +1056, PP2 +398 MiB over all load states of
today's boots (bs1 to depth 98304, bs2-4 to 96880); inside one request instance
of single-sequence chunks the steepest rise is ~330-377 MiB per 16k chunk.

THE FIX. ``p_card_reference_from_logs(windows=True)`` -- chosen by the launcher
when every log carries H55 windows and ``#969N ADMIT`` -- takes the chunk-0
point only from a single-sequence chunk, normalised with ITS measured transient;
the growth RATE from single-sequence chunks of one request instance (a new
start at 0 is a new instance, rids repeat), the growth CAP from the base over
all windows of any load, pooled over all boots. ``PCardReference.growth_cap_mib``
caps rate x chunks at the measured maximum; empty for the older references, which
stay byte-identical.

Hermetic: no GPU, no torch device.
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest

from sglang.srt.planner import p_card_chunk as pc

FIX = os.path.join(os.path.dirname(__file__), "fixtures")
BOOT = "dkrnfh91dprsavisbar1dauer09280831"
MODEL = "Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist"
ROW_MIB = 2.4172
CUT = (29, 11, 8)


def _text(sub, name):
    with open(os.path.join(FIX, sub, name + ".P.lines")) as fh:
        return fh.read()


def _ref(windows):
    return pc.p_card_reference_from_logs(
        [(BOOT, _text("p_card_242b", BOOT))], stage_layers=CUT, row_mib=ROW_MIB,
        support=pc.P_TRANSIENT_SUPPORT_FNFL2, model=MODEL, windows=windows,
    )


_H55 = (
    "[2026-09-28 08:00:{s:02d} PP0] WEG2-VRAM-PEAK rank=0 phase=chunk rows={rows} n=1 "
    "t0_unix_ms=1 t_unix_ms=2 window_ms=1 peak_allocated_mib={peak} peak_reserved_mib=0 "
    "start_allocated_mib=0 transient_mib=0 allocated_mib={end} reserved_mib=0 "
    "card_free_start_mib=0 card_free_mib=0 card_total_mib=0"
)


def _ext(s, reqs):
    body = ", ".join(f"('{r}', {a}, {b}, {a}, {b - a})" for r, a, b in reqs)
    return f"[2026-09-28 08:00:{s:02d} PP0] #969 EXTENT n=1 fwd=1 reqs=[{body}]"


def _adm(s, bs):
    return f"[2026-09-28 08:00:{s:02d} PP0] #969N ADMIT slot=0 fwd_ct=1 bs={bs} extend=16384 rids=[x]"


class TestTheHighWaterMarkMixesTheLoadIntoTheGrowth:
    def test_the_old_way_reads_a_bs4_transient_as_growth(self):
        old = _ref(False)
        # the befund, reproduced from the metal: 1942 MiB per 16k on PP0
        assert old.growth_mib_per_token[0] * 16384 == pytest.approx(1942.0, abs=0.5)
        assert old.growth_cap_mib == ()

    def test_the_windows_measure_the_base_rate_and_its_cap(self):
        new = _ref(True)
        rate = [g * 16384 for g in new.growth_mib_per_token]
        assert all(300.0 <= r <= 400.0 for r in rate), rate
        assert new.growth_cap_mib == (1211.0, 1050.0, 398.0)
        assert new.growth_measured_chunks == (4, 4, 2)
        assert new.seats == 6 and new.mamba_slots == 32

    def test_the_chunk0_point_is_a_single_sequence_chunk_with_its_own_transient(self):
        old, new = _ref(False), _ref(True)
        # the old way normalised with the builtin support point (3697 on PP0);
        # the window's own transient at the chunk-0 point is smaller
        assert new.headroom0_mib[0] < old.headroom0_mib[0]
        assert new.headroom_mib == old.headroom_mib


class TestInstancesAndPooling:
    def test_a_restart_at_zero_is_a_new_instance_of_the_same_rid(self):
        text = "\n".join([
            _adm(0, 1), _ext(0, [("r", 0, 16384)]),
            _H55.format(s=1, rows=16384, peak=1000, end=100),
            _adm(2, 1), _ext(2, [("r", 16384, 32768)]),
            _H55.format(s=3, rows=16384, peak=1400, end=400),
            _adm(4, 2), _ext(4, [("r", 32768, 36000), ("r", 0, 13152)]),
            _H55.format(s=5, rows=16384, peak=1900, end=700),
            _adm(6, 1), _ext(6, [("r", 13152, 29536)]),
            _H55.format(s=7, rows=16384, peak=1950, end=900),
        ])
        ws = pc.seq_windows(text, 16384)[0]
        assert [w.inst for w in ws] == ["r#0", "r#0", "", "r#1"]
        assert [w.persist_mib for w in ws] == [0.0, 300.0, 600.0, 800.0]
        # the instance r#1 has no single-sequence chunk 0: it gives no rate,
        # but its base (800) is the cap
        rate, sat, top = pc.window_growth(ws, 16384)
        assert rate * 16384 == pytest.approx(300.0)
        assert top == 800.0 and sat == 3

    def test_a_boot_that_saw_only_chunk_1_does_not_cut_the_saturation(self):
        def boot(rise1, rise2):
            lines = [_adm(0, 1), _ext(0, [("r", 0, 16384)]),
                     _H55.format(s=1, rows=16384, peak=1000, end=100),
                     _adm(2, 1), _ext(2, [("r", 16384, 32768)]),
                     _H55.format(s=3, rows=16384, peak=1000, end=100 + rise1)]
            if rise2 is not None:
                lines += [_adm(4, 1), _ext(4, [("r", 32768, 49152)]),
                          _H55.format(s=5, rows=16384, peak=1000, end=100 + rise2)]
            return "\n".join(lines)

        pooled = []
        for i, text in enumerate((boot(360, None), boot(340, 690))):
            for w in pc.seq_windows(text, 16384)[0]:
                pooled.append(type(w)(**{**{f: getattr(w, f) for f in w.__struct_fields__},
                                         "inst": f"b{i}/{w.inst}"}))
        rate, sat, top = pc.window_growth(pooled, 16384)
        assert rate * 16384 == pytest.approx(360.0)
        assert top == 690.0 and sat == 2

    def test_only_logs_of_the_multi_sequence_form_take_the_window_way(self):
        assert pc.logs_carry_seq_windows([(BOOT, _text("p_card_242b", BOOT))])
        old = [(b, _text("p_card_h41", "fnFL2" + b)) for b in ("x163", "x164", "x165")]
        assert not pc.logs_carry_seq_windows(old)
        assert not pc.logs_carry_seq_windows([])


class TestTheCapInTheSolve:
    def _solve(self, reference):
        return pc.solve_p_card(
            reference=reference, support=pc.P_TRANSIENT_SUPPORT_FNFL2,
            fractions=[0.324, 0.637, 0.733887], lru_rows=[32, 32, 32], stage_layers=list(CUT),
            chunk=16384, kv_mib=[1904.0, 816.0, 544.0], num_experts=512, row_mib=ROW_MIB,
            draft_on_p=False, draft_mib_last_stage=0.0, cards=["nvml1", "nvml0", "nvml2"],
            near_oom_mib=400.0, prompt_tokens=262144, lmem_fixed=True, seats=6,
            mamba_slots=32, mamba_mib_per_slot=(1.5602 * 22, 1.5602 * 8, 1.5602 * 6),
        )

    def test_growth_at_262144_is_the_measured_maximum_not_rate_times_chunks(self):
        ref = _ref(True)
        fits = self._solve(ref)
        assert [f.growth_mib for f in fits] == [1211.0, 1050.0, 398.0]
        uncapped = self._solve(pc.msgspec.structs.replace(ref, growth_cap_mib=()))
        assert all(u.growth_mib > f.growth_mib for u, f in zip(uncapped, fits))

    def test_the_shipped_reference_has_no_cap_and_solves_as_before(self):
        assert pc.P_CARD_REFERENCE_FNFL2.growth_cap_mib == ()
        fits = self._solve(pc.P_CARD_REFERENCE_FNFL2)
        g = pc.P_CARD_REFERENCE_FNFL2.growth_mib_per_token
        sat = pc.P_CARD_REFERENCE_FNFL2.growth_measured_chunks
        assert [f.growth_mib for f in fits] == pytest.approx(
            [g[s] * sat[s] * 16384 for s in range(3)])
