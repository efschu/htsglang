"""Letzte Boots: tok/s of a finished boot out of VictoriaMetrics (Nutzer 02.10.: the table lost prefill and decode
once a boot fell out of the 16-min ring)."""

from rigdash import vmpush


def _iv(e, tok, dur, steady, seat_s=None, busy=None):
    return {"e": e, "tok": tok, "dur": dur, "steady": steady, "seat_s": seat_s, "busy": busy if busy is not None else dur}


def test_decode_sums_add_counts_each_settled_steady_interval_once():
    s = vmpush.decode_sums_empty()
    ivs = [_iv(10, 100, 1.0, False), _iv(11, 120, 1.0, True, seat_s=3.0), _iv(12, 120, 1.0, True, seat_s=3.0),
           _iv(13, 50, 1.0, False)]
    s = vmpush.decode_sums_add(s, ivs, settled_before=12.5)        # 13 not settled yet
    assert s["tok"] == 240 and s["dur"] == 2.0 and s["seat_s"] == 6.0 and s["busy"] == 2.0 and s["last_e"] == 12
    s = vmpush.decode_sums_add(s, ivs, settled_before=20.0)        # same ring again: nothing twice
    assert s["tok"] == 240 and s["last_e"] == 13


def test_decode_sums_seeded_from_vm_continue_after_restart():
    seed = dict(vmpush.decode_sums_empty(), tok=1000.0, dur=10.0, last_e=50.0)
    s = vmpush.decode_sums_add(seed, [_iv(49, 99, 1.0, True), _iv(51, 100, 1.0, True)], settled_before=60.0)
    assert s["tok"] == 1100.0 and s["dur"] == 11.0


def test_boot_rates_from_prefill_slowest_rank_and_decode_sums():
    ser = {("weg2_rank_prefill_new_tokens_total", "P", "tp0pp0"): [(1, 1000.0), (2, 216156.0)],
           ("weg2_rank_prefill_compute_ms_total", "P", "tp0pp0"): [(2, 58395.4)],
           ("weg2_rank_prefill_new_tokens_total", "P", "tp0pp1"): [(2, 216156.0)],
           ("weg2_rank_prefill_compute_ms_total", "P", "tp0pp1"): [(2, 38442.0)],
           ("weg2_rank_prefill_new_tokens_total", "D", "tp0pp0"): [(2, 500.0)],       # below the noise floor
           ("weg2_rank_prefill_compute_ms_total", "D", "tp0pp0"): [(2, 900.0)],
           ("weg2_boot_decode_tokens_total", "", ""): [(1, 30000.0), (2, 65000.0)],
           ("weg2_boot_decode_seconds_total", "", ""): [(2, 544.0)],
           ("weg2_boot_decode_seat_seconds_total", "", ""): [(2, 1500.0)],
           ("weg2_boot_decode_busy_seconds_total", "", ""): [(2, 500.0)]}
    r = vmpush.boot_rates_from(ser)
    assert r["prefill"]["P"]["rank"] == "tp0pp0" and round(r["prefill"]["P"]["tps"]) == 3702
    assert "D" not in r["prefill"]
    assert round(r["decode"]["gen_tps_boot"], 1) == 119.5 and r["decode"]["seats_boot"] == 3.0


def test_boot_rates_from_without_decode_sums_says_none():
    assert vmpush.boot_rates_from({})["decode"] is None


def test_decode_sum_lines_carry_model_and_boot():
    lines = vmpush.decode_sum_lines(dict(vmpush.decode_sums_empty(), tok=5.0), "NF", "boot-x", 1000)
    assert lines[0] == 'weg2_boot_decode_tokens_total{boot="boot-x",model="NF"} 5.0 1000'
