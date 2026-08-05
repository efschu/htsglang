# Window 10 -- per-arm results

Record reference (#424 int8_decode, barlink BAR1, pin 1960957e3b):
  s14 bs=1 tick 126.8 / client 120.5 tok/s, ms/Verify 30.37, accept 3.85
  s14 bs=1 A-vs-A floor IN ITS OWN BOOT: 104.46 / 118.65 / 106.07 (33.9 %)
  bench.sh narrative decode_TPS 86.46, code 112.18, PP 1638.99
  max_total_num_tokens 431360, ctx 131072, mrr 16

| arm | tree | config | s14 tick | s14 client | ms/Verify | accept | floor spread | bench narr | bench code | PP | tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|
| arm1_pin_record_bl | pin 84fff442e1 | record | 106.8 | 94.8 | 37.27 | 3.98 | 87.0-107.7 (23.8 %) | 62.86 (CV 26.1 %) | 93.14 (CV 1.6 %) | 1302 | 431360 |
| arm2_int_record_bl | int 548f4cee5c | record | 99.9 | 95.1 | 38.84 | 3.88 | 70.4-94.1 (33.6 %) | 67.81 (CV 1.7 %) | 90.85 (CV 1.1 %) | 1289 | 350400 |
| arm0_int_today_every32 | int 548f4cee5c | today | 82.3 | 83.4 | 38.89 | 3.20 | 93.6-101.7 (8.6 %) | 67.42 (CV 2.5 %) | 88.13 (CV 1.9 %) | 1290 | 602944 |
| arm0_int_today_517on | int 548f4cee5c | today | 102.9 | 102.0 | 38.88 | 4.00 | 91.0-94.1 (3.4 %) | 68.51 (CV 4.0 %) | 88.58 (CV 0.9 %) | 1291 | 602944 |
| arm0_int_today_wdoff | int 548f4cee5c | today | 72.3 | 74.3 | 44.59 | 3.22 | 64.0-84.2 (31.6 %) | 58.92 (CV 2.2 %) | 79.33 (CV 1.2 %) | 1298 | 602944 |

## py-spy leaf census (20 samples of the TP0 scheduler thread under a bs=1 load)

* **arm1_pin_record_bl** (barlink dev, gate registry EMPTY)
    * 12/20 synchronize (streams.py:108)
    * 2/20 common_template (flashinfer_backend.py:7395)
    * 1/20 set_last_decode_finish_time (req_time_stats.py:830)
* **arm2_int_record_bl** (barlink dev, #517 on)
    * 14/20 synchronize (streams.py:108)
    * 3/20 build_dcp_weighted_kv_indices (owner.py:529)
    * 1/20 plan (prefill.py:1993)
* **arm0_int_today_every32** (barlink dev, EVERY=32)
    * 11/20 synchronize (streams.py:108)
    * 1/20 prepare_for_draft (base_spec_worker.py:352)
    * 1/20 run (jit.py:743)
* **arm0_int_today_517on** (barlink dev, #517 on)
    * 13/20 synchronize (streams.py:108)
    * 3/20 replay (graphs.py:139)
    * 2/20 common_template (flashinfer_backend.py:7429)
* **arm0_int_today_wdoff** (barlink dev, watchdog OFF)
    * 14/20 check_aborted (barlink_device.py:1552)
    * 2/20 replay (graphs.py:139)
    * 1/20 foreach_copy (buffers.py:47)
