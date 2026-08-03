# Herkunft dieser Fixtures

Echte Serverzeilen des s12-Laufs vom 2026-07-30, nicht nachgebaut.

* Lauf: `gpu-battery-results/2026-07-30_bar1/s12_prefill_kurve`
  (Qwen3.6-27B-FP8, tp3 `--rank-gpu-id 0,1,2 --rank-tp-ratio auto-performance`,
  `--kv-cache-dtype fp8_e4m3`, NEXTN k=3, `chunked_prefill_size=2048`,
  `--max-running-requests 16`, PVE-Host 192.168.0.1, Port 30030).
* Serverlogs auf dem Host: `/root/battery-bar1/s12.<arm>.<sessions>.log`.
* `<arm>_<sessions>.log` ist daraus die Auswahl
  `grep -E "Prefill rank batch|Decode batch|barlink-BAR1: Aufbau"`, die
  Aufbau-Zeilen auf die ersten drei (ein Rang je Zeile) gekuerzt. Sonst
  unveraendert, inklusive Zeitstempel und Rangpraefix.
* `punkte.jsonl` sind die Punkte (`folge`) 1, 2, 5 und 6 aus
  `gpu-battery-results/2026-07-30_bar1/s12_prefill_kurve/punkte.jsonl`, ohne
  `output_sample` (Modellausgabe gehoert nicht in eine Parser-Fixture).
* Vier Punkte, nicht acht: bar1 und Grundlinie bei 1 und bei 8 Sessions sind
  genau die vier, auf denen die Kernaussage von #293 Schritt 1 steht.

Die Zahlen, die `test_s12_log_analyse.py` gegen diese Dateien prueft, sind
dieselben, die im Abschnitt "#293 Schritt 1" von
`docs/dev/INTEGRATION_R3_VALIDATION.md` stehen. Wer eine davon aendert, muss
die andere mitaendern -- das ist der Zweck der Kopplung.

## Task #315 addendum (2026-07-30)

The capture predates the #295 German-to-English translation of
`barlink_bar1.py`. The three `barlink-BAR1: Aufbau in ...` lines at the top of
`bar1_1.log` and `bar1_8.log` were mechanically retranslated in place to the
current `barlink-BAR1: setup in ...` wording via
`test/registered/unit/distributed/_bar1_marker_source.py`, with every
captured number (ms, MiB, KiB, byte count, rank, timestamp) kept unchanged.
`grep -E "Prefill rank batch|Decode batch|barlink-BAR1: Aufbau"` above is the
capture command as it was actually run on 2026-07-30; a re-capture today
would grep for `"barlink-BAR1: setup"` instead. The `Prefill rank batch` /
`Decode batch` lines are untouched.

## Task #358 addendum (2026-07-31)

The transport formerly called HTCCL is called barlink now. The log lines in
`bar1_1.log` / `bar1_8.log` and the grep expressions quoted above carry the
new vocabulary for the same reason as the #315 addendum: the parser under
test reads the emitter's current format strings. No captured number changed.

## Task #525 addendum (2026-08-03)

`punkte.jsonl` itself was never actually committed: `.gitignore`'s blanket
`*.jsonl` rule (still there for the `gpu-battery-results/` run dumps) silently
ate `git add` on this file too, so the four tests that read it
(`TestBefund::test_bar1_wait_grows_about_twice_as_fast_as_the_baseline`,
`::test_batches_overlap_from_four_sessions_on`, `TestBericht::
test_the_tables_render_and_do_not_judge`,
`TestHarnessLuecke::test_the_run_recorded_no_accept_at_all`) ran for months
against `lade_punkte() == {}` -- `auswerten()`'s `points.get((arm, sessions))
or {}` fallback then windows every large batch in the log INCLUDING the
warmup instead of the last `requests` of the point, which is a different
(and wrong) set of numbers, not a crash. Restored byte-for-byte (`decode[].
output_sample` stripped only) from
`/spinning/gpu-battery-results/2026-07-30_bar1/s12_prefill_kurve/punkte.jsonl`
`folge` 1/2/5/6 -- the exact four points this file already claimed to carry --
and `.gitignore` now carries a narrow `!`-exception for this one fixture path
so it cannot be re-swallowed by the blanket rule.
