# Herkunft dieser Fixtures

Echte Serverzeilen des s12-Laufs vom 2026-07-30, nicht nachgebaut.

* Lauf: `gpu-battery-results/2026-07-30_bar1/s12_prefill_kurve`
  (Qwen3.6-27B-FP8, tp3 `--rank-gpu-id 0,1,2 --rank-tp-ratio auto-performance`,
  `--kv-cache-dtype fp8_e4m3`, NEXTN k=3, `chunked_prefill_size=2048`,
  `--max-running-requests 16`, PVE-Host 192.168.0.1, Port 30030).
* Serverlogs auf dem Host: `/root/battery-bar1/s12.<arm>.<sessions>.log`.
* `<arm>_<sessions>.log` ist daraus die Auswahl
  `grep -E "Prefill rank batch|Decode batch|HTCCL-BAR1: Aufbau"`, die
  Aufbau-Zeilen auf die ersten drei (ein Rang je Zeile) gekuerzt. Sonst
  unveraendert, inklusive Zeitstempel und Rangpraefix.
* `punkte.jsonl` sind die Punkte 1, 5 und 6 desselben Laufs aus
  `punkte.jsonl`, ohne `output_sample` (Modellausgabe gehoert nicht in eine
  Parser-Fixture).
* Vier Punkte, nicht acht: bar1 und Grundlinie bei 1 und bei 8 Sessions sind
  genau die vier, auf denen die Kernaussage von #293 Schritt 1 steht.

Die Zahlen, die `test_s12_log_analyse.py` gegen diese Dateien prueft, sind
dieselben, die im Abschnitt "#293 Schritt 1" von
`docs/dev/INTEGRATION_R3_VALIDATION.md` stehen. Wer eine davon aendert, muss
die andere mitaendern -- das ist der Zweck der Kopplung.
