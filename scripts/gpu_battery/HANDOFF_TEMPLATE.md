# Uebergabe an den Fixer

Auszufuellen bei jedem FAIL oder STOP, bevor der Executor endet.

Das Ziel ist eine einzige Eigenschaft: **der Fixer beginnt ohne einen neuen
Lauf.** Ein Bericht, der ihn zwingt, den Boot zu wiederholen, hat das Boot-
Fenster zweimal gekostet und nichts gespart. Alles unten Aufgefuehrte ist zum
Zeitpunkt des Fehlschlags vorhanden — spaeter ist es teilweise weg.

Bei mehreren Fehlschlaegen: pro Fehlschlag ein Block.

---

## Kopf

| Feld | Wert |
|---|---|
| Lauf-Verzeichnis | `BATTERY_RUN=` |
| Worktree / Commit | `WT=` / `git -C $WT rev-parse HEAD` |
| Schritt | z. B. `s04_boot_c` |
| Verdikt | FAIL oder STOP |
| Verdikt-Zeile | die vollstaendige `BATTERY-…`-Zeile, woertlich |
| Zeitstempel Start / Ende | ISO, aus `state.json` |
| Dauer / Budget | `<dauer>s` von `<timeout_s>s` |
| Versuch | 1 oder 2 (aus `state.json`, `attempts`) |
| Treiber / torch / NCCL | aus `s00_preflight/preflight.json` |

## 1. Exaktes Kommando

Woertlich, mit gesetzter Umgebung — so, wie es lief:

```bash
BATTERY_RUN=<…> WT=<…> bash run_step.sh <schritt>
```

Abweichende Variablen (`P2P_BASELINE`, `GDR_TSV`, `SMOKE_MODEL`, `PORT`, …)
nennen. Keine gesetzt: ausdruecklich „keine".

## 2. Artefakte

Alle Pfade absolut. Die Datei, an der der Check haengengeblieben ist, zuerst.

| Datei | Bedeutung | vorhanden |
|---|---|---|
| `<run>/<schritt>/step.log` | stdout/stderr des Schrittskripts | ja/nein |
| `<run>/<schritt>/server.log` | Serverlog (bei Boot-Schritten) | ja/nein |
| `<run>/<schritt>/<ergebnis>.json` | das gepruefte Artefakt | ja/nein |
| `<run>/<schritt>/vram.csv`, `vram_summary.txt` | MIN-frei je Karte | ja/nein |
| `<run>/<schritt>/cards.txt` | zur Laufzeit aufgeloeste Kartenreihenfolge | ja/nein |
| `<run>/<schritt>/pyspy-*.txt` | Stack-Dumps (bei Haenger) | ja/nein |
| `<run>/<schritt>/check.err` | stderr des Checks | ja/nein |
| `<run>/state.json` | Verdikte, Versuche, Historie | ja |

**Fehlt eine Datei, ist ihr Fehlen der Befund** — als solchen melden, nicht als
Formfehler.

## 3. Log-Beweis: die Zeilen, nicht das Log

Nie ein ganzes Log einfuegen. Diese Greps laufen lassen und je Treffer die
Zeile mit Zeilennummer zitieren (`grep -n`, maximal fuenf Zeilen Kontext):

```bash
S=<run>/<schritt>
grep -n "CUDA out of memory\|torch.OutOfMemoryError" $S/server.log | head -5
grep -n "Traceback (most recent call last)" $S/server.log | head -5
grep -n "NCCL error\|Watchdog caught collective" $S/server.log | head -5
grep -n -iE "error|assert|refus|reject" $S/step.log | head -20
tail -40 $S/server.log
```

Zusaetzlich, je nach Schritt:

* **s04_boot_c** — der Tensorname, an dem der Drafter-Load scheiterte:
  `grep -n -iE "dflash|draft|tensor|key" $S/loader_lines.txt | head -20`
* **s01 / s06** — die NCCL-Transportwahl:
  `grep -n -E "via (P2P|SHM|NET)" $S/results/run.log $S/nccl_debug.log | head -20`
* **s07** — das `traceback`-Feld der fehlgeschlagenen Zeile in
  `offload_register_gpu.json` (steht schon im JSON, nicht neu erzeugen)
* **s08** — `errors` und `neutrality_violations` aus `dispatcher_tables.json`,
  vollstaendig; beide sind kurz und beide sind der Befund
* **s10-s12 (Host-Schritte)** — das Serverlog liegt auf dem HOST unter
  `/root/battery-bar1/`, im Lauf-Verzeichnis stehen nur Grep-Ergebnis und Tail.
  Also:
  ```bash
  cat $S/htccl_lines.txt | head -40        # s11: Aufbau, ERREICHT, Riegel
  cat $S/belege/*.txt | grep ERREICHT      # s12: Arm-Beleg je Punkt
  tail -40 $S/server.log                   # begrenzter Tail, nicht das Log
  ```
  Dazu je Schritt das eine JSON: `driver_state.json` (`missing` nennt genau,
  was fehlt), `bar1_e2e.json` (`riegel`, `gruppen`, `graph_check`),
  `prefill_kurve.json` (`abbruch`, `reihenfolge`, `grundlinie_abweichung_pct`).
  Bei einem Haenger laufen die py-spy-Dumps auf dem Host und landen als
  `$S/pyspy-host-*.txt`; die zugehoerigen PIDs stehen in `$S/host_pids`.
  **Host-Locks mit pruefen** — CT999 und Host haben getrennte `/tmp`.

## 4. Bei einem Haenger: der Stack, vor dem Kill

`run_step.sh` dumpt bei Zeitueberschreitung jeden registrierten Prozess nach
`<schritt>/pyspy-<pid>.txt`. Diese Dumps gehoeren in die Uebergabe — ein
Haenger ohne Stack muss reproduziert werden, ein gedumpter nicht.

Fehlt ein Dump, obwohl der Schritt haengte: sagen, warum (py-spy nicht
installiert, Prozess schon weg, PID nicht registriert). Das ist selbst ein
Befund ueber die Batterie.

## 5. Kartenzustand zum Zeitpunkt des Fehlschlags

```bash
nvidia-smi --query-gpu=index,name,pci.bus_id,memory.used,memory.total,utilization.gpu \
           --format=csv > <run>/<schritt>/nvidia-smi-after.csv
cat <run>/<schritt>/nvidia-smi-after.csv
ls -d /tmp/gpu-card-*.lock 2>/dev/null && cat /tmp/gpu-card-*/info 2>/dev/null
cat /spinning/gpu-arb/holder 2>/dev/null
# Bei s10-s12 zusaetzlich die HOST-Seite -- eigenes /tmp, eigene Locks:
ssh -i /root/.ssh/id_root@proxmox -o BatchMode=yes -o ConnectTimeout=10 \
    root@192.168.0.1 'ls -d /tmp/gpu-card-*.lock 2>/dev/null; \
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader'
```

Dazu aus `vram_summary.txt` die **MIN-freien MiB je Karte** ueber den ganzen
Lauf. Bei allem, was nach Speicher riecht, ist das die erste Zahl, die der
Fixer braucht — und die einzige, die nachtraeglich nicht mehr erhebbar ist.

## 6. Karten-Identitaet

Aus `s00_preflight/preflight.json` bzw. `<schritt>/cards.txt`: je Karte
NVML-Index, CUDA-Index, PCI-Adresse, UUID, Name.

Ohne diese Tabelle ist jeder Kartenindex im Bericht mehrdeutig — die CUDA- und
die NVML-Reihenfolge unterscheiden sich auf diesem Rig, und die Zuordnung kann
sich mit Treiber oder Boot verschieben.

## 7. Was der Check konkret bemaengelt hat

Die Verdikt-Zeile woertlich, dazu ein Satz, welche Bedingung im Check-Skript
gebrochen ist (Datei und Funktion, z. B.
`checks/check_common.py::check_accept_artifact`, Bedingung „Positionskurve
deckt 0..K-1 ab"). Der Fixer soll die Assertion lesen koennen, ohne sie zu
suchen.

## 8. Was NICHT gemacht wurde

Ausdruecklich auflisten, damit der Fixer nichts doppelt vermutet:

* Retry: ja/nein (und wenn ja: unveraendert?)
* Rezepte/Skripte veraendert: **nein** (falls doch: sofort melden, das
  Ergebnis ist dann nicht vergleichbar)
* Prozesse gekillt: welche PIDs, mit oder ohne Dump
* Nachgelagerte Schritte: nicht gefahren (welche)

## 9. Wiederaufnahme fuer den Fixer

Der Fixer setzt nach dem Fix genau hier wieder an — ohne die gruenen Schritte
zu wiederholen:

```bash
export BATTERY_RUN=<dasselbe Lauf-Verzeichnis>
cd <WT>/scripts/gpu_battery
/spinning/htsglang-gpu/.venv/bin/python battery_state.py plan
bash run_step.sh <der gescheiterte Schritt>
```

Gruen gewordene Schritte bleiben gruen. Besteht nach dem Fix Zweifel an einem
frueheren gruenen Schritt, wird er ausdruecklich erzwungen — nicht durch
Loeschen von Artefakten:

```bash
bash run_step.sh <schritt> --force
```

Aktuellen Stand jederzeit:

```bash
/spinning/htsglang-gpu/.venv/bin/python battery_state.py status
```
