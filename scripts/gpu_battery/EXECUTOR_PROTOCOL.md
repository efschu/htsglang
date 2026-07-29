# Executor-Protokoll

Du faehrst die GPU-Testbatterie ab. Du diagnostizierst nicht, du reparierst
nicht, du optimierst nicht. Du fuehrst Schritte aus, liest Verdikte und
berichtest. Alles Uebrige macht danach ein Fixer-Agent.

Dieses Protokoll gilt woertlich. Wenn eine Situation hier nicht steht, ist die
richtige Reaktion **anhalten und berichten** — nicht improvisieren.

---

## 0. Start

```bash
export BATTERY_RUN=/spinning/gpu-battery-results/$(date +%Y-%m-%d_%H%M)
export WT=/spinning/wt-gpu-battery
mkdir -p "$BATTERY_RUN"
cd "$WT/scripts/gpu_battery"
PY=/spinning/htsglang-gpu/.venv/bin/python
$PY battery_state.py --run-dir "$BATTERY_RUN" init
$PY battery_state.py --run-dir "$BATTERY_RUN" plan
```

**Wiederaufnahme statt Neustart:** Setze `BATTERY_RUN` auf das VORHANDENE
Lauf-Verzeichnis. Gruene Schritte werden dann uebersprungen, ohne dass du
etwas tun musst.

Der `plan`-Aufruf sagt dir, welche Schritte zu fahren sind. **Fahre genau
diese, in genau dieser Reihenfolge.**

## 1. Die Schleife

Fuer jeden Schritt aus dem Plan, **streng nacheinander, nie zwei gleichzeitig**:

```bash
BATTERY_RUN=$BATTERY_RUN bash run_step.sh <schritt>
```

Das Kommando gibt genau eine Verdikt-Zeile aus. Es ist die Zeile, die mit
`BATTERY-` beginnt:

| Zeile | Bedeutung | Deine Reaktion |
|---|---|---|
| `BATTERY-PASS <schritt>` | bestanden | naechster Schritt |
| `BATTERY-SKIP <schritt>: …` | war schon gruen | naechster Schritt |
| `BATTERY-FAIL <schritt>: …` | echter Testfehler | **ANHALTEN**, Abschnitt 3 |
| `BATTERY-STOP <schritt>: …` | Umgebung/Vorbedingung | **ANHALTEN**, Abschnitt 3 |
| `BATTERY-GATE <schritt>: …` | Report-Gate | **ANHALTEN**, Abschnitt 2 |

Zusaetzlich kann `BLOCKED` auftauchen: eine Voraussetzung ist nicht PASS. Das
ist ein STOP. Nicht ueberspringen.

Es gibt keine sechste Moeglichkeit. Kommt etwas anderes, ist das ein STOP.

## 2. Das Report-Gate

Nach `s02_boot_a` erscheint eine `BATTERY-GATE`-Zeile, auch bei PASS. Dann:

1. anhalten,
2. das Ergebnis melden (Verdikt, Accept-Zahlen je Prompt gegen die
   Referenzspalte, Positionskurve, MIN-freie MiB je Karte),
3. auf Freigabe warten, bevor `s03_boot_b` startet.

Boot A ist der einzige Boot, dessen Ausgang aendert, was die anderen bedeuten.
Deshalb liest jemand ihn, bevor die naechsten drei Boot-Fenster verbraucht
werden.

## 3. Bei FAIL oder STOP

In dieser Reihenfolge, ohne Abweichung:

1. **Sofort anhalten.** Kein weiterer Schritt.
2. **Nicht debuggen.** Keine Logs waelzen, keine Hypothese bilden, keine
   Datei aendern, kein Rezept anfassen, keine Flags variieren, kein zweiter
   Versuch mit anderen Werten.
3. **Nicht wiederholen** — mit genau einer Ausnahme, Abschnitt 4.
4. **Locks freigeben.** `run_step.sh` tut das selbst; kontrolliere es:
   ```bash
   ls -d /tmp/gpu-card-*.lock 2>/dev/null
   ```
   Liegt noch ein Lock von `holder=gpu_battery` mit **deiner** PID herum,
   entferne es. **Fremde Locks werden nie angefasst**, egal wie alt.
5. **Abschlussbericht schreiben** nach `HANDOFF_TEMPLATE.md`.

## 4. Die einzige erlaubte Wiederholung

Nur Schritte, die in `BATTERY.md` als **wiederholbar** markiert sind, duerfen
**genau einmal** unveraendert erneut laufen: s00, s01, s06, s07, s08, s09, s10,
s11.

Die vier Boots (s02–s05) sind **nicht wiederholbar**. Jeder Start verbraucht
ein Boot-Fenster aus einem festen Budget, und ein gescheiterter Boot ist ein
Ergebnis, kein Unfall. **s12 ist ebenfalls nicht wiederholbar**: der Schritt
kostet acht Boots, das ist keine unbeaufsichtigte Entscheidung.

Bedingungen fuer den einen Retry:

* der Schritt ist als wiederholbar markiert,
* **unveraendert** — dasselbe Kommando, dieselben Flags, dieselbe Umgebung,
* danach ist Schluss. Ein zweiter FAIL ist endgueltig.

Pruefe, ob ein Schritt wiederholbar ist:

```bash
$PY battery_state.py field <schritt> retryable    # 1 = ja, 0 = nein
```

Der Retry laeuft mit demselben Kommando wie der Erstlauf. `state.json`
protokolliert ihn als zweiten Versuch.

## 5. Was du niemals tust

* **Nie von der Schrittliste abweichen.** Keine zusaetzlichen Schritte, keine
  Umordnung, kein Ueberspringen eines FAIL.
* **Nie ein Rezept editieren.** Weder `scripts/dual_group/r7c/*`, noch
  `scripts/p2p_readiness/*`, noch die Schritt- oder Check-Skripte. Ein
  veraendertes Rezept liefert ein Ergebnis, das mit nichts vergleichbar ist.
* **Nie ein fremdes Lock brechen.** `/tmp/gpu-card-N.lock` mit fremdem
  `holder` ist tabu — abgelaufener Herzschlag hin oder her. Das ist eine
  Operator-Entscheidung.
* **Nie breit killen.** Kein `pkill python`, kein `pkill sglang`, kein
  `killall`. Auf dieser Kiste laufen fremde Server. Nur PIDs, die in
  `<schritt>/pids` stehen.
* **Nie ohne py-spy-Dump killen.** `run_step.sh` dumpt vor jedem Kill. Musst du
  ausnahmsweise selbst killen: erst
  `py-spy dump --pid <pid> > <schritt>/pyspy-<pid>.txt`, dann `kill`.
* **Nie unbegrenzt warten.** Kein `sleep` ohne Grenze, kein `wait` ohne
  Timeout, kein `curl` ohne `-m`. Ein Aufruf, der ewig blockiert, macht dich
  handlungsunfaehig, ohne dass es jemand sieht.
* **Nie einen Serverlog vollstaendig lesen oder zitieren.** Die Checks greppen
  die Dateien. Berichte Pfade und einzelne Zeilen.
* **Nie Zahlen deuten.** Du meldest, was der Check sagt, und die Werte aus den
  Artefakten. Ob eine Accept-Zahl „gut" ist, ist nicht deine Frage.
* **Nie das Zeitbudget dehnen.** Ueberschreitung ist STOP. `run_step.sh`
  erzwingt es; unterlaufe es nicht mit eigenen Hintergrundlaeufen.

## 6. Fortschritt melden

Nach jedem Schritt genau eine Zeile:

```
<schritt>: <VERDIKT> (<dauer>s) — <artefakt-verzeichnis>
```

Kein Zwischenkommentar, keine Vermutung, keine Prognose fuer den naechsten
Schritt.

### 6b. Ergebnis-Tabelle (NUR haiku-Schritte — Nutzer-Vorgabe 2026-07-29)

Gilt fuer die als haiku markierten Schritte (s00, s01, s06, s07, s08) und
**ausnahmsweise auch fuer s12** (siehe 6c — der Nutzer will dieser Messung live
zusehen). Die uebrigen sonnet-Boot-Schritte melden weiterhin nur die eine Zeile
aus 6.

Nach jedem haiku-Schritt — und auch ZWISCHEN Runden/Teilmessungen eines
Schritts, wenn dort bereits Ergebnis-Dateien im Artefakt-Verzeichnis
liegen — gib zusaetzlich zur Verdikt-Zeile eine kompakte Markdown-Tabelle
der Ergebnisse aus, damit der Nutzer live zusehen kann. Regeln:

- NUR wenn Ergebnisse existieren (results-JSON/TSV im Schritt-Verzeichnis).
  Keine Ergebnisse -> keine Tabelle, keine Platzhalter-Tabelle.
- Inhalt: die numerischen Kennzahlen der Ergebnis-Dateien des Schritts,
  eine Zeile je Messpunkt/Paar, Spalten = Feldnamen wie sie in der Datei
  stehen (nichts umbenennen, nichts umrechnen). Bei Leitern die Groesse
  als erste Spalte. Maximal ~20 Zeilen; mehr Punkte -> die ersten und
  letzten Zeilen plus eine Zeile "... (N weitere, siehe <datei>)".
- REINE DARSTELLUNG: keine Bewertung, keine Auffaelligkeits-Markierung,
  keine Interpretation, kein Vergleich mit Erwartungen — das Urteil kommt
  ausschliesslich vom Check-Skript.
- Quelle sind die persistierten Dateien, NIEMALS Server-/Skript-Logs in
  den Kontext ziehen, um Zahlen zu extrahieren.
- Die Tabelle ersetzt nichts: Verdikt-Zeile (6), Checks und Abbruchregeln
  gelten unveraendert.

### 6c. Live-Tabelle bei s12 (der einzige Hintergrundschritt)

s12 laeuft ueber eine Stunde und schreibt nach **jedem** Sessionzahl-Paar
`zwischentabelle.md` neu. Damit der Nutzer live mitliest, ist s12 der eine
Schritt, den du im Hintergrund startest und dessen Tabelle du pollst:

```bash
BATTERY_RUN=$BATTERY_RUN bash run_step.sh s12    # im Hintergrund starten
# und dann, mit Abstand (z. B. alle 3-5 min), NUR diese Datei lesen:
cat "$BATTERY_RUN/s12_prefill_kurve/zwischentabelle.md"
```

Regeln dabei, alle unveraendert aus 6b: nur ausgeben, wenn die Datei existiert
und Zeilen hat; reine Darstellung, keine Bewertung, kein Vergleich mit einer
Erwartung; Quelle ist ausschliesslich diese Datei, **niemals** ein Serverlog.
Ob die Kurve flach bleibt oder steigt, sagst du **nicht** — das Verdikt kommt
vom Check, die Deutung vom Leser.

Auch hier gilt: nie zwei Schritte gleichzeitig, und nie unbegrenzt warten. Das
harte Budget aus der Schritt-Tabelle laeuft im Hintergrundlauf weiter.

## 6d. Die drei Host-Schritte (s10-s12)

Diese Schritte steuern den PVE-Host ueber ssh. Was das fuer dich aendert:

* **Zwei Lock-Namensraeume.** Du pruefst am Ende **beide**:
  ```bash
  ls -d /tmp/gpu-card-*.lock 2>/dev/null || echo "Container: keine Locks"
  ssh -i /root/.ssh/id_root@proxmox -o BatchMode=yes -o ConnectTimeout=10 \
      root@192.168.0.1 'ls -d /tmp/gpu-card-*.lock 2>/dev/null' \
      || echo "Host: keine Locks"
  ```
  Fremde Locks werden auf **keiner** Seite gebrochen.
* **Jedes ssh mit Frist.** Kein `ssh` ohne `timeout`/`ConnectTimeout`, kein
  `curl` ohne `-m`. Die Skripte halten sich daran; wenn du selbst eine Zeile
  absetzt, auch.
* **py-spy laeuft auf dem HOST**, vor jedem Kill:
  ```bash
  ssh -i /root/.ssh/id_root@proxmox -o BatchMode=yes root@192.168.0.1 \
    '/spinning/subvol-999-disk-0/spinning/htsglang-gpu/.venv/bin/py-spy dump --pid <pid>'
  ```
  Die PIDs, die ein Schritt gestartet hat, stehen in
  `<schritt>/host_pids`. Nur diese, nie ein Muster.
* **Logs bleiben auf dem Host** (`/root/battery-bar1/`). Die Checks lesen das
  Grep-Ergebnis und einen begrenzten Tail aus dem Lauf-Verzeichnis. Zieh ein
  Serverlog **nicht** in deinen Kontext, auch nicht „nur kurz".
* **Betrachter beenden ist eine Nutzer-Entscheidung.** Meldet s10
  `BATTERY-STOP ... Prozesse halten die Module`, ist das Ende des Schritts. Du
  setzt `BAR1_VIEWER_KILL_OK=1` **nur** auf ausdrueckliche Freigabe, und die
  gilt nur fuer diesen Lauf.
* **s10 stellt den Treiber nicht zurueck.** Das ist Absicht. `s10_restore.sh`
  faehrst du nur, wenn es dir ausdruecklich gesagt wird.

## 7. Abschluss

Wenn alle geplanten Schritte PASS oder SKIP sind:

```bash
$PY battery_state.py --run-dir "$BATTERY_RUN" status
ls -d /tmp/gpu-card-*.lock 2>/dev/null || echo "keine Locks offen"
```

Dann den Abschlussbericht schreiben: die Statustabelle, das Lauf-Verzeichnis
und je Schritt Dauer und Artefaktpfad. Bei einem FAIL oder STOP stattdessen
das vollstaendige `HANDOFF_TEMPLATE.md`.
