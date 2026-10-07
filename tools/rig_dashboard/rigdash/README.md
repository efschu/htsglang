# rigdash — Pflegehinweise

## Zeitreihen-DB, Tabs, Expertenansicht (Nutzer 01.10., Konzept `/spinning/gpu-arb/docs/DASHBOARD-REDESIGN-1001.md`)

**VictoriaMetrics** (Order 01.10. ~07:40Z) ist die Zeitreihen-DB des Rigs. Sie läuft auf CT999 unter `:8428`, dazu
`node_exporter` (CT999 und Proxmox-Host 192.168.0.11:9100) und `nvidia_gpu_exporter` (NVML). Grafana läuft daneben
unter `:3000` mit der Tafel „Rig – Verlauf“. Installiert wird mit `deploy/vm/install_vm.sh` bzw.
`deploy/grafana/install_grafana.sh`, beide idempotent. Units und `scrape.yml` liegen im Repo.

- Der Probennehmer schreibt die IPC alle 5 s nach VM (`vmpush.Bridge`): `weg2_front_*`, `weg2_rank_*` und
  `weg2_flip_*`, die Flips zum Zeitpunkt des Flips. Labels sind `model`, `boot`, `group`, `rank`, `dir`, `kind`. **Nie eine rid als Label.**
- Der rigdash liest per PromQL (`vmpush.VmClient`):
  - die TTFT-Kacheln (`/api/live` → `vm`),
  - den TTFT-Verlauf (`/api/history` → `ttft`).
- Neue Größen kommen als Metrik nach VM, nicht als neue Spalte in `history.sqlite` und nie aus einem Log.
- Die Grafana-Tafel ändert man in `deploy/grafana/make_dashboard.py` und danach mit `install_grafana.sh`.
  In der Grafana-Oberfläche lässt sie sich nicht ändern (`allowUiUpdates: false`).

**`/api/live` einmal je Sekunde für alle** (`server.LiveCache`):

- `?lean=1` liefert Boots, die nur eine Tabellenzeile sind, ohne Kurven.
- `?dev=0` lässt Features und Image-Änderungen weg.
- Die Antwort geht gzip-komprimiert.
- Eine Zoom-Anfrage rechnet für sich allein.

**Seite:**

- Tabs: Überblick (Vorgabe), Boots, Verlauf, Karten, Entwicklung (nur Rig-Ausgabe).
- Gezeichnet wird nur der sichtbare Tab. Ein Abschnitt mit unverändertem HTML wird nicht angefasst (`put()`).
- Die **Expertenansicht** (`body.expert`) zeigt alles mit Klasse `x`, alle kv-Zeilen der Kacheln und die Quell-Plaketten. Die Standardansicht zeigt nur kv-Zeilen mit Klasse `keep`.
- **Regel für neue Messwerte:** Sie kommen mit `x` (bzw. ohne `keep`) in die Expertenansicht, nie ungefragt in den Standard (Memory `dashboard-redesign-ttft-tabs-expert-1001`).
- Im Überblick steht je Modell die Kennzahlzeile: TTFT, Decode, Prefill P/D, Flipzeit, Warteschlange, Tode/Hänger.

**Feature-Liste aktuell halten:** Der Tab „Entwicklung“ zeigt den Stand von `features.json`. Er ist gelb ab 12 h oder
wenn danach gebootet wurde, rot ab 24 h. Darunter stehen die **Commits der Image-Linie (48 h) ohne Baustein**,
aus git gerechnet (`features.new_commits`). Der Zähler steht im Tab-Titel. Eintragen wie unten mit `features_update.py`.

## Flipzeit (EINE Definition, Nutzer 06.10.2026)

P→D = letzter P-Chunk fertig → erstes Decode-Token erzeugt; D→P = letztes Decode-Token erzeugt → erster
Prefill-Chunk beginnt zu rechnen (erster Forward auf PP0, `flip_user_time.prefill_start_source=pp_first_forward`,
nie der Leg-1-Dispatch). Ausnahme: kein Flip zählt, wenn kein Prefill oder Decode ansteht (Leerlauf-Flip,
D→P: `flip_user_time.idle_flip`). Layer-Tausch, Vorlauf, Nachlauf sind nur die ZERLEGUNG der einen Zahl, nie eine eigene
Flipzeit. Es gibt EINE Berechnung, `flipzeit.py`: `ipcboot.flip_views` misst je Flip, die Historie schreibt jeden
gezählten Flip einmal als Marke `flip_t2t` (offen, vorläufig, ohne Endpunkt, Leerlauf: nie), und jede Zahl der Seite
(Kachel Überblick, Kachel Verlauf, Diagramm, Boot-Liste) ist `flipzeit.tile` über diese Marken. Nur das FENSTER
unterscheidet sich und steht in der Beschriftung: Überblick = letzte 60 min des Modells, Verlauf = gewählter
Bereich/Zoom, Boot-Liste = ganzer Boot. Tests: `tests/test_flipzeit_1006.py`.

**Phasenleiste mit echter Arbeitszeit (WACH-OHNE-ARBEIT-0929):** Die Rang-Zeilen kommen zu spät
(P nach dem Pipeline-Durchlauf, D einen Pass später). P-Arbeit zeichnet deshalb von
`TIMING-FLUSH-WAIT t_unix_ms` am Kopf des Forwards bis + `FWD-TIMING-PREFILL total_ms` (gepaart
über #new-token und gpu-ms; die Timing-Zeile kommt erst am Kopf des nächsten Forwards und korrigiert
dann rückwirkend), D-Extends von `HOST-ANON-PASS phase=EXTEND wall_ms` (unter 50 ms verworfen).
Ohne Anker bleibt die alte Zeichnung. Zwischen `WEG2-FLIP done woke=D` und dem ersten TP0-
Decode-Token steht **Flip-Nachlauf D** statt „wach, keine Arbeit“, aufgeteilt nach
`WEG2-POST-WAKE-PASS n=0` (Lesungen vor Pass 0, prepare = Park-Resume, run = Re-Extend).
Gilt für NF und 27B.

**TODO (27B-Review 29.09.):** auch das ist reine Anzeige aus Log-Zeilen. Umstellen auf
`events.jsonl`, sobald die Front Flip- und Pass-Ereignisse (FLIP done, erster Decode, POST-WAKE-PASS,
Forward-Start/-Ende je Rang) dort schreibt; dann entfallen FWD-/FLUSH-/ANON-Paarung und Log-Scan.

## Deploy-Linie (Order 29.09.): `desk/dashboard-ipc-0929`

Deployt wird der rigdash nur aus der Linie `desk/dashboard-ipc-0929`. Wer ihn ändert (Features-Sitz, Dashboard-Sitz, 27B), setzt darauf auf oder übernimmt die Linie ff und pusht dorthin.

`deploy/install.sh` verweigert mit rc 3, wenn eine der beiden Bedingungen nicht erfüllt ist:
1. Die Revision ist ein Nachfahre der Linienspitze auf origin.
2. Die Revision enthält das laufende Release (`/opt/rigdash/current`).

Ein bewusster Rückschritt geht nur mit `RIGDASH_DEPLOY_ROLLBACK=1`, der Grund gehört ins Entscheidungslog. Nur prüfen, ohne etwas zu ändern: `deploy/install.sh --check <rev>`. `RIGDASH_REPO=<worktree>` wählt das Repo, wenn das Skript außerhalb eines Checkouts liegt.

Anlass: Der Features-Sitz deployte von `desk/dashboard-features-0929`. Ein Deploy von dort hätte Stufe 1 von DASHBOARD-AUS-IPC still wieder entfernt.

## Arbeit zur Zeit, in der sie geschah (Nutzer 30.09. ~16:45Z: „die ganze zeit 16k/s prefill“)

`activity.py` legt jede Arbeit auf ihre echte Zeit. Ein Prefill-Chunk läuft über `prefill.last {t, gpu_ms}`
(PP0-Start bis Ende auf der letzten PP-Stufe), Decode ist Δ`decode.tokens` zwischen zwei Rang-Uhren ohne Flip-/Extend-Fenster,
Flips kommen aus den Ereignissen. Grund: Der kumulative Zähler springt um einen ganzen Chunk; Δ Zähler / Δ Probe gab 16.384 tok/s
und P/D-Überlappung. Kacheln: „Rate laufender Schub“, „beste Schub-Rate“, „Rate Schübe 60 s“ (je Tokens / Wanduhr des Schubs, P/D), Decode nur über stetige Proben, je Stream
nur mit Decode davor und danach. Verlauf: 1-s-Modellzeilen unter dem Präfix `mi.`; die alten `m.`-Zeilen aus dem falschen Instrument
werden nicht mehr gezeigt. Flipzeit: siehe Abschnitt „Flipzeit“ oben (eine Definition, `flipzeit.py`).
Audit aller Werte: `/spinning/gpu-arb/docs/DASHBOARD-PLAUSI-AUDIT-0930.md`; Tests `tests/test_activity_0930.py`.

### D-Prefill: Admit-Extends sind keine Prefill-Rate (Auftrag 880, Nutzer 03.10. „6 token/s prefill in D???“)

Auf D beginnt jeder von P übernommene Request mit einem Extend von meist **1 neuem Token** (Rest aus dem Cache). y8vb `D.log`:
3029 von 3051 `Prefill batch`-Zeilen haben `#new-token: 1`; der Stock-Wert `input throughput (token/s)` einer solchen Zeile ist
1 Token / Wanduhr seit der letzten Zeile = 5,8 tok/s. Im Dashboard stand die Zahl nicht als Stock-Wert, aber derselbe Fehler steckte in den
D-Kennzahlen: die 1-Token-Chunks hängen (Abstand < 1,5 s) zu einem „Schub“ über Minuten zusammen (Tokens / Wanduhr = wenige tok/s) und
die Tile „D-Prefill seit Boot“ teilte alle D-Tokens durch Boot-Wanduhr bzw. alle Rechenzeit. Jetzt (`activity.WIDE_MIN_TOK = 64`):

* `activity.chunks(..., kind="wide"|"admit")` trennt VOR der Schub-Bildung; `Model.dwide` / `Model.dadmit`. Die Phasenleiste behält alle Chunks.
* D-Rate (Kachel „D-Prefill“, KPI, Kurve `D_prefill_rate`, Phasen-Segment) nur aus Chunks mit mindestens 64 neuen Tokens; ohne solchen Chunk „–“.
* Getrennt benannt: „D-Admit/Extend (Tokens je Request)“ = Anzahl Extends, Tokens, Ø Tokens je Request (`prefill.D.admit`, Segment `admit_n/admit_tok`).
* „D-Prefill + Admit seit Boot“: Summe aller D-Tokens bleibt, die Raten (GPU-Zeit und Wandzeit) gelten für die Chunks ≥ 64 Tokens im 16-min-Ring
  (`totals.d_split`), weil die kumulativen Zähler keine Chunk-Breite vor dem Ring tragen.
* Log-Pfad (`live.py`): `wall_confounded_tps` und die D-Raten nehmen Zeilen mit < 64 Tokens nicht mehr auf, `prefill.D.now.admit` zählt sie.
* Grenze: Die Trennung geschieht je 1-s-Probenschritt (Ø neue Tokens je Chunk im Schritt); ein Admit im selben Schritt wie ein breiter Chunk zählt mit dem breiten (±1 Token).
* P ist nicht betroffen (Chunks von 1–16k Tokens, jede Zeile zählt).

## Glatt und ehrlich, Sitze, Zoom (Nutzer 30.09. ~21:05Z / ~21:10Z)

„der decode durchsatz … nicht durchgehend sondern extrem sprunghaft“. Gemessen am Dienst (NF y5i): zwei Ursachen.
1. **Probenschlupf.** Der 1-s-Sampler läuft unter Last länger als 1 s. Etwa 28 % der 1-s-Zeilen hatten keine eigene Probe
   und wurden als Lücke geschrieben. Jetzt zählt eine Sekunde als beobachtet, wenn sie zwischen zwei Proben liegt, die höchstens
   3 s auseinander sind (`Model.coverage`). Die Arbeit darin legen die Rang-Uhren fest. Pegel (KV) halten die letzte Probe,
   NVML-Reihen im Verlauf höchstens 2 s. Die Summe der Leistung gilt nur über alle Karten, nie über einen Teil davon.
2. **Wanduhr-Nenner.** Ein 5-s-Eimer mit 2 s D-Extend zeigte den halben Decode-Durchsatz. Gezeichnet wird jetzt die Rate
   *während* die Phase rechnete (`*_rate` = Tokens / `*_busy`). Decode-Zeit: stetige Proben voll; Anfang und Ende einer Strecke
   nach Δ`decode.gpu_ms` × gemessenes Wand/GPU-Verhältnis. Ein Eimer ohne diese Arbeit ist eine Lücke. Die Wanduhr-Rate bleibt
   für tok/s/W, Energie und Tokenzählung und liegt im Verlauf als ausgeblendete Reihe bei.
   Im Verlauf stehen die Anteile (`dec_busy`, `p_busy`, `d_busy`, `dec_seat`) je 1-s-Zeile. Geteilt wird erst je gezeigtem
   Eimer, deshalb stimmt es auf jeder Stufe. `dec_bs_min/_max` fassen per MIN/MAX zusammen.

**Sitze** (Nachtrag 21:10Z): Batchgröße der Decode-Runden aus Δ`decode.gpu_ms_by_bs`, nach Rundenzeit gewichtet (sonst `decode.running`).
Sitze werden nur über die Decode-Zeit gemittelt, Schlaf/Flip/Extend zählen also nicht als 0 Sitze. Je Stream = Tokens / Sitz-Sekunden,
also gilt Gesamt = je Stream × Ø Sitze exakt.

**Zoom** (`static/zoom.js`, ohne Bibliothek). Ziehen in einer Zeitgrafik (Verlauf, Karten-Verlauf, Kurven der Boot-Karte,
Phasenleiste) zoomt alle Grafiken auf diesen Bereich. Zurück geht per Doppelklick, per Knopf „Zoom zurück“/„ganz heraus“ in der
Zoom-Leiste oder per Esc. Auf einer fokussierten Grafik: + / − / ← / → / 0. Touch: waagerecht ziehen.
Der Verlauf lädt `api/history?from=&to=` im passenden Raster neu (ab ≤ 24 min 1 s). Die Boot-Karte lädt `api/live?zoom=t0,t1`
mit 1/2/5-s-Eimern (`series_zoom`); die 60-s-Kacheln bleiben auf „jetzt“. Die Kartenbalken oben sind Momentwerte ohne Zeitachse.
Tests: `tests/test_rates_glatt_0930.py`.

## Tiefe beim Hover: „Token x–y (n neu)“ (Nutzer 02.10. ~12:04Z / ~12:15Z)

Der Prefill-Durchsatz fällt mit der Kontexttiefe, darum zeigt jedes Prefill-Segment der Phasenleiste (und der letzte Schub
der Prefill-Kachel, die P→D-Flipzeile „nach Prefill …“) x = Start-Tiefe (Präfix), y = End-Tiefe, n = neu gerechnete Token,
dazu tok/s über die ersten und letzten 15 % der Token (`activity.prefill_depth`). Quellen, beste zuerst: rankstats
`prefill.last.ext` [[rid, start, end]] je Chunk; events `request_done` `prefill.{P,D}` (rid-genau, aber erst am Anfrageende);
sonst Δ`prefill.cached_tokens` des Schubs (#cached-token steht nur am ersten Chunk einer Anfrage). Ein Decode-Segment zeigt
tok/s je Batchgröße (Proben mit nur einer bs, Δ`gpu_ms_by_bs`) und je Anfrage Token x–y (n neu) mit tok/s: aus rankstats
`decode.reqs` [[rid, prompt, out]] je Probe, sonst geschätzt aus `request_done` (Ø der Anfrage, linear). `prefill.last.ext`
und `decode.reqs` baut der Port-Sitz (02.10.); bis dahin greifen die Rückfälle. Texte DE/EN im Block `DEPTH-BEGIN` von
`static/index.html`. Tests: `tests/test_depth_1002.py`.

## Probennehmer im eigenen Prozess, Zähler statt Momentproben (Nutzer 30.09. ~21:40Z)

„der probenehmer sollte doch nicht an zu viel last scheitern? der sollte das doch irgendwie parallel davon tun können?“
Die Lesungen liefen bisher als Threads im Webserver-Prozess. Unter 5 dauerabfragenden `/api/live`-Clients schrieb
der NVML-Thread 287 s lang keine Zeile (GIL).
- Mit `--state-dir` startet der Webserver `python -m rigdash.sampler` als Kindprozess (`sampler.Supervisor`) und
  überwacht ihn. Der Sampler liest den IPC-Ring, NVML, den Host und den Verlauf; er schreibt `history.sqlite` und `ring.sqlite`.
  Der Webserver liest nur (`IpcBoots(role="reader")`). Ist der Sampler tot oder still, zeigt die Seite ein rotes Banner,
  und `/api/health` meldet `sampler.ok=false`. Der Supervisor startet ihn neu. Stirbt der Webserver, beendet sich der
  Sampler selbst (Eltern-PID). Die systemd-Unit bleibt dieselbe, deshalb gibt ein Rollback nie einen zweiten Schreiber.
- Zähler statt Momentproben: Das Δ der Rang-Zähler zwischen zwei Lesungen gilt über die Rang-Uhr (`activity.coverage`,
  bis 30 s). Die Leistung kommt aus `nvmlDeviceGetTotalEnergyConsumption`: Energie-Δ auf die überdeckten Sekunden
  (`history.spread_counter`). Die Host-CPU kommt aus dem /proc/stat-Δ über alle Sekunden seit der letzten Lesung.
  Eine verspätete Lesung verliert damit nichts.
- Pegel ohne Zähler (Temperatur, Takt, Last, Speicher, KV) werden bei einer übersprungenen Sekunde gehalten und
  gezählt: `held` in `/api/health` (Ziel 0). Das `_hold` der Ansicht ist nur Notnagel und wird als `view_filled` gezählt.
Tests: `tests/test_sampler_prozess_0930.py`, `tests/test_rates_glatt_0930.py::TestCounterBooking`.

Folgeauftrag 30.09. ~22Z: Der Sampler liest auch `sources.py` (Karten, PCIe, docker, gpuq, Fronts) und die
Rang-Dateien für die Felder, und er führt die Energie-Schleife. Die Karten kommen jetzt per NVML im selben Prozess
statt per `nvidia-smi`-Fork, jede Sekunde. `power.draw` ist das Δ des Energie-Zählers über das Intervall.
Der Webserver liest nur `ring.sqlite` (`SourcesReader`, `EnergyReader`, Rang-Tabelle) und `history.sqlite`;
selbst misst er nichts mehr.

## Nur IPC, kein Boot-Log (Nutzer 29.09. über 27B; Rüge und Order 30.09.)

„das dashboard soll auch aus der inter prozess kommunikation gespeist werden, nicht aus logs“ -- seit 30.09. ohne Ausnahme:

- `ipcboot.py` (`IpcBoots`) ersetzt `live.LiveLogs` als Quelle der Boot-Karten. Jede Sekunde eine Probe je Boot-Zustandsordner
  `/spinning/docker-acceptance/<line>/state/<boot_id>/` (`state.json`, `events.jsonl`, `rankstate/<G>/*.rankstats`) in einen
  16-min-Ring; Raten, Fenster, Schübe, Kurven, Phasenleiste und Energie-Zuordnung sind Deltas dieser Proben.
- `ipcfields.field()` kennt keinen Log-Rückfall: ein Feld ist `ipc`, `fehlt` (die Seite zeigt „fehlt in IPC“ und im Titel den
  Schreiber aus `ipcfields.MISSING_WRITER`) oder `leer` („noch kein Flip in diesem Boot“, „keine Raten: Boot beendet“).
- `live.py`, `parse.py`, `stops.py:HarnessLogs` startet der Server nicht mehr (Altbestand für Tests).
  `tests/test_ipcboot_0930.py` wird rot, sobald ein verdrahtetes Modul ein `*.log` öffnet oder der Server einen Log-Leser startet.
- Inventar mit IPC/FEHLT je Anzeige und der FEHLT-Liste: `/spinning/gpu-arb/docs/DASHBOARD-AUS-IPC-INVENTAR-0929.md`.
- `tests/test_no_new_log_parsers.py` bleibt: kein neuer Regex.

## Ruhige Anzeige (Nutzer 29.09.: „ständig verschiebt sich das nach oben/unten“)

`refresh()` baut keine Abschnitte per `innerHTML` neu, sondern arbeitet das neue HTML per
`morph()` Knoten für Knoten ein (Kinder mit `id` werden über die id gepaart — neue Karten, Banner
und Tabellen brauchen deshalb eine stabile `id`). Das Element auf Lesehöhe steht nach dem
Neuzeichnen an derselben Stelle (`holdAnchor`, Browser-`overflow-anchor` ist dafür aus), Abschnitte
schrumpfen beim Refresh nicht (min-height ratscht, Auf-/Zuklappen gibt frei), `<details open>`
bleibt offen. 10 s nach Scrollen/Tippen/Klicken und solange etwas markiert ist, wird nicht neu
gezeichnet; dazu der Schalter „Live-Update pausieren“ im Kopf. Event-Handler in neu gezeichneten
Teilen als Property (`el.onmousemove = …`), nicht `addEventListener` — der Knoten bleibt ja.
Prüfung headless: `python3 deploy/scroll_hold.py http://127.0.0.1:8891/ 300 --stress`
(Sprung der Leseposition muss 0 px sein).

Der Dienst selbst ist in `server.py` und `deploy/rig-dashboard.service` beschrieben
(LAN :8890, läuft aus `/opt/rigdash/current`, Deploy per `deploy/install.sh <rev>`).
Diese Datei beschreibt nur, was Agenten und Operatoren **pflegen** müssen.

## Features-Karte (Nutzer-Order 29.09.: „das muss immer aktuell gehalten werden“)

Quelle: `/spinning/gpu-arb/docs/features.json`. Je Feature eine Zeile mit
`id, modell (27B|NF|beide), titel, fertig, zweige[{branch, sha}], schalter[…], gewinn[…],
aus_begruendung, verantwortlich`.

**Nicht eintragen, rigdash rechnet es selbst:**

* **im Image**: ein Commit aus `zweige` ist Vorfahr der Image-REV des Modells
  (`state.json.rev`), sonst gleiche `git patch-id` auf der Image-Linie (27B-Picks unter
  neuem sha), sonst gleicher Betreff. `zweige` sind *Alternativen* derselben Änderung
  (Original + Picks); bei mehreren Commits den letzten des Features eintragen.
* **aktiv**: aus `groups.<P|D>.launch` (argv + Schalter-Env) im `state.json` des laufenden
  oder letzten Boots des Modells — nie aus Log-Text. Fehlt der Schnappschuss (Image vor
  dem Launcher-Commit `groups.<G>.launch`), aus der Profil-Datei (Kommentare zählen nicht).
  Ein Schalter, der nirgends gesetzt ist, hat seinen `default`.

**Wann eintragen** (mit dem CLI, nie per Hand-Edit ohne `check`):

```bash
U=/opt/rigdash/current/rigdash/features_update.py
# Feature fertig/gebaut (upsert nach id; --zweig/--schalter ersetzen die Listen)
python3 $U set --id H106 --modell NF --titel "…" --fertig ja \
    --zweig desk/nf-…=<sha> --schalter SGLANG_X=env:D:1:aus --verantwortlich NF-Implementierer
# 27B hat es gepickt (neuer sha, alte bleiben)
python3 $U add-zweig --id H106 --zweig desk/27b-unified-0926=<sha>
# Gewinn -- gemessen (Boot + Quelle), gerechnet (Planer/Modell) oder unbelegt
python3 $U gewinn --id H106 --modell NF --metrik Flipzeit --vorher 3.1 --nachher 2.4 --einheit s \
    --art gemessen --quelle "fliptimes" --boot <boot_id>
# im Image, aber aus: warum
python3 $U begruendung --id H63 --text "…"
python3 $U check
```

Regeln: Gewinne strikt je Modell — bei `modell=beide` trägt jeder Gewinn `--modell`
(sonst verweigert das CLI). Schalter: `NAME=art:gruppe:an_wert:default`, `art` env|flag,
`gruppe` P|D|beide|front|launcher (front/launcher: nicht im Gruppen-Schnappschuss, rigdash
liest das Profil), `default` an|aus. 27B trägt seine Zeilen selbst ein.

## Features Soll/Ist (Nutzer-Rüge 29.09.: „Bugfixes sind keine Features“)

Oben die **Produkt-Features** F1–F24 (`features.json` → `produkt`), darunter die Commits/Fixes
als **Bausteine** (`features`, Karte oben). Je Produkt-Feature: `id, nr, titel, soll`
(ein messbarer Satz), `ist.{27B,NF}` = `{status, wert, grund, beleg, belegt_am, quelle}`,
`bausteine` (ids aus `features`), optional `kreuztabelle` (F2), `untertabelle` (F12 Formate,
F23 Prompt-Längen), `matrix` (F24 Form × bs × Tiefe × Text), `marker` (Instrument je Format).

* **Ist nur mit Beleg**, sonst `unbelegt`. Jeder Sitz schreibt nur seine Spalte: NF die NF-,
  der 27B-Sitz die 27B-Zellen (`import-27b` liest `features_27b_ist_0929.md`, Quelle je Zeile).
* **zuletzt belegt** (`belegt_am`, ISO): ist der Beleg älter als der Start des letzten Boots
  dieses Modells, zeigt das Dashboard die Zelle gelb „Ist veraltet, neu messen“ (27B/Nutzer
  29.09.: neue Erkenntnisse fallen nicht hinten runter). `produkt-ist` ohne `--belegt-am` = jetzt.
* **Matrix-Zellen** sind nur Messungen (`wert` oder `ungültig` = EOS unter 500 Token);
  eine fehlende Zelle ist „ungemessen“, nie interpoliert. Werte ohne Tiefe (Agentenlast)
  stehen als Randwert `gemischt`.
* **Wert im aktuellen Boot** rechnet rigdash selbst (live.py/state.json/Profil) und zeigt
  das Instrument dazu; wo keins existiert, steht der fehlende Marker da (`MISSING` in
  features.py). Je Format: NF INT4/NVFP4, 27B INT8/NVFP4/W4A8 (W4A8 = 3080-Rang eines
  NVFP4-Boots). Das Format kommt aus `PROFILE_FORMAT` des Profils (inkl. `source`-Basis).
* Ein neuer Baustein ohne Produkt-Feature ist nur ein Hinweis, kein Schreibverbot;
  `set --produkt F8` hängt ihn im selben Schritt an.

```bash
python3 $U produkt-ist --id F1 --modell NF --status fertig+aktiv --wert "…" --beleg "Boot …" [--belegt-am 2026-09-29T06:00Z]
python3 $U kreuz --modell NF --a kvonly --b dcp --status "nur Desk" --note "F15"
python3 $U matrix --id F24 --modell NF --form "Form A" --bs 1 --tiefe kurz --text code --wert "131,9 tok/s" --boot x177 --beleg "…" [--ungueltig]
python3 $U zeile-ist --id F23 --zeile 97k --modell NF --status fertig+aktiv --wert "24,18 s" --beleg x175
python3 $U import-27b [--md /spinning/gpu-arb/docs/features_27b_ist_0929.md]
python3 $U boot-override --boot <boot_id> --lifecycle "stopped (geplant)" --beleg "…"   # state.json bleibt unberührt
python3 $U md --out /spinning/gpu-arb/docs/FEATURES-SOLL-IST-0929.md   # Tabelle als Markdown, Werte vom laufenden rigdash
```

## Verlauf in unserem eigenen Teil (Nutzer 30.09. ~15:30Z, ersetzt „Verlauf wie Grafana“ vom 29.09.)

Die Grafana-Vorlage (`/spinning/gpu-arb/docs/vorlagen/dashboard-vorlage-grafana-0929.png`) war ein
Stilbeispiel, keine Kopiervorlage. Der dunkle Nachbau oben auf der Seite ist weg; seine Inhalte stehen
verteilt in unserem Teil, im Seitendesign (dieselben Tokens, Schrift, Karten, hell und dunkel):

- **Verlauf** (`#verlauf`, unter den Boot-Karten): Kacheln Decode je Stream p50/p90, Prefix-Cache-Treffer,
  Flipzeit P→D; Diagramme Prefill-Durchsatz (P, D), Decode-Durchsatz (alle Streams + je Stream),
  Input-Tokens aus Cache / neu gerechnet / Übergabe, KV-Belegung (D, P), Flipzeit je Flip.
  Modell 27B|NF und Zeitraum 15m…7d oben in der Karte.
- **Karten** (`#gpus-card`): Kacheln Leistungsaufnahme (Summe aller Karten), heißeste GPU, Host-CPU;
  Diagramme **Leistungsaufnahme (Power draw)** als Summe aller Karten mit den Einzelkarten dünn
  darunter, Temperatur, SM-Takt, Host CPU & Speicher. Die 15-min-Sparklines je Karte sind entfallen.
- Stil: eine y-Achse je Diagramm (keine Doppelachse: Prefill und Decode getrennt), Einheit an der Achse,
  Fläche unter der Linie, feines Raster, Endwert als Punkt, Legende mit dem letzten Wert, Cursor über alle
  Diagramme gekoppelt. Farben: P blau (`--s1`), D orange (`--s2`), Decode grün (`--s3`), Karten violett/
  gelb/magenta (`--s7/--s4/--s5`, dataviz-Referenzpalette, validiert), Summe in Textfarbe.
- Die Boot-Kachel-Sparklines (`spark()`) tragen denselben Stil: Fläche, Viertelraster, −15/−10/−5/jetzt,
  letzter Wert als Punkt und Zahl.

### Quelle der Modellreihen: IPC, nicht Log (Nutzer 30.09.: „keine IPC über Logs“)

`history.Recorder.ingest_ipc` liest alle 5 s mit einem **eigenen** `IpcStates` (nicht dem des
Log-Sammlers, der erst nach einer Runde über alle Logs pollt) jede Boot-Zustandsablage und darin
`rankstate/<G>/*.rankstats` (weg2.rankstats/1, Timer-geschrieben). Je Gruppe zählt der erste Rang (TP0/PP0):

| Reihe | Rechnung |
|---|---|
| P-/D-Prefill tok/s | Δ`prefill.new_tokens` / Δ`ts` |
| Decode tok/s | Δ`decode.tokens` / Δ`ts` |
| Decode je Stream | das / `decode.running`, nur wenn Δ`decode.gpu_ms` ≥ 50 % der Wanduhr (sonst Flip/Leerlauf im Intervall) |
| KV-Belegung D / P | `sched.full_token_usage` (Pegel) |
| Input-Tokens | P: `cached_tokens` = aus Cache, `new_tokens` = neu gerechnet P; D: `new_tokens` = neu gerechnet D, `cached_tokens` = Übergabe P→D |
| Cache-Stufen | `state.json front.served_tokens.*.cached_tier` |
| Flipzeit | `flipzeit.py` über Marken `flip_t2t` (aus `events.jsonl` flip_begin/flip_done/flip_user_time + rankstats) |

Eine lebende, aber ruhende Gruppe liefert 0 (durchgehende Linie), ein Rang mit einer Datei älter als
20 s liefert nichts (Lücke). `m.<Modell>.ipc` = 1 markiert jedes Intervall, in dem der Sampler einen
lebenden Boot las; nur innerhalb solcher Intervalle überbrückt die Seite eine Lücke der je-Stream-Linie.
Ein Boot, den rigdash nicht live gesehen hat (Dienst war aus), wird einmal aus seinen Logs nachgetragen,
nur bis zur ersten IPC-Probe; das Etikett sagt dann „rankstats (IPC) · ältere Abschnitte aus Log (Übergang)“.

**Offene Lücke (Vorschlag an den Implementierer-Sitz, nicht hier gebaut):** Der Rang kann bei D nicht
trennen, ob `prefill.cached_tokens` die Übergabe P→D ist oder ein echter Präfix-Treffer einer
D-direkt-Anfrage. rigdash zählt D-cached darum konservativ als Übergabe (nie als Cache). Ein Zähler
`prefill.cached_tokens_handoff` in rankstats (Anteil der `cached_tokens`, deren rid ein P-Leg-1 hatte,
das Wissen `Pending.leg1_ran` liegt in der Front und müsste mit dem Leg-2-Request an D gehen) macht es exakt.

### Zwei Ausgaben aus einem Code: `--edition rig|release`

`rig` (Vorgabe) liefert die ganze Seite. `release` (Env `RIGDASH_EDITION=release`) ist die veröffentlichte
fLLiper-Ausgabe: `server.edition_page` schneidet jeden Block `<!--DEV:BEGIN-->…<!--DEV:END-->` aus
(HTML, CSS `/* … */` und JS `// …`), also Entwicklungsstand, Startflags, Features Soll/Ist, Bausteine,
Image-Änderungen, Container, GPU-Fensterplan, Letzte Boots und die LAN-Links. Es bleibt keine leere Hülle.
`/api/live` antwortet ohne `features`, `image_changes`, `gpuq`; `/api/launch`, `/weg2` und `/api/weg2/*`
geben 404. Neue Entwicklungsteile gehören in einen DEV-Block (`tests/test_edition_0930.py` prüft das).
**Ausnahme Profil-Reiter (Nutzer-Entscheid 05.10., Auftrag 1984):** Reiter, Panel, CSS und die Module `profil.js`, `profil_balken.js`,
`hwprofil.js`, `modellprofil.js` stehen NICHT in einem DEV-Block; `/api/profil/*` (laden, bearbeiten, speichern, exportieren, Trockenlauf,
Balken-`recompute`), `GET /api/hwprofil` (nur Anzeige, spricht gpuq nicht an) und `/api/modellprofil/*` (liest nur `config.json` und Köpfe
unter den Modellwurzeln) antworten auch in `release`, weiter nur im LAN (Proxy 403). Rig-Betrieb bleibt zu:
`POST /api/hwprofil/measure|cancel` (bucht gpuq) 403 mit Klartext, Kartenplaner, Startzeile, `/api/launch`, `/api/weg2/*` 404
(`tests/test_profil_release_edition_1984.py`).

### Profil-Editor deployen: Planer-Module stagen und Unit-Flags (Auftrag 1984)

`install.sh` legt `python/sglang` der DASHBOARD-Revision nach `/opt/rigdash/planner`; die ist eine Dashboard-Linie und trägt die Rechenmodule des
Editors nicht (`weg2/profile_json, refusals, profile_catalog(+_curated), model_profile, card_identity, topology`, `rigmon/hardware_profile`,
`planner/profile_couplings, expert_residency, pp_cut`). Sie liegen auf den Python-Release-Zweigen (`desk/profil-editor-release-27b-1005`,
`-nf-1005`). `deploy/stage_profil_modules.sh <rev>` legt den vollen Baum `python/sglang` der Revision unter
`/opt/rigdash/kartenplan/profil/releases/<sha>` ab und schaltet `profil/current` (der Kartenplaner-Baum `kartenplan/current` bleibt unberührt):

    deploy/stage_profil_modules.sh --check   fa9e7d5c4c    # prüfen, nichts schreiben (Voreinstellung); Dashboard-Revision -> REFUSED, Exit 3
    deploy/stage_profil_modules.sh --dry-run fa9e7d5c4c    # dazu die Aktionen, die --apply ausführt
    deploy/stage_profil_modules.sh --apply   fa9e7d5c4c    # schreibt, nur unter --root (Standard /opt/rigdash/kartenplan), idempotent (Lead)
    deploy/stage_profil_modules.sh --unit-flags            # die Unit-Zeilen

Danach setzt der Lead in der Unit `RIGDASH_PROFIL_TREE` (= `--profil-tree`, EIN Baum für Editor, Modellprofil, Hardwareprofil und den Kopplungs-Worker;
`--hw-tree` überstimmt ihn nur für die Hardware), `--couplings-python`/`RIGDASH_COUPLINGS_PYTHON` (Python der sglang-Umgebung), bei Bedarf
`--profiles-release-dir`, `--profile-dir`, `--model-root`, `--edition release` und, nur in der Rig-Ausgabe, `--hw-measure-tree/--hw-python/--hw-prefix`.
**MemoryMax:** der Worker (`import sglang`) liegt im cgroup der Unit (gemessen RSS 612 MiB), die Unit stand bei MemoryCurrent 487 MiB / Peak 715 MiB
gegen `MemoryMax=1G`: auf 2G heben. Der Worker rechnet auch das Topologie-Urteil des Trockenlaufs (Op `topology`); fehlt er, bleibt die Notiz
"Topologie für N Karte(n) nicht geprüft" (Tests: `tests/test_profil_staging_1984.py`, `tests/test_profil_topology_child_1984.py`).

### Speicher und Stufen

- `history.py`: `history.sqlite` im `--state-dir`. Tiers p0 (1 s NVML, 5 s Host und Modell), p1 (10 s)
  und p2 (60 s). Aufbewahrung 3 h / 3 d / 30 d, Deckel 256 MB. Es gibt keinen zusätzlichen Dienst.
  Jede Reihe ist eine Rate oder ein Pegel, nie ein Zähler. Deshalb bleibt das Mittel über jede Stufe richtig.
- `cacheacct.py` rechnet die Cache-Falle: „aus Cache“ = Prefix-Treffer bei der Annahme. Die Übergabe P→D
  ist eine eigene Reihe und nie Cache. Für Boots ohne IPC-Probe bleibt die Paarung der `WEG2-SERVED`-Zeilen
  per rid (Etikett „aus Log (Übergang)“).
- Neue Reihen: in `history.view` den Namen aufnehmen. Den Schreiber in `Recorder` setzen, nie aus
  einem neuen Log-Regex (`tests/test_no_new_log_parsers.py`).
- Diagramme: `static/grafik.js` mit uPlot 1.6.32, lokal eingebettet (`static/uplot.*`, MIT). Es gibt kein CDN.


## Kartenplaner (Item 510, Reiter "Kartenplaner", nur Rig-Ausgabe)

Optimale Startkonfiguration für ein gewähltes Modell/Profil (27B INT8, NF INT4 abl, 27B NVFP4 Dual, 27B FP8, 27B GGUF UD-IQ4_XS)
auf 1..6 gewählten Karten (Katalog `kartenplan_catalog.py`, PCIe je Karte: Gen, Lanes, Resizable BAR, über Chipsatz).

* Urteil "geht / geht nicht": die Original-Planerfunktionen `card_identity.arch_gate/order_cards/uncalibrated_message` und
  `topology.plan_topology` (`kartenplan_gate.py`, per Dateipfad geladen, synthetische Karten, keine GPU, kein Launcher).
* Plan: Aufzeichnung des Planers beim echten Boot (`kartenplan_data/*.json`: vram_plan.json, Budgetzeilen, argv/env, Rang-Log-Posten,
  Flag-Erklärungen, `planer_nachrechnung` = `launcher.budgets_from_dc` im Kindprozess gegen die Boot-Zahlen).
* Andere Karten/Zahlen: der Planer verweigert (HW-COUNT/HW-ARCH/HW-UNCALIBRATED/HW-TOPOLOGY); es gibt dann nur eine gekennzeichnete NÄHERUNG.
* Records erneuern (Schreibtisch; liest Logs, darum außerhalb dieses Pakets): `cd tools/rig_dashboard; python3 -m kartenplan_build.records;
  python3 -m kartenplan_build.bridge --trees-root <Ordner mit <rev>/python/sglang>`.
* VRAM-Balken je Karte und Phase (Auftrag 880): ein Balken = die Karte in dem Zustand, in dem die Zeilengruppe (P bzw. D) wach ist. Die Posten liegen als
  zusammenhängende Blöcke in fester Reihenfolge **gemeinsam (Treiber) → P → D** (Server: `kartenplan._annotate_segments`, stabil nach `SEG_ORDER`;
  die Klammer unter dem Balken zeigt die Blöcke). Mauszeiger/Tipp auf einen Posten: Name, MiB/GiB, Anteil an der Karte, Phase, Herkunft
  (**gemessen** = Rang-Log/NVML, **Planerwert** = vram_plan bzw. Budgetzeile) und eine Ein-Satz-Erklärung (`SEG_WHAT`).
  Überlauf: ist die Summe größer als die Karte, wächst der Balken (Skala = Summe), die Kartenkante bleibt markiert, der Überstand ist schraffiert und ein
  Hinweis „Karte N: X MiB über dem VRAM – Profil passt nicht“ nennt die größten Posten. Ein negativer Rest im Rang-Budget (Dual-Record: Posten
  überlappen, der Planer schließt die Karte trotzdem) ist **kein** Misfit: gelb schraffiert, Hinweis „Rang-Budget um X MiB überbucht“
  (`overlap_mib` / `hard_over_mib`). Live-NVML-Balken können nicht überlaufen und bleiben unverändert.
* Ansehen ohne den Dienst: `python3 -m rigdash.kartenplan_preview --port 18890 --tree <baum>/python`, dann `http://127.0.0.1:18890/#t=kartenplan`.
* Deploy-Vorschlag: `deploy/install_510.sh --check` / `deploy/install_510.sh` (Lead).


## Modellprofil schätzen (PROFIL-EDITOR S3, Auftrag 960, Nutzer 03.10.: „button zum modelprofil erstellen und werte aus dem modell am desk schätzen“)

`POST /api/modellprofil/schaetzen` mit `{"path": "<Modellverzeichnis oder .gguf>", "draft_path"?, "kv_dtype"?: "auto|fp8_e4m3",
"mamba_ssm_dtype"?: "float32|bfloat16", "gguf_file"?, "registry"?: true|"kennung"}` liefert `{ok, profile, registry_fields?, elapsed_s, cached}`;
`profile` ist `flliper.model/1` (jeder Wert `{v, src}` mit Quelle `config|Index|geschätzt|stat`), `registry_fields` die aus dem Schätzprofil
abgeleiteten Felder einer `form.ModelProfile`-Zeile.  `GET /api/modellprofil/modelle` listet die Modellverzeichnisse unter den Wurzeln (nur `stat`).

* **Der Schätzer ist nicht hier.**  `sglang/srt/weg2/model_profile.py` (reine Standardbibliothek) wird wie das Gate des Kartenplaners per Dateipfad aus
  dem Planer-Baum geladen (`MODELLPROFIL_TREE`, sonst `KARTENPLAN_TREE`, sonst die Kandidaten in `kartenplan.TREE_CANDIDATES` und `<repo>/python`).
  Fehlt die Datei im Baum, antwortet die Route 503 mit dem Namen der Datei.  Der Baum unter `tests/fixtures/modellprofil/` ist eine Kopie von
  `desk/profil-s3-modell-1003`.
* **Gelesen wird nur `config.json` und die Kopfzeilen** (8 Byte + JSON je Shard bzw. der GGUF-Kopf), nie ein Gewicht.  Der Pfad muss unter einer
  Modellwurzel liegen (`--model-root`, wiederholbar, oder `RIGDASH_MODEL_ROOTS`; Standard `/spinning/llm_stuff/club-3090/models-cache`); relative
  Pfade, `..`, NUL und Symlinks aus der Wurzel hinaus werden mit 400 abgewiesen.  Antworten werden je Pfad und Dateistand (Größe, mtime) gemerkt.
* **Nur im LAN** (auch in der Edition `release`, seit 05.10.): über den Proxy 403 (die Route liest Dateien unter den Modellwurzeln).  Körper höchstens 64 KiB.
* `static/modellprofil.js` (`window.ModellProfil`): `liste()`, `schaetzen(path, opts)`, `zeilen(profil)` (Zeilen `{gruppe, label, wert, roh, src, hinweis}`),
  `tabelle(profil)` (HTML-Baustein, escaped), `bytes(n)`.  Die Oberfläche baut Auftrag 930; dieses Modul zeichnet nichts selbst.

## Hardwareprofil messen und anzeigen (Auftrag 950, Profil-Editor S2; nur Rig-Ausgabe, nur LAN)

Routen und JSON, keine Oberfläche (die baut der Profil-Editor, Auftrag 930; `static/hwprofil.js` ist das Anzeige-Modul zum Einhängen).

* `GET /api/hwprofil` → `{profile, problems, window, job, gpuq, owner, window_len}`. `profile` ist `flliper.hardware/1`: eine **Sicht** (keine vierte
  Messdatei) über den Karten-Probe-Cache (`card_probe-*.json`), das Stufe-0-Profil (`hw_profile-*.json`) und NVML. Jeder Zahlenwert ist
  `{v, src, at, probe, note}` mit `src` = `gemessen` | `NVML` | `Datenblatt` | `geschätzt` | `nicht gemessen` (dann `v: null` und `note` = Grund).
  Gebaut wird in `sglang/srt/rigmon/hardware_profile.py` des Planer-Baums (per Dateipfad geladen, kein `import sglang` in diesem Prozess).
* `POST /api/hwprofil/measure` `{"cards": [<NVML-Index>, ...]}` bucht **selbst** ein gpuq-Fenster (Eigentümer `profil-editor`, nur diese Karten,
  15 min (Auftrag 1006: alle Rechenformate inkl. nativ W4A4 + BAR1-Strecke je Paar in Kindprozessen), ohne `not_before`, exklusiv; `mib` nur wenn der Body es verlangt) und misst darin. Antwort `action`:
  `messung_gestartet` (Kindprozess läuft, Fenster geht danach SOFORT zurück, auch nach Fehler) · `wartet` (Fenster `pending`: Status, **nichts
  gemessen**, Buchung bleibt; erneuter Druck nimmt sie wieder auf) · `abgelehnt` (unplanbar, Karte belegt trotz Fenster, zu wenig Restzeit, gpuq weg;
  HTTP 409) · `laeuft_bereits`. Das gpuq-Token verlässt den Prozess nie; die Buchung steht zusätzlich in `<state-dir>/hwprofil_window.json`, damit
  ein Neustart ein verwaistes Fenster zurückgibt.
* `POST /api/hwprofil/cancel` gibt ein wartendes Fenster zurück.
* Messumfang (Auftrag 1006, `card_probe --run`): je Karte SM-Zahl, L2, membw/GEMV, bf16, fp8, int8 W8A8, NVFP4 W4A8 (nur sm_8x), W4A16 Marlin, W4A4 nativ
  (nur sm_12x; ältere Karten tragen den Grund), H2D/D2H Bandbreite und Latenz (Median 4 kB, Minimum im Hover); je geordnetem Paar Host-Staging/p2p und die
  **BAR1-Strecke** (`rigmon/bar1_probe.py`: ein Kindprozess je Karte, Produktions-Transport mit Byte-Beweis, `--no-bar1` schaltet ab). Ein Wert, den ein
  Mikrobench nicht liefern kann, bleibt „nicht gemessen“ mit Grund. `profile.bar1` = `{measured, complete, pairs_measured, pairs_total, note}`.
* Kein Hintergrund-Poller: nur wer die Seite bedient fragt. Ein laufendes Fenster, das nach 180 s nicht benutzt wurde, geht beim nächsten Aufruf zurück.
* Dienst-Parameter (Deploy durch den Lead): `--hw-tree` (gestagter Baum mit `hardware_profile.py` + `weg2/card_identity.py`, `deploy/stage_hwprofil.sh`),
  `--hw-measure-tree` (voller sglang-Baum für den Kindprozess), `--hw-python` (Interpreter mit torch + sgl_kernel; ohne sgl_kernel bleiben die Arme
  int8/W4A16 leer und der Lauf meldet das als Warnung), `--hw-prefix` (z. B. `systemd-run --scope -q -p MemoryMax=6G`: der Dienst hat
  `MemoryMax=1G`, torch/CUDA gehört in einen eigenen cgroup-Rahmen). Env: `HWPROFIL_TREE`, `HWPROFIL_MEASURE_TREE`, `HWPROFIL_PYTHON`, `HWPROFIL_PREFIX`.
* Einhängen in eine Seite: `<div id="x"></div><script src="hwprofil.js"></script><script>HwProfil.mount(document.getElementById("x"))</script>`;
  `HwProfil.render(antwort)` liefert nur den HTML-Text.
