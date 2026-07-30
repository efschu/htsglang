# GPU-Testbatterie

Dreizehn Schritte, ~5 h 09 min Kartenzeit, jeder mit einem maschinellen
Erfolgskriterium. Gebaut, damit ein guenstiges Modell sie mechanisch abfahren
kann: Schritt ausfuehren, Check laufen lassen, PASS = weiter, FAIL/STOP =
anhalten und strukturiert melden. Der Executor urteilt nie. Jede Entscheidung
steckt vorab in einem Check-Skript.

Wer die Batterie faehrt, liest **EXECUTOR_PROTOCOL.md** — das hier ist die
Referenz, das Protokoll ist die Anweisung.

---

## Einmal vorbereiten

```bash
export BATTERY_RUN=/spinning/gpu-battery-results/$(date +%Y-%m-%d_%H%M)
export WT=/spinning/wt-gpu-battery              # der Worktree unter Test
mkdir -p "$BATTERY_RUN"
cd "$WT/scripts/gpu_battery"
/spinning/htsglang-gpu/.venv/bin/python battery_state.py --run-dir "$BATTERY_RUN" init
```

`BATTERY_RUN` ist die Klammer um alles: Artefakte, Logs, Zustand. Ein zweiter
Lauf am selben Tag bekommt ein eigenes Verzeichnis, eine WIEDERAUFNAHME
bekommt das alte.

## Jeder Schritt, immer gleich

```bash
BATTERY_RUN=$BATTERY_RUN bash run_step.sh <schritt>
```

`run_step.sh` macht in dieser Reihenfolge und ohne Rueckfrage: Wiederaufnahme-
Gate, VRAM-Korridor, Locks nehmen (oder bewusst nicht), Schritt unter hartem
Timeout, py-spy-Dump vor jedem Kill bei Zeitueberschreitung, Locks freigeben,
Check, Verdikt in `state.json` schreiben, Verdikt-Zeile ausgeben.

Exit-Code 0 = PASS oder SKIP, 1 = FAIL, 2 = STOP.

---

## Wiederaufnahme und Auswahl

Gruene Schritte werden **per Default nicht wiederholt**. Ein Lauf, der bei
Schritt 6 abbrach, wird nach dem Bugfix so fortgesetzt:

```bash
export BATTERY_RUN=/spinning/gpu-battery-results/<der alte Lauf>
bash run_step.sh s06        # macht bei s06 weiter, s00-s05 bleiben gruen
```

Was gefahren wird, ist frei waehlbar. Der Plan sagt vorher, was passieren
wuerde:

```bash
PY=/spinning/htsglang-gpu/.venv/bin/python
$PY battery_state.py plan                       # alles Offene
$PY battery_state.py plan --only s01,s06,s08    # nur diese drei
$PY battery_state.py plan --from s06            # ab hier
$PY battery_state.py plan --to s05              # nur die Boot-Queue
$PY battery_state.py plan --skip s03,s04        # ohne die teuren Boots
$PY battery_state.py status                     # was steht wo
```

**Berechtigter Zweifel an einem gruenen Schritt** — ein gruener Lauf, dem man
nach einem Fix oder einem Treiberwechsel nicht mehr traut — wird ausdruecklich
erzwungen, nicht durch Loeschen von Dateien:

```bash
bash run_step.sh s02 --force          # laeuft trotz PASS erneut
$PY battery_state.py plan --force s02,s03
$PY battery_state.py plan --rerun-all # alles noch einmal
```

Jeder Wiederholungslauf zaehlt in `state.json` als neuer Versuch und schiebt
das alte Verdikt in `history`. Ein erzwungener Lauf ist damit im Nachhinein von
einem Erstlauf unterscheidbar.

**Abhaengigkeiten sind Artefakt-Abhaengigkeiten, keine Reihenfolge.** s08
braucht die Dateien von s01 und s06 — sonst nichts. Es laesst sich Wochen
spaeter allein nachfahren, ohne einen einzigen Boot zu wiederholen. Ein Schritt,
dessen Voraussetzung nicht PASS ist, meldet `BLOCKED` statt still zu laufen.

**Ausnahme: der BAR1-Block (s10-s12) ist eine Kette.** Ohne gepatchten Treiber
gibt es keinen Direktpfad, und ohne einen Lauf, der belegt ueber ihn ging, gibt
es keine Kurve zu messen. s11 setzt s10 PASS voraus, s12 setzt s11 PASS voraus.
s00-s09 haengen umgekehrt an **keinem** BAR1-Schritt: die Batterie bleibt auf
einem Serientreiber vollstaendig fahrbar.

---

## Die Schritte

Reihenfolge wie unten. Karten-Identitaet wird ueberall zur Laufzeit ueber
PCI/NVML aufgeloest; nirgends steht ein fester Index.

### S0 — `s00_preflight` · haiku · ~3 min · wiederholbar

Karten-Inventar (PCI, UUID, NVML↔CUDA-Join), VRAM-Korridor, Lock-Zustand,
Pflichtdateien, Treiber-/torch-/NCCL-Version.

* **Vorbedingung** keine. Dies ist die Vorbedingung aller anderen.
* **Kommando** `bash run_step.sh s00`
* **Erfolg** `check_s00_preflight.py`: ≥2 Karten, jede mit PCI+UUID+CUDA-Index
  (der PCI-Join MUSS gelingen — ohne ihn meint jeder spaeter genannte
  Kartenindex moeglicherweise ein anderes Stueck Silizium), jede ≥400 MiB frei,
  kein fremdes Lock, alle Pflichtdateien da, nvidia-smi/curl/py-spy vorhanden.
* **Abbruch** jeder Befund hier ist STOP, nie FAIL: es wurde noch nichts
  getestet, also kann nichts fehlgeschlagen sein.
* **Artefakte** `s00_preflight/preflight.json`, `inventory.json`

### S1 — `s01_p2p_reprobe` · haiku · ~10 min · wiederholbar

Das Re-Probe-Paket nach dem Treiber-Update: capability matrix → d2d bench →
NCCL-Transport-Check.

* **Vorbedingung** s00 PASS. **Locks nimmt run_all.sh selbst** — run_step.sh
  prueft nur, dass sie frei sind. Wuerde es sie halten, braeche das Werkzeug
  an der eigenen Lock-Nahme ab.
* **Kommando** `bash run_step.sh s01`
  (optional `P2P_BASELINE=<altes nccl_transport.json>` fuer die Diff-Spalte)
* **Erfolg** `check_s01_p2p_reprobe.py`: Envelopes stimmen; `can_access_peer`
  fuer JEDES gerichtete Paar entschieden (None ≠ False); wo Peer-Zugriff moeglich
  ist, sind die **effektiven Apertur-Felder befuellt** (die nominellen 256 MiB
  sind eine Obergrenze, keine Nutzbarkeitszusage, und jeder Konsument ignoriert
  sie); die 255/256/257-MiB-Klammer liegt in der Leiter; jedes NCCL-Paar hat
  einen Transportbefund; und die Dateien werden mit den **echten
  #279-Ladern** geladen — null Aperturen oder null Profile ist FAIL.
* **Abbruch** Timeout/Absturz eines Paares = FAIL. Kein results-Verzeichnis =
  STOP.
* **Nicht bewertet** ob P2P anspringt. „Nirgends P2P" ist ein vollstaendig
  erhobenes Ergebnis. `verdict_diff.md` fuellt der Leser, nicht der Executor.
* **Artefakte** `s01_p2p_reprobe/results/{capability_matrix,d2d_bench,nccl_transport}.json`,
  `run.log`, `verdict_diff.md`

### S2 — `s02_boot_a` · sonnet · ~35 min · NICHT wiederholbar · **REPORT-GATE**

r7c Boot A, Qwen3.6-27B-FP8: der Ein-Achsen-Falsifikator. A ist zuerst, weil
als einziger Boot sein Ausgang aendert, was die anderen bedeuten.

* **Vorbedingung** s00 PASS, Korridor gruen, Locks frei.
* **Kommando** `bash run_step.sh s02`
* **Erfolg** `check_s02_boot_a.py`: Arm fuer alle fuenf Inhalte
  (alphabet, squares, repeat, code, prose); `accept_len_mean` je Arm eine echte
  Zahl (None ist das eigene Abbruchkriterium des Rezepts: Sonde aus oder
  Spec-Pfad laeuft nicht); `rounds > 0`; **Positionskurve vorhanden** und
  Positionen 0..K-1 abgedeckt; **Referenzspalte** vorhanden und mit Quelle
  benannt; MIN-freie-MiB je Karte erhoben; kein OOM/NCCL/Traceback im Log.
* **Abbruch** Server nicht oben, OOM, Traceback = FAIL (der Boot lief und hat
  etwas gesagt). Kein server.log = STOP (das Rezept lief gar nicht).
* **Nicht bewertet** die HOEHE der Accept-Zahlen. Reproduktion (2,6–3,3,
  Position 0 ~65 %) und Falsifikation (~1,5, Position 0 24–45 %) sind beide
  Ergebnisse.
* **REPORT-GATE** Nach S2 wird angehalten und berichtet, PASS oder nicht,
  bevor S3 startet. So steht es in der R7c-Queue.
* **Artefakte** `s02_boot_a/{accept.json,accept.txt,reference_column.json,vram.csv,vram_summary.txt,cards.txt,server_info.json,server.log,step.log}`

### S3 — `s03_boot_b` · sonnet · ~40 min · NICHT wiederholbar

r7c Boot B, Huihui-AWQ-MTP (INT4-Koerper, BF16-Kopf): die zweite Haelfte von
As Frage. A bewegt die Ziel-Quantisierung, B hebt nur den Kopf.

* **Vorbedingung** s00 PASS. (Artefakt-seitig unabhaengig von S2; die
  Reihenfolge ist inhaltlich, nicht technisch.)
* **Kommando** `bash run_step.sh s03`
* **Erfolg / Abbruch** wie S2.
* **Vorab bekanntes Risiko** AWQ × uneven TP × MTP wurde auf diesem Branch nie
  gebootet. Lehnt der Load die Form ab, ist das EIN verbrauchter Boot und die
  Antwort lautet „nicht auf diesem Vehikel". Der Executor tunt die Ratio nicht
  im Fenster — das waere ein anderes Experiment.
* **Artefakte** wie S2, unter `s03_boot_b/`

### S4 — `s04_boot_c` · sonnet · ~45 min · NICHT wiederholbar

r7c Boot C, GGUF-Q3-Ziel plus quantisierter DFLASH-Q8_0-Drafter solo auf einer
3080. Dritter, weil der Boot mit der hoechsten Retry-Wahrscheinlichkeit — und
ein Retry ist billiger, wenn die Accept-Fragen geklaert sind.

* **Vorbedingung** s00 PASS.
* **Kommando** `bash run_step.sh s04`
* **Erfolg** wie S2, **plus**: `loader_lines.txt` belegt einen geladenen
  Drafter. Ein Server, der nur auf dem Ziel hochkam, bestuende alle
  Accept-Pruefungen und haette Cs Frage trotzdem nicht beantwortet.
* **Abbruch** Drafter-Load wirft auf einen Tensornamen → FAIL, Name melden.
  OOM auf der Wirtskarte → FAIL; `RESERVE_HOST` anzuheben ist Sache des
  NAECHSTEN Laufs, nicht eines Retries im Fenster.
* **Ausdruecklich kein Abbruch** inkohaerenter Output. Das ist ein Ergebnis.
* **Artefakte** wie S2 plus `loader_lines.txt`, unter `s04_boot_c/`

### S5 — `s05_boot_d` · sonnet · ~30 min · NICHT wiederholbar

r7c Boot D, Lane-Re-Seed A/B auf der Runde-7b-Konfiguration. Der eine Boot,
von dem bekannt ist, dass er hochkommt.

* **Vorbedingung** s00 PASS.
* **Kommando** `bash run_step.sh s05`
* **Erfolg** `check_s05_boot_d.py`: drei Inhalte (squares, code, prose),
  **beide** Arme je Inhalt (ein A/B mit einem Arm ist kein A/B),
  `accept_len_mean` und `decode_ms_mean` in beiden Armen echte Zahlen (der
  Preis ist der Punkt des Boots und steht in `decode_ms_mean`),
  `reseed_forwards` im Re-Seed-Arm gesetzt, `output_identical` als Bool
  vorhanden; MIN-frei je Karte; kein Fatal im Log.
* **Ausdruecklich kein Fehler** `output_identical == False`. Das IST die
  Messung.
* **Artefakte** `s05_boot_d/{reseed.json,reseed.txt,vram_summary.txt,server.log,…}`

### S6 — `s06_nccl_reference` · haiku · ~15 min · wiederholbar

Die NCCL-/System-RAM-Referenzmessung im #279-Format
(`new_nccl_reference_envelope`, schema_version 1).

* **Vorbedingung** s00 PASS.
* **Kommando** `bash run_step.sh s06`
* **Was gemessen wird** je Kartenpaar, in gepinnten Subprozessen: `all_reduce`
  ueber die Leiter 64 KiB / 1 MiB / 8 MiB / 64 MiB und `send_recv` in **beide**
  Richtungen, je in zwei Armen (`idle` und `host_stream_64mib`), **p50 UND p99
  in jeder Zeile**. Beidseitig, weil der #278-Abschluss genau daran scheiterte,
  dass die Last-Achse p50-gegen-p99 asymmetrisch erhoben war und damit
  unbrauchbar wurde.
* **Erfolg** `check_s06_nccl_reference.py`: Envelope korrekt; jede Zeile hat
  alle zehn Pflichtfelder (eine Zeile mit neun wird vom Lader verworfen, eine
  Datei aus solchen Zeilen laedt als leer); p99 ≥ p50; `transport` gefuellt;
  beide Arme ueber DIESELBEN (op, Paar, Groesse)-Schluessel; `send_recv` in
  beiden Richtungen je Paar; und `load_nccl_reference()` liefert measured-
  Profile ohne Fehler.
* **Abbruch** abgebrochenes Paar = FAIL.
* **Artefakte** `s06_nccl_reference/{nccl_reference.json,nccl_debug.log}`

### S7 — `s07_offload_register_gpu` · haiku · ~12 min · wiederholbar

Offload-Register auf echtem Silizium: `CudaDeviceOps`, echte Posten-Groessen,
Rueckhol-Latenzen je Klasse (#286-Restliste 1 und 4).

* **Vorbedingung** s00 PASS.
* **Kommando** `bash run_step.sh s07`
* **Was laeuft** alle drei Bewegungsrouten auf der groessten Karte (zur
  Laufzeit aufgeloest): `tensor` (pinned Pool + async H2D) fuer
  lane_workspaces/kv_shadow/experts, `tag` (#93-Tag-Pools ueber den echten
  memory-saver) fuer graph_rungs/gdn_state_sets, `suspend` (#89) fuer
  cold_lane. Je Posten 256 MiB, 5 park/wave_in-Zyklen, p50/p99.
  Der Park wird EXPLIZIT angefordert: das Register parkt bedarfsgesteuert, und
  unter `auto` verweigert `park()` ohne Saettigungsdruck — korrekt, aber dann
  misst der Lauf das Tor statt der Bewegung. Die Sonde setzt deshalb jede
  gemessene Klasse ueber den granularen Klassen-Knopf
  (`--lane-offload-class-policy`-Syntax) auf `ram` und legt die aufgeloeste
  Policy-Karte ins Artefakt. Die Planphase (`--dry-run`) faehrt denselben
  Ablauf gegen `FakeDeviceOps` — ohne Karte, als Selbsttest der Sonde.
* **Erfolg** `check_s07_offload_register_gpu.py`: `device_ops ==
  "CudaDeviceOps"` (eine Validierung mit FakeDeviceOps validiert nichts);
  **alle drei Routen** gruen; je Zeile eine echte Groesse, die
  `resolve_size_bytes` bestaetigt; Zustandsfolge enthaelt wirklich `parked`
  (ein still no-oppender Park gibt dasselbe zurueck wie ein arbeitender);
  Rueckhol-Latenz > 0 ueber ≥3 Zyklen; null park/wave_in-Fehler;
  Negativkontrolle verweigert (ein Posten bleibt auf `auto` ohne Sensor und
  muss abgelehnt werden — sonst belegt der Lauf nicht, dass die explizite
  Policy den Park zugelassen hat); `latency_term_ms` erhoben — genau diese
  Zahl liest der #279-Dispatcher.
* **Abbruch** memory-saver nicht verfuegbar = STOP (zwei von drei Routen waeren
  ungetestet, und ein gruenes Verdikt auf einem Drittel des Registers waere
  schlimmer als keines). Der memory-saver braucht seinen Preload-Hook in
  LD_PRELOAD, sonst stirbt jede `region()`; die Sonde setzt ihn selbst und
  startet sich einmal neu (LD_PRELOAD liest der Linker beim Prozessstart).
* **Artefakte** `s07_offload_register_gpu/offload_register_gpu.json`

### S8 — `s08_dispatcher_tables` · haiku · ~3 min · wiederholbar · **CPU-only**

Die gemessenen Ratentabellen in den #279-Dispatcher laden und die
Placeholder-Neutralitaet nachpruefen.

* **Vorbedingung** s01 UND s06 PASS. Keine Karte, kein Lock, kein Korridor.
* **Kommando** `bash run_step.sh s08` (optional `GDR_TSV=<#278-Matrix>`)
* **Erfolg** `check_s08_dispatcher_tables.py`: alle drei Quellen vorhanden und
  fehlerfrei geladen; >0 measured-Profile UND >0 effektive Aperturen (null
  Aperturen heisst: die capability matrix hat nichts beigetragen und jeder
  direkte Pfad ist unbegrenzt — die stille Variante des Fehlers); die absichtlich
  platzhalter-kontaminierte Klasse entscheidet ueberall STATUS_QUO, auch fuer
  `protected`; **Saettigungs-Sensor und Latenz-Term wurden dabei NICHT
  konsultiert** (die Sonden-Hooks WERFEN bei Beruehrung, damit „nicht
  konsultiert" bewiesen und nicht vermutet ist); und die vollstaendig gemessene
  Klasse entscheidet mindestens einmal einen echten Pfad — sonst wurden die
  Tabellen geladen und ignoriert.
* **Warum der Schritt existiert** `load_rate_tables` degradiert fehlende
  Quellen laut, aber fehlerfrei zu Platzhaltern, und Regel 1 haelt dann alles
  auf dem Status quo. Zur Laufzeit richtig — hier nicht von einem Lauf ganz
  ohne Karten unterscheidbar. Und: die Neutralitaet wurde bisher NUR mit
  Platzhaltern getestet. Erst jetzt gibt es measured-Kandidaten, die die Regel
  verletzen koennten.
* **Artefakte** `s08_dispatcher_tables/dispatcher_tables.json`

### S9 — `s09_sensor_smoke` · sonnet · ~15 min · wiederholbar

gdn-/KV-Druck-Leiter: Flags auf einem echten Boot, Sensor an echter Belegung.
Kleines Modell (Qwen3.5-4B), eine Karte, TP=1.

* **Vorbedingung** s00 PASS.
* **Kommando** `bash run_step.sh s09`
* **Erfolg** `check_s09_sensor_smoke.py`: alle vier Leiter-Flags kommen von
  `/get_server_info` zurueck (Argument-Validierung ist CPU-getestet — neu ist,
  dass die Werte den Scheduler erreichen); zwei identische Greedy-Generierungen
  sind identisch (die Leitern sollen inert sein); ≥20 Belegungs-Samples aus
  `sglang:token_usage` mit Maximum > 0 (eine Nulllinie hiesse, die Last hat den
  Pool nie erreicht); der Sensor liefert eine Lesung mit Verdikt, endlicher
  Belegung und Trend — und zweimal dieselbe aus derselben Reihe.
* **Ausdruecklich NICHT getestet** die Verdrahtung des Sensors an die
  Scheduler-Belegung und jede echte Bewegung von State-Sets oder KV. Beides
  existiert noch nicht; es sind die offenen Posten, die diese Zahlen
  vorbereiten.
* **Artefakte** `s09_sensor_smoke/{sensor_smoke.json,server_info.json,server.log}`

---

## Der BAR1-Block (s10-s12) — laeuft auf dem PVE-HOST

Diese drei Schritte fahren nicht dort, wo die Batterie fahrt. Der gepatchte
Treiber, das Halter-Modul und `/dev/dmabuf_holder` liegen auf dem PVE-Host;
CT999 kann das Halter-Gerat nicht einmal oeffnen (Major 10 steht nicht in der
Device-Allowlist des Containers). Die Schritte steuern den Host also ueber ssh,
waehrend Batterie, Zustand und Artefakte im Container bleiben.

Was daraus folgt und in jedem der drei Skripte steht:

* **Zwei `/tmp`-Namensraeume.** `/tmp/gpu-card-N.lock` in CT999 und das
  gleichnamige Verzeichnis auf dem Host sind **verschiedene** Locks. Deshalb
  werden beide genommen: die Container-Seite von `run_step.sh`
  (`locks="battery"`), die Host-Seite vom Schritt selbst ueber
  `battery_host.sh`. Ein fremdes Lock wird auf **keiner** Seite gebrochen.
* **Jedes ssh hat eine Frist.** Ein blockierendes ssh in einem Bash-Aufruf
  macht den Executor handlungsunfaehig, ohne dass es jemand sieht.
  `host_ssh` traegt immer `timeout`, `ConnectTimeout` und Keepalives.
* **py-spy laeuft auf dem HOST**, vor jedem Kill, aus dem Container-venv (der
  Host hat kein eigenes). Getoetet wird nur die eigene Prozessgruppe, nie ein
  Muster.
* **Serverlogs bleiben auf dem Host** (`/root/battery-bar1/`). In das
  Lauf-Verzeichnis kommen das Grep-Ergebnis und ein begrenzter Tail — nie das
  ganze Log, nie in den Kontext eines Agenten.
* **Pfadabbildung.** Das Wurzel-Dateisystem des Containers ist auf dem Host
  `/spinning/subvol-999-disk-0`, in beide Richtungen beschreibbar. Deshalb
  schreibt der Messtreiber auf dem Host direkt in das Lauf-Verzeichnis — nur so
  ist die Live-Tabelle von s12 live.

Umgebungsvariablen (alle mit Vorgabe, alle in `battery_host.sh` dokumentiert):
`BAR1_HOST`, `BAR1_HOST_KEY`, `BAR1_HOST_SUBVOL`, `BAR1_HOST_WT`,
`BAR1_NV_SOURCE`, `BAR1_HOLDER_KO`, `BAR1_EXTCACHE`, `BAR1_PORT`,
`BAR1_VIEWER_KILL_OK`, `S12_SESSIONS`, `S12_POINT_SECONDS`,
`S12_BASELINE_TOL_PCT`.

### S10 — `s10_bar1_driver` · sonnet · ~6 min · wiederholbar

Gepatchter Treiber plus Vorbedingungen des Direktpfads. Rezept aus
`04_BETRIEB.md` der P2P-Uebergabe, mit einem bewussten Unterschied: die
PCI-Adressen fuer den Reset werden zur Laufzeit aus `lspci` aufgeloest statt
hineingeschrieben — Enumeration verschiebt sich zwischen Boots und
Treiberzustaenden.

* **Vorbedingung** s00 PASS.
* **Kommando** `bash run_step.sh s10`
* **Idempotent** Liegt das gepatchte Modul schon **mit** Regkey und Halter vor,
  wird nichts entladen: der Schritt prueft und vermerkt
  `reload_performed=false`. Den Treiber zu wechseln, um zu zeigen, dass man ihn
  wechseln kann, wuerde genau den Zustand kosten, den s11 und s12 brauchen.
* **Erfolg** `check_s10_bar1_driver.py`: Regkey in
  `/proc/driver/nvidia/params`; `strings nvidia.ko | grep -c SMALLBAR_P2P` = 37
  (voller Patch, nicht die 1 des minimalen); **die srcversion des GELADENEN
  Moduls stimmt mit der der `.ko` ueberein** — die Regkey-Zeile allein belegt
  nur einen Parameter, nicht die Identitaet des Moduls; `dmabuf_holder` geladen
  **und** `/dev/dmabuf_holder` da; `nvidia_uvm` geladen; alle drei Karten
  enumerieren mit UUID und PCI-Adresse.
* **Abbruch** Host nicht erreichbar, Rechenprozesse auf den Karten, oder
  Betrachter halten die Module = STOP (Umgebung). Fehlender Regkey, minimaler
  Patch, fremdes Modul, fehlender Halter, verlorene Karte = FAIL.
* **Betrachter** `nvtop` und Verwandte halten die Module. Sie werden **nur** mit
  `BAR1_VIEWER_KILL_OK=1` beendet, und diese Freigabe gilt **nicht dauerhaft**
  (05_FALLEN). Ohne sie haelt der Schritt an und nennt die PIDs.
* **Rueckstellung** Die Batterie endet per Default **ohne** Rueckstellung: s11
  und s12 laufen unter dem Patch, und die NCCL-Koexistenz ist gemessen
  (37,11 / 64,65 / 359,35 µs gegen 37,89 / 65,25 / 357,63 µs — alle innerhalb
  der Streuung). Wer den Serientreiber zurueck will:
  `bash s10_restore.sh` — eigenes Kommando, kein Batterie-Schritt.
* **Artefakte** `s10_bar1_driver/{driver_state.json,remote_probe.sh,remote_reload.sh,host/*}`

### S11 — `s11_bar1_e2e` · sonnet · ~25 min · wiederholbar

Ein Standardlauf ueber den Direktpfad, Ende zu Ende. Die Frage ist **nicht**, ob
er bootet.

* **Vorbedingung** s10 PASS. Die BAR1-Integration muss im Arbeitsbaum unter Test
  liegen (`htccl_bar1.py`, `benchmark/bar1_graph_check.py`), sonst STOP mit dem
  Hinweis auf `BAR1_HOST_WT`.
* **Kommando** `bash run_step.sh s11`
* **Tor zuerst** `benchmark/bar1_graph_check.py 0,1,2`. `GRAPH_FREIGABE=1` ohne
  diesen Beleg ergibt Zahlen von einem Betriebspunkt, den niemand verteidigen
  kann.
* **Erfolg** `check_s11_bar1_e2e.py`: alle Gate-Faelle des Tors bestanden;
  **je Gruppe** `ERREICHT=bar1` — mit `SGLANG_UNEVEN_DCP=1` sind es zwei
  (`tp:0`, `dcp:0`), und eine davon auf gloo macht den Lauf gemischt (genau das
  ist einmal passiert und hat eine ganze Messung gekostet); je Gruppe eine
  `HTCCL-BAR1: Aufbau in`-Zeile; Smoke-Antwort kohaerent (die Zahlen 1..20 in
  Folge, **gezaehlt**, nicht beurteilt) und `spec_accept_length` als Zahl; kein
  OOM/NCCL/Watchdog im gegreppten Log.
* **Der Riegel als Sonderfall** `htccl._select` bricht **laut** ab, statt unter
  einer Graph-Aufzeichnung still auf die host-gestaffelte gloo-Ebene
  auszuweichen. Solange `all_gather` in `htccl_bar1` fehlt, endet der
  Standardlauf mit
  `RuntimeError: HTCCL: 'all_gather' mit <n> Byte waehrend einer
  CUDA-Graph-Aufzeichnung ...`. Der Check gibt das als **eigene FAIL-Meldung**
  mit `RIEGEL`, Operation und Groesse aus — richtiges Verhalten des Codes und
  ein FAIL dieses Schritts zugleich. Genau dieses Szenario nimmt die parallele
  BAR1-Integration ab.
* **Nicht bewertet** die HOEHE von `spec_accept_length`, die Aufbaudauer, jeder
  Durchsatz.
* **Artefakte** `s11_bar1_e2e/{bar1_e2e.json,graph_check.txt,htccl_lines.txt,smoke.json,server_info.json,server.log,remote_*.sh}`

### S12 — `s12_prefill_kurve` · sonnet · ~70 min · NICHT wiederholbar

**Die** Messung. Ueber den Host-Weg ist die Prefill-Kurve flach
(1190/1097/1144/1105/1122 tok/s ueber 1/2/4/8/16 Sessions) bei 65-90 %
Kollektivzeit — mehr Sessions bringen nichts, weil die Kollektive saettigen,
nicht die Rechenwerke. Ist der Deckel der Umweg, faellt er mit ihm.

* **Vorbedingung** s11 PASS.
* **Kommando** `bash run_step.sh s12`
* **Verschraenkt, nicht blockweise** Je Sessionzahl laufen bar1 und Grundlinie
  hintereinander (A,B,A,B), dann die naechste Sessionzahl. Acht Boots fuer vier
  Punkte je Arm — das ist der Preis dafuer, nicht zwei Nachmittage zu
  vergleichen (Messregel 5). Blockweise waere je Arm ein Boot und wertlos.
* **Die Arme unterscheiden sich in genau drei Variablen**
  (`SGLANG_HTCCL`, `SGLANG_HTCCL_TRANSPORT`, `SGLANG_HTCCL_GRAPH_FREIGABE`,
  dazu die Treiberquelle). Beide Bootskripte kommen aus **einer** Vorlage, damit
  sie nicht auseinanderlaufen; ein Test diffed sie und laesst genau zwei
  Zeilen Unterschied zu.
* **Was gemessen wird** je Arm und Sessionzahl 1/4/8/16 ein Prefill-Punkt
  (Zeitbox 15 s, Vorlauf 8 s, Zaehler ist `prompt_tokens` **aus der Antwort**,
  je Anfrage ein eindeutiger Kopf plus `/flush_cache`, damit nicht der
  Radix-Cache gemessen wird), dazu je Boot ein Decode-Punkt bei bs=1 und bs=16
  mit `ms/Token`, `ms/Verify` und `spec_accept_length`. Rohzeilen je Anfrage
  werden persistiert, nicht nur die Endzahl.
* **Erfolg** `check_s12_prefill_kurve.py`: je geplanter Sessionzahl eine Zahl in
  **beiden** Armen; die Arme haben wirklich abgewechselt (sonst FAIL
  „blockweise"); **je Punkt der Transport-Beleg** — bar1-Punkte mit allen
  Gruppen auf `ERREICHT=bar1`, Grundlinien-Punkte ohne jede HTCCL-Gruppe; je Arm
  ein Decode-Punkt bei bs=1 und bs=16; je Arm ein persistiertes Ausgabe-Sample
  (ein schneller Muell-Lauf sieht in einer Durchsatztabelle gut aus); und die
  **Grundlinie reproduziert** die bekannten Zahlen innerhalb `±5 %`.
* **Abbruch** Reproduziert die Grundlinie nicht, ist das **STOP**: diese
  Umgebung ist nicht die, aus der die bekannten Zahlen stammen. Vorbehalt dazu:
  die bekannten Werte stammen aus einem **anderen** Messprogramm. Reisst das
  Tor, ist der erste Verdacht das Messprogramm, nicht das Rig — deshalb STOP
  (Operator entscheidet) und nicht FAIL, und deshalb ist die Toleranz ueber
  `S12_BASELINE_TOL_PCT` einstellbar. Der A/B-Vergleich selbst haengt daran
  nicht: beide Arme werden mit demselben Programm gemessen.
* **Ausdruecklich NICHT bewertet** ob die Kurve flach bleibt oder steigt, und um
  wieviel bar1 die Grundlinie schlaegt. Flach ist ein Befund, steigend ist ein
  Befund. Ein Check, der das beurteilte, entschiede die Frage, fuer die die
  Messung existiert. Das Ergebnis-JSON traegt beide Kurven nebeneinander.
* **Live mitlesen** Nach jedem Sessionzahl-Paar wird
  `zwischentabelle.md` aus den persistierten Punkten neu geschrieben. Der
  Executor gibt sie aus (EXECUTOR_PROTOCOL 6b/6c).
* **Nicht abgedeckt** Rechen- gegen Wartezeit je Rang. Das braucht den Profiler
  und ist eine eigene Uebung.
* **Artefakte** `s12_prefill_kurve/{prefill_kurve.json,zwischentabelle.md,punkte.jsonl,roh_*.jsonl,belege/*,logs/*,remote_*.sh}`

---

## Zeitbudget

| Schritt | Modell | erwartet | hartes Budget |
|---|---|---:|---:|
| s00_preflight | haiku | 3 min | 5 min |
| s01_p2p_reprobe | haiku | 10 min | 30 min |
| s02_boot_a | sonnet | 35 min | 60 min |
| s03_boot_b | sonnet | 40 min | 65 min |
| s04_boot_c | sonnet | 45 min | 70 min |
| s05_boot_d | sonnet | 30 min | 50 min |
| s06_nccl_reference | haiku | 15 min | 30 min |
| s07_offload_register_gpu | haiku | 12 min | 20 min |
| s08_dispatcher_tables | haiku | 3 min | 10 min |
| s09_sensor_smoke | sonnet | 15 min | 30 min |
| s10_bar1_driver | sonnet | 6 min | 15 min |
| s11_bar1_e2e | sonnet | 25 min | 45 min |
| s12_prefill_kurve | sonnet | 70 min | 150 min |
| **Summe** | | **5 h 09 min** | |

Die harten Budgets liegen deutlich ueber den Erwartungen, damit ein langsamer
Load nicht mit einem Haenger verwechselt wird. Ueberschreitung ist STOP, nie ein
Grund, laenger zu warten — mit py-spy-Dump jedes registrierten Prozesses vor
dem Kill.

## Was die Batterie bewusst NICHT tut

* **Sie bewertet keine Messwerte.** Kein Check hat eine Meinung zu einer
  Accept-Hoehe, einer Bandbreite oder ob P2P anspringt. Geprueft wird, dass die
  Messung stattgefunden hat, vollstaendig ist und von ihrem echten Konsumenten
  geladen werden kann. Die Deutung ist Leserarbeit — deshalb ist sie nicht in
  einem Skript vergraben, das sie stillschweigend faellt.
* **Sie fuellt `verdict_diff.md` nicht.** Die acht „kein P2P"-Altverdikte
  brauchen ein Urteil je Zeile; ein ueberworfenes Verdikt bekommt eine eigene
  Aufgabe, bevor Platzierungs-Code angefasst wird. Diese Datei protokolliert
  das Delta, sie autorisiert keine Umbauten.
* **Sie tunt nichts im Fenster.** Kein Ratio, kein `RESERVE_HOST`, kein
  Kontext. Ein Boot, der die Form ablehnt, ist ein verbrauchter Boot mit einer
  Antwort — kein Startpunkt fuer Parametersuche.
* **Sie testet keine Verdrahtung, die es nicht gibt.** Sensor an Scheduler
  (Erg. 9), Admissions-Hook am Scheduler (Erg. 8), echte Bewegung der
  GDN-State-Sets: alles offene Posten. Sie zu „testen" hiesse, ein gruenes
  Verdikt fuer nicht vorhandenen Code zu vergeben.
* **Sie faehrt keine Perf-Regression.** ms/Runde gegen einen Rauschboden ist
  eine eigene, laengere Uebung mit verschraenkter Messung und fixiertem Takt.
  Eine Batterie mit 20-Sekunden-Messpunkten kann sie nicht liefern und soll es
  nicht vorgeben. **s12 ist die eine Ausnahme** und traegt den Preis dafuer:
  acht Boots, verschraenkte Arme, ein Reproduktions-Tor gegen die bekannte
  Grundlinie — und trotzdem kein Urteil ueber die Form der Kurve.
* **Sie stellt den Treiber nicht zurueck.** Nach s10 bleibt der gepatchte
  Treiber geladen; das ist Absicht und belegt (NCCL-Koexistenz gemessen).
  `s10_restore.sh` ist ein Operator-Kommando, kein Schrittende.
* **Sie bricht keine fremden Locks** und killt nichts, was sie nicht selbst
  gestartet hat.
