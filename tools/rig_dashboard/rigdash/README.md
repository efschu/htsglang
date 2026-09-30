# rigdash — Pflegehinweise

## Flipzeit (Nutzer-Korrektur 29.09.)

Flipzeit = `WEG2-FLIP begin` → erstes Decode-Token (P→D) bzw. erste PP0-`Prefill batch` (D→P);
`flip_total` (reconciled) ist nur der Layer-Tausch darin. Jeder Wert nennt sein Instrument.
Ein 27B-Boot führt mit `flip_total`, bis ein 27B-Boot unter der neuen Definition gemessen ist
(`FIRST_TOKEN_HEADLINE_FOR_27B` in `live.py`) — sonst sähe die 27B-Historie wie ein Rückschritt aus.

**TODO (27B-Review 29.09.):** `live.py` liest die Flip-Zeiten aus Log-Zeilen (nur Anzeige für
Menschen, keine Steuerung). Umstellen auf `events.jsonl` der Front, sobald die Front die
Flip-Ereignisse (begin/done/erstes Token) dort schreibt; dann entfällt der Log-Scan.

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

## Aus IPC, nicht aus Logs (Nutzer 29.09. über 27B)

Die Order lautet: „das dashboard soll auch aus der inter prozess kommunikation gespeist werden, nicht aus logs“. Das Inventar mit jeder Kachel, ihrer Quelle heute, der IPC-Quelle und dem Stand liegt in `/spinning/gpu-arb/docs/DASHBOARD-AUS-IPC-INVENTAR-0929.md`.

- `ipcstate.py` liest je Boot `state.json`, `events.jsonl` und `stop_request.json` unter `/spinning/docker-acceptance/<line>/state/<boot_id>/` (IPC-STATE-PLAN §2.2) und ordnet sie über `state.json.tag` dem Log-Boot zu.
- Aus IPC kommen:
  - Kopf (REV, Profil, Image, lifecycle, Topologie, Modell)
  - Startform (`groups.<G>.launch`)
  - geplanter Stopp oder Tod (`stops.classify_ipc`)
  - Front-Zustand und Warteschlange (`/weg2/state`, sonst der `state.json front`-Spiegel)
  - Feature-Werte „Transport“ und „bedient“
- Was noch aus einem Log kommt, trägt sichtbar **„aus Log (Übergang)“**.
- `tests/test_no_new_log_parsers.py` friert alle heutigen Regex-Literale ein. Ein neuer Log-Parser macht den Test rot, sein Weg ist: zuerst schreibt die Quelle (Front, Launcher, Rang), dann wird der Leser umgestellt, dann fällt der Parser.

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
| Flipzeit | `events.jsonl flip_first_work` |

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
