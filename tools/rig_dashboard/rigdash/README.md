# rigdash — Pflegehinweise

## Flipzeit (Nutzer-Korrektur 29.09.)

Flipzeit = `WEG2-FLIP begin` → erstes Decode-Token (P→D) bzw. erste PP0-`Prefill batch` (D→P);
`flip_total` (reconciled) ist nur der Layer-Tausch darin. Jeder Wert nennt sein Instrument.
Ein 27B-Boot führt mit `flip_total`, bis ein 27B-Boot unter der neuen Definition gemessen ist
(`FIRST_TOKEN_HEADLINE_FOR_27B` in `live.py`) — sonst sähe die 27B-Historie wie ein Rückschritt aus.

**TODO (27B-Review 29.09.):** `live.py` liest die Flip-Zeiten aus Log-Zeilen (nur Anzeige für
Menschen, keine Steuerung). Umstellen auf `events.jsonl` der Front, sobald die Front die
Flip-Ereignisse (begin/done/erstes Token) dort schreibt; dann entfällt der Log-Scan.

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
