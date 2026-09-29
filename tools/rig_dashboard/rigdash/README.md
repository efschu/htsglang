# rigdash — Pflegehinweise

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
