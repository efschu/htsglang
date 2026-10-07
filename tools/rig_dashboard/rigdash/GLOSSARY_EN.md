# Dashboard glossary DE -> EN (rigdash)

Single word list for the English UI. Part 1 (static JS/HTML) created it; Part 2 (Python texts, `weg2/propose*.py`,
`profile_couplings.py`) and Part 3 (catalog files) append to it and use the same terms. Style: terse developer English.

Not translated (API contract `flliper.*/1`, logic keys, CSS): JSON keys and field names (`werte`, `herkunft`, `verdikt`,
`zustand`, `formen`, `ausgang`, ...), enum values that the JS compares (`geht`, `geht_mit_force`, `verweigert`,
`unbelegt`, `vorgeschlagen`, `einfach`, `experte`, `GESTOPPT`, `TOT`, `HAENGT`, ...), env/flag names, CSS classes, element IDs,
file names. The JS shows an English display label for such a value and keeps the German value as the key.

## General terms

| DE | EN |
|---|---|
| geht | ok |
| nur mit --force | only with --force |
| mit --force | with --force |
| verweigert | refused |
| Force ungeprüft | force unchecked |
| nicht beurteilt | not judged |
| kein Lauf | no run |
| Planer-Rechnung | planner estimate |
| Planer-Rechnung: passt / passt nicht / unbelegt | planner estimate: fits / does not fit / unverified |
| Orakel / Orakel-Fehler | oracle / oracle error |
| Trockenlauf | dry run |
| Neu prüfen | Re-check |
| Vorschlag | proposal (button: Propose) |
| unbelegt | unverified |
| belegt | verified |
| geborgt | borrowed |
| geschätzt | estimated |
| gemessen / nicht gemessen | measured / not measured |
| gerechnet / nicht gerechnet | computed / not computed |
| Datenblatt | datasheet |
| Herkunft | source |
| Quelle | source |
| Beleg | evidence |
| Zustand (je Wert) | state |
| Urteil / Verdikt | judgement / verdict |
| vorgeschlagen | proposed |
| vom Launcher gelöst | solved by the launcher |
| von Ihnen übersteuert | overridden by you |
| ungeprüft seit Ihrer Änderung | unchecked since your change |
| Standard (nicht im Profil) | default (not in the profile) |
| Übernehmen | Take over |
| Hinweis | note |
| Sperre | block |
| Ablehnung / Ablehnungscode | refusal / refusal code |
| Abhängigkeiten / Kanten | dependencies / edges |
| Betriebsform | operating form |
| Einzelkarte / nur TP / Flip PP/TP / Dual PP/TP | single card / TP only / Flip PP/TP / Dual PP/TP |
| Rang | rank |
| Karte | card |
| Karten-Balken | card bars |
| Gewichte / Experten / Draft / KV / Mamba-Zustand | weights / experts / draft / KV / Mamba state |
| Aktivierung | activation |
| Festposten | fixed items |
| Reserve / Frei | reserve / free |
| Rest | remainder |
| Überlauf | overflow |
| Kartenende / Kartengrenze | end of card / card limit |
| Korridor | corridor |
| Sitze (gleichzeitig) | seats (concurrent seats) |
| Kontext | context |
| Anfrage | request |
| Leerlauf | idle |
| Eimer (Zeitraster) | bucket |
| Übergabe P->D | handoff P->D |
| neu gerechnet | recomputed |
| aus Cache | from cache |
| Schub | burst |
| Wanduhr | wall clock |
| Rechenzeit / Rechenrate | compute time / compute rate |
| Flipzeit | flip time |
| Vorlauf / Nachlauf | lead / tail |
| Zerlegung | decomposition |
| ruht / schläft | idle / asleep |
| Aushungerungs-Klemme | starvation clamp |
| Eintrittsstufe | entry stage |
| fehlt in IPC | missing in IPC |
| Expertenansicht / Einfach / Experte | expert view / Simple / Expert |
| Startzeile | launch line |
| Fensterplan | window plan |
| Verlauf | history |
| Überblick | overview |
| Entwicklung | development |
| Sitzungen | sessions |
| Bausteine | building blocks |
| Soll / Ist | target / actual |
| Laufbericht | run report |

## Literal strings shared across parts (JS and Python/catalog must agree)

| Where | English text |
|---|---|
| `flipzeit.DEFINITION["P>D"]` = `FLIP_DEF` (index.html, grafik.js) | last P chunk done → first decode token produced |
| `flipzeit.DEFINITION["D>P"]` | last decode token produced → first prefill chunk starts computing (first forward on PP0) |
| `FLIP_EXCEPTION` (index.html) | No flip counts when no prefill or decode is pending (idle flip). |
| history.py `NO_DATA_LABEL` (grafik.js accepts both) | no data (before IPC recording) |
| ipcstate.py log source label (grafik.js accepts both) | from log (transition) |
| hardware profile source tags (hwprofil.js accepts both) | measured / NVML / datasheet / estimated / not measured |
| model profile source tags (`src`) (modellprofil.js shows `geschätzt` as `estimated`) | config / Index / estimated / stat |
| `profile_couplings.BAR_SEGMENTS` labels (profil_balken.js `SEGS`) | Weights / Experts (resident) / Draft/MTP / KV / Mamba/GDN state / Activation / Fixed items; tail: Reserve / Free; Overflow |
| bar segment origins | approximation (browser), computed, input/computed, not computed |
| alarm states `a.state` (index.html accepts both) | GESTOPPT=STOPPED, TOT=DEAD, HAENGT=HANGING, WARNUNG=WARNING |
| `weg2line.py` launch placeholder | `<fenster-id>` stays (the page text uses the same token) |

## Number and time format

Numbers use `en-US` (`1,234.5`), clock times `en-GB` (24 h). The decimal comma is gone everywhere in the JS.

## Appended by Part 2 / Part 3

(empty; append below, keep alphabetical order of the DE column within a section)
