# Dashboard glossary DE -> EN (merged: part 1 UI terms + part 2 Python terms)

Merge note (part 4): Part 1 (static JS/HTML) and Part 2 (Python texts) each created this file. Both term lists are kept
below unchanged, Part 2 table first as "Python-text terms", then the Part 1 sections. Where both define the same German
term the English wording agrees except: `unverified` (Part 2: unverified / Part 1: unverified), `Vorschlag` (proposal in both),
`Verdikt` (verdict in both; Part 1 lists "judgement / verdict" for Urteil). The catalog glossary lives separately in
`python/flliper/srt/pdflip/GLOSSARY_EN.md`.

## Part 2: Python-text terms

One English term per German term, used by the UI (static/*.js, index.html), the Python texts (rigdash/*.py,
pdflip/propose*.py, planner/profile_couplings.py) and the catalog texts. Machine-readable names (JSON keys, flags,
env names, refusal codes, enum values such as `state`/`force_state`) are NOT translated.

| German | English |
|---|---|
| Karte | card (GPU card) |
| Rang | rank |
| Lauf | run |
| Boot | boot |
| Trockenlauf, Dry-Run | dry run |
| Orakel | oracle (the launcher dry run that answers for the planner) |
| Verdikt, Urteil | verdict |
| Verweigerung, Ablehnung | refusal |
| verweigert, abgelehnt | refused |
| forcebar | forceable (can be overridden with `--force`) |
| nicht forcebar | not forceable |
| Force uebergeht | force overrides / passes |
| Abhaengigkeit | dependency |
| Kante (Kantenkatalog) | edge (edge catalog) |
| Betriebsform, Form | operating mode (form) |
| Einzelkarte | single card |
| nur TP | TP only |
| Flip PP/TP | flip PP/TP |
| Dual PP/TP | dual PP/TP |
| Hardwareprofil | hardware profile |
| Modellprofil | model profile |
| Startprofil, Serverprofil | server profile |
| Profil-Planer, Planer | profile planner, planner |
| Vorschlag | proposal |
| vorgeschlagen | proposed |
| uebersteuert | overridden |
| geloest (vom Launcher) | solved (by the launcher) |
| unbelegt | unverified (no evidence / not measured) |
| belegt | verified |
| geborgt | borrowed |
| Datenblatt | datasheet |
| gemessen / geschaetzt | measured / estimated |
| Geheimnis | secret |
| Laufbericht | run report |
| Gewichte | weights |
| Experten | experts |
| Sitze (gleichzeitig) | seats (concurrent) |
| Kontext | context |
| Reserve | reserve |
| Balken | bar |
| Ueberlauf | overflow |
| (P-)Schnitt | (P) cut |
| Wert | value |
| Zeile | row / line |
| Schalter | switch |
| Erklaerung | explanation |
| Konsequenz | consequence |
| Grund | reason |
| Hinweis | note |
| Pfad | path |
| Treiber | driver |
| Baum | tree (source tree) |
| Zustand | state |
| Ausgang | outcome |
| Messergebnis | measurement result |
| Passung | fit |
| Eingabe | input |
| Anwender | user |

## Part 1: UI terms (rigdash)

Single word list for the English UI. Part 1 (static JS/HTML) created it; Part 2 (Python texts, `pdflip/propose*.py`,
`profile_couplings.py`) and Part 3 (catalog files) append to it and use the same terms. Style: terse developer English.

Not translated (API contract `flliper.*/1`, logic keys, CSS): JSON keys and field names (`values`, `source`, `verdict`,
`state`, `forms`, `outcome`, ...), enum values that the JS compares (`geht`, `ok_with_force`, `verweigert`,"
`unverified`, `vorgeschlagen`, `einfach`, `expert`, `GESTOPPT`, `TOT`, `HAENGT`, ...), env/flag names, CSS classes, element IDs,
file names. The JS shows an English display label for such a value and keeps the German value as the key.

## General terms

| DE | EN |
|---|---|
| geht | ok |
| only with --force | only with --force |
| mit --force | with --force |
| verweigert | refused |
| force unchecked | force unchecked |
| not judged | not judged |
| no run | no run |
| Planer-Rechnung | planner estimate |
| planner computation: fits / does not fit / unverified | planner estimate: fits / does not fit / unverified |
| oracle / oracle error | oracle / oracle error |
| Trockenlauf | dry run |
| Re-check | Re-check |
| Vorschlag | proposal (button: Propose) |
| unbelegt | unverified |
| belegt | verified |
| geborgt | borrowed |
| estimated | estimated |
| measured / not measured | measured / not measured |
| computed / not computed | computed / not computed |
| Datenblatt | datasheet |
| Herkunft | source |
| Quelle | source |
| Beleg | evidence |
| state (per value) | state |
| judgement / verdict | judgement / verdict |
| vorgeschlagen | proposed |
| solved by the launcher | solved by the launcher |
| overridden by you | overridden by you |
| unchecked since your change | unchecked since your change |
| default (not in the profile) | default (not in the profile) |
| Take over | Take over |
| Hinweis | note |
| Sperre | block |
| Ablehnung / Ablehnungscode | refusal / refusal code |
| dependencies / edges | dependencies / edges |
| Betriebsform | operating form |
| single card / TP only / Flip PP/TP / Dual PP/TP | single card / TP only / Flip PP/TP / Dual PP/TP |
| Rang | rank |
| Karte | card |
| card bars | card bars |
| weights / experts / draft / KV / Mamba state | weights / experts / draft / KV / Mamba state |
| Aktivierung | activation |
| Festposten | fixed items |
| Reserve / Frei | reserve / free |
| Rest | remainder |
| overflow | overflow |
| Kartenende / Kartengrenze | end of card / card limit |
| Korridor | corridor |
| seats (concurrent) | seats (concurrent seats) |
| Kontext | context |
| Anfrage | request |
| Leerlauf | idle |
| bucket (time grid) | bucket |
| handoff P->D | handoff P->D |
| neu gerechnet | recomputed |
| aus Cache | from cache |
| Schub | burst |
| Wanduhr | wall clock |
| Rechenzeit / Rechenrate | compute time / compute rate |
| Flipzeit | flip time |
| warmup / tail | lead / tail |
| Zerlegung | decomposition |
| idle / asleep | idle / asleep |
| Aushungerungs-Klemme | starvation clamp |
| Eintrittsstufe | entry stage |
| fehlt in IPC | missing in IPC |
| expert view / Simple / Expert | expert view / Simple / Expert |
| Startzeile | launch line |
| Fensterplan | window plan |
| Verlauf | history |
| overview | overview |
| Entwicklung | development |
| Sitzungen | sessions |
| Bausteine | building blocks |
| expected / actual | target / actual |
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

## Part 4 additions (display words found in the leftover pass)

| DE | EN | Where |
|---|---|---|
| value refusal (forcebar) / not forcebar | value refusal (forceable) / not forceable | `pdflip/refusals.py` `CLASS_LABEL`, `force_scope` (register display texts; codes, `klass` keys, `source`, `enforced_by` unchanged) |
| blocked / unchecked (force_state in the run report) | blocked / unchecked | `profil.py` `_STATE_DISPLAY` (API values stay German) |
| none / occupied / development (transport, confidence) | none / verified / development | `kartenplan_transport.py` |
| gerechnet (bar origin) | computed | `profile_couplings.py` |
| last 60 min / zoomed section | last 60 min / zoomed section | `flipzeit.window_label` |
| P active / D active | P active / D active | `ipcboot.PHASE_LABEL` |
| Vision load / compute / unload | Vision load / encode / unload | `ipcboot.VIS_NAME` |
| Layer-Tausch | layer swap | `ipcboot` phase `sub` |
| yes / tight / no (hw_fit level, shown) | yes / tight / no | `profil_planer.js`, `propose_verdict.py` |
| Planer-Rechnung | planner calculation (Python texts) / planner estimate (chip label); both appear, same meaning | |

