# F0-H: Post-Freeze-Fixes portiert, Linie NF (08.10.2026)

Zweig `desk/flliper-nf-f0h-1008` ab `desk/flliper-nf-1007` @ ffea1c00ff (umbenannter NF-Baum). Quelle: `a452294dd2..origin/desk/nf-int22-1008 @ 6b3bd1a6df`, alle 19 Fix-Commits (ohne Merge-Commits der Sammel-Integs), Liste `deskq/FIXES-NACH-FREEZE-1007.md` Nr. 1-18 (Sammel-Integs Nr. 3, 6, 10, 12, 17, 19 werden nicht einzeln portiert; Nr. 16 = L3-Fix, Nr. 18 = PP-ROOM-VOTE). 27B: keine Post-Freeze-Fixes, nichts zu tun.

## Verfahren (Rename-Regel des Kits, kein Handumbenennen)
Je Fix-Commit gehen die Dateiversionen vor dem Commit (Parent) und nach dem Commit des alten Baums durch die Kit-Schritte des F0-E-NF-Laufs auf einer Mini-Wurzel aus nur diesen Dateien: `rename_to_flliper.py apply --weg2 --ident-map merged_0928`, `ident_fix.py` (FIXMAP wie F0-E inkl. ident_map_1007, `IDENT_FIX_WEB=1`, refined, collision_ok-Dateien der NF-Linie), `english_audit extract` + Translation-Memory (kein Modellaufruf, kein Request an 30030/30099) + `pinned_units` + `apply`. Die Differenz Parent-umbenannt -> Fix-umbenannt wird per `git merge-file` dreiseitig in den umbenannten Baum gemergt. Werkzeuge: `deskq/done/f0h-port-tools-1008/` (portlib.py, port_one.py, chain.sh, commit.py, verify_final.py, finalize via Commit-Nachrichten).

Beweise des Verfahrens:
* Kalibrierung: dieselbe Pipeline auf dem Parent-Stand reproduziert die Dateien des umbenannten Baums byte-gleich (ee0683f429, 621a24ea06, 8ac55dc893; im 24ed9b508b-Fall weicht nur ab, was ein noch nicht portierter Vorgaenger-Commit liefert).
* Endprobe (`verify_final.py`): alle 43 Dateien des Deltas `a452294dd2..int22` aus der Pipeline auf dem int22-Stand sind **byte-gleich** zum portierten Baum (43 IDENTICAL, 0 abweichend, inkl. catalog.json).
* Konflikte: 0 (alle 19 Portierungen `MERGED-CLEAN`/`NEW`). Uebersetzungs-Einheiten (log/help/doc) der 43 Dateien: 3, alle TM-Treffer, 1 gepinnt Deutsch; nichts unuebersetzt. Kommentare/Docstrings bleiben Deutsch wie im ganzen Baum (Prosa nach dem Release).
* Namen im neuen Schema (aus der Regel): `FLLIPER_PDFLIP_ENABLE_FORM_A_ADMIT_ROOM_FIRST`, `FLLIPER_PDFLIP_ENABLE_CAPPARK_FLIP_HOLD`, `FLLIPER_PDFLIP_ENABLE_W3_SPILL_ANCHOR_POOL` (Default aus), Marker `#580 GRAMMAR-INTAKE PENDING` unveraendert; Spiegel geprueft: `name_compat.canonical_env_name("SGLANG_WEG2_ENABLE_W3_SPILL_ANCHOR_POOL") == "FLLIPER_PDFLIP_ENABLE_W3_SPILL_ANCHOR_POOL"`, `canonical_env` faltet den alten Namen.

## Tabelle: Reihenfolge, alter SHA -> neuer SHA, Tests
Testsatz je Commit: eigene Tests + Tests, die die geaenderten Module nennen (Module mit <=110 bzw. ab Commit 9 <=40 nennenden Testdateien), im umbenannten Baum auf dem Stand des Commits, `pytest_gedeckelt.sh`, leeres HOME. "nicht rot auf Basis" = Rote, die auf der Basis ffea1c00ff mit denselben Dateien nicht rot sind (`res/red_base.json`: 151 Rote dort).

| Nr | alt | neu | Listeneintrag | Betreff | Dateien | Ergebnis | nicht rot auf Basis |
|---|---|---|---|---|---|---|---|
| 1 | ee0683f429 | 4bbc6f7aa3 | Nr. 1 (H98e) | H98e: a Form A load-back past a taken ADMIT decides its room from the  | 6 | 46 failed, 1070 passed, 22 warnings, 27 subtests passed | test_pdflip_scheduling_slice_a_0907.py::test_t1_p_drains_the_whole_backlog_before_the_flip_at_p_concurrency |
| 2 | 17bf3e03f5 | 12c276fa90 | Nr. 1 (H98e, Katalog) | CATALOG-1008: Profil-Katalog fuer SGLANG_WEG2_ENABLE_FORM_A_ADMIT_ROOM | 1 | 1 failed, 175 passed, 1 skipped, 23 warnings, 6 subtests passed | keine |
| 3 | 43c0181dc9 | 78cbf9317f | Nr. 2 (H110) | H110: token-cut admission prices the rows this pass allocates (load-ba | 3 | 46 failed, 1076 passed, 20 warnings, 27 subtests passed | test_nf_form_a_admit_room_first_h98e.py::AdmitRoomFirstTest::test_a_real_shortfall_costs_the_shortfall_not_the_drain |
| 4 | 8a50452564 | 67b04baf7e | Nr. 3 (H98e-Test im int16-Integ) | H98e test: shortfall price now includes the H110 first chunk (per rank | 1 | 123 passed, 17 warnings, 6 subtests passed | keine |
| 5 | e0e23e110a | a6e6d5a42d | Nr. 5 (W98) | W98: die Hand-off-Dateien der Arena sind der Arena-Schreiber -- int16  | 2 | 45 failed, 1935 passed, 14 skipped, 27 warnings, 175 subtests passed | keine |
| 6 | 8ac55dc893 | 551f17d84e | Nr. 4 (#580/#791b) | #580 nach #791b: grammatik-fertige Requests kommen nach dem Memo-Drain | 3 | 48 failed, 146 passed, 17 warnings, 81 subtests passed | keine |
| 7 | 371e388a4d | 02ffbb2eff | Nr. 7 (CAPPARK-FLIP-HOLD) | CAPPARK-FLIP-HOLD: #248h capacity re-queue does not run while a flip p | 3 | 8 failed, 461 passed, 1 xfailed, 22 warnings | keine |
| 8 | b48335c795 | 59559eeff6 | Nr. 7 (user_flipzeit) | USER-FLIPZEIT: user_flipzeit_ms beside the begin->done total (instrume | 3 | 4 failed, 168 passed, 17 warnings | keine |
| 9 | 621a24ea06 | 3935bab854 | Nr. 7 (client_ip) | request_done.client_ip: origin address of a request (first X-Forwarded | 8 | 1 failed, 56 passed, 17 warnings | keine |
| 10 | 48274837e6 | ca8972dc94 | Nr. 7 (client_ip, letzte XFF-Adresse) | client_ip: take the LAST X-Forwarded-For address (the peer the owui pr | 2 | 1 failed, 10 passed, 17 warnings | keine |
| 11 | 74453ef6dd | 4673838d64 | Nr. 8 (pre_wait_ms) | USER-FLIPZEIT: pre_wait_ms diagnosis, D>P without a proven waiter arri | 2 | 4 failed, 169 passed, 17 warnings | keine |
| 12 | a2455041e0 | f5bc3357c0 | Nr. 8 (user_flipzeit nach Nutzerdefinition) | USER-FLIPZEIT: D>P span starts at max(last D token of THIS D phase, fi | 3 | 5 failed, 168 passed, 17 warnings | test_pdflip_flipzeit_dp_user_0930.py::test_without_a_park_the_last_served_leg2_is_the_decode_end |
| 13 | 265e273594 | e3446e4988 | Nr. 9 (PW BACKUP-WALL) | PW BACKUP-WALL: one free basis -- a short peel's remainder is known-un | 7 | 16 failed, 555 passed, 1 xfailed, 22 warnings | keine |
| 14 | bc583417b0 | 82f34e0e5f | Nr. 9 (PW-R D-WALL-HEAD) | PW-R D-WALL-HEAD: a D head the backup wall makes unservable is answere | 3 | 6 passed, 17 warnings | keine |
| 15 | fc0adfc41e | eec6245570 | Nr. 9 (DQH D-HEAD-HOLD) | DQH D-HEAD-HOLD: a refused D head is re-evaluated on a state change or | 3 | 5 passed, 17 warnings, 3 subtests passed | keine |
| 16 | ccf0daefbe | 16ac83976b | Nr. 9 (#1537 Ratchet) | #1537 ratchet: name PW-R's group MIN in _get_new_batch_prefill_raw (en | 1 | 19 passed, 20 warnings | keine |
| 17 | bff5913a62 | 5992773f79 | Nr. 11 (UD-H) | UD-H: on the local-PP floor a refused write_back leaf with host-only c | 3 | 52 passed, 17 warnings, 2 subtests passed | keine |
| 18 | 24ed9b508b | a51b4f600a | Nr. 18 (PP-ROOM-VOTE) | PR PP-ROOM-VOTE: every PP stage applies one agreed room R_m in pass m  | 8 | 2 failed, 241 passed, 18 warnings, 2 subtests passed | keine |
| 19 | 7c85710e30 | 98cc90807a | Nr. 16 (L3 W3-Spill) | W3-ANCHOR-POOL: the W3 arena spill works through the HostPoolGroup's a | 3 | 2 passed, 18 warnings | keine |

Die Reihenfolge ist die der Sammel-Integs (int16 -> int17 -> int18 -> int19/20 -> int21 -> int22), Seitenzweige in der Reihenfolge ihrer Merges.

## Roten, die nicht aus dem Port stammen (belegt)
* Nr. 3: `test_nf_form_a_admit_room_first_h98e::test_a_real_shortfall_costs_the_shortfall_not_the_drain` rot, bis Nr. 4 (alt 8a50452564) den Preis des H110-Erstchunks nachzieht - dieselbe Wechselwirkung wie im alten Integ.
* Nr. 12 und danach: `test_pdflip_flipzeit_dp_user_0930::test_without_a_park_the_last_served_leg2_is_the_decode_end` (alt `test_weg2_flipzeit_dp_user_0930`) ist auch auf dem ALTEN `nf-int22 @ 6b3bd1a6df` rot (gleiche Assertion `('flip_begin', 1100) == ('last_d_served', 21100)`), gruen auf a452294dd2 (`res/flip_old_int22.log`, `flip_old_base.log`, `flip_new_final.log`). Der Fix a2455041e0 verwirft das veraltete letzte D-Token einer frueheren Phase, der Test von 09-30 erwartet es noch. **Quell-Befund fuer den NF-Sitz**, hier nicht veraendert.
* Nr. 1: `test_pdflip_scheduling_slice_a_0907::test_t1...` einmal rot im breiten Nachbarlauf (asyncio-Zeitlauf mit sleep(0.25) unter Last), in allen spaeteren Laeufen und im Endlauf gruen.

## Gates auf dem portierten Baum (Endstand) gegen die umbenannte Basis ffea1c00ff
| Satz | portiert | Basis | rote Namen |
|---|---|---|---|
| Kit-Testsatz (`run_tests.sh`, tests_all.txt + compat_shims) | 10 failed, 1162 passed, 9 skipped, 3 errors | 10 failed, 1162 passed, 9 skipped, 3 errors | identisch |
| gateF (Planer/Katalog/Compat, 26 Dateien, HOME leer, Dockerdir wie F0-E) | 6 failed, 429 passed, 21 skipped | 6/429/21 (Aufzeichnung F0-E `gateF2_new.log`, gleicher Baum; nicht neu gefahren) | identisch (TestDryRunGolden x4, apc nf_abl dry-run, `test_build_and_shipped_catalog_agree`) |
| rigdash/tests (HWPROFIL_TREE=COUNTING_TREE) | 3 failed, 964 passed, 15 skipped | 3 failed, 964 passed, 15 skipped | identisch (Playwright fehlt) |
| rigmon test_hardware_profile_persist_1006 + _950 | 3 failed, 52 passed | 3 failed, 52 passed | identisch |
| 50 Dateien, die in einem Commit-Lauf rot waren (Endstand vs Basis, gleicher Lauf) | 151 failed, 678 passed, 5 skipped | 150 failed, 679 passed, 5 skipped | genau 1 Rotes mehr: der Befund Nr. 12 oben |

## Offen / nicht belegt
* Kein Metallbeweis (keine GPU, kein Boot); die Metall-Spalten der Fix-Liste gelten weiter.
* `catalog.json`: nur die umbenannte Differenz des alten Rebuilds (Eintrag `FLLIPER_PDFLIP_ENABLE_FORM_A_ADMIT_ROOM_FIRST`, Zeilenverschiebungen von environ.py) ist uebernommen; ein voller Katalog-Neubau ist F0-F (der Test `test_build_and_shipped_catalog_agree` bleibt wie auf der Basis rot).
* Das Profil `nf-int4-h6-abl-xc-w3sp.env` (Fix-Liste Nr. 16) steht in keinem Commit des alten Baums, nur als Datei in `/spinning/gpu-arb/docker/profiles/`; in-tree Konvertierung der Profile = F0-G.
* `tools/owui_proxy/*` (neu mit Nr. 7/client_ip): Hostpfade `/spinning/htsglang-gpu`, `/opt/owui_proxy`, `htsglang front` im Test-Docstring bleiben (R2: Hostpfade und Evidenz nicht umbenannt).
* Zweite Runde fuer Fixes ab int23: `python port_one.py <sha>` je Fix-Commit in der Reihenfolge der Liste, danach `verify_final.py` gegen den neuen Quellkopf (Pfade in portlib.py: F0E-Laufverzeichnis fuer collision_ok.json/identfix_merged.json/uf.translated.jsonl bleibt Voraussetzung).
