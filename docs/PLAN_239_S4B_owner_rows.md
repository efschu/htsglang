# #239 S4b (F14, vorher als zweites 'F13' registriert) – Entwurf nach Code-Lage (28.09., NF-Implementierer)

Basis: desk/nf-s4b-239-f13-0928 auf d39836412c.

## Befund, der den Umfang ändert
- plan_s3_251.md §F4: **Host-Anteil 0 ist die Optimum-Form** bei x1/x2. Der Host (TP0) hat dann 0 FA-Zeilen, die Worker tragen ALLE KV-Bytes.
- Heute ist es umgekehrt: TP0 ist der einzige Rang mit Host-Tier-Bytes (ArenaMHAHostPool, Arena-Rebind, R12-Schatten entscheidet das Host-Leben). Die Worker sind byteless (Null-Tier, plain Pools mit 0 GB).
- Unter dem Schnitt gilt das nur noch für Mamba/QSA/Draft. Für KV liegen die Bytes bei den Workern, die Baum-Autorität bleibt aber bei TP0.

## Teile
1. **Store-Fenster** (FERTIG, abdd7ec919). Owner-Zeilen-Fenster: identitätsadressiert, Läufe `[k*S+lo, k*S+hi)`. S=64 = Seite, also ein Lauf je Slot.
2. **Backend und Attach.**
   - Unter owner ctx + kanonischer Seite: KV-Fenster = `owner_row_window` (hi>lo) oder KV-ENTHALTUNG (hi==lo).
   - Bei Enthaltung macht set/get auf dem KV-Key kein IO; exists bleibt die Datei-Vollständigkeit.
   - Worker mit Zeilen bekommt ein echtes File-Backend mit NUR dem KV-Fenster (kein Mamba-Blob, keine QSA-/Draft-Seite).
   - Der Attach-Riegel page_size!=1 fällt nur für die Owner-Zeilen-Form.
   - `_page_backup`: owner_mask=None unter der Owner-Zeilen-Form. Jede Seite wird von jedem Owner geschrieben, und das Fenster schneidet die Bytes.
   - Batch-C-Pfade (pageio, Arena): Identitätsfenster über Pack/Unpack oder den Python-Pfad (v1).
3. **L2 (Arena) mit Owner-Zeilen.**
   - ArenaMHAHostPool.bind nimmt Owner-Zeilen-Extents: k/v-Views je Layer wie heute (ganze Seite), own_extents = Owner-Zeilen.
   - Backup Karte→Slot und Load Slot→Karte über die Owner-Abbildung (global → kompakt, wie `_dcp_kv_transfer_pairs`).
   - Ein Slot ist complete, wenn alle Owner geschrieben haben (Arena-Header, gemeinsam).
   - TP0 mit Anteil 0: KV-Pool byteless, claimt aber mit (Buchhaltung). Das Rebind-Verdikt liest „Slot complete“ aus dem gemeinsamen Arena-Header, also ohne Kollektiv.
4. **R12-Schatten.**
   - TP0 bleibt Baum-Autorität.
   - Das KV-Transit/Rebind-Verdikt stützt sich auf die Slot-Vollständigkeit statt auf TP0-Bytes.
   - Die Worker behalten ihre KV-Zeilen, bis TP0s Verdikt kommt.
5. **Loadback #988/H105, Tail-Adopt/Handoff.**
   - Der KV-Teil läuft je Owner mit Vollständigkeit als MIN (existiert als S3d-Vote).
   - Mamba bleibt TP0.
   - Der #243-Halter zählt Worker-Seiten.
6. **#1424d/g auf Worker-Bäumen.** PROOF-CUT mit Worker-Stimmen; Orphan-Pass auch auf Worker-Arenen.
7. **F14 wired=True.** Riegel fällt, dann Slice-Smoke und Serving-Boot.

## Teil 3b – Experten-Eigentum zur 5090 (Nutzer-Korrektur 28.09. 11:00Z, fester Teil der Zielform)
Legt der Schnitt die FA-KV auf die 3080er, verlieren deren Karten Expertenzeilen. Die 5090 gewinnt den Platz, den die KV freigibt. Das Eigentum (`--rank-moe-ratio`) ist deshalb eine freie Variable des Solves, nicht die Form-A-Konstante 183/137/168.

**Solve** (Desk-Dry-Run `tmp/r989/uneven_dcp/own_dryrun_239.py`; echter Checkpoint, `plan_d_residency`, Budget rc12r [26328,17664,17864], FR_D 0,06/0,51/0,48, Scratch 90/48/48; keine GPU):
- Gitter:
  - r0 183..327 in 16er-Schritten, r1 in 12er-Schritten, Summe 488;
  - Schnitte kein / [0,a,64-a] mit a ∈ {0,16,32,48,64} / [16,24,24];
  - verworfen wird jede Form mit Planer-Verweigerung (W130 u. a.) oder Deckel < Scratch+2.
- Ziel: min max_r T_r bei bs1, Nebenblick bs4. T_r = Fehlgriffzeilen × 48 Layer × c_r.
  - Fehlgriffe je MoE-Layer und Runde bei gleichverteiltem Zugriff: (E_r − held_r)·(1 − (511/512)^(bs·4·10)).
  - c_r = 0,1 ms (5090) / 0,2 ms (3080). **UNGEMESSEN, Hochrechnung.**
- x1-Regel über die Fehlgriff-LAST: kein Worker trägt mehr Fehlgriff-Zeit als in Form A heute (9,39 / 30,33 ms bei bs1).

| Form | ratio | Schnitt | lokale Experten | Deckenzeilen | T_bs1 je Rang [ms] | max bs1 | max bs4 |
|---|---|---|---|---|---|---|---|
| Form A heute | 183/137/168 | – | 193/145/177 | 119/132/135 | 26,7 / 9,4 / 30,3 | 30,3 | 108,3 |
| Schnitt, altes Eigentum | 183/137/168 | joint → 0/46/18 | 193/145/177 | 146/113/128 | – (W130 Weg2DCardNearOom verweigert) | – | – |
| **Schnitt + Eigentum (neu)** | **215/117/156** | **0/48/16** | **226/124/165** | **146/112/128** | **28,9 / 8,7 / 26,7** | **28,9** | **103,2** |
| nächster Kandidat | 215/105/168 | 0/64/0 | 227/111/177 | 146/106/135 | 29,3 / 3,6 / 30,3 | 30,3 | 108,3 |

- Ergebnis: Mit dem Schnitt wird die 5090 frei von FA-KV, ihr Deckel steigt 119 → 146. Sie übernimmt 33 lokale Experten, dafür geben TP1 21 und TP2 12 ab.
- Die Fehlgriff-Zeit fällt im Maximum um ~5 %, der Engpass wandert von TP2 zu TP0. Das ist eine Hochrechnung, die Messung macht der S4a-Bogen.
- Ohne die Eigentumsverschiebung verweigert der Planer den Schnitt (W130).

**Laufzeit-Stütze (geprüft am Code, kein neuer Pfad):**
- Das Eigentum ist heute schon ein freier Parameter: `--rank-moe-ratio` → `planner/placement.py` (`_build_cost_model`, `compute_placement_struct`) → Experten-Karte (#75).
- Store-Slots und 1:1-Platztausch (#82) leiten sich je globale Id aus dieser Karte ab.
- Frühere Formen liefen mit anderen Vektoren (z. B. 3991/1000/1000 in xsn293), die Mechanik hängt also nicht an 183/137/168.
- P-Seite: P (PP3) hat eigenes Eigentum je PP-Stufe; der Flip tauscht je globale Id Karte↔Store. Ein anderes D-Eigentum ändert nur, welche Ids die 5090 in D hält, nicht den Pfad.
- **Offen (Metall):** mehr Experten auf der 5090 heißt mehr Bytes 5090↔Store je Flip. Die Flipzeit misst der S4a-Bogen.
- Aufwand: kein Code, nur das Profil. Dry-Run erneut, sobald Teil 3 (L2 mit Owner-Zeilen) die Worker-Budgets ändert.

**Profil:** `/spinning/gpu-arb/docker/profiles/nf-s4a-cut.env` → `--rank-moe-ratio 215,117,156`, `--d-kv-token-cut 0,48,16` (statt `joint`, weil der Joint-Solver ohne Eigentum den verweigerten Schnitt 0/46/18 wählt), Kommentare F13 → F14.

## Aufwand (Schätzung, keine Messung)
- Teil 2: ~0,5 DT.
- Teile 3+4: ~1,5–2 DT (Arena-Pfade mit kompakter Abbildung, Rebind-Verdikt).
- Teil 5: ~0,5–1 DT.
- Teil 6: ~0,5 DT.
- Gesamt ~3–4 DT statt 1–1,5, dazu 2–3 Slice-Smoke-Boots.
