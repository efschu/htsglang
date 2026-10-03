# Form-Messmatrix v3: Format des Kalibrier-Eingangs (verbindlich für 27B und Next Flash)

Stand 29.09. (27B-FORM-MESSMATRIX). Code: `python/sglang/srt/weg2/form_measures.py`.
NF stellt `nf_decode_matrix.py` auf dieses Format um (Absprache 29.09.); es gibt keinen zweiten Importer.

## 1. Ein Kalibrierlauf = ein Verzeichnis

Je Arm (eine Form, ein Boot, D-only nach M1-Regel):

| Datei | Inhalt | Pflicht |
|---|---|---|
| `<arm>_ladder.jsonl` | eine JSON-Zeile je Stream und je Gruppe (unten) | ja |
| `<arm>_server.log` | Server-Log des Arms; gelesen wird NUR die Rundenzeile `Decode rank batch … gpu-ms: X (compute C, wait W)` und `Decode batch … accept len: A` | ja (Rundenzeit) |
| `moe_heat/moe_heat_<group>_tp<r>_*.json` | #276-Hitze-Records (`SGLANG_DEBUG_MOE_HEAT`), NF | nur NF |

Die Zuordnung Arm -> (Identität, Form des Profil-Records `FORM_MATRIX_AXES`) liefert der Aufrufer als
`--identity-map` (JSON `{arm: {identity: {...}, form: "<name>"}}`).

## 2. `<arm>_ladder.jsonl`

`role: "stream"` (eine Zeile je Anfrage):

| Feld | Typ | Bedeutung |
|---|---|---|
| `point` | str | `<text>@<depth>`, z.B. `code@32k`; text ∈ code/prose/thinking, depth ein Tiefenpunkt |
| `kind` | str | text (redundant zu point; gewinnt, wenn gesetzt) |
| `target_tokens` | int | Kontexttiefe in Token (wird nach OBEN auf den Tiefenpunkt gebuckelt) |
| `bs` | int | gleichzeitige Anfragen der Gruppe |
| `temp` | str | `kalt` / `warm` |
| `rep` | int | Wiederholung (Gruppen-Schlüssel mit point/bs/temp) |
| `t_send`, `t_end` | float | Unix-Sekunden; das Fenster [min t_send, max t_end] der Gruppe wählt die Rundenzeilen |
| `completion_tokens` | int | erzeugte Token; < 95 % des Lauf-Maximums = EOS-kurz -> Gruppe verworfen |
| `short` | bool | optional, vom Treiber gesetzt (ignore_eos-Leiter) |
| `error` | str/null | gesetzt -> Gruppe verworfen |

`role: "group"` (eine Zeile je Gruppe): `point`, `bs`, `temp`, `rep`, `streams_short` (int; > 0 -> Gruppe verworfen).

Die Treiber müssen `ignore_eos` senden (27B `decode_ladder_cw.py` seit 29.09.). Eine Gruppe mit einem kurzen Stream misst nicht ihr bs.

## 3. Zellwert (was der Import schreibt)

- `round_ms_median`, `n`, `p10`, `p90`: Runde = langsamster Rang der Runde (worauf die Runde wartet).
- `compute_ms[r]`, `wait_ms[r]`: Median je Rang, nur wenn jede Rundenzeile den Split trägt (`SGLANG_DEBUG_COLLECTIVE_CLOCK_GRAPH_NODES=1`).
- `accept_len`: Median der `accept len` im Fenster.
- Unter `min_rounds` (5) Runden: `ungemessen` mit Grund. Ein leerer Lauf überschreibt nie `gemessen` oder `unmoeglich:*`.

## 4. Next-Flash-Fehlgriffe: NUR aus Records, nie aus Log-Zeilen

- `misses_per_seat[r]`: Quelle ist der #276-Hitze-Record (`pool_heat.py`, Kind `moe_heat` v1):
  - `not_local` = Lanes an Experten, die der Rang nicht besitzt.
  - `steps` = erfasste Schritte.
  - `form_measures.heat_misses()` summiert je Rang und liefert `misses_per_step`.
  - **Grenze:** Der Record entsteht je PHASE (D-Schlaf), nicht je Zelle. Einer Zelle zuordenbar ist er nur, wenn der Arm genau eine Zelle je Phase misst. Sonst bleibt das Feld leer (= ungemessen).
  - Zellgenau braucht es einen Zähler-Schnappschuss je Leiter-Gruppe: NF-Instrument (z.B. `flush` auf Anforderung oder ein Zählerstand in state.json).
- `miss_cost_ms[card]`: Quelle soll die Rundenzerlegung `DECODE-ROUND-COST` (H23) mit Worker-Attention sein.
  - **Heute ist das nur eine Log-Zeile** (`wake_round_census.py`), kein Record. Deshalb liest der Import sie nicht.
  - Nötig ist ein Record-Produzent: DECODE-ROUND-COST als JSON-Record bzw. Event in `events.jsonl`, je Rang mit `gpu_ms/compute_ms/miss_ms`. Das liefert NF.
- Beide Felder bleiben bis dahin leer. Die Planer-Saat (0,1/0,2 ms) wird nie als Messung eingetragen.

## 5. Kompatibilität mit dem Belegungsplan (VRAM-VERTRAG-0929)

Eine Form des Records `FORM_MATRIX_AXES` ist eine Belegungsform: dieselben Achsen `roles/weights/tokens/moe_ratio/owned_cut/fr_d/scratch_rows/kv_stage/precision/spec/transport`.
Der Belegungsplan soll seine Form unter demselben Namen und denselben Achsen führen, dann ist die Zelle der Zeit-Tabelle direkt der Plan-Form zuordenbar.
`capacity[form][depth] = max bs` im Record ist der Platz, den der Belegungsplan für die Form vorsieht; Zellen darüber sind `unmoeglich:kapazitaet`.
