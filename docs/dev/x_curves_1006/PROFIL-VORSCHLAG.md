# X aus Profilkurven — Profil-Vorschlag und Schnittstelle (06.10.2026)

Teil F des AUFTRAG-x-kurven-1006. **Nur Vorschlag**: `docker/profiles/*.env`
gehören dem 27B-Sitz / Nutzer und werden hier nicht geändert. Code: Branch
`desk/x-curves-1006` (Basis cand4c `369f31e2e2`).

## 1. Die zwei Formen (ÄNDERUNG 06.10. ~11:00Z: „kein live und kein curve-capped, nur fixed|curve“)

| Form | Profilzeilen | Wirkung |
|---|---|---|
| fest | `--x-mode fixed` `--tp-prefill-max-tokens N` | X = N für den ganzen Boot; keine Live-Samples, keine Neulösung, keine Hysterese |
| Kurve | `--x-mode curve --x-curves <Datei>` | X je Anfrage aus der Kurvendatei; D's W50-Riegel = Hüllkurve der Datei (der Launcher setzt ihn, harte Grenze) |
| Kurve mit Deckel | `--x-mode curve --x-curves <Datei> --x-ceiling-tokens N` | Nutzerentscheid 06.10. (NF-Sitz): der manuelle Deckel klemmt X von oben, die Kurve entscheidet darunter; D's W50-Riegel = der NIEDRIGERE von Hüllkurve und Deckel (nie unter dem Start-X, H84). Ohne Deckel wie die Zeile darüber |
| (ohne Flag) | kein `--x-mode` | die Front genau wie vor dem Flag (heute: Live-Neulösung, `SGLANG_WEG2_ENABLE_X_COST_LINE` Default True). Nicht still geändert — was der Default werden soll, entscheidet der Nutzer |

`--x-mode live` und `--x-mode curve-capped` werden abgewiesen (argparse-Choices; im Code W196).

**Manueller Deckel in `curve` — entschieden 06.10. (NF-Sitz): erlaubt, klemmt von oben; Riegel = min(Hüllkurve, Deckel).** Beleg, warum der Riegel hart ist: der W50-Riegel ist
eine echte D-Grenze: D verweigert jede Anfrage, deren ungecachter Umfang nach
dem Prefix-Match über seinem eigenen `--tp-prefill-max-tokens` liegt
(`python/sglang/srt/managers/scheduler.py:14089` `_weg2_x_refuses`, Meldung
`W50 Weg2TpPrefillExceeded` Z. 14242-14247; Ausnahmen davor: Kapazitäts-Park
Z. 14225 und RESUME-VIA-P Z. 14239), die Front leitet sie dann durch P um
(`python/sglang/srt/weg2/front.py:12802` `WEG2 W50-REROUTE ... reason=x_refusal`,
X-REQUEUE; die zweite Verweigerung nach P ist terminal W35/W53). Den Riegel setzt
der Launcher (`launcher.py` `resolve_x_ceiling`, D bekommt `--tp-prefill-max-tokens`
= Deckel bzw. Start-X). In `curve` bleibt er hart: der Launcher setzt ihn auf die
Hüllkurve (`x_curves.resolve_launch`, `ceiling_for_d = envelope`), die Front
klemmt jedes Anfrage-X darunter (`front.py` `_x_route_of`, `cap = tp_prefill_max_tokens`
= Hüllkurve ≤ `x_ceiling_tokens`). Ob man zusätzlich einen manuellen, niedrigeren
Deckel neben `curve` setzen darf, ist entschieden (ja): `x_curves.resolve_launch`
rechnet die Hüllkurve unter dem Deckel (`x_envelope(curves, cap=Deckel)`), das ist
`ceiling_for_d`; `resolve_x_ceiling` hebt ihn wie bisher nie unter das Start-X.
Launcher-Zeile `X MODE: ... manual ceiling --x-ceiling-tokens N BINDS|does not bind
(curves' own envelope E)`; ROUTE-VERDICT `... clamp=cap, ceiling=N)`. W195 entfällt.

`--x-curves-beyond clamp|refuse` (optional, Default `clamp`): eine Anfrage
tiefer als die Kurve reicht wird an der tiefsten Zeile bepreist und in der
ROUTE-VERDICT benannt (`clamp`), oder mit W193 (503) abgewiesen (`refuse`).

## 2. NF: `docker/profiles/nf-int4-h6-abl.env` (Vorschlag)

```diff
+PROFILE_X_CURVES=/opt/htsglang/profiles/nf/x_curves.nf-int4-h6-abl.RTX5090-RTX3080-RTX3080.json   # Kurvendatei aus calib_x_<s9wwu9>.jsonl (tools/build_x_curves.py)
 ...
 PROFILE_ARGS=(
 ...
   --x-ceiling-tokens 12288                             # RC2 (H84); unter curve: manueller Deckel, Riegel = min(Hüllkurve, 12288)
+  --x-mode curve                                       # X-CURVES 1006: X je Anfrage aus den Profilkurven (Nutzer 06.10.), nicht live
+  --x-curves "$PROFILE_X_CURVES"                       # EINE Kurvendatei je Modell x Form x Hardware; D's W50-Riegel = ihre Hüllkurve
 )
```

NF trägt heute kein `--tp-prefill-max-tokens` im Profil (Start-X = Launcher-Default
4096) — dort fällt nichts weg. Das Start-X bleibt nur noch der Boden des
X-SOLO-Bandes (`_x_band_floor`), solange `SGLANG_WEG2_X_BAND_FOLLOWS_PRICE`
es nicht ohnehin auf das geltende X hebt.

`_form SGLANG_WEG2_ENABLE_X_COST_LINE 1` kann stehen bleiben: unter
`--x-mode curve` ruft die Front `resolve_x_live` / `_resolve_x_cost_line`
nicht mehr auf. Der Schalter wirkt dann nur noch auf die P-Kostenring-Lesung
(Instrument) und auf `_x_excursion_band_off` (X-SOLO-Bandboden = geltendes X).

## 3. 27B (Flip-Formen): `docker/profiles/27b.env`, `profiles_release/27b-base.env` (Vorschlag)

```diff
+PROFILE_X_CURVES=/opt/htsglang/profiles/27b/x_curves.27b.RTX5090-RTX3080-RTX3080.json
 ...
-  # RC7-X (X-Review V, Operator 25.09.): Start-X ausdruecklich 4096 -- ...
-  --tp-prefill-max-tokens 4096
   --x-ceiling-tokens 12288                             # unter curve: manueller Deckel, Riegel = min(Hüllkurve, 12288)
+  --x-mode curve                                       # X-CURVES 1006 (Nutzer 06.10.): X je Anfrage aus den Profilkurven
+  --x-curves "$PROFILE_X_CURVES"
```

Die feste `--tp-prefill-max-tokens 4096`-Zeile fällt weg (Nutzerentscheid 1).
Der Launcher-Default ist ebenfalls 4096; ohne die Zeile ändert sich das
Start-X also nicht, es ist nur kein „Dauerwert X“ im Profil mehr. Wer bis zur
27B-Kurve weiterbooten will: `--x-mode fixed --tp-prefill-max-tokens 4096`
(fest, ohne Live-Neulösung) oder vorläufig gar nichts (ohne Flag wie heute).
`--x-ceiling-tokens 12288` darf in beiden Profilen stehen bleiben (Nutzerentscheid
06.10.): unter `curve` klemmt er X von oben; ohne die Zeile entscheidet allein die
Hüllkurve.

**Dual-Profile** (`27b-nvfp4-dual*.env`): keine X-Zeilen, keine Änderung
(Nutzerentscheid 1).

## 4. Kurvendatei (Format `weg2-x-curves/1`)

JSON, eine Datei je Modell x Form x Hardware; Schema = `msgspec`-Structs in
`python/sglang/srt/weg2/x_curves.py` (unbekannte Felder werden abgewiesen):

```json
{
 "format": "weg2-x-curves/1",
 "identity": {"model": "<Checkpoint-Verzeichnisname>",
              "form": "arch=..,experts=..,draft=..,kv=..,flip=.. (Residue-Achsen der WEG2-FORM-Zeile)",
              "hardware": "RTX5090,RTX3080,RTX3080",
              "created": "2026-10-06T13:10:00Z", "source": "<Boot-Tag>"},
 "p_curve": {"rows": [{"depth": 0, "n": [512, 2048, 4096], "ms": [161.0, 283.8, 447.7]}, ...],
             "chunk_tokens": 4096, "instrument": "P Prefill rank batch gpu_ms, max over ranks, chunks==1",
             "depth_interp": "linear", "points": 1234},
 "d_curve": {... wie p_curve ...},
 "flip_price": {"depth": [4120, 200120], "seconds": [4.0, 6.0], "attribution": "calib_prefix", "pairs": 14}
}
```

* `rows[].ms[i]` = EIN Prefill-Forward (ein Chunk) von `n[i]` neuen Tokens bei
  `depth` gecachten Tokens. Eine Anfrage = Summe ihrer Chunks, jeder bei seiner
  eigenen Tiefe (`chunk_tokens` > 0).
* `flip_price` = D→P + P→D in Sekunden, nach Kontexttiefe der Anfrage, die den
  Flip auslöste.
* Identität: `model` = `form.model_key`, `form` = die Residue-Achsen der
  WEG2-FORM-Zeile (wie der X-COST-LINE-/PARK-RT-Schlüssel), `hardware` =
  `card_identity.inventory_signature` der geordneten Karten. Prüft der
  Launcher (alle drei) und die Front (Modell + Form) — fremd = W192.

Bau: `tools/build_x_curves.py calib_x_<Boot>.jsonl --out <Datei> --model … --form … --hardware … --p-chunk-tokens 4096 --d-chunk-tokens 4096 --cap 12288`
(druckt die Plot-Prüfung als Text: alle Zeilen, ms/Token, Monotonie-Warnung, Flip-Preis, X nach Tiefe x k).

## 5. Schnittstelle für das Dashboard (später, NACH fable-planner)

Nicht gebaut, nur festgehalten:

* **Profilzeilen**: `--x-mode {fixed,curve}`, `--x-curves <Pfad>`,
  `--x-curves-beyond {clamp,refuse}`,
  `--tp-prefill-max-tokens <N>` (nur bei `fixed` sinnvoll). Abhängigkeiten:
  `--x-curves`/`--x-curves-beyond` nur bei `curve` (sonst W194);
  `curve` + `--x-ceiling-tokens` = Deckel (Riegel = min(Hüllkurve, Deckel)); `curve` ohne
  Datei = W190; `live`/`curve-capped` = abgewiesen (W196).
* **Kurvendatei lesen**: `x_curves.load(path)` → `XCurves`; Anzeige mit
  `x_curves.text_table(curves, cap=…)` oder direkt aus den Zeilen (Plot P/D ms
  über n je Tiefe, Flip-Preis über Tiefe, X über Tiefe für k = 1, 2, 4).
* **Live-Zustand** `/weg2/state` (nur bei ausdrücklichem `--x-mode`):
  `x_mode`, `x_curves_source` (`<Datei>@<Boot-Tag>`), `x_curves_envelope`,
  `x_curves_last` = `{x, depth, k, why, clamp}` der letzten Anfrage;
  `x_tokens` = Boot-X (Hüllkurve bzw. festes X) wie bisher.
* **Logzeilen**: Launcher `X MODE: …`, `X PROVENANCE: …; x-mode=…`, `X CEILING: … [x-mode=…]`;
  Front `WEG2 X MODE <mode>: …`, `WEG2 ROUTE-VERDICT … (#1290); x_mode=… X_req=… (curve …, why, clamp=…) depth=… k=… price=…s notes=… curves=…`.
