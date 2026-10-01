# efeu-TP14: Qwen3.8-35B-A3B-Distill Q3_K_M mit fLLiper — Plan (01.10.2026)

Zweig: `desk/efeu-35b-q3-fllliper-1001` (htsglang), Basis `backup/gguf-q4-bringup-651-efschu` @ `421f882723`
(baumgleich mit `feat/gguf-q4-bringup-651` @ `00a1c50fcb`, nur Autor-Mail verschieden; die Backup-Linie trägt die konfigurierte Repo-Mail).
Laptop-Arbeitsbaum: `/root/efeu35q3/` (Modelle, Logs, Ergebnisse, Skripte), Serving-Baum bleibt `/root/651-p2/sglang_src` (Overlay, kein git — wird in den Zweig gesichert, s. §9).
Werkzeuge im Zweig: `docs/dev/651/efeu35q3/`.

Dieser Plan ist das Arbeitsdokument; der Stand steht in §11 (Ergebnisse) und wird beim Umsetzen nachgetragen.

---

## 0. Vorarbeit — was existiert und hier NICHT neu gebaut wird

- #651/#655 (FINAL_651.md, HANDOFF_655.md): Laptop serviert Qwen3.6-35B-A3B-UD-Q4KM-noQ6K über `htsglang-ondemand.service`
  (Port 31651 → Backend 31661, Laden auf Anfrage, Parken nach 60 s). Guard v2 (numpy-Orakel), wedge_policy (cp 256),
  min-KV-Gate, Mamba-Slots 4, Tool-Parser qwen3_coder.
- **Wichtig, im Auftrag nicht erwähnt:** Seit 09.08. ist im Dienst `40-kt655.conf` AKTIV (Nutzer-Order, CPU-Experten über
  kt_kernel LLAMAFILE): alle 256 Experten auf der CPU, Graphen AUS, CTX 16384, Decode ~8,6 tok/s statt 15,5. Grund war
  GTT-Platz für Kontext. Das ist der eigentliche Rückfall-Zustand, nicht „Q4 mit Graphen“.
- kt_kernel-Messung (421f882723): CPU-Experten kosten 46 % Decode, Prefill unverändert (121 vs 123 tok/s @978 Tok);
  Teilaufteilung GPU/CPU-Experten auf GGUF unmöglich (ggml_moe_a8_vec kennt keine −1-Experten).
- PP=2 CPU+iGPU (dichtes Vehikel Qwen2.5-1.5B): 0,69x Prefill an der 1:1-Teilung, monoton besser bis 0/28 (iGPU allein);
  Ko-Lauf-Steuer auf der iGPU +17 %/Layer. Für den 35B fehlt die GDN-CPU-Kernelfamilie.
- Spec/NEXTN zahlt nicht (212 ms/Schritt vs 64,8 ms).
- HiCache im Baum vorhanden (HiRadixCache + Mamba-Host-Pool + HiCacheFile mit Modell-Identitäts-Hash), war nur aus RAM-Not AUS.
- Kernel-Gesetze gfx1103: Q6_K (dequant/MMVQ/moe_a8) nondeterministisch falsch; Q5_K seltene Fehlstarts; IQ* MMQ kaputt;
  Q2_K/Q3_K/Q4_K/Q8_0 sauber. int32-topk-Fix (b7a46481c3) im Serving-Baum vorhanden (gguf.py:1189).
- MES-Hang („MES failed to respond … REMOVE_QUEUE“ → GPU-Reset) unter Prefill-Last; Mitigation cp 256, kein Fix.

## 1. Checkpoint

1. Download direkt auf dem Laptop von HF (`curl -C -`, ~23 MB/s), SHA256 gegen `SHA256SUMS` des Repos
   (`48a5197d…c4a9`, 17 663 774 592 B = 16 845 MiB).
2. Inventar aus dem Header (`gguf_inventory.py`, funktioniert schon auf der Teildatei). **Gemessen:**
   753 Tensoren, F32 310, Q3_K 301, Q4_K 124, Q5_K 6, Q6_K 1, Q8_0 11, keine IQ*, kein BF16.
   - Q6_K: genau `output.weight` (lm_head, 248320×2048).
   - Q5_K: Layer 0 und 1 von `attn_qkv`, `ffn_down_exps`, `ffn_down_shexp`.
   - Q8_0: der komplette MTP-Block blk.40.
   - neu gegenüber Unsloth-Q4: `ssm_alpha/ssm_beta` sind Q3_K (dort F32), `attn_gate` Q3_K, `ssm_out` Q3_K, `token_embd` Q3_K.
3. Derivat (Rückfall, s. §1a): Q6_K → Q8_0 (Pflicht nach Gesetz), Q5_K → Q8_0 (6 Tensoren, ~+0,2 GB; entfernt die zweite
   bekannte Fehlerklasse komplett), `ssm_out` Q3_K → Q8_0 (sonst wird out_proj beim Laden zu dichtem bf16 entpackt, weil
   head_v_dim 128 kein Vielfaches des K-Quant-Blocks 256 ist: 503 MB bf16 statt 267 MB Q8_0 je Decode-Runde zu lesen).
   Werkzeug `requant_no_q6k.py` wird verallgemeinert (Typliste statt fest Q6_K), Herkunftsdokument wie
   DERIVED_CHECKPOINT_PROVENANCE.md mit SHA beider Dateien.

### 1a. Q6_K-Wurzel statt Umgehung (Nutzer-Order 01.10. ~09:55Z, PFLICHT)
Der Nutzer verwirft „Q6_K nach Q8_0 requantisieren“ als Endzustand: Q6_K auf gfx1103 ist ein Softwarefehler.
Vorgehen, jeweils gemessen mit `q6k_rootcause.py` (echte Q6_K-Bytes, je Fehllauf Blockindex, Welle, Speicherplatz,
implizites d' gegen das d aller Blöcke, dazu mmvq/mmq/moe_vec mit UNTERSCHIEDLICHEN Experteninhalten):
- V0 Ist: Override 11.0.0 + gfx1100-Build (Serving heute).
- V1 Override 11.0.2 + gfx1102-Build (gfx1102 hat wie gfx1103 das kleine Registerfile; torch-Rad und rocBLAS bringen gfx1102 mit).
- V2 gfx11-generic-Build.
- V3 nativer gfx1103-Build ohne Override (nur Erweiterung, torch ohne eigene Kernel).
- V4 Takt fest niedrig (power_dpm_force_performance_level) als Physik-Gegenprobe.
- Aus der Fehlersignatur (ganze 32er-Läufe = eine Welle, eine Store-Anweisung) die Kernelstelle eingrenzen; ISA gfx1100 vs
  gfx1103 vergleichen; Fix im Kernel oder im Build, mit Falsifikator (Fehler muss vor dem Fix reproduzierbar sein).
Das Q8_0-Derivat bleibt nur Rückfall und wird so benannt.

## 2. Architektur
`general.architecture = qwen35moe`, block_count 41 (40 + MTP), 256 Experten / 8 aktiv, 16 Q-Köpfe / 2 KV, head 256,
GDN: conv 4, 16 Gruppen, inner 4096, state 128, dt_rank 32, full_attention_interval 4, rope 1e7, Sektionen [11,11,10,0].
Gleich wie Qwen3.6-35B-A3B. Unterschiede nur in der Quantisierung (s. §1.2) — die Adapter-Risiken daraus:
- gemischte Quant-Typen in einem fusionierten Linear: `in_proj_qkvz` = attn_qkv (Q4_K/Q5_K) + attn_gate (Q3_K);
  der Q4-Unsloth-Checkpoint hatte beide Q8_0, der Mischpfad (`shard_weight_type`) lief auf dem Laptop nie.
- `in_proj_ba` quantisiert (Q3_K) statt F32-Carve-out.
Beides wird nicht vorab umgebaut, sondern durch das Kohärenz-Gate (§4) geprüft; erst ein Befund begründet einen Eingriff.
Tokenizer/Chat-Template: aus der GGUF (Distill-Template); `--tokenizer-path` zeigt auf ein Verzeichnis mit passendem
config/tokenizer (Qwen3.6-35B-A3B-Basis, Vokabular 248320 identisch prüfen). mmproj (858 MiB): NICHT bereitstellen —
seine bloße Anwesenheit schaltet Multimodal ein (model_config.py), Nutzen für den Coding-Agenten null, kostet Kontext.

## 3. Speicherbudget (29 GiB GTT, 29,5 GiB RAM, eine Speicherbank)
Derivat ~16,9–17,1 GiB statt 22,7 GiB → ~5,5 GiB mehr Luft. Verwendung in dieser Reihenfolge:
1. kt-Offload fällt weg (Graphen wieder AN, Decode ×1,8).
2. MES-Hang-Abstand (freier Speicher nie unter ~2 GiB).
3. Kontext: Ziel 32k (Agent-Systemprompt ~17k + Arbeit), danach Rest.
4. HiCache-L2 klein (Staging) + L3 auf NVMe (§6a).
Chunked-Prefill-Größe nach Messung (256 / 512 / 1024), Abbruch beim ersten Reset; dmesg vor jeder Hypothese.

## 4. Kohärenz-Gate vor jeder Leistungszahl
- Guard v2 grün (auf dem jeweils benutzten Erweiterungs-Build).
- probe.py: feste Proben, Temperatur 0, thinking aus, 8/8 greedy-deterministisch, Inhalt richtig.
- llama.cpp-CPU (`/root/651-p2/llama.cpp`) auf DERSELBEN Datei: Token-Übereinstimmung der ersten N greedy Tokens
  für 3–4 Prompts.
- Ohne Gate keine Zahl; „HTTP 200“ ist kein Beleg.

## 5. Messung bs1
- Decode tok/s mit Graphen (Betriebspunkt) und eager (Referenz), `bench_decode.py` (Inter-Token-Verteilung, ms/Runde).
- Prefill tok/s bei 512 / 2k / 8k Prompt (eindeutige Prompts, kein Präfix-Treffer), TTFT-basiert.
- Prefill-Dauerlast (Soak, ≥10 min), dmesg-Reset-Zähler vorher/nachher.
- Jede Zahl mit Instrument + Nenner + Energieprofil (platform_profile steht auf low-power/power-saver — wird mit notiert).

## 6. fLLiper-Form
- P-Layout: iGPU (+ CPU-Stufe, nur wenn gemessen zahlt). D-Layout: iGPU allein. Flip auf demselben Speicher.
- CPU-Stufe für den 35B-Hybrid: die einzigen lauffähigen CPU-Rechenwege sind (a) kt_kernel CPU-Experten (LLAMAFILE,
  liest die GGUF direkt) und (b) PP=2 mit CPU-Rang (braucht GDN-CPU-Kernel, existieren nicht). Messplan: (a) im
  Prefill gegen iGPU-solo (2–3 Kandidaten: 0 / alle CPU-Experten; Teilung unmöglich, s. §0), Runden-ms im Ko-Lauf.
- Flip-Kosten-Tabelle je Posten: Graphen (Capture-ms vs. Park), Mamba-/KV-Zustand, Dequant-Workspace; Park auf Platte
  (eigene Park-Datei, nie Swap), Reload-ms.
- Ehrliche Form, wenn die CPU-Stufe nicht zahlt: P == D == iGPU, Flip-Maschinerie mit Kostenmessung trotzdem fahren
  oder begründet als No-op dokumentieren.

### 6a. Persistenter Präfix-KV: HiCache L2 + L3 auf NVMe (Nutzer-Order 01.10. ~09:55Z, PFLICHT)
Ein Coding-Agent schickt ~17k Token Systemprompt; bei ~150 tok/s sind das ~2 min Prefill je kaltem Turn.
- L3 = Datei-Backend auf NVMe (`/var/lib/hicache/kv`), L2-Host nur Staging (auf der APU ist Host-RAM = GPU-Speicher).
- Präfix einmal vorfüllen, über Anfragen, Dienst-Neustarts und den Flip wiederverwenden.
- Mamba/GDN-Zustand muss mit dem KV verankert sein (Hybrid): vorhandene Maschinerie des Baums (HiRadixCache/
  unified_radix_cache + MambaPoolHost + HiCacheFile-Komponentenpools) benutzen; die Rig-Teile (L3P-Persistenzindex,
  Anker je Chunk) nur portieren, wo die Laptop-Maschinerie die Anforderung messbar nicht erfüllt. Keine neue Maschinerie.
- Messen: Rückladen 17k-Präfix aus L3 (ms) gegen Neu-Prefill; TTFT des zweiten Agent-Turns; dasselbe nach Dienst-Neustart.
- Budget: L3-Größe auf NVMe (766 GiB frei), L2-Staging in MiB.

## 7. Dienst
- Neuer Checkpoint per Drop-in (`20-model.conf` MODEL=…, Name `qwen38-35b-a3b`), kt-Drop-in AUS, Graphen AN,
  großzügige Kaltlade-Zeiten. Rückfall: Drop-ins des Q4-Zustands als `.q4-fallback` aufheben, Umschalter-Skript.
- omp/efeu-Agent auf das neue Modell erst nach Kohärenz + Soak.

## 8. Prüfung gegen die Vorab-Rechnung des Operators
Decode ~17–21 tok/s bs1 (aktive Bytes ~1,5 GB/Runde), Decke ~40 tok/s; Prefill ~140–180 tok/s iGPU solo;
CPU-Stufe höchstens +20–40 %, wahrscheinlich ~0. Wird mit Messzahlen bestätigt oder widerlegt (§11).

## 9. Sicherung
Laptop-Serving-Baum (`/root/651-p2/sglang_src`, kein git, weicht in ~37 Dateien vom Zweig ab) und alle Laptop-Skripte
werden in den Zweig gesichert (`docs/dev/651/efeu35q3/laptop_tree/` als Byte-Kopien + SHA256SUMS), nie wieder Einzelkopie.

## 10. Entscheidungen
Laufend in `/spinning/gpu-arb/docs/ENTSCHEIDUNGEN-0927.md` mit Präfix „efeu-TP14:“.

## 11. Ergebnisse (wird nachgetragen)

Stand 01.10. ~14:05Z. Zweig `desk/efeu-35b-q3-fllliper-1001` (Belege: `docs/dev/651/efeu35q3/results/`).
Alle Zahlen bs1, ein Request, Energieprofil **balanced** (wenn nicht anders genannt), A/A'-Rauschboden angegeben.

### 11.1 Checkpoint und Architektur
- HF-Download auf dem Laptop, SHA256 OK. qwen35moe, identisch zu Qwen3.6-35B-A3B. Q6_K nur lm_head, Q5_K nur Layer 0/1.
- **Kein Derivat nötig**: Q6_K-Wurzel gefunden (11.2). Das Requant-Werkzeug (`requant_types.py`) bleibt nur Rückfall, nicht gebaut.
- Tokenizer/Template aus dem Distill-Repo, bewährte config.json. Kein mmproj.

### 11.2 Q6_K-Wurzel (Nutzer-Order) — Software, nicht Physik
- Ubuntu-clang 21 erzeugt für gfx11 real-true16: D16-Load in eine VGPR-Hälfte rast gegen VALU-Schreiben der anderen Hälfte.
  Messung (2,1 M Elemente, 20 Starts je Typ): gfx1100-Build Q6_K 20/20 falsch (bis 1,6, Inf); gfx1102 + Override 11.0.2 genauso;
  gfx11-generic und `-real-true16`: alle Typen 0/20. Override ist NICHT schuld.
- Zweiter Fehler beim ersten Boot: WARP_SIZE Host 64 / Gerät 32 → getuntes K-Quant-MMVQ (M=2..8) Launch-Failure; Legacy/MoE-vec Doppelwelle.
- Fix: `sgl-kernel/rocm-gguf-gfx11` (ohne true16, WARP_SIZE_GGUF, moe_vec überspringt id<0, Marker-Op). Strenger Guard v2 (jede Nichtdeterminismus = FAIL): neuer Build PASS, alter FAIL.
- gguf.py hebt die Q6_K-Eindämmung nur mit Marker auf → lm_head bleibt Q6_K gepackt (417 MB statt 1017 MB je Token).

### 11.3 Kohärenz
- probe_q38: 8/8 + 8/8, Runden identisch, 8x Greedy mit Logprobs 1/1 verschieden (auch über den Dienst und unter kt-Teilung).
- llama.cpp-CPU gleiche Datei: 6/9 exakt, Abweichungen erst nach 6/22/34 Token an Beinahe-Gleichständen (Eis: −0,75 vs −0,91 nats).

### 11.4 Leistung bs1 (gegen Vorab-Schätzung)
| Größe | Messung | Instrument / Nenner | Schätzung |
|---|---|---|---|
| Decode, Graphen, balanced | 20,20 / 20,05 tok/s (49,5 ms/Token) | bench_decode, Inter-Token-Median, 2×254 Token | 17–21 → **bestätigt** |
| Decode, Graphen, low-power | 11,30 / 11,19 tok/s | gleich | — (Profil halbiert!) |
| Decode, performance | 20,52 / 20,44 tok/s | gleich | — |
| Decode über Dienst (HiCache an) | 19,81 / 19,77 tok/s | gleich, Front-Tür 31651 | — |
| Prefill cp256 | 100,9–107,4 tok/s (512/2k/6k) | bench_prefill, TTFT max_tokens=1, eindeutige Prompts | 140–180 → **widerlegt** |
| Prefill cp512 (Betriebspunkt) | 109,9–114,0 tok/s; Soak 745 s 111,5 tok/s, 0 Resets | dto. + dmesg | 140–180 → widerlegt |
| Kalt-Prefill 20k-Systemprompt | 216,6 s (92 tok/s, HiCache an, lange Attention) | prefix_reuse_bench TTFT | — |
- Prefill-Kostenmodell (Rang-Zeilen): ~0,33 s je Chunk + ~8,2 ms je Token → Asymptote ~122 tok/s; Rechenzeit dominiert.
- Restgröße im Decode: GDN-out_proj (Q3_K, head_v 128 < Block 256) wird beim Laden dicht bf16 entpackt: 503 MB je Token
  (25 % der gelesenen Bytes). Mit Aktivierungs-Permutation statt Spalten-Retiling bliebe er Q3_K (108 MB) → erwartet ~+20 % Decode. Offen.

### 11.5 fLLiper
- P-Layout iGPU + CPU (kt-Experten 128/256): Prefill 2k **154,3 / 158,0 tok/s** gegen 114,0 / 113,2 solo (+37 %), 512er-Chunk 3,13 s statt 4,43 s, kohärent.
  Vorab-Erwartung „+20–40 %, wahrscheinlich ~0“: der obere Rand trifft, „~0“ ist widerlegt; die geteilte TDP frisst den Gewinn NICHT auf.
- D-Layout iGPU solo: 20,2 tok/s; mit aktiver Teilung 11,9 tok/s → D muss solo sein.
- Flip-Kosten-Tabelle:

| Posten | Größe | resident / Park / Neubau | gemessen |
|---|---|---|---|
| Decode-Graph | 0,15 GB | resident | Capture 2,47 s |
| CPU-Expertenhälfte (P-exklusiv) | ~6,6 GB RAM | müsste je Flip geladen/verworfen werden | Laden ~13,5 s (Load 88,95 s mit vs 75,4 s ohne kt) |
| GPU-Expertenhälfte (D-exklusiv im Split) | ~6,6 GB GTT | dto. Gegenrichtung | nicht gemessen |
| Mamba-Slots (4), KV-Pool | 0,3 / 0,66 GB | phasengeteilt, kein Flip | — |
- Beide Layouts gleichzeitig resident: passt nicht (Teilung allein 26,5 GB von 29,5 GB belegt). Ski-Rental: Ersparnis 2,39 ms/Token,
  Hin+Rück ~27 s → lohnt erst ab ~11k neuen Prefill-Token je Phase. Mit L3 sind Agent-Turns fast immer darunter.
- **Verdikt:** CPU-Stufe zahlt in P (+37 %), Laufzeit-Flip mit Expertenumzug NICHT gebaut (offener Punkt mit Kostenmodell);
  der Dienst fährt die D-optimale Form (iGPU solo, Graphen) und macht den Prefill über L3 klein. PP mit CPU-Rang bleibt mangels GDN-CPU-Kernel zu.

### 11.6 Persistenter Präfix (Nutzer-Order)
- Vorhandene Maschinerie (UnifiedRadixCache KV+MAMBA, HiCacheFile, Chunk-Anker), geschlossen: libssl-dev, Torch-Fallback der kvcacheio-Direct-Transfers.
- 20019-Token-Systemprompt: kalt 216,6 s; zweiter Turn 1,74 s; neue Sitzung gleicher Systemprompt 1,33 s; **nach Neustart 5,31 s** (L3-Rückladen ≈ 4,8 s statt 216 s); Ausgabe aus L3 = kalte Ausgabe.
- L3-Budget: 827 MB / 20073 Dateien je 20k Token (page_size 1) auf 46-GB-Loop-Image → ~1,1 M Token Verlauf; Verdrängung/Kappung im Datei-Backend ungeprüft (offen).

### 11.7 Dienst
- `htsglang-ondemand` + `50-q38.conf`, `/root/efeu35q3/switch_model.sh q38|q4` (Rückfall = alter Zustand inkl. omp-Registry). Wake 146 s, Idle-Park 600 s.
- omp: `omp --model local/qwen38-35b-a3b` (32k); alte ID als Alias.

### 11.8 Coding-Agent (omp, volle Werkzeugliste ≈ 17k-Token-Systemprompt)
- Zum ersten Mal läuft omp auf diesem Laptop mit vollem Werkzeugsatz (#655 musste auf 2 Werkzeuge kürzen, ~10k Prompts wedgten). 0 GPU-Resets.
- Phase 1 kalt: 195 s (davon ~170 s Prefill); Modell antwortete ohne Werkzeug (6765 richtig, Datei nicht geschrieben) → Abnahme-Kriterium FAIL.
  Wiederholt mit ausdrücklicher Werkzeug-Anweisung: Datei geschrieben und ausgeführt, aber Off-by-one (10946) — Modellqualität (Q3-Distill), nicht Serving.
- Phase 2 (neuer omp-Prozess, gleiches Verzeichnis): Präfix 16896 Token aus dem Cache, Fehler gelesen, Datei korrigiert, „sum is 15“ — 41 s. PASS.
- Befund: omp schreibt das Arbeitsverzeichnis früh in den Systemprompt → anderes Verzeichnis = Präfix-Divergenz = kalter 17k-Prefill (~3 min).
  Gleiches Verzeichnis = Treffer. Offener Punkt: omp-Prompt-Layout (Umgebung ans Ende) oder Ein-Verzeichnis-Workflow.

### 11.9 Offene Punkte
1. Laufzeit-Flip P(kt 128/128)↔D(solo) mit Expertenumzug (Kostenmodell 11.5); Schwelle ~11k neue Prefill-Token.
2. GDN-out_proj Q3_K quantisiert lassen (Aktivierungs-Permutation) → erwartet ~+20 % Decode.
3. L3-Kappung/Verdrängung des Datei-Backends prüfen (46-GB-Loop-Image), page_size>1 für weniger Dateien (20k Dateien je 20k Token).
4. cp1024 nicht gemessen (erwartet +3 %).
5. Kalt-Prefill mit HiCache 92 tok/s @20k gegen 111 tok/s @6k ohne: Anteil Schreib-Overhead vs. lange Attention nicht getrennt.
6. omp-Prompt-Layout (Verzeichnis früh im Systemprompt) bricht sitzungsübergreifende Wiederverwendung.
