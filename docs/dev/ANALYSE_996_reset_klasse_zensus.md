# ANALYSE #996 — RESET-KLASSE: ZENSUS statt Einzelwurzel

Datum: 2026-08-28. Baum `/spinning/wt-943boot`, Branch `merge/flip-window-0828`,
HEAD `cf16281b3f`. READ-ONLY am Code; AST/grep/LSP; keine GPU, kein Boot.
Hermetische Laeufe mit `CUDA_VISIBLE_DEVICES=""`.

Nutzer-Order 2026-08-28: "reset klasse pruefen lassen, nicht wieder eins nach dem
anderen wurzel finden." Dies ist die Klassen-Antwort: Reset-Site, Ueberlebenden-
Zensus, Vier-Bucket-Klassifikation, Ratchet-Verdikt.

Beleg-Stufe durchgehend **DESK-BEWIESEN**. Kein Metall in dieser Session; das
Wort "gefixt" faellt nirgends.

---

## 0. Vier Praemissen, die der Code widerlegt

| # | Praemisse | Befund | Beleg |
|---|---|---|---|
| P1 | "`cutover_participants.py` … importiert von `schedule_batch.py`" | **FALSCH.** Kein Produktionsmodul importiert es. `schedule_batch.py:1904` ist ein Kommentar. Einzige Importeure: zwei Testdateien. Die Registry ist PRAESENT-ABER-UNVERDRAHTET. | `grep -rn "from sglang.srt.managers.cutover_participants\|import cutover_participants" python/` → 0 Treffer. Importeure: `test/registered/unit/managers/test_cutover_participants_859.py:29`, `test_cutover_discovery_diff_859.py:23` |
| P2 | R12-EVAL: "`_live_reqs` … **chunked_req** bleibt draussen" | Fuer `chunked_req` **FALSCH**. Er wird enumeriert UND abgeraeumt. Nur `waiting_queue` stimmt. | `phase_flip_runtime.py:1361-1364`; `:1853-1856` (`scheduler.chunked_req = None`) |
| P3 | "#856-Reset nullt nur die POOLS; Request-Objekte werden per Objekt-Chirurgie weitergereicht" | Fuer **retracted Residents** zu pessimistisch: `reset_for_retract` nullt 33 Felder, darunter `last_node`, `prefix_indices`, `mamba_pool_idx`, `swa_uuid_for_lock`. Die Chirurgie sitzt nicht bei den Residents, sondern bei den **Nicht-Residents**. | `schedule_batch.py:1846-1959`; Details §2 |
| P4 | "#955 `_escalated`-Latch nie geclearet" (als Defekt) | Der Latch wird bewusst **nur** von einem bedienten Round-Trip geloescht; ein Void loescht ihn nie, und das ist die ausgeschriebene Invariante, nicht eine Luecke. | `pp_admission_congruence.py:560-575`; Clear-Sites `:592`, `:661`, `:982` (letztere via `record_return_trip`, `:964-996`) |

P1 ist die folgenreichste und P3 verschiebt die Suche: nicht der Reset ist zu
schwach, sondern **die Halter-Menge ist groesser als die Menge, die der Reset
kennt.**

---

## 1. Die Reset-Site, exakt — was der Code TUT

Zwei getrennte Stufen, nicht eine. Der Release laeuft **vor** `_cutover`, nicht
darin.

### Stufe 1 — Release (`_release_residents_for_cutover`, `phase_flip_runtime.py:9175`)

| Schritt | file:line | Wirkung |
|---|---|---|
| Gesetz: retract STRIKT vor reset | `phase_flip_runtime.py:1500` / `:1501` | Umkehrung reproduziert #825 (Crash auf allen drei Raengen) |
| `_retract` → `retract_all` | `phase_flip_runtime.py:1947` | pro Req: `release_req` → `release_kv_cache` → `cache_finished_req` → `dec_lock_ref` |
| Seam-Stempel | `phase_flip_runtime.py:1983` `seam_readmit_epoch`; `:1989` `reissue_seam_grant` | |
| `consume_retracted_from_live_universe` | `phase_flip_runtime.py:1772-1857` | `filter_batch` auf `running_mbs`, `last_mbs`, `running_batch`, `last_batch`; `chunked_req = None` (`:1855`) |
| `drop_prefix_tree_returning_rows` | `phase_flip_runtime.py:1612` | `tree.evict` (gibt Rows zurueck) → `reset()` (neuer Root) |
| `readmit_seam_residents` | Aufruf `phase_flip_runtime.py:9458`, Impl `scheduler.py:5157 ff` | `_add_request_to_queue(req, is_retracted=True)` (`:5194`) in `waiting_queue`, Block nach vorn (`:5203-5208`) |

### Stufe 2 — `_cutover` (`phase_flip_runtime.py:2922-3662`)

Direkte Zuweisungen: `ps` (`:3006`/`:3016`), `tp_group` (`:3019`), `tp_cpu_group`
(`:3020`), `attn_tp_group` (`:3021`), `attn_tp_cpu_group` (`:3022`), `pp_group`
(`:3023`), `dp_tp_group` (`:3025`), `request_receiver` (`:3039`),
`output_streamer` (`:3059`), `load_inquirer` (`:3065`),
`batch_result_processor` (`:3078`), `spec_algorithm` (`:3267`), `draft_worker`
(`:3268`), `model_worker` (`:3269`), `phase_flip_active_stack` (`:3270`),
`_decode_steps_this_phase = 0` (`:3314`).

Indirekt: `init_pp_loop_state()` (`:3231` → `scheduler_pp_mixin.py:6233-6401`)
baut den ganzen PP-Ring neu, inkl. `_pp_chunked_req_before_by_slot`
(`:6399-6401`) und `pp_flip_forget_ring_scoped_slots` (`:6299` →
`:1315-1317`); `install_resident_set` /
`promote_slot_zero_to_running_batch` (`:3231`/`:3236` →
`phase_flip_resident_carry.py:664-699`, `:897-911`);
`reset_stale_batch_flags` (`:3328` → `phase_flip_draft_bootstrap.py:686-713`);
`rebind_for_cutover` (`:3614`).

### Der Geltungsbereich

**`_live_reqs` (`phase_flip_runtime.py:1280-1365`) IST die Definition von "was
der Cutover anfasst":** `running_mbs[*]` (`:1324`), `last_mbs[*]` (`:1348`),
`running_batch`/`last_batch` (`:1350`), `chunked_req` (`:1361`).

**Nicht darin:** `waiting_queue`, `grammar_manager.grammar_queue`,
`adder.can_run_list`, `_pp_parked_continuations`, `PPAdmissionCongruenceGuard`,
`admission_limiter`, `gdn_slot_executor`, Disagg-Queues.

`waiting_queue` wird im Cutover **nur entdupliziert**, nie zurueckgesetzt:
`_consume_carried_from_waiting_queue` (`phase_flip_resident_carry.py:702-739`,
Zuweisung `:739`) entfernt die gerade carried Requests. Die Felder der
verbleibenden Mitglieder bleiben unberuehrt.

---

## 2. Zensus — Vier Buckets

### RESET (wird genullt)

`Req.reset_for_retract` (`schedule_batch.py:1846-1959`) nullt **33 Felder**, u.a.
`prefix_indices` (`:1908`), `last_node` (`:1911`), `cache_protected_len`
(`:1912`), `num_matched_prefix_tokens` (`:1913`), `swa_uuid_for_lock` (`:1914`),
`extend_range` (`:1916`), `mamba_pool_idx` (`:1935`), die sechs
`mamba_*`-Handles (`:1938-1943`), die vier `kv_*`-Zaehler (`:1945-1948`),
`is_retracted`/`retracted_stain` (`:1927-1928`).
`req_pool_idx` wird eine Stufe frueher genullt, in `ReqToTokenPool.free`
(`mem_cache/memory_pool.py:452`, erreicht ueber `mem_cache/common.py:1826`).

Scheduler-seitig: `chunked_req` (`phase_flip_runtime.py:1855`), der PP-Ring
(`scheduler_pp_mixin.py:6233-6401`), `batch_is_full` auf allen erreichbaren
Batches (`phase_flip_draft_bootstrap.py:691-713`), die Ring-Slots
`_pp_flip_arm_mb_id`/`_pp_flip_arm_epoch`/`_pp_flip_resume_slot`
(`scheduler_pp_mixin.py:1315-1317`), Radix-Device-Tier
(`phase_flip_runtime.py:1612`).

### RE-DERIVED

| Posten | Neuberechnung | Vorbehalt |
|---|---|---|
| Die **acht co-derivierten Match-Felder** (`prefix_indices`, `last_node`, `last_host_node`, `best_match_node`, `host_hit_length`, `swa_host_hit_length`, `mamba_host_hit_length`, `mamba_branching_seqlen`) | EIN Tupel-Assign aus EINEM `match_result` in `init_next_round_input`, `schedule_batch.py:1392-1408` | **Zwei Bedingungen, beide verletzbar** — siehe B4-1 |
| `cur_batch`, `_uniform_prefetch_ballot`, `_pass_prefetch_verdicts` | Pro Pass unbedingt ueberschrieben (`scheduler.py:5846`, `:5870`) | — |
| `_seam_debt_lapse_announced`, `_seam_transport_debt_since` | `scheduler.py:12419-12421` aus `_in_tp_now` | Selbstkorrigierend |
| `_pp_parked_continuations` | Tabelle wird pro Lap **ersetzt** (`scheduler_pp_mixin.py:1954`) | Nur bei ankommender Nachricht MIT Key (`:1951-1952`) — siehe B4-5 |

### CARRIED-BY-DESIGN (mit Entscheidungs-Beleg)

| Posten | file:line | Beleg |
|---|---|---|
| `last_node` bei Prefix-Truncation nicht geloescht | `schedule_batch.py:1830-1836` | "kein Reading, sondern ein RESOURCE HANDLE — Nulling wuerde den Lock-Ref leaken" |
| `last_node`/`prefix_indices` beim #984-Give-back nicht geloescht | `scheduler_pp_mixin.py:3207-3211` | "sie sind, was das naechste Angebot als executed meldet" |
| Carried Chunk behaelt seine EINE Admission-Ref | `scheduler_pp_mixin.py:3220-3231` | #990-Ownership-Guard, R12/Boot-9: sonst trifft ein `inc` zwei `dec` |
| `_escalated` nur per bedientem Round-Trip loeschbar | `pp_admission_congruence.py:560-575`, Clear `:982` via `record_return_trip:964-996` | Ausgeschriebene Invariante. **Korrigiert P4** |
| `_PREMISE_DEAD_STAMP` ueberlebt einen Void | `scheduler_pp_mixin.py:2459-2465`, Selbst-Invalidierung `:2483-2499` | Generation-Vergleich statt Clear |
| `_pp_flip_armed_passes` nicht am Cutover geloescht | `scheduler_pp_mixin.py:1303-1309` | "IS DELIBERATELY NOT CLEARED HERE, and that omission is the point" |
| Disagg-Queues nie geleert | Gate `phase_flip_runtime.py:2602-2603` (`guards.append("PD disaggregation")`) | Der Flip **armt gar nicht**, wenn `disaggregation_mode != NULL`. Abwesenheit mit verweigerndem Gate belegt |
| `retraction_count`, `solo_oom_count`, `prefill_attempt_count` | `schedule_batch.py:1116-1128`, `:1169` | Explizit "preserved across retracts" |

### UEBERLEBT-UNGEPRUEFT — **DIE BEFUNDLISTE**, nach Schaerfe

| # | Posten | file:line | Warum ungeprueft | Waere welcher Korpus-Defekt |
|---|---|---|---|---|
| **B4-0** | **Der Ratchet kann nicht rot werden.** Der Discovery-Diff filtert `undeclared` auf die Menge der bereits DEKLARIERTEN und assertiert dann deren Deklariertheit. Tautologie. `known` ist ein Dead Store. | `test/registered/unit/managers/test_cutover_discovery_diff_859.py:54-64`; `known` berechnet `:50-53`, nie gelesen | **Bewiesen, nicht ungeprueft** — §3.2 | Der Ratchet selbst. INDIKATOR-GESETZ: das Gruen misst nichts |
| **B4-1** | **Die acht Match-Felder werden NICHT garantiert re-derived.** Zwei unabhaengige Wege daran vorbei: (i) fuenf Praedikate vor dem Aufruf, eines ein `break`, das den Rest der Queue ueberspringt; (ii) ein Call-Site **ohne `tree_cache`**, der den `match_prefix`-Block komplett auslaesst. | Assign: `schedule_batch.py:1392-1408`, gegated `if tree_cache is not None` (`:1378`). Guards: `scheduler.py:9304` (continue), `:9307` (continue), `:9317` (continue), `:9338-9343` (**break**), `:9363-9371` (continue), Call `:9377`. Tree-lose Call-Site: `scheduler.py:8986` `self.chunked_req.init_next_round_input()` | Erreichbarkeit des Schadens nicht per Call-Graph geschlossen | **#965** ("sechs stale Felder aus EINEM match_result") — und #965 wurde in `truncate_prefix_to` (`schedule_batch.py:1837-1841`) gefixt, **nicht** in `reset_for_retract`. Geschwister-Site uebersehen |
| **B4-2** | **`storage_hit_length` hat ueberhaupt keine Clear-Site.** Zwei Schreiber im ganzen Baum: `__init__` und der Prefetch-Pfad. Weder `reset_for_retract` noch `truncate_prefix_to` fassen es an. | `schedule_batch.py:960` (init), `scheduler.py:9375` (`= loaded_tokens`). `grep -rn "\.storage_hit_length\s*=" managers/` → genau diese zwei | Ob ein stale Wert konsumiert wird, nicht getracet | **#937** (Prefetch-Zustand ohne Generation-Bindung) |
| **B4-3** | **`admission_limiter` ueberlebt den Layoutwechsel voellig unbeachtet.** Sein `ceiling` ist `max_running_requests`, waehrend `_cutover` `pp_max_micro_batch_size` neu setzt (`:3095-3100`). Kein Flip-Modul kennt ihn. | Objekt `scheduler.py:4373`; `current` `admission_limiter.py:150`; Mutation `scheduler.py:10195-10196`, `:10327`. `grep -rn "admission_limiter" managers/phase_flip_*.py` → **nur ein Kommentar** in `phase_flip_resident_carry.py:154`, kein Code | Ob die Sitzzahl nach dem Flip falsch ist, nicht gemessen | **#984** (Seat-Asymmetrie), exakt dieselbe Form |
| **B4-4** | **`phase_flip_draft_cold_armed` ist ein Latch ohne jede Clear-Site.** Drei Vorkommen im ganzen Baum: Name, Read, Set-True. Nie `False`, nie `delattr`. | `phase_flip_draft_bootstrap.py:786` (Name), `:970` (Read), `:1019` (`setattr(..., True)`). `grep -rn "phase_flip_draft_cold_armed\|COLD_ARMED_ATTR" .` → genau diese drei | Ob ein zweiter Flip den Draft dauerhaft kaltstellt, nicht gemessen | **#955** (Latch nie geclearet), reinste Form |
| **B4-5** | **`_pp_parked_continuations` wird nach pp→tp nicht mehr ersetzt.** Ersetzung nur bei ankommender Ring-Nachricht mit Key; in der TP-Phase laeuft der PP-Ring nicht. Kein Cutover-Clear. | Ersetzung `scheduler_pp_mixin.py:1951-1956`; Clear `:1983-1996` nur von `scheduler.py:9698` (gegated `pp_rank==0`). `grep -rn "parked_continuation" managers/phase_flip_runtime.py` → **0 Treffer** | Folge ist Fehlpriorisierung, kein Crash | #984-nah, niedrige Schaerfe |
| **B4-6** | **`gdn_slot_executor` wird einmal lazy gebaut und nie invalidiert.** Kein Flip-Modul kennt ihn. | `scheduler.py:583` (None), `:7278` (`if ... is None`, lazy build). `grep -rn "gdn_slot_executor" managers/phase_flip_*.py` → **ABSENT** | Ob der Slot-Ladder-Treiber layout-abhaengig ist, nicht geprueft | **#963** (rang-/layout-divergenter Zustand) |
| **B4-7** | **`seam_readmit_epoch` wird nie zurueckgesetzt** — und der Doc-Kommentar behauptet das Gegenteil. Verbraucht wird ein SEPARATES Boolean. | Set `phase_flip_runtime.py:1983`; Doc-Behauptung `schedule_batch.py:988-990`; echter Spend `phase_purity.py:1189`/`:1203` (`seam_grant_consumed`) | Funktional gedeckt durch das Boolean; der Kommentar ist falsch | #920-nah; primaer ein **falscher Kommentar** dieser Kampagne |
| **B4-8** | **Ein toter Daempfer-Term.** `_arming_condition_persists` liest `getattr(scheduler, "grammar_queue", None)`. Das Attribut existiert auf dem Scheduler nicht — die Queue liegt auf `grammar_manager`. Der Term ist immer `None`. | Lesestelle `phase_flip_runtime.py:10068`; echte Queue `scheduler.py:11073`, `:11377` (`self.grammar_manager.grammar_queue`) | Bewiesen per grep; Wirkung = ein Damper-Kriterium weniger | INDIKATOR-GESETZ: misst nicht, was es behauptet |

**Erreichbarkeit von B4-1, ehrlich (Kompensator-Erreichbarkeit):** Der Schaden
verlangt einen Konsumenten der acht Felder zwischen Baum-Drop und
Re-Derivation. #965 benennt diesen Konsumenten praezise
(`schedule_batch.py:1812-1828`): `needs_host_load_back()` bleibt wahr,
`init_load_back` laeuft auf stale `best_match_node`/`host_hit_length`,
`prefix_indices = cat([...])` hinterlaesst ein **Loch**, und
`prepare_for_extend` dimensioniert den Cross-Stage-Tensor, als waere es
zusammenhaengend. Das ist die still-falscher-Kontext-Klasse. Ich habe den
Call-Graph vom Cutover zu diesem Konsumenten **nicht** geschlossen — deshalb
Bucket 4 und nicht "Befund".

---

## 3. Ratchet — Verdikt

### 3.1 Cut 1 traegt

`REGISTRY` (`cutover_participants.py:92-241`), 14 Teilnehmer, jeder mit Hook +
Reachability-Probe oder erklaerter `gap`; Symbol-Existenz getestet; eigene
Can-fail-Proofs (`test_cutover_participants_859.py:130-147`).
Hermetisch verifiziert: **44 passed in 11.82 s**
(`PYTHONPATH=/spinning/wt-943boot/python`, `CUDA_VISIBLE_DEVICES=""`) — kein
Symbol-Drift. **Das funktioniert.**

### 3.2 Cut 2 misst nichts — Beweis

Testkoerper woertlich aus `test_cutover_discovery_diff_859.py:42-64`, hermetisch
nachgefahren:

```
real module              -> GREEN
module + 2 NEW writes    -> GREEN
module + only-new source -> GREEN
```

```python
known = set(MUTATED_STATE) | set(NOT_PARTICIPANTS)          # :50  — nie wieder gelesen
for p in REGISTRY: known.add(p.name)                         # :52-53
undeclared = {w for w in written
              if w in MUTATED_STATE or w in NOT_PARTICIPANTS}  # :54-58  ← filtert auf DEKLARIERTES
for name in undeclared:
    if name in NOT_PARTICIPANTS: continue                    # :62-63
    assert name in MUTATED_STATE, name                       # :64  — tautologisch wahr
```

Gemeintes Praedikat ist die Negation (`if w not in known`). Die beiden
`test_can_fail_*` (`:138-147`) pruefen nur den Helfer `discover_cutover_writes`,
nicht die Assertion — die #719-Form: Instrument belegt, Konsument tot.

### 3.3 Zweiter Defekt: falscher Scope

`_cutover_source()` (`:33-36`) liefert `inspect.getsource(phase_flip_runtime)` —
das **ganze Modul**, 11k+ Zeilen.

| Scope | Attribut-Schreibungen | undeklariert |
|---|---|---|
| ganzes Modul | 175 | **163** |
| Cutover-Call-Closure (11 benannte Funktionen) | — | **17** |

Das erklaert vermutlich die Invertierung: mit korrektem Praedikat und
modulweitem Scope waere der Test dauerhaft rot mit 163 Posten. **Das Praedikat
allein zu reparieren erzeugt keinen Ratchet, sondern einen abgeschalteten Test.**

Die 17 im echten Scope: `_decode_steps_this_phase`,
`_fence_persisted_nothing_streak`, `_last_retract_writeback_report`,
`attn_tp_cpu_group`, `attn_tp_group`, `batch_result_processor`, **`chunked_req`**,
`dp_tp_group`, `draft_worker`, `load_inquirer`, `model_worker`,
`output_streamer`, `pp_group`, `ps`, `request_receiver`, `tp_cpu_group`,
`tp_group`. Davon 11 Kommunikator-/Worker-Rebindings, 3 Seam-Buchhaltung —
und **`chunked_req` ist ein echter Teilnehmer**, dasselbe Feld, das Boot 14
gekillt hat, in keiner der drei Deklarationslisten.

### 3.4 Verdikt: **die Registry KANN es tragen**, in drei Schritten, alle im Testbaum

1. **Praedikat invertieren** (`test_cutover_discovery_diff_859.py:54-58` →
   `if w not in known`) plus ein Can-fail-Test, der ein gepflanztes Attribut
   durch die ASSERTION treibt, nicht nur durch den Helfer.
2. **Scope auf die Call-Closure** (`:33-36`): Quelle einer benannten
   `SEAM`-Funktionsmenge, deklariert in `cutover_participants.py`, damit die
   Liste selbst reviewbar ist. Ohne diesen Schritt ist Schritt 1 ein
   163-Posten-Rotlicht.
3. **Die 17 einmal deklarieren**; `chunked_req` als echter Teilnehmer mit Hook
   `consume_retracted_from_live_universe` und Probe `log:RESIDENTS RELEASED`
   (die Zeile existiert, `phase_flip_runtime.py:9290`).

### 3.5 Die Grenze des Ratchets, und die kleinste Alternative

Ein Diff ueber **Zuweisungen** sieht nur, was der Cutover SCHREIBT. B4-1 bis B4-8
sind Zustaende, die er **nicht anfasst** — eine Auslassung ist so nicht
detektierbar.

Kleinste Alternative fuer die Auslassungs-Haelfte: `_live_reqs`
(`phase_flip_runtime.py:1280`) ist bereits die einzige Autoritaet fuer "wer ist
resident". Ein Deklarations-Test, der die vier Routen darin gegen die
**Halter-Menge** im Scheduler diffed — `waiting_queue`,
`grammar_manager.grammar_queue`, `chunked_req`, `running_mbs`, `last_mbs`,
`running_batch`, `last_batch`, `cur_batch`, `_pp_parked_continuations`,
`admission_limiter`, `gdn_slot_executor`, Disagg-Queues — und fuer jeden
nicht-enumerierten Halter eine explizite Zeile "bewusst draussen, weil …"
verlangt, macht B4-3, B4-5, B4-6 und B4-8 zu Desk-Zeit statt Boot-Zeit.
Diese Haelfte fehlt heute vollstaendig.

---

## 4. Die Klasse, in einem Satz

Der #856-Reset ist **korrekt fuer seinen Geltungsbereich**, und sein
Geltungsbereich ist `_live_reqs`. Die ~15-Instanzen-Klasse entsteht nicht daran,
dass er zu wenig nullt, sondern daran, dass **die Menge der Halter groesser ist
als die Menge, die er kennt** — und dass der eine Mechanismus, der genau das
haette melden sollen, seit seiner Erstellung tautologisch gruen ist.

Der Korpus zerfaellt in zwei Familien, nicht eine:

- **Familie A, Seam** (#920, #911, #937, #938): Binding-/Generation-Zustand am
  Cutover. Deckung: `cutover_participants.REGISTRY` — vorhanden, Enforcement tot.
- **Familie B, Void-/Park-Loop** (#951, #955, #959, #963, #965, #969, #984,
  #990, #992, #993, #994, Boot-9, Boot-14) in `scheduler_pp_mixin.py` /
  `schedule_policy.py` / `pp_admission_congruence.py`. Deckung: **keine
  Registry.** Das ist die groessere und die ungedeckte Haelfte.

Struktureller Nebenbefund zu Familie B (Sweep-belegt): die drei "Void-Pfade" sind
keine drei Implementierungen. `_pp_void_retracted_pass`
(`scheduler_pp_mixin.py:7753`) setzt nur ein Flag; die Entsorgung laeuft immer
ueber `_pp_void_own_batch` (`:7882`) — **und der ist auf PP0 unerreichbar**
(`:3805` `not is_first_rank`; `pp_upstream_void_pending` `:1565` liefert auf PP0
`False`). PP0 entsorgt in `_pp_absorb_void_output` (`:9174`), einer **zweiten,
duplizierten** Schleife. Zwei Kopien derselben Entsorgungslogik, rang-getrennt,
ist die strukturelle Vorbedingung fuer jede kuenftige "Clear-Site auf nur einem
von zwei Pfaden"-Instanz.

---

## 5. Was ich NICHT pruefen konnte

- **Erreichbarkeit von B4-1** vom Defektpfad — verlangt einen Call-Graph vom
  Baum-Drop zum `init_load_back`-Konsumenten. Nicht geschlossen.
- **Ob B4-3 (`admission_limiter`) nach dem Flip real falsch sitzt** — verlangt
  eine Messung der Sitzzahl ueber einen Flip.
- **Ob B4-4 (`phase_flip_draft_cold_armed`) den Draft dauerhaft kaltstellt** —
  verlangt zwei aufeinanderfolgende Flips auf Metall.
- **Ob B4-6 (`gdn_slot_executor`) layout-abhaengigen Zustand haelt** — verlangt
  Lesen von `gdn_slot_runtime.py`, nicht getan.
- **Alles Metall.** Keine GPU beruehrt, kein Boot, keine Laufzeit-Bestaetigung
  irgendeines Bucket-4-Postens. Die Karten hielt in dieser Session ein anderer
  Agent.
