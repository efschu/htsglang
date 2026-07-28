# DESIGN #201 — Hierarchische Parallelitaet: uneven Layer-Split (PP) x partielles TP x uneven DCP, cross-rig ueber RDMA

Status: **ANALYSE + DESIGN, KEIN BAU**. Reine CPU-Lesearbeit, keine GPU-Boots (Karten gehoeren #199).
Code-Anker read-only aus `/spinning/wt-merge-probe` (`integration/r3-probe`, merge-base `82e7cdcff9`).
Datum 2026-07-26.

Nutzer-Auftrag woertlich: *"uneven layer split, also groessere karte mehr layer, und partielles
tensor parallel mit layer split auch mit rdma und auf rig zwei theoretisch wieder ein uneven tp
dcp, wenn hier schnittpunkte auftreten gleich kompatibel implementieren. also tensor parallele
layer als split auf allen ebenen."*

---

## 0. Kurzfassung (das Ergebnis vorweg)

1. **Der Mechanismus fuer uneven Layer-Split existiert bereits** — `SGLANG_PP_LAYER_PARTITION`
   (`distributed/utils.py:1197-1233`), stock upstream, im Fork unveraendert. Es fehlt kein Kernstueck,
   nur ein Planner + CLI + die Vertraeglichkeit mit den Fork-Features.
2. **PP im Fork ist zu 100 % stock upstream.** Kein Fork-Delta in Stage-Bildung, Layer-Zuweisung
   oder Aktivierungs-Transport. Dafuer lehnen **acht** Argparse-Stellen `pp_size > 1` hart ab —
   praktisch jedes Fork-Feature.
3. **Die Link-Rechnung faellt eindeutig fuer PP aus** — eine Stage-Grenze kostet pro Token
   **20 KiB statt 7,5 MiB** (Faktor 384) und **2 Nachrichten statt 128 Kollektive** (Faktor 64).
4. **Aber: der Link ist bei bs=1 gar nicht das Problem.** Die 128 Kollektive kosten gemessen
   3,39 ms von einem 145-ms-Schritt = **2,3 %**. Wer sie komplett loescht, gewinnt maximal 2,3 %.
   Der Straggler-Compute kostet den Rest — und **PP behebt den Straggler nicht, es macht ihn
   additiv statt maximal.**
5. **Die Kernfrage "ab welchem N kippt es" hat eine N-unabhaengige Antwort:** im
   bandbreiten-gebundenen Decode schlaegt 2-Stufen-PP das schnelle Rig allein nur dann, wenn
   `B_stage2 > B_stage1`. Hier ist `B_2/B_1 = 0,19`. **Es gibt kein N.** PP gewinnt hier nie gegen
   das Haupt-Rig allein — solange das Modell aufs Haupt-Rig passt.
6. **Wogegen PP sehr wohl gewinnt: das heutige cross-rig TP=4.** Sofort bei N=1, um Faktor
   **2,3-3,2x** (16-22 tok/s gegen gemessene 6,9), und der Vorsprung waechst mit der Batchgroesse,
   weil TPs Link-Last linear mit N skaliert und PPs nicht.
7. **Empfehlung: BAUEN, aber in dieser Reihenfolge und mit diesem Rahmen** — V1 rein intra-Rig
   als Einzelteil ([[einzelteil-vor-verbund]]), Nutzen ist **Kapazitaet**, nicht Tempo, und der
   erste Schritt ist eine **Messung mit null Zeilen Code**. Details §5.

---

## 1. Ist-Zustand PP im Fork/Upstream

### 1.1 Stage-Bildung und Rang-Layout

`pp_size` wird geparst in `server_args.py:1090-1096` (Alias `--pipeline-parallel-size`), dazu
`pp_max_micro_batch_size` (`:1096`) und `pp_async_batch_depth` (`:1101`).

**PP ist die AEUSSERE Dimension** des Rang-Gitters:

```python
# model_executor/model_runner.py:1555-1557
init_distributed_environment(
    world_size=self.tp_size * self.pp_size,
    rank=self.tp_size * self.pp_rank + self.tp_rank, ...)
```
also `rank = pp_rank * tp_size + tp_rank` (gleiche Formel in `managers/tp_worker.py:312,560,636`).

Daraus folgt die Gruppen-Geometrie in `distributed/parallel_state.py`:
- **TP-Gruppen sind zusammenhaengende Bloecke** (`:2383-2392`): Raenge `[i*tp_size, (i+1)*tp_size)`.
  Eine TP-Gruppe liegt also vollstaendig **innerhalb einer Stage** — das ist die gute Nachricht
  fuer den Nutzer-Wunsch "auf rig zwei wieder ein uneven tp dcp".
- **PP-Gruppen sind gestreift** (`:2636-2654`): eine PP-Gruppe je TP-Spalte,
  `ranks = range(pp_group_idx, world_size, tp_size)`, `rank_in_group == pp_rank`,
  `use_custom_allreduce=False`.
- **DCP-Gruppen werden INNERHALB der TP-Gruppen ausgeschnitten** (`:2424-2433`) — DCP ⊂ TP ⊂ Stage.
  Auch das faellt unter PP strukturell korrekt heraus.
- Strukturcheck: `world_size != tp_size * pp_size` → `RuntimeError` (`:2360-2365`).

Prozess-Spawn: `entrypoints/engine.py:657-661` (`for pp_rank ... for tp_rank ...`), GPU-Zuweisung
`server_args.py:5369-5387 gpu_id_for_rank()`.

### 1.2 Layer-Zuweisung — hier sitzt der Ratio-Hebel, und er existiert schon

`distributed/utils.py:1197-1233`, **verifiziert byte-identisch zu upstream**:

```python
def get_pp_indices(num_hidden_layers, pp_rank, pp_size) -> Tuple[int, int]:
    partition_list_str = os.getenv("SGLANG_PP_LAYER_PARTITION", None)
    if partition_list_str is not None:
        partitions = [int(layer) for layer in partition_list_str.split(",")]
        if len(partitions) != pp_size:      raise ValueError(...)
        if sum(partitions) != num_hidden_layers: raise ValueError(...)
        start_layer = sum(partitions[:pp_rank])
        end_layer   = start_layer + partitions[pp_rank]
    else:
        base_layers = num_hidden_layers // pp_size
        remainder   = num_hidden_layers % pp_size
        # Distribute the extra layers to the last 'remainder' partitions
        ...
```

- Default: **gleichmaessig**, `num_layers // pp_size`, Rest auf die **letzten** Stages.
- **Der uneven-Hebel ist schon da**: `SGLANG_PP_LAYER_PARTITION="52,12"` ist heute gueltiger,
  upstream-getesteter Code. Das ist der sglang-Port von vLLMs `VLLM_PP_LAYER_PARTITION`.
- Er ist **nicht** in `environ.py` registriert (rohes `os.getenv`) und hat **kein CLI-Flag**.
  Das ist die gesamte fehlende Oberflaeche fuer "groessere Karte mehr Layer" — mehr nicht.

Konsument `utils/common.py:1494-1536 make_layers()`: nicht-eigene Layer werden zu
`PPMissingLayer`-Identity-Stubs, **globale Layer-Indizes bleiben erhalten** (`layers/utils/common.py:109-127`).
Modell-Aufrufstellen identisch in ~60 Modellen (`models/llama.py:396-412`, `models/qwen2.py:343`, ...).

Nachgelagert: `model_runner.py:939-941` (`start_layer`/`end_layer`/`num_effective_layers`),
KV-Pool auf das Stage-Fenster gesizt (`model_runner_kv_cache_mixin.py:2094-2099`),
SWA/Full-Attn-Listen pro Stage neu gefiltert (`model_runner.py:1254-1274`).

### 1.3 Aktivierungs-Transport zwischen Stages

Nutzlast = `PPProxyTensors` (`forward_batch_info.py:1564-1591`). Pro Mikrobatch vorwaerts
**zwei** Tensoren (`models/llama.py:429-462`):

```python
if not self.pp_group.is_last_rank:
    return PPProxyTensors({"hidden_states": hidden_states, "residual": residual})
```
je `[num_tokens, hidden_size]` im Modell-dtype. Der `residual` faehrt mit, weil die RMSNorm mit
dem Residual-Add fusioniert ist (Upstream-`FIXME` an Ort und Stelle). DSV4/mHC packt beides in
einen Tensor (`runner/base_runner.py:109-124`) — der Beleg, dass 1 statt 2 Tensoren machbar ist.

**Der Draht** (`parallel_state.py:1719-1841 send_tensor_dict`/`recv_tensor_dict`):

```python
group          = self.device_group     # NCCL
metadata_group = self.cpu_group        # gloo
if dst is None: dst = (self.rank_in_group + 1) % self.world_size
p2p_works = self.send_object(metadata_list, dst=dst, ...)      # gepicklet ueber gloo
for tensor in tensor_list:
    comm_group = metadata_group if tensor.is_cpu else group
    work = send_func(tensor, self.ranks[dst], group=comm_group) # torch.distributed.isend
```

Also: **`torch.distributed.isend/irecv` auf dem NCCL-`device_group`** fuer die Tensoren, **gloo**
fuer die gepicklete Metadaten. Bemerkenswert: `GroupCoordinator.send()/recv()` (`:1843-1868`)
bevorzugen zwar `pynccl_comm`, liegen aber **nicht** auf dem PP-Scheduler-Pfad.

**Zwei Dinge, die fuer die Kostenrechnung zaehlen:**
- *Send-Allgather-Optimierung* (`scheduler_pp_mixin.py:1019-1023,1073-1078`): nur `1/attn_tp_size`
  jedes Aktivierungstensors quert die Stage-Grenze, die Empfaenger-Stage gathert intern.
  Das drueckt die Grenzlast noch einmal um den TP-Grad der Stage.
- *Metadaten-Pickle pro Crossing*: `send_object` (`:1548-1632`) schickt Groessen-Tensor + Payload
  ueber gloo — also **2 zusaetzliche Nachrichten pro Crossing**, obwohl die Formen pro bs statisch
  sind. Cachebar; heute nicht gecacht. Relevant, weil es das "eine Nachricht"-Argument von 2 auf
  ~4 Nachrichten verwaessert (§3.3).

### 1.4 Scheduling / Mikrobatching — 1F1B existiert bereits

`scheduler.py:4834-4856` → `event_loop_pp()`; Implementierung `managers/scheduler_pp_mixin.py:67-174`.
Es ist **echtes interleaved Mikrobatching**, kein Ein-Batch-Betrieb:

```python
self.pp_loop_size: int = self.ps.pp_size + self.server_args.pp_async_batch_depth  # :559
while True:
    for mb_id in range(self.pp_loop_size):                                        # :89-93
```
Default `pp_async_batch_depth=0` → genau `pp_size` Mikrobatches in Flight. Pro Mikrobatch:
Requests empfangen → weiterreichen → schedulen → `_pp_recv_proxy_tensors()` (blockierend) →
`_pp_launch_batch` → Ergebnis des *anderen* Mikrobatches verarbeiten → async isend an die
naechste Stage.

Requests-Cap pro Mikrobatch: `scheduler.py:974-980` setzt automatisch
`pp_max_micro_batch_size = max(max_running_requests // pp_size, 1)`. **Das ist die Stelle, an der
die Batch-Amortisation verloren geht** — siehe §3.5.

**Autoregressive Rueckschleife**: `_pp_send_output_to_next_stage` (`:1154-1187`); da
`send_tensor_dict`s Default `dst = (rank_in_group+1) % world_size` ist, ist der "Nachfolger" der
letzten Stage **Rang 0**. Inhalt `{"next_token_ids": ...}` (+ Logprobs auf Anforderung).
Rang 0 konsumiert in `_pp_prep_batch_result` (`:1114-1147`).

**CUDA-Graphs funktionieren mit PP fuer Decode** (`runner/decode_cuda_graph_runner.py:905-910,1332-1339`;
Puffer `runner_utils/buffers.py:310-313`). Nur *piecewise* Graphen werden unter PP still
abgeschaltet (`server_args.py:5184`). Das ist fuer §3.4 entscheidend.

### 1.5 Verbote — acht Argparse-Stellen lehnen `pp_size > 1` ab

| Kombination | Stelle | Kern der Meldung |
|---|---|---|
| **PP + Overlap-Scheduler + Spekulation** | `server_args.py:9647-9650` | `assert (self.disable_overlap_schedule and self.speculative_algorithm is None)` — harter Assert, **keine** Auto-Abschaltung |
| **PP + MTP-Modelle** | `model_runner.py:950-957` | `"PP is not compatible with MTP models."` |
| PP + Prefill-Context-Parallelism | `server_args.py:7790` | `"PP is not supported with context parallelism"` |
| PP + elastic EP | `server_args.py:8155` | `"PP size should be set to 1 under elastic EP"` |
| PP + PD-Multiplexing | `server_args.py:9691-9694` | `pp_size == 1` erzwungen |
| PP + piecewise CUDA-Graph | `server_args.py:5184` | still abgeschaltet |
| PP + Modell ohne `pp_proxy_tensors` | `model_runner.py:776-784` | `"Pipeline Parallel is not compatible with this model."` |
| **Fork:** `--rank-gpu-id` | `server_args.py:5784-5789` | `"not compatible with --pp-size > 1 ... Only pure tensor parallelism is supported."` |
| **Fork:** `--weightless-kv-fastlane` | `server_args.py:3994-3999` | dito |
| **Fork:** `--enable-kv-session-offload` | `server_args.py:4290-4294` | dito |
| **Fork:** `--speculative-draft-placement solo` | `server_args.py:4442-4446` | dito |
| **Fork:** Hibernate | `model_loader/hibernate.py:100-121` | `"#89 hibernate V1 is scoped to pure single-node Tensor Parallelism"` |

**Nicht** verboten und damit bereits nutzbar: PP + DP-Attention, **PP + DCP**, PP + einfaches EP,
PP + volle Decode-CUDA-Graphen.

### 1.6 Fork-Delta: keines

`git diff 82e7cdcff9 HEAD` ueber die PP-tragenden Dateien:
- `managers/scheduler_pp_mixin.py`: **0 Aenderungen**, taucht im Diff nicht auf.
- `distributed/utils.py`: +1102 Fork-Zeilen, aber `grep -E '^[+-].*(get_pp_indices|PP_LAYER|pp_rank|pp_size)'`
  liefert **nichts** — die Fork-Additionen sind uneven-TP / `--rank-kv-ratio` / DCP-Solver.
- `distributed/parallel_state.py`: +348 Zeilen, alles HTCCL und kv-session-offload; der
  `_PP`-Block (`:2636-2654`) und `send_tensor_dict`/`recv_tensor_dict` sind unveraendert.
- `models/llama.py`, `utils/common.py`: PP-Pfad unberuehrt.

**Konsequenz fuer die Aufwandsschaetzung:** wir bauen nicht gegen eigenen Code, sondern erweitern
stock-upstream-Code. Das ist billiger *und* riskanter (kein eigener Testbestand, aber
Upstream-Regressionen treffen uns).

---

## 2. Die Schnittpunkte — Feature fuer Feature

Rahmen: "PP-Stage" = ein Rang haelt nur `[start_layer, end_layer)`, und die TP-Gruppe umspannt
genau eine Stage.

### A) Uneven TP (`--rank-tp-ratio`) — NEEDS PARAMETERIZATION

- Partition-Mathematik `distributed/utils.py:598,618,662` (`partition_units`/`sizes`/`offsets`),
  TP-Wrapper `:671,693,705`. Die Funktionen sind **layer-agnostische reine Funktionen** — gut.
- Der Ratio ist **prozess-globaler Singleton**: `_TP_PARTITION_RATIOS` (`utils.py:86-90`),
  installiert einmal in `scheduler.py:4923 set_tp_partition_ratios(...)`. Der umgebende
  `configure_scheduler_process` (`:4861`) **bekommt `pp_rank`** (`:4867`), benutzt ihn aber nie.
- Laenge ist an die volle TP-Groesse gepinnt: `server_args.py:5677`
  (`if len(self.rank_tp_ratio) != self.tp_size: raise`).
- **Stiller Fehlmodus:** `tp_partition_sizes` faellt bei Laengen-Mismatch auf den geraden Split
  zurueck (`utils.py:687-690`) statt zu erroren.
- **LUECKE / kleiner Bug:** ein blankes `--rank-tp-ratio` **ohne** `--rank-gpu-id` hat heute
  **keinen** PP-Reject — der Check liegt hinter dem `rank_gpu_id`-Early-Return
  (`server_args.py:5659-5670`). Es wuerde unter PP durchlaufen und still gerade splitten.
  Das ist unabhaengig von #201 ein Sofort-Fix ([[bugs-immer-prioritaet]]).

Noetig: Ratio je Stage (`pp_rank`-gekeyt oder Liste von Vektoren), Validierung gegen die
**Stage-TP-Breite**. Keine Aenderung an der Partition-Mathematik.

### B) Uneven DCP + Owner-Rule — NEEDS PARAMETERIZATION (Allokator schon COMPATIBLE)

- Owner-Rule `layers/dcp/owner.py:25-27`, backend-agnostisch:
  `rank owns slot L <=> (L % cp_S) in [cp_lo, cp_hi)`.
- **Die DCP-Gruppe faellt unter PP GRATIS korrekt heraus** (`parallel_state.py:2424-2433`
  schneidet innerhalb der ohnehin stage-lokalen TP-Gruppen).
- **Die entscheidende Antwort auf die Nutzer-Sorge ("der KV liegt pro Stage nur fuer DEREN
  Layer — Wechselwirkung mit dem Token-Split"): es gibt KEINE Wechselwirkung.** Die beiden
  Achsen sind orthogonal: der Layer-Split partitioniert die **Layer-Achse** des KV-Pools, der
  DCP-Token-Split die **Token-Achse**. Der Allokator ist auf **beiden** Achsen schon
  parametriert: `mem_cache/memory_pool.py:1555-1572` nimmt `start_layer`/`end_layer`, jeder
  Zugriff ist `self.k_buffer[layer_id - self.start_layer]` (`:2062,2076,2123,2276,2451,2523,2598`),
  und `model_runner_kv_cache_mixin.py:2094-2099` uebergibt das Stage-Fenster bereits.
  Die Owner-Rule rechnet ausschliesslich in Slot-Indizes *innerhalb* eines Layers.
- Was **nicht** stage-faehig ist: der Token-Vektor `_CP_TOKEN_RATIOS` (`utils.py:162`) und
  `uneven_dcp_owner_bounds` (`:300-326`) sind Prozess-Globals ohne `pp_rank`-Dimension, und die
  Pool-Sizing-Formel `model_runner_kv_cache_mixin.py:2390-2407` teilt eine **weltweit**
  min-reduzierte `max_total_num_tokens` durch einen stage-lokalen Ratio — inkommensurable
  Einheiten, sobald zwei Stages verschiedene Vektoren haben.
- Der DCP-Collective-Guard (`layers/dcp/collective_guard.py:87`) all-gathert die Zahl der
  dispatchten Full-Attention-Layer ueber die stage-lokale `cp_group` — ueberlebt den Split, wird
  aber **sofort laut**, wenn zwei Raenge derselben Gruppe je verschieden viele Layer halten.
  Nuetzlich: das ist ein eingebauter Falsifikator fuer einen falschen Stage-Split.

### C) KV-Kapazitaet / `--rank-kv-ratio` — NEEDS PARAMETERIZATION

- Profiling ist bewusst **rang-lokal** unter uneven Budgets
  (`model_runner_kv_cache_mixin.py:139-164`, Kommentar `:146-152`: *"the classic min-reduce of
  free bytes would collapse every rank onto the weakest GPU"*).
- `bytes → pages` ist **schon layer-parametriert**: `pool_configurator.py:241-251` filtert
  `effective_layer_ids` gegen `[start_layer, end_layer)` und nutzt `num_effective_layers`.
- **Aber `max_total_num_tokens` wird uniform erzwungen — per `ReduceOp.MIN` ueber die
  WELT-Gruppe, an drei Stellen:** `model_runner_kv_cache_mixin.py:2974-2988` (uneven-DCP-Zweig),
  `:3011-3027` (generischer Zweig), `:1103-1111` (Mamba-State). Der Kommentar bei `:3011` nennt
  den PP-Fall sogar ausdruecklich: *"Sync across PP ranks (each may have different layer counts)"*.
- **Warum das unter uneven Layer-Split aktiv schaedlich ist:** eine kurze Stage hat pro Token
  billigeren KV und wuerde eine viel groessere Token-Kapazitaet profilieren; die Welt-MIN wirft
  das weg und pinnt alle auf die tiefste Stage. Genau das Gegenteil dessen, was uneven Layer-Split
  bezweckt.
- Noetig: die KV-Geometrie-MIN wandert auf die **DCP/TP-Gruppe**, dazu eine separate
  Cross-Stage-Reconciliation, die unterschiedliche Kosten pro Token beruecksichtigt.

### D) Spec / MTP / EAGLE + Solo-Placement — HARD CONFLICT

Die direkte Antwort auf die Nutzer-Frage *"Draft-Modell auf welcher Stage? Verify braucht ALLE Layer!"*:

- **Verify ist ein voller Ziel-Forward, dessen Logits SYNCHRON auf demselben Rang gelesen werden**
  — `eagle_worker_v2.py:2609 verify()` → `:2673 self.target_worker.forward_batch_generation(...)`,
  Konsum der `logits_output` in `:2674`, danach Grammar-Bitmask und Rejection-Sampling lokal.
  Unter PP produziert **nur die letzte Stage** Logits; fruehere geben `PPProxyTensors` zurueck
  (`model_runner.py:490`). Die Verify-Schleife muesste als Pipeline-Rundreise neu geschrieben werden.
- **Der MTP/NEXTN-Kopf haengt hinter dem letzten Layer** → er landet zwangslaeufig auf der
  **letzten Stage**. Bei der naheliegenden Anordnung (schwaches Rig zuletzt) laeuft der Draft
  k-mal seriell **auf der schwaechsten Karte** — der genaue Gegenentwurf zum Sinn von Spec.
  **Design-Konsequenz mit Zahl: die Stages so ordnen, dass das schwache Rig NICHT letzte Stage
  ist.** Zusatzlast der letzten Stage: `lm_head` = 5120 x 248320; bei fp8 1,27 GB pro Token
  gelesen = 2,06 ms auf der 2080 Ti (616 GB/s) — also ungefaehr **eine ganze zusaetzliche
  Layer-Aequivalenz** (τ1 ≈ 2,23 ms/Layer, §3.2), plus derselbe Betrag noch einmal fuer jeden
  Draft-Schritt.
- Solo-Placement setzt voraus, dass **ein** Rang einen ungeshardeten Draft ueber **alle** Layer
  haelt (`model_runner.py:373 compute_draft_solo_role`, Schatten auf `meta` `:1897,1917-1919`).
  Reject `server_args.py:4442-4446`.
- Upstream verbietet PP+Spec ohnehin hart (`server_args.py:9647-9650`) und PP+MTP-Modelle
  (`model_runner.py:950-957`).

**Verdikt: nicht parametrierbar, ein eigener Umbau.** Fuer #201 explizit ausserhalb V1-V3.

### E) CUDA-Graph-Plan als Gruppen-Entscheidung — NEEDS PARAMETERIZATION

- `model_runner.py:1110 _harmonize_cuda_graph_plan`, gerufen `:1210`; Kommentar `:1205`:
  *"The graph plan must be a GROUP decision, not a per-rank one."* Regel (`:1128`): pro Phase
  gathern, bei Uneinigkeit fuer **alle** deaktivieren (Minimum regiert).
- Die Gruppe ist `tp_group.cpu_group` bzw. `attention_tp_group.cpu_group` (`:1176-1200`) — unter
  PP also **genau eine Stage**. Zwei Stages verhandeln unabhaengig.
- **Fuer Kollektive ist das harmlos** (keine queren die Stage-Grenze). **Fuer den P2P-Fahrplan
  ist es das nicht**: eine gecapturte Stage replayt eine fixe send/recv-Sequenz bei fixen Formen,
  waehrend die eager Nachbar-Stage sie pro Batch neu herleitet — dieselbe Divergenz-Klasse, gegen
  die diese Funktion existiert, eine Ebene hoeher.
- Noetig: zweistufige Verhandlung (erst intra-Stage, dann cross-Stage) — oder bewusst die
  Entscheidung "Graph-Plan ist pro Stage frei, aber die P2P-Formen sind vertraglich fixiert".
  **Letzteres ist der eigentliche Gewinn (§3.4) und sollte die gewaehlte Richtung sein.**

### F) HTCCL-Transport-Registry — HARD CONFLICT heute, auf UCX loesbar

Die Nutzer-Intuition (*"Stage-Grenze = neue Kommunikations-Klasse: Punkt-zu-Punkt statt Kollektiv"*)
trifft exakt den kritischen Punkt.

| Ebene | Implementierte Ops |
|---|---|
| `HTCCLCommunicator` (`htccl.py`) | `all_reduce:210`, `all_gather:281`, `reduce_scatter:317`, `all_gather_into_tensor:358`, `reduce_scatter_tensor:371`, `broadcast:382` |
| `device`-Transport (`htccl_device.py:1086`) | `{all_reduce, all_gather, reduce_scatter, broadcast}` |
| `shm` (`htccl_shm.py:169`) | `{all_reduce}` |
| `ucx` (`feat/htccl-ucx-l2`, `htccl_ucx.py:153`) | `{all_reduce, all_gather, broadcast, reduce_scatter}` + async `:1046,1078,1100` |

**Es gibt nirgends `send`/`recv`.** Und der P2P-Pfad des GroupCoordinators hat **keinen
HTCCL-Zweig** (`parallel_state.py:1843-1869`):

```python
def send(self, tensor, dst=None):
    pynccl_comm = self.pynccl_comm
    if pynccl_comm is not None and not pynccl_comm.disabled: pynccl_comm.send(tensor, dst)
    else: torch.distributed.send(tensor, self.ranks[dst], self.device_group)
```

Das ist **entscheidend**, denn wenn HTCCL an ist, wird pynccl **absichtlich nicht gebaut**
(`parallel_state.py:564-573`, `should_build_pynccl` `:248`, Docstring `:259`: *"NCCL cannot span
vendors."*). Eine PP-Stage-Grenze unter HTCCL faellt also auf `torch.distributed.send` auf dem
`device_group` durch — genau den NCCL/RCCL-Kommunikator, der cross-vendor nicht existieren kann.
Ergebnis: Haenger oder Crash, **ohne** `_htccl_unsupported`-Guard, der es benennen wuerde.

**Der Weg ist aber kurz:** die UCX-Bindings tragen bereits getaggtes send/recv —
`htccl_ucx_bindings.py:307-312` (`ucp_tag_send_nbx`/`ucp_tag_recv_nbx`), gewrappt als `post_send:519`
/ `post_recv:532` / `wait:562`, heute nur private Plumbing der Kollektive
(`htccl_ucx.py:441,452,472`). Es fehlt: ein `send`/`recv`-Seam-Op, Aufnahme in `HTCCL_OPS` (`:153`)
und ein Dispatch-Zweig in `parallel_state.py:1843/1855`. **Das ist die einzige echte
Neubau-Komponente des ganzen Vorhabens** — und sie ist zugleich der fehlende L2-Baustein aus
[[nordstern-rdma-tp5]] (hierarchischer Transport).

Nebenbefund: HTCCL wird heute fuer **jeden** GroupCoordinator mit `world_size > 1` konstruiert,
auch fuer die `"pp"`-Gruppe (`parallel_state.py:502-528`), und unterdrueckt dort pynccl — d. h. der
Fehlmodus ist bereits scharf, nur unerreichbar, weil alle Fork-Features PP ablehnen.

### G) Expert-Offload / Weightless-Lane / Hibernate

- **G1 MoE-Expert-Offload (#77): COMPATIBLE.** Die Layer-Iteration ist ein **Modul-Walk**, kein
  Index-Range: `model_runner.py:1688-1694` (`for module in self.model.modules(): if isinstance(module, FusedMoE) ...`).
  `self.model` enthaelt auf einem PP-Rang bereits nur die Stage-Layer → natuerlich stage-lokal.
  Die Hotset-Datei ist auf **absolute** `layer_id` gekeyt (`expert_offload.py:803-808`) und failt
  hart bei Miss — unter PP korrekt, sofern der Offline-Trace dieselbe absolute Nummerierung hat.
  Anpinnen, nicht fixen. Orthogonaler Blocker: braucht `--disable-cuda-graph`, ausser
  `SGLANG_MOE_OFFLOAD_CUDA_GRAPH=1` (`fused_moe_triton/layer.py:490-501`).
- **G2 Weightless-KV-Lane: HARD CONFLICT.** Praemisse woertlich (`distributed/utils.py:187-190`):
  *"One rank (the 'head rank') holds ALL attention heads and runs Q/O-proj + FFN + GDN as pure
  TP=1"*; der Worker laeuft ueber **jeden** Attention-Layer des gespiegelten Voll-Modells
  (`model_runner.py:3628-3631`). Das ist die exakte Negation eines Stage-Subsets. Eine PP-faehige
  Variante ("ein gewichtstragender Rang **pro Stage**") waere ein anderes Feature, keine
  Parametrierung. Reject `server_args.py:3994-3999`.
- **G3 Hibernate: NEEDS PARAMETERIZATION, mechanisch trivial.** Shard-Datei heisst
  `rank{tp_rank}_{...}.pt` (`hibernate.py:74-77`) — **kein `pp_rank`**; die Identitaets-Fingerprint
  (`:86-98`) enthaelt `tp_size, dcp_size, rank_tp_ratio, rank_gpu_id, context_length`, aber
  **weder `pp_size` noch die Layer-Partition** → ein Restore ueber eine *andere* Layer-Aufteilung
  wuerde grob matchen und den falschen Shard laden. Fix: `pp_rank` in den Dateinamen, `pp_size` +
  Partition in die Identitaet, `_assert_v1_scope` (`:100-121`) lockern.

### Zusammenfassung der Vertraeglichkeit

| Feature | Verdikt |
|---|---|
| KV-Allokator (Layer-Achse) | **COMPATIBLE** — schon parametriert |
| DCP-Gruppenbildung | **COMPATIBLE** — faellt per Konstruktion heraus |
| MoE-Expert-Offload | **COMPATIBLE** |
| Uneven TP (`--rank-tp-ratio`) | NEEDS PARAM — globaler Singleton → pro Stage |
| Uneven DCP Token-Vektor + Owner-Bounds | NEEDS PARAM — globaler Singleton → pro Stage |
| KV-Kapazitaet (`max_total_num_tokens`) | NEEDS PARAM — Welt-MIN → Gruppen-MIN + Reconciliation |
| CUDA-Graph-Plan | NEEDS PARAM — zweistufige Verhandlung / P2P-Formvertrag |
| Hibernate | NEEDS PARAM — trivial |
| **HTCCL-Transport** | **HARD CONFLICT** — kein P2P; UCX-Bindings tragen es aber schon |
| **Spec / MTP / Solo-Draft** | **HARD CONFLICT** — Verify ist ein Ein-Rang-Voll-Forward |
| **Weightless-KV-Lane** | **HARD CONFLICT** — Praemisse ist die Negation von PP |

---

## 3. Die Kernrechnung: Stage-Grenze gegen TP-Gruppe

### 3.1 Eingangsgroessen (alles gemessen, nichts geschaetzt)

**Modell** Qwen3.6-27B-FP8 (`.../models-cache/Qwen3.6-27B-FP8/config.json`):
`num_hidden_layers=64`, `hidden_size=5120`, `head_dim=256`, `num_attention_heads=24`,
`num_key_value_heads=4`, `vocab_size=248320`, Gewichte **28,47 GB** → **0,445 GB/Layer**,
`embed_tokens` und `lm_head` je 1,27 GB (fp8).
`layer_types`: `full_attention_interval=4` → **16 Full-Attention-Layer, 48 Linear-Attention (GDN)**.

**Link** (40G RoCE, HTCCL/UCX L2, cross-rig, beide Richtungen live —
`/spinning/wt-ucx-l2/FEATURES_VS_UPSTREAM.md:600-660`):
8 KiB all_reduce **26,5 us**, Barrier **5,2-5,5 us**, roher Link-RTT **~1,5 us**,
`ucx_perftest` unidirektional 3413 MB/s = **27,3 Gbit/s**, 4 MiB-Peak 21,19 Gbit/s.

**Decode-Baselines**, Qwen3.6-27B-FP8, bs=1, **ohne** Spec:

| Konfiguration | Modus | tok/s | ms/Token | Quelle |
|---|---|---|---|---|
| Haupt-Rig TP=3 uneven | CUDA-Graphs | **37,8** | 26,5 | `RESULT_concurrency_sweep.md` |
| Haupt-Rig TP=3 uneven | CUDA-Graphs | 44,2 | 22,6 | `RESULTS.md` (anderer ctx/kv-dtype) |
| Haupt-Rig TP=3 uneven | eager | 12,5 | 80,0 | `RESULTS.md` |
| 5090 solo (gequetscht) | eager | 15,3 | 65,4 | `RESULTS.md` |
| **cross-rig TP=4** (6,4,4,2) | eager | **6,9** | **145** | `crossrig_tp4/SUMMARY.txt` |
| cross-rig TP=4 + NEXTN-Spec | eager | 16,3-18,4 | — | `FEATURES_VS_UPSTREAM.md:659` |

**Auslastung** waehrend der cross-rig-Messung: Haupt-Rig **10-13 %/GPU**, 2080 Ti **50-55 %**.

Erster harter Befund: **cross-rig TP=4 (145 ms) ist 1,8x LANGSAMER als das Haupt-Rig allein bei
gleichem Modus (eager, 80 ms) — obwohl es eine GPU mehr hat.** Die vierte Karte kostet, sie
zahlt nicht ein.

### 3.2 Bytes und Nachrichten pro Token

**TP-Gruppe cross-rig.** Pro Layer 2 All-Reduce (nach Attn-Out-Proj, nach MLP-Down-Proj) →
**128 Kollektive/Token**. Nutzlast je Kollektiv bei bs=1: `5120 x 2 B = 10,0 KiB`.
Der UCX-Default ist ein Ein-Schritt-Vollaustausch, d. h. der Rig-2-Rang sendet 3x und empfaengt 3x:

```
6 x 10,0 KiB x 128  =  7,50 MiB pro Token ueber den Link
128 x 26,5 us       =  3,39 ms pro Token, synchron blockierend
```

**PP-Stage-Grenze.** Pro Mikrobatch vorwaerts `hidden_states` + `residual`:

```
2 x 5120 x 2 B      =  20,0 KiB pro Token pro Grenze   (bf16-Aktivierungen)
```
plus Rueckschleife `next_token_ids` (8 B + Metadaten). Mit der Send-Allgather-Optimierung
(`scheduler_pp_mixin.py:1019`) bei attn_tp=3 auf der Empfaenger-Stage: **6,7 KiB**.
Mit dem mHC-Trick (Residual in hidden_states gefaltet, `runner/base_runner.py:109-124`): **10 KiB**.

| | cross-rig TP=4 | PP 2 Stages | Faktor |
|---|---|---|---|
| Bytes/Token ueber den Link | 7,50 MiB | 20,0 KiB (6,7 KiB mit Allgather) | **384x (1150x)** |
| Nachrichten/Token | 128 Kollektive | 2 P2P (heute: ~4, s. u.) | **64x (32x)** |
| Link-Zeit/Token | 3,39 ms | ~0,05 ms (heute ~0,2 ms) | **68x (17x)** |
| Wire-Zeit bei 27,3 Gbit/s | 2,20 ms | 0,006 ms | 384x |

Die Klammerwerte "heute" beruecksichtigen das ungecachte Metadaten-Pickle (§1.3): pro Crossing
gehen 2 gloo-Nachrichten (Groesse + Payload) mit, obwohl die Formen pro bs statisch sind. Cachen
ist ein kleiner, klar umrissener Fix und verdoppelt den Vorteil.

### 3.3 Ehrliche Einordnung: der Link ist bei bs=1 NICHT das Problem

3,39 ms von einem gemessenen 145-ms-Schritt sind **2,3 %**. Wer die Kollektive vollstaendig
loescht, gewinnt hoechstens 2,3 %.

Das deckt sich mit dem, was #198 selbst festgehalten hat: *"straggler compute (not communication)
is the next order of magnitude"* — und mit der Auslastung: die 2080 Ti rechnet ~52 % von 145 ms
= **~75 ms**, die Haupt-Rig-Karten je ~15 ms. Der Verbund wartet auf **eine** Karte.

**Und genau das behebt PP nicht.** Unter TP ist die Schrittzeit `max_i(compute_i) + comm`; unter
PP ist sie `Σ_s(stage_time_s)`. PP macht den Straggler-Beitrag **additiv statt maximal**. Bei
perfekt balanciertem Split mit Arbeitsanteil `f_s` und Kapazitaet `B_s` gilt

```
T_PP = Σ_s W_s/B_s  =  S · W/(Σ_s B_s)         (balanciert: alle Stage-Zeiten gleich)
T_TP = W/(Σ_s B_s)  + comm                     (ideal, verlustfrei balanciert)
```

**PP ist also per Konstruktion exakt S-mal schlechter als ideales TP.** Es gibt keinen Term, der
das kompensiert; PP kauft ausschliesslich, dass `comm` verschwindet — und `comm` sind hier 2,3 %.

### 3.4 Wo PP trotzdem gewinnt: der Graph-Plan

Der Grund, warum das cross-rig TP=4 heute **145 ms statt 80 ms** braucht, ist nicht der Draht,
sondern dass die **gruppenweite Graph-Verhandlung** (`model_runner.py:1110-1200`, Minimum regiert)
alle vier Raenge in den eager-Modus zwingt, weil der sm75-Rang nicht capturen kann. Gemessener
eager-Aufschlag auf demselben Rig: **12,5 gegen 44,2 tok/s = 3,5x** (`RESULTS.md`).

**Unter PP ist die Verhandlungsgruppe die Stage** (`:1176-1200` gathert ueber `tp_group.cpu_group`,
das unter PP eine Stage ist), und volle Decode-Graphen sind mit PP unterstuetzt
(`decode_cuda_graph_runner.py:905-910`). Also: **Stage 0 behaelt ihre Graphen, waehrend Stage 1
eager laeuft.** Das ist der einzige strukturelle Hebel im ganzen Vorhaben, der eine
Groessenordnung bewegt — und er kommt nicht vom Link, sondern von der Entkopplung der
Faehigkeits-Verhandlung.

Preis: der P2P-Fahrplan muss vertraglich fixiert werden (§2E), sonst replayt eine gecapturte
Stage feste Formen gegen eine eager Nachbarin.

### 3.5 Die Bubble-Rechnung und die Frage "ab welchem N kippt es"

Per-Layer-Zeiten. Haupt-Rig mit Graphen: `26,5 ms / 64 = 0,414 ms/Layer` (τ0).
Fuer die 2080 Ti (τ1) zwei Schaetzer, bewusst als Korridor:

- **(a) optimistisch, Bandbreiten-Roofline mit derselben Effizienz.** Haupt-Rig erreicht
  28,47 GB / 3312 GB/s = 8,60 ms Roofline gegen 26,5 ms real → 32,4 % Effizienz. 2080 Ti bei
  616 GB/s: 46,2 ms Roofline → **142,6 ms Voll-Modell → τ1 = 2,23 ms/Layer**.
- **(b) pessimistisch, aus der Live-Messung abgeleitet.** Rig 2 hielt 12,5 % der Gewichte und
  rechnete ~75 ms → Voll-Modell ~600 ms → **τ1 = 9,38 ms/Layer**. Enthaelt eager **und** den
  fp8-Dequant-Fallback auf sm75 (kein natives fp8-GEMM).

Der Faktor 4 zwischen (a) und (b) **ist** der sm75-fp8-Handicap. Bei einem dtype mit nativem
Kernel gilt (a); bei fp8 auf Turing gilt (b). Das ist die groesste Unsicherheit der Rechnung und
zugleich ein Hinweis: **die Modell-/dtype-Wahl entscheidet hier mehr als die Topologie.**

Balancierter Split `L1 · τ1 = (64 − L1) · τ0`:

| | L1 (Layer auf Rig 2) | T_PP | tok/s | vs. cross-rig TP=4 (6,9) | vs. Haupt-Rig allein (37,8) |
|---|---|---|---|---|---|
| (a) τ1=2,23 | 10 | 44,7 ms | **22,4** | **+3,2x** | −41 % |
| (b) τ1=9,38 | 3 | 53,4 ms | **18,7** | **+2,7x** | −51 % |

(In (b) ist die Stage-Reihenfolge bereits optimiert: schwaches Rig **zuerst**, damit `lm_head`
und ein etwaiger MTP-Kopf auf dem Haupt-Rig bleiben — sonst kaemen 8,7 ms dazu und es faellt auf
16,1 tok/s.)

**Gewonnene Kapazitaet:** (a) 10 x 0,445 = **4,45 GB**, (b) 3 x 0,445 = **1,33 GB**.
**Preis pro verlagertem GB:** `(τ1 − τ0)/0,445 GB` = **4,1 ms/Token/GB** (a) bzw.
**20,1 ms/Token/GB** (b). Als Vergleichsanker: naives Gewichts-Streaming aus Host-RAM ueber den
PCIe-Pfad dieses Rigs (Gen4 x4, ~6 GB/s real, [[rig-interconnect-p2p]]: kein P2P, alles PHB, GPU0
auf x4) kostet ~167 ms/Token/GB. **Cross-rig-PP ist pro GB also 8-40x billiger als naives
Host-Streaming** — gegen #77s *sparse* MoE-Offload (nur geroutete Experten, geprefetcht,
effektiv ~13 ms/Token/GB) ist es dagegen etwa ein Gleichstand bis 3x besser.

#### Die N-Frage, sauber beantwortet

Im bandbreitengebundenen Decode ist die Stage-Zeit von der Mikrobatch-Groesse fast unabhaengig
(die Gewichte dominieren). Mit `S` Stages, `N` Sessions und dem existierenden Cap
`pp_max_micro_batch_size = max_running_requests // pp_size` (`scheduler.py:974-980`):

```
Durchsatz_PP  = (N/S) · (B_1 + B_2)          [ein Mikrobatch je Stage-Zeit 1/(B_1+B_2)]
Durchsatz_Rig1 =  N   ·  B_1
PP gewinnt  <=>  (B_1+B_2)/S > B_1  <=>  B_2 > (S−1)·B_1
```

Bei `S=2` muss die zweite Stage also **mindestens so schnell sein wie das ganze erste Rig**.
Gemessen ist `B_2/B_1 = 26,5/142,6 = 0,19`.

> **Es gibt kein N.** PP schlaegt das Haupt-Rig allein bei diesem Modell nie — weder bei 1 noch
> bei 100 Sessions. Der Grund ist nicht die Bubble im engeren Sinn, sondern dass
> Pipeline-Mikrobatching die **Batch-Amortisation zerstoert**: N Sessions in S Mikrobatches
> aufgeteilt lesen die Gewichte S-mal pro Runde, waehrend ein Batch von N auf einer TP-Gruppe sie
> einmal liest. Gemessener Beleg fuer die Staerke dieser Amortisation (Haupt-Rig, Graphen, nospec):
> N=1 37,8 → N=2 81,7 (2,16x, superlinear) → N=4 156,7 → N=8 270,7 tok/s.

**Die Gegenrechnung, die PP rettet — aber nur gegen cross-rig TP:** TPs Link-Last skaliert
**linear mit N**, PPs praktisch nicht.

| N | TP=4 cross-rig Link-Kosten/Schritt | PP Link-Kosten/Schritt | Haupt-Rig-Schritt (Referenz) |
|---|---|---|---|
| 1 | 5,6 ms | 0,21 ms | 26,5 ms |
| 4 | 12,2 ms | 0,25 ms | 25,5 ms |
| 8 | **21,0 ms** | 0,30 ms | 29,6 ms |
| 16 | **38,6 ms** | 0,39 ms | ~35 ms (extrapoliert) |
| 32 | **73,8 ms** | 0,58 ms | ~45 ms (extrapoliert) |

(TP: `7,5 MiB · N / 3,41 GB/s + 3,39 ms`; PP: `40 KiB · N / 3,41 GB/s + 0,2 ms`.)

> **Ab N≈8 uebersteigt allein die Link-Last des cross-rig-TP 70 % einer kompletten
> Haupt-Rig-Schrittzeit; ab N≈16 uebersteigt sie den ganzen Schritt.** Cross-rig TP ist damit
> durchsatz-feindlich; cross-rig PP ist es nicht. **Das ist die eigentliche Rechtfertigung fuer
> PP: nicht "schneller als das Haupt-Rig", sondern "die einzige cross-rig-Topologie, die mit der
> Batchgroesse skaliert".**

#### Und ohne RDMA?

Auf dem gemessenen 1-GbE-Pfad (112 MiB/s, TCP-p50 78 us, [[nordstern-rdma-tp5]]):
TP `7,5 MiB/0,112 GB/s + 128·78 us = 67 + 10 = 77 ms/Token`. PP `20 KiB/0,112 GB/s + 4·78 us
= 0,17 + 0,31 = ~0,5 ms/Token`. **Faktor 154.**

> **Auf 1 GbE ist PP die einzige lauffaehige cross-rig-Topologie ueberhaupt.** Das entkoppelt den
> [[nordstern-rdma-tp5]]-Pfad vom ConnectX-Bring-up: L0 cross-rig **PP** braucht kein RDMA.

### 3.6 Ein modellspezifischer Befund fuer den Split-Planer

Bei Qwen3.6-27B ist der Speicher **pro Layer nicht uniform**:
- **16 Full-Attention-Layer** tragen den wachsenden KV: `2·4·256 = 2048 Elemente/Token/Layer`,
  fp8 → 2 KiB. Bei ctx 8192 und 8 Sessions: **134 MB pro Full-Layer**.
- **48 Linear-Attention-Layer (GDN)** tragen einen **pro-Sequenz** festen Zustand
  (`mamba_ssm_dtype=float32`, `linear_num_value_heads=48`, `linear_value_head_dim=128`,
  `linear_key_head_dim=128`) — kontextlaengen-unabhaengig, aber sessionzahl-abhaengig; grob
  ~25 MB pro Linear-Layer bei 8 Sessions. Das ist die Ursache des `mamba-capped 8200` in
  `crossrig_tp4/SUMMARY.txt`.
- `embed_tokens` (Stage 0) und `lm_head` (letzte Stage) sind **je 1,27 GB und keine Layer**.

> **Design-Regel: der Split-Ratio muss auf BYTES pro Layer rechnen (Gewichte + KV@ctx +
> State x Sessions + die beiden Nicht-Layer-Bloecke), nicht auf Layer-ZAHL.** Bei Periode 4 hat
> zwar jeder zusammenhaengende Bereich der Laenge L genau ⌊L/4⌋..⌈L/4⌉ Full-Layer, ist also
> zufaellig fair — aber Gemma-4 und MoE-Modelle mit ungleichen Layer-Groessen sind es nicht.
> Der Mechanismus muss allgemein sein, auch wenn dieses eine Modell gnaedig ist.

---

## 4. Zielbild

### 4.1 Konfiguration

```
Stage 0 = Rig 2  (2080 Ti + Vega 64), intern TP=2 uneven, L1 Layer, eager
Stage 1 = Rig 1  (5090 + 2x 3080),    intern TP=3 uneven + uneven DCP, 64−L1 Layer, CUDA-Graphs
```

Bewusst **schwaches Rig zuerst**: `embed_tokens` (billig) landet dort, `lm_head` + Sampling +
ein spaeterer MTP-Kopf bleiben auf dem starken Rig (§2D, §3.5).

Layer-Zahl gewichtet nach **Bytes/Layer geteilt durch Stage-Durchsatz**, nicht nach Kapazitaet
allein und nicht nach Rechenleistung allein — beide Nebenbedingungen zugleich:
```
minimiere  max_s (bytes_s / B_s)   u.d.N.  bytes_s <= vram_s
```

### 4.2 Flags

| Flag | Rolle | Bemerkung |
|---|---|---|
| `--pp-layer-ratio a,b,...` | uneven Layer-Split, analog `--rank-tp-ratio` | duenner CLI-Wrapper um das **schon existierende** `SGLANG_PP_LAYER_PARTITION`; zusaetzlich `auto-capacity` / `auto-performance` wie beim TP-Planer (`uneven_perf.py:2368`) |
| `--rank-tp-ratio` | pro Stage, als Liste von Vektoren oder `pp_rank`-gekeyt | heute prozess-globaler Singleton (`utils.py:86`) |
| `--rank-kv-ratio` | dito, pro Stage | |
| `--stage-map` | Rang→Rig/Host-Zuordnung fuer cross-rig | erst ab V3 noetig; intra-Rig reicht `--rank-gpu-id` nach Lockerung von `server_args.py:5784` |
| `SGLANG_HTCCL_PP_TRANSPORT` | P2P-Lane fuer die Stage-Grenze | `ucx` / `torch` (Default `torch` = heutiges Verhalten) |

### 4.3 Guards (fail fast, nach dem Muster der acht existierenden Rejects)

- `sum(pp_layer_ratio) == num_hidden_layers` und `len(...) == pp_size` (upstream hat das schon,
  `utils.py:1214-1216`) — plus: **jede Stage >= 1 Layer** und `num_hidden_layers >= pp_size`
  (`utils/common.py:1509`).
- **HTCCL + `pp_size > 1` ohne P2P-Lane → harter Fehler mit Namensnennung**, statt des heutigen
  stillen Durchfalls auf `torch.distributed.send` (§2F). Das ist der wichtigste neue Guard.
- Spec/MTP, Weightless-Lane, kv-session-offload, Solo-Draft: **Rejects bleiben** (§2D, §2G2).
- Der offene Sofort-Fix: blankes `--rank-tp-ratio` ohne `--rank-gpu-id` braucht einen
  PP-Reject (§2A).
- Gruppen-uniform zu entscheiden: Graph-Plan pro Stage **plus** ein cross-Stage-Formvertrag fuer
  die P2P-Puffer; KV-Kapazitaets-MIN auf Gruppen- statt Weltebene.

### 4.4 Migrationspfad in Scheiben ([[einzelteil-vor-verbund]])

| Scheibe | Inhalt | Neuer Code |
|---|---|---|
| **S0 — MESSUNG** | `pp_size=2` **intra-Rig**, stock, `SGLANG_PP_LAYER_PARTITION`, ohne jedes Fork-Feature, `--disable-overlap-schedule`, kein Spec. Misst τ0/τ1 empirisch, prueft das Modell aus §3.5, deckt die Upstream-Bruchstellen auf. | **null Zeilen** |
| **S1** | `--pp-layer-ratio` als CLI + Planner (Bytes/Layer, §3.6); Hibernate-Keying (§2G3) | klein |
| **S2** | uneven TP + uneven DCP **pro Stage** (Singletons → `pp_rank`-gekeyt) + KV-Kapazitaets-MIN auf Gruppenebene (§2A/B/C) | mittel |
| **S3** | HTCCL-P2P-Lane ueber UCX (`send`/`recv`-Seam-Op + Dispatch in `parallel_state.py:1843/1855`) — **die einzige echte Neubau-Komponente** (§2F) | mittel |
| **S4** | cross-rig PP=2 ueber RoCE; Graph-Plan zweistufig + P2P-Formvertrag (§2E) | mittel |
| **S5** | PP x Spec/MTP — Verify als Pipeline-Rundreise (§2D) | gross, **zurueckstellen** |

---

## 5. Aufwand und Empfehlung

| Scheibe | Aufwand | Risiko | Empfehlung |
|---|---|---|---|
| S0 Messung | **~0,5 Tag, kein Code** | sehr niedrig | **SOFORT, sobald die Karten frei sind** |
| S1 Layer-Ratio + Planner | ~1-2 Tage | niedrig (Mechanismus existiert) | bauen, nach S0 |
| S2 Per-Stage-Ratios + KV-Kapazitaet | ~3-5 Tage | mittel (drei Singletons + drei MIN-Reduces, [[geteilter-puffer-familie]]-Muster) | bauen, wenn S0 traegt |
| S3 HTCCL-P2P-Lane | ~2-4 Tage | mittel (Bindings da, Semantik neu) | **eigenstaendig wertvoll** — auch ohne PP der fehlende L2-Baustein aus [[nordstern-rdma-tp5]] |
| S4 cross-rig PP | ~3-5 Tage | hoch (Graph/P2P-Vertrag, zwei Hosts) | erst nach S3 + S0-Zahlen |
| S5 PP x Spec | ~2 Wochen+ | sehr hoch | **zurueckstellen** |

### Empfehlung

**Bauen — aber mit korrigierter Zielsetzung und in dieser Reihenfolge.**

1. **Zuerst messen, nicht bauen.** S0 kostet null Zeilen Code, weil
   `SGLANG_PP_LAYER_PARTITION` bereits existiert. Es liefert genau die Zahl, an der die ganze
   Rechnung haengt (τ1, Korridor Faktor 4 zwischen Roofline und Live-Messung) und pruefte das
   Modell aus §3.5 gegen die Wirklichkeit. Alles andere sollte davon abhaengen.
2. **Die Zielsetzung muss "Kapazitaet" heissen, nicht "Tempo".** Die Rechnung ist eindeutig:
   PP schlaegt das Haupt-Rig allein nie, solange das Modell aufs Haupt-Rig passt — N-unabhaengig,
   weil `B_2 > (S−1)·B_1` auf dieser Hardware unerfuellbar ist. Wer #201 als Tempo-Feature
   verkauft, verkauft es falsch. Als **Kapazitaets**-Feature ist es dagegen mit 4-20 ms/Token/GB
   gegen 167 ms/Token/GB (naives Host-Streaming) klar im Recht — und gegen das heutige cross-rig
   TP=4 gewinnt es sofort um 2,7-3,2x.
3. **S3 (HTCCL-P2P) vorziehen und eigenstaendig rechtfertigen.** Es ist die einzige echte
   Neubau-Komponente, es ist der als fehlend benannte hierarchische Transport aus
   [[nordstern-rdma-tp5]] (Leitplanke 4), und es macht cross-rig **ohne RDMA** tragbar (§3.5,
   Faktor 154 auf 1 GbE). Dieser Wert ist unabhaengig davon, ob PP je in Produktion geht.
4. **S5 (Spec x PP) zurueckstellen.** Verify ist ein Ein-Rang-Voll-Forward mit synchronem
   Logit-Konsum (`eagle_worker_v2.py:2673`); das ist ein Umbau, keine Parametrierung. Da Spec
   auf diesem Rig 2,0-2,4x bringt (`crossrig_tp4/SUMMARY.txt`), heisst das konkret: **jede
   PP-Konfiguration verliert heute den Spec-Gewinn**. Das gehoert offen in die Bewertung —
   PP=2 mit 22 tok/s gegen Haupt-Rig+Spec mit ~75 tok/s ist der ehrliche Vergleich.
5. **Ein Sofort-Fix unabhaengig von #201:** blankes `--rank-tp-ratio` ohne `--rank-gpu-id` hat
   keinen PP-Reject und wuerde still gerade splitten (§2A, `server_args.py:5659-5670`).

### Geltungsbereich dieser Analyse ([[stichprobenbreite-schlussfolgerung]])

Ein Modell (Qwen3.6-27B-FP8, 64 Layer, hybrid GDN/Full-Attn), ein Rig-Paar, ein Link (40G RoCE),
Decode bei bs=1 bis N=8, reine Lesearbeit ohne einen einzigen PP-Boot. Die Aussage
"PP verliert gegen das Haupt-Rig allein" gilt **fuer Modelle, die aufs Haupt-Rig passen** — genau
dort, wo Kapazitaet kein Argument ist. Fuer 122B-A10B (heute #77-Host-Offload) ist die Rechnung
nicht gemacht und koennte anders ausgehen; das waere die naechste zu quantifizierende Frage.
τ1 ist der schwaechste Punkt (Korridor Faktor 4) und genau das, was S0 fuer null Aufwand klaert.

---

# TEIL 2 — Bestandskarte + Slice 1 (Auftrag #201 Phase 1, 2026-07-28)

Basis: `origin/integration/r3-probe-next2` = `c626de2e52`, Worktree `/spinning/wt-201`,
Branch `feat/uneven-pp-slice1`. Zeilennummern in diesem Teil beziehen sich auf **diesen**
Stand (Teil 1 stand auf `integration/r3-probe`, die Nummern sind dort verschoben).

## 6. Bestandskarte

### 6.1 Upstream-PP in diesem Baum — unveraendert gegenueber Teil 1

Nachgeprueft, keine Abweichung zur Analyse aus Teil 1:

| Gegenstand | Fundstelle (c626de2e52) |
|---|---|
| `pp_size` / `pp_max_micro_batch_size` / `pp_async_batch_depth` | `server_args.py:1091-1102` |
| Rang-Gitter `rank = pp_rank * tp_size + tp_rank` | `model_runner.py:844-845`, `:1598-1599` |
| Layer-Zuteilung `get_pp_indices` + `SGLANG_PP_LAYER_PARTITION` | `distributed/utils.py:1288-1324` |
| Konsument `make_layers` (PPMissingLayer-Stubs, globale Layer-Indizes) | `utils/common.py:1744-1785` |
| 1F1B-Mikrobatching `event_loop_pp` | `managers/scheduler_pp_mixin.py:68-174` |
| Mikrobatch-Cap `max_running_requests // pp_size` | `managers/scheduler.py:997-1000` |
| Stage-p2p: `send_tensor_dict`/`recv_tensor_dict` | `distributed/parallel_state.py:1757-1880` |
| Send-Allgather-Optimierung (nur `1/attn_tp_size` quert die Grenze) | `parallel_state.py:1795-1812`, Aufruf `scheduler_pp_mixin.py:1067,1154` |
| Prozessschnitt: ein Scheduler-Prozess je (pp_rank, tp_rank) | `entrypoints/engine.py` Spawn-Schleife, Platzierung `server_args.gpu_id_for_rank` |

Der Transport ist weiterhin `torch.distributed.isend/irecv` auf dem NCCL-`device_group`,
Metadaten gepicklet ueber gloo. **Kein HTCCL-P2P** — der Befund aus Teil 1 §2F steht
unveraendert und bleibt der Blocker fuer Slice 2 (Cross-Rig).

### 6.2 GDN/Hybrid x PP — das Verdikt

**Die Frage entscheidet sich an der Modelldatei, nicht am PP-Kern.**

| Modelldatei | Architektur | PP |
|---|---|---|
| `models/qwen3_5.py` (Qwen3.5 / **Qwen3.6**, GDN + Full-Attn) | `Qwen3_5ForConditionalGeneration`, `Qwen3_5MoeForConditionalGeneration` | **faehig** |
| `models/qwen3_next.py` (Qwen3-Next, GDN + Full-Attn) | `Qwen3NextForCausalLM` | **harter Abbruch** |

Qwen3-Next bricht zuerst an `models/qwen3_next.py:1165`:
`assert self.pp_group.is_first_rank and self.pp_group.is_last_rank` — ohne Meldung, in
`__init__`, also **vor** dem Runner-Gate `model_runner.py:803-806`. `make_layers` wird dort
ohne `pp_rank`/`pp_size` gerufen (`:1070-1072`), und `forward` hat kein `pp_proxy_tensors`
(`:1220-1227`).

Qwen3.5/3.6 erfuellt den Vertrag vollstaendig und strukturgleich zu `models/llama.py`:
`embed_tokens` nur auf Rang 0 / sonst `PPMissingLayer` (`qwen3_5.py:1386-1399`), `make_layers`
mit `pp_rank`/`pp_size` (`:1427-1434`), `norm` nur auf der letzten Stage (`:1436-1439`),
`start_layer`/`end_layer` als Properties (`:1451-1457`), `forward(..., pp_proxy_tensors=...)`
mit Rueckgabe von `PPProxyTensors` (`:1460-1479`, `:1511-1517`), Layer-Schleife ueber das
Stage-Fenster (`:1483`), Gewichts-Loader ueberspringt fremde Layer (`:1560-1566`).
Die Einstiegs-Architekturen erben `forward` von `models/qwen3_vl.py:1400-1440`, das
`pp_proxy_tensors` durchreicht.

**Mamba/GDN-State-Pool ist stage-lokal und korrekt indiziert** — das war die eigentliche
Sorge, und sie traegt nicht:
- `MambaPool` wird auf `len(mamba_layer_ids)` gesizt (`mem_cache/memory_pool.py:385`),
  alle Puffer `(num_mamba_layers, size+1, ...)` (`:437,460,476,481,489,511`).
- Uebersetzung global -> dicht ueber ein Dict, nicht ueber `layer_id - start_layer`:
  `self.mamba_map = {layer_id: i for i, layer_id in enumerate(mamba_layer_ids)}`
  (`memory_pool.py:1124`), benutzt in `mamba2_layer_cache` (`:1214-1218`).
- Die uebergebene Layer-Liste ist an **allen drei** Konstruktionsstellen aufs Stage-Fenster
  gefiltert: `model_runner_kv_cache_mixin.py:1943-1947`, `:2598-2604`, `:2632-2645`.
- Die Full-Attention-KV-Seite symmetrisch: `model_runner_kv_cache_mixin.py:3192-3210`,
  Mapping `memory_pool.py:2866-2867`, angewandt `:2922-2951`.
- Der Backend-Dispatch testet **Mitgliedschaft globaler** Layer-IDs
  (`hybrid_linear_attn_backend.py:821-828`) — stage-agnostisch und damit sicher.

Der Baum traegt die Upstream-Fixes dafuer: `16802fb6b2 [FIX] fix mambaish model pp kv cache
compute (#17334)` und `9b4dd27478 [Fix] Fix Qwen3.5 MoE model loading and Mamba cache sharding
in PP mode (#21448)`. Es existiert sogar ein PP-Genauigkeitstest fuer genau dieses Modell:
`test/registered/pp/test_pp_single_node_extra.py:251-300` (`TestQwen35PPAccuracy`, tp2/pp1 gegen
tp1/pp2 auf gsm8k) — in CI als "too flaky" uebersprungen, also gebaut, aber nicht bewacht.

**Zwei Restdefekte, beide nicht toedlich:**
1. `configs/mamba_utils.py:178-187` `mamba_cache_per_req` multipliziert mit der **globalen**
   Zahl der GDN-Layer. Verbraucht in `model_runner_kv_cache_mixin.py:1491,1506,1574,1619-1620,
   1697,1706`. Unter PP rechnet sich also jede Stage den State **aller** Stages an und
   unterdimensioniert `max_mamba_cache_size` um grob den Faktor `pp_size`. Konservativ und
   korrekt, aber verschwenderisch — der einzige Ort, an dem globale Indizierung ueberlebt hat.
   Kandidat fuer Slice 3.
2. `memory_pool.py:2935-2937` `_wait_for_layer` rechnet `layer_id - self.start_layer` ueber
   **alle** Stage-Layer, waehrend der darunterliegende Full-KV-Pool nach der dichten
   **Full-Attention**-Nummer indiziert wird. Zwei Indexraeume. Nur scharf, wenn ein
   `layer_transfer_counter` haengt (HiCache layer-wise transfer) — beruehrt #212, nicht mich.

**Es gibt nirgends einen Reject, der PP zusammen mit mamba/GDN/hybrid/linear-attn nennt.**
Die Kombination ist absichtlich erlaubt.

### 6.3 #154 gegengeprueft: der MTP-Block zaehlt NICHT als Layer

Die Lehre ist im Baum bereits kodifiziert, an zwei Stellen:
- `model_loader/gguf_registry.py:226-236`: `n_blocks = kv("block_count") - kv("nextn_predict_layers")`,
  mit dem Kommentar, dass ein MTP-tragendes Qwen3.6-27B-GGUF 65 meldet und der Draft als
  `blk.64` liegt.
- `uneven_perf.py:1426-1433`: `n_layers = block_count - nextn`.

`configs/model_config.py:1088` setzt `num_hidden_layers` auf die reine Backbone-Zahl; der
MTP-Block lebt in `num_nextn_predict_layers`. **Konsequenz fuer `--pp-layer-ratio`: die Summe
muss 64 sein, nicht 65.** Das ist als Test festgenagelt
(`test_sum_mismatch_is_rejected_by_get_pp_indices`).

### 6.4 Der #202-Reject: was er schuetzt, und was kontrolliert faellt

Vorher (`server_args.py:6546-6551` bzw. `:6666-6671` im Ausgangsstand):

```
--rank-tp-ratio + pp_size > 1  -> hart abgelehnt
--rank-gpu-id   + pp_size > 1  -> hart abgelehnt
```

**Was er wirklich schuetzt, ist eine PLATZIERUNGS-Kollision, keine Partitions-Frage.**
`gpu_id_for_rank` (`server_args.py:6127-6146`) gab `self.rank_gpu_id[tp_rank]` zurueck — ohne
`pp_rank`. Unter einer Pipeline haette also **jede** Stage dieselben physischen Karten
adressiert: der gesamte Pipeline-Stapel auf einer Stage-Breite Karten, jede Karte mit
`pp_size` vollen Gewichts-Slices und `pp_size` KV-Pools. Genau das ist der Fehlmodus, und er
ist kein Argument gegen uneven TP je Stage — er ist ein Argument dafuer, dass die Platzierung
den Stage-Index kennen muss.

Der zweite Schutz ist echt und bleibt: `--rank-tp-ratio` ist ein prozess-globaler Singleton
(`distributed/utils.py:_TP_PARTITION_RATIOS`, installiert in `scheduler.py`
`configure_scheduler_process`), und `configure_scheduler_process` bekommt `pp_rank` zwar
uebergeben, benutzt ihn aber nicht. Solange das so ist, kann jede Stage nur **denselben**
Ratio-Vektor fahren. Fuer strukturell gleiche Stages ist das korrekt; verschiedene Vektoren je
Stage sind Slice 3.

**Kontrolliert geoeffnet wird genau eine Tuer:** eine Pipeline, deren Stages **je eine eigene,
disjunkte Gruppe physischer Karten** besitzen. Ausgedrueckt als welt-langes `--rank-gpu-id`
(`pp_size x tp_size` Eintraege in Weltrang-Ordnung `pp_rank * tp_size + tp_rank`). Alles andere
bleibt ein harter Reject mit benanntem Grund. Details in §7.2.

### 6.5 Der Rest der PP-Reject-Matrix in diesem Baum

| Kombination | Fundstelle | Charakter |
|---|---|---|
| PP + Overlap-Scheduler **oder** Spekulation | `server_args.py:11197-11200` | harter Assert, **keine** Auto-Abschaltung — jeder PP-Boot braucht `--disable-overlap-schedule` und darf kein Spec fahren |
| PP + MTP-Modelle | `model_runner.py:970-979` | `"PP is not compatible with MTP models."` |
| PP + Modell ohne `pp_proxy_tensors` | `model_runner.py:803-806` | Signatur-Introspektion, `**kwargs` genuegt **nicht** |
| PP + Context-Parallelism | `server_args.py:8949` | |
| PP + elastic EP | `server_args.py:9314` | |
| PP + PD-Multiplexing | `server_args.py:~10992` | |
| PP + `--enable-dsa-cache-layer-split` | `server_args.py:7887-7891` | |
| PP + DFLASH / DSpark | `arg_groups/speculative_hook.py:175-178`, `:320-323` | |
| PP + CPU-Graph-Runner | `model_executor/cpu_graph_runner.py:610` | |
| PP + piecewise CUDA-Graph | `server_args.py:5963` | still abgeschaltet |
| `num_hidden_layers >= pp_size` | `utils/common.py:1759` | |

Volle **Decode**-CUDA-Graphen sind unter PP unterstuetzt; nur piecewise faellt weg. Das ist der
Hebel aus Teil 1 §3.4 und bleibt gueltig.

Nebenbefund (nicht meiner, #212 gemeldet): `model_runner.py:1284,1291`
`adjust_hybrid_swa_layers_for_pp` iteriert `range(self.start_layer, self.end_layer + 1)` —
inklusive `end_layer`, also ein Layer zu weit. Latent, weil kein GDN-Hybrid als
`is_hybrid_swa` gilt (`configs/model_config.py:2043-2078`); scharf fuer SWA-Modelle unter PP.

## 7. Slice 1 — Umsetzung

### 7.1 `--pp-layer-ratio`

`server_args.py`: Flag-Definition direkt an `pp_size` (`:1103-1122`), Handler
`_handle_pp_layer_ratio` als letzter Schritt von `_handle_pipeline_parallelism`.
Duenner, validierter CLI-Mantel um `SGLANG_PP_LAYER_PARTITION` — derselbe Weg, den der
PD-Topologie-Planer (`disaggregation/topology.py:750-754`) schon benutzt; die beiden werden
gegeneinander verriegelt statt sich zu ueberschreiben.

Validierung, alles fail-fast vor dem Gewichts-Load:
- `pp_size > 1` noetig, Laenge `== pp_size`, jeder Eintrag `>= 1`.
- Summe gegen `config.json`, **best effort**: bei einem GGUF-Checkpoint gewinnt die
  `block_count` der Datei ueber die Sibling-Config (`gguf_registry.py`), also dort nur
  Warnung statt Reject. Die verbindliche Pruefung bleibt, wo die echte Tiefe existiert:
  `get_pp_indices`.
- Konflikt mit `--disaggregation-prefill-layer-split` und mit einem bereits abweichend
  gesetzten `SGLANG_PP_LAYER_PARTITION`.

Ohne Flag wird die Umgebung **nicht angefasst** — der Default-Pfad ist byte-gleich.

### 7.2 Geoeffnete Tuer und neue Reject-Matrix

| pp_size | `--rank-tp-ratio` | `--rank-gpu-id` | Ergebnis |
|---|---|---|---|
| 1 | egal | Laenge `tp_size` | unveraendert |
| >1 | gesetzt | fehlt | **Reject**: nennt die verlangte Laenge `pp_size x tp_size` und die Weltrang-Ordnung |
| >1 | gesetzt | Laenge `pp_size x tp_size`, Stage-Gruppen disjunkt | **zugelassen** |
| >1 | gesetzt | Stages teilen sich eine Karte | **Reject**: nennt die Stages und die geteilten Karten |
| >1 | egal | Laenge weder `tp_size` noch Welt | **Reject** mit der Welt-Formel |
| >1 | `auto` / `auto-performance` | egal | **Reject**: der Planer leitet **einen** Vektor aus allen Karten ab, unter Pipeline spannt der ueber alle Stages (Slice 3) |

Disjunktheit ist die Zulassungsbedingung, keine Stilregel: zwei Stages auf einer Karte halten
beide Gewichts-Slices und beide KV-Pools dort und laufen trotzdem nacheinander — der Split
kostet Speicher, ohne Kapazitaet zu kaufen. Ko-Lokation **innerhalb** einer Stage bleibt
erlaubt (der bestehende Multi-Rank-pro-GPU-Modus).

Mitgezogen, weil die Tuer sonst hinter der ersten Zeile wieder zufaellt:
`gpu_id_for_rank` indiziert jetzt nach Weltrang (`world_rank(pp_rank, tp_rank)`),
`--rank-gpu-memory-mib` als Liste ist welt-lang, `_rank_mem_fraction_static` hat einen Eintrag
je Weltrang, und `apply_rank_memory_budget(tp_rank, pp_rank)` liest ihn dort
(`managers/scheduler.py:configure_scheduler_process`).

### 7.3 Was Slice 1 ausdruecklich NICHT tut

- Kein per-Stage-Ratio: `_TP_PARTITION_RATIOS` bleibt prozess-global, alle Stages fahren
  denselben Vektor (§6.4).
- Keine Aenderung an der Welt-MIN-Reduktion von `max_total_num_tokens`
  (Teil 1 §2C) — unter uneven Layer-Split pinnt sie weiterhin alle Stages auf die tiefste.
  Das ist der groesste offene Posten von Slice 3.
- `mamba_cache_per_req` bleibt global gerechnet (§6.2 Defekt 1).
- Kein HTCCL-P2P, also kein Cross-Rig (Slice 2).

## 8. Aufwand Slice 2 und Slice 3 — mit einer Korrektur an Teil 1

### 8.1 Slice 2 (Cross-Rig-Grenze) — deutlich billiger als Teil 1 §5 annahm

Teil 1 hat den Cross-Rig-Schritt an S3 (HTCCL-P2P-Lane, 2-4 Tage) und S4 (3-5 Tage) gehaengt,
weil HTCCL kein `send`/`recv` kennt. **Diese Kopplung gilt fuer die geplante Konfiguration
nicht.**

- Stage 1 auf Rig 2 ist zunaechst die **2080 Ti — eine NVIDIA-Karte**. HTCCL existiert fuer
  Cross-**Vendor**-Gruppen. Eine NVIDIA↔NVIDIA-Pipeline ueber die 40G-Strecke braucht kein
  HTCCL, sondern faellt auf genau den Pfad, den PP ohnehin benutzt:
  `torch.distributed.isend/irecv` auf dem NCCL-`device_group`, Metadaten ueber gloo
  (`parallel_state.py:1757-1880`). HTCCL wird nur konstruiert, wenn `SGLANG_HTCCL=1` gesetzt
  ist — fuer diesen Lauf wird es das nicht.
- **Mehr-Knoten-PP ist bereits verdrahtet, ohne eine Zeile Neucode.**
  `entrypoints/engine.py:1547-1580 _calculate_rank_ranges`: bei `pp_size=2, nnodes=2` faehrt
  Knoten 0 genau `pp_rank 0` und Knoten 1 genau `pp_rank 1`. Das ist exakt die Rig-Aufteilung.
- Der `nnodes > 1`-Reject fuer `--rank-gpu-id` (`server_args.py:~6684`) **bleibt richtig** und
  muss nicht fallen: jeder Knoten platziert nur seine eigenen Raenge in seiner eigenen lokalen
  Geraeteansicht, dafuer ist `--base-gpu-id` das passende Werkzeug. Welt-langes `--rank-gpu-id`
  ist ein Ein-Knoten-Konstrukt.

**Aufwand Slice 2: ~1-2 Tage, und der groesste Teil davon ist Messung, nicht Bau.** Die offenen
Posten sind Integration und Betrieb, nicht Mechanismus:
1. Start-Rezept fuer zwei Knoten (`--nnodes 2 --node-rank 0/1 --dist-init-addr` ueber
   `<RDMA_R1>`), Rig 2 hat kein installiertes sglang (nur `PYTHONPATH=<RIG2_SGLANG_SRC>`) und
   der Sync-Stand muss stimmen.
2. NCCL ueber RoCE ohne GDR: laeuft, kostet Host-Bounce. Nach Teil 1 §3.2 ist die Grenzlast
   ~20 KiB/Token — der Aufschlag ist irrelevant, das ist gerade der Punkt von PP.
3. `--pp-layer-ratio` gegen die 2080-Ti-Kapazitaet: bei Qwen3.6-27B-Q3 traegt sie nach
   Auftragsrahmen ~12-16 der 64 Layer, also `--pp-layer-ratio 50,14` als Startpunkt.
4. Der Graph-Plan wird pro Stage verhandelt (Teil 1 §3.4) — die sm75-Karte zwingt das Haupt-Rig
   **nicht** mehr in den eager-Modus. Das ist der erwartete Groessenordnungs-Gewinn gegenueber
   dem heutigen cross-rig TP=4 und muss dort gemessen werden.

Der HTCCL-P2P-Baustein (Teil 1 §2F) bleibt noetig — aber erst, wenn die **Vega 64** eine Stage
tragen soll. Er ist damit von Slice 2 entkoppelt und behaelt seine eigenstaendige Begruendung
aus [[nordstern-rdma-tp5]].

### 8.2 Slice 3 (uneven TP auf beiden Stages) — ~3-5 Tage, unveraendert das teuerste Stueck

Die Substanz aus Teil 1 §2A/B/C steht; Slice 1 hat nur die Tuer geoeffnet, nicht den Raum
moebliert. Vier Posten, in dieser Reihenfolge:

1. **`_TP_PARTITION_RATIOS` je Stage** (`distributed/utils.py`, installiert in
   `scheduler.py:configure_scheduler_process`). Der Prozess bekommt `pp_rank` bereits
   uebergeben und benutzt ihn nicht — die Aenderung ist mechanisch, die Frage ist die
   CLI-Form (Liste von Vektoren gegen `pp_rank`-gekeytes Mapping). **~0,5 Tag.**
2. **`max_total_num_tokens`-MIN von der Welt- auf die Gruppen-Ebene**
   (`model_runner_kv_cache_mixin.py`, drei Stellen: uneven-DCP-Zweig, generischer Zweig,
   Mamba-State). Das ist der grosse Posten: unter uneven Layer-Split pinnt die Welt-MIN heute
   alle Stages auf die tiefste — das genaue Gegenteil des Feature-Zwecks. Braucht eine
   Cross-Stage-Reconciliation, die verschiedene KV-Kosten pro Token beruecksichtigt.
   **~2 Tage, und der [[geteilter-puffer-familie]]-Falsifikator gehoert davor.**
3. **`mamba_cache_per_req` stage-lokal** (`configs/mamba_utils.py:178-187`, §6.2 Defekt 1).
   Jede Stage rechnet sich heute den GDN-State aller Stages an. Klein, aber es kostet unter PP
   real Kapazitaet. **~0,5 Tag.**
4. **`--rank-tp-ratio auto` je Stage** — der Planer muesste je Stage-Kartengruppe getrennt
   laufen und das Layer-Gewicht der Stage kennen. Erst sinnvoll, wenn 1-3 stehen. **~1 Tag.**

Nicht in Slice 3, und begruendet ausgeschlossen: PP x Spec/MTP (Teil 1 §2D — Verify ist ein
Ein-Rang-Voll-Forward, upstream ohnehin hart verboten in `server_args.py:11197-11200`) und die
Weightless-KV-Lane (Teil 1 §2G2 — ihre Praemisse ist die Negation eines Stage-Subsets).

## 9. Smoke intra-Rig — Ergebnis (2026-07-28)

Konfiguration: `pp_size=2`, TP=1 je Stage, Stage 0 = 5090 (44 Layer),
Stage 1 = 3080 (20 Layer), `Qwen3.6-27B-MTP-Q3_K_M-GGUF`, `--enable-metrics`,
`--disable-overlap-schedule`, kein Spec. Rezept in `docs/rig-runbook.md` §4.7.

**Urteil: PP x GDN-Hybrid laeuft.** Boot bis "fired up" ~100 s, 9313-Token-Prefill +
700 Token Greedy-Decode in 16,6 s, Dauerzustand **44,2 tok/s** mit `cuda graph: True`.
Die Ausgabe war ueber alle 700 Token kohaerent, Zitat:

> "A pipeline stage boundary moves less data than a tensor-parallel group because it only
> transmits the hidden states and residual connections for a single microbatch across stages,
> whereas a tensor-parallel group requires two all-reduce communication operations per layer
> [...]"

Belege zum Layer-Split: `--pp-layer-ratio 44,20` wurde uebernommen
(`uneven pipeline layer split over 2 stages`), Platzierung
`Pipeline placement: {0: [0], 1: [1]}`, und der KV folgt dem Split exakt —
2,99 GB K auf der 44-Layer-Stage gegen 1,36 GB auf der 20-Layer-Stage
(Verhaeltnis 2,2 = 44/20) bei identischem `max_total_num_tokens=142714`.

**Bestaetigt aus Teil 1 §3.4:** volle Decode-CUDA-Graphen laufen unter PP, die
Graph-Verhandlung ist stage-lokal.

**Praezisierung zu Teil 1 §2C (Welt-MIN):** die MIN pinnt beide Stages auf dieselbe
Token-Zahl; in dieser Konfiguration bindet ohnehin die tiefe Stage, die MIN kostet also
nichts. Der Schaden entsteht erst, wenn die kurze Stage deutlich mehr Token finanzieren
koennte — dann wirft die Welt-MIN das weg. Bleibt Slice-3-Posten, aber ohne die
Dramatik, die Teil 1 unterstellt hat.

### 9.1 Zwei Befunde, die der Smoke aufgedeckt hat

**(a) Mixed-Arch-PDL-Bug — NICHT PP-spezifisch, NICHT von #201 verursacht.**
Boot A (dieselbe Topologie, aber `--base-gpu-id 0` statt `--rank-gpu-id`) stirbt auf der
3080 in `layers/fused_qk_rmsnorm_rope_gate.py`:

```
PTXASError: Modifier '.launch_dependents' requires .target sm_90 or higher
ptxas ... --gpu-name=sm_86
```

Ursache: `_ENABLE_PDL` ist eine **Modul-Konstante**, einmal aus
`cuda_sm_at_least(9)` berechnet — und `cuda_sm_at_least` hat `device_id=0` als Default.
Auf diesem Rig ist Geraet 0 die sm_120-5090, also meldet jeder Prozess PDL-Faehigkeit,
auch der auf `cuda:1`. Der Kernel emittiert `griddepcontrol`, `ptxas` uebersetzt fuer das
**echte** Geraet sm_86 und bricht ab.

Das trifft **jeden** Prozess, dessen Rechen-Geraet nicht Geraet 0 ist und sich in der
Architektur davon unterscheidet — unabhaengig von PP. Der Grund, warum es hier zuerst
auftaucht: `--rank-gpu-id` erzwingt `SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS=1`
(`entrypoints/engine.py:636-650`), jeder Prozess sieht dann seine eigene Karte als
`cuda:0`, und alle Standard-Rezepte dieses Rigs benutzen `--rank-gpu-id`. Eine Pipeline mit
`--base-gpu-id` ist die erste Konfiguration ohne diese Isolation. **Nicht in #201 gefixt**
(fremder Code, eigener Bug); Workaround und Fundstelle stehen im Runbook §4.7.

**(b) Per-Stage-Budgets brauchen keinen TP-Ratio — im Slice gefixt.**
`--rank-gpu-memory-mib` als Liste verlangte `--rank-tp-ratio`. Die Begruendung ("bei
gerader TP sind alle Raenge strukturell gleich, nimm einen Skalar") ist eine Aussage ueber
eine reine TP-Gruppe. Eine Pipeline ist keine: die Stages halten **per Konstruktion**
verschiedene Layer-Zahlen, unter `--pp-layer-ratio` absichtlich. Bei `tp_size=1` je Stage
liesse sich der geforderte Ratio nicht einmal ausdruecken. Ein Skalar, der auf die 3080
passt, laesst die 44-Layer-Stage verhungern (gemessen: 175 MiB zu wenig bei 15000 MiB).
Gate jetzt auf `pp_size == 1` beschraenkt.

Zusaetzlich fielen drei Worker-seitige Stellen auf, die die Per-Rank-Vektoren noch nach
`tp_rank` indizierten (`model_runner_kv_cache_mixin.py`: Budget-Auswahl, GPU-Zuordnung der
Fehlermeldung, Ko-Lokations-Zaehler). Unter PP griff damit **jede** Stage auf den Eintrag
von Stage 0 zu — Stage 1 pruefte ihre 3080 gegen das 5090-Budget. Behoben ueber
`_rank_vector_index()`.

---

# TEIL 3 — Slice 2: die Stage-Grenze cross-rig (Auftrag #201, 2026-07-28)

## 10. Ergebnis vorweg

Die Grenze laeuft ueber die 40G-Strecke. `pp_size=2`, `nnodes=2`, Stage 0 auf einer
rig-1-3080, Stage 1 auf der 2080 Ti von Rig 2. **Kein einziger Mechanismus fehlte** —
Teil 2 §8.1 hat richtig vorhergesagt, dass Slice 2 Integration und Messung ist, nicht Bau.
Der gesamte Code-Anteil sind drei Dinge: ein Boot-Rezept (`scripts/pp/`), ein
env-gegateter Grenz-Zaehler (`SGLANG_PP_BOUNDARY_STATS`) und ein Ping-Pong-Werkzeug.
Runbook §4.9.

Vehikel **Qwen3.5-4B safetensors fp16**, nicht das 9B aus dem Auftragsvorschlag: ein
Qwen3.5-9B safetensors existiert auf diesem Rig nicht, nur das GGUF, und GGUF laeuft
auf sm75 gar nicht (#212). Das 4B ist der groesste safetensors-Checkpoint, der auf
BEIDEN Rigs liegt, und laut Runbook §4.8 genau der, der allein nicht auf die 2080 Ti
passt — die Pipeline kauft hier also etwas, das die Karte allein nicht kann.

## 11. Die Zahlen, und wogegen sie ehrlich stehen

Alle drei Arme: Qwen3.5-4B fp16, triton, ctx 16384, kein Spec (PP verbietet es),
`--disable-overlap-schedule`, 18-s-Decode-Fenster, Median aus drei Laeufen,
Ausgabe jedes Mal kohaerent.

| Arm | Karten | Decode | ms/Token | 8k-Prefill TTFT |
|---|---|---|---|---|
| solo | 1x 3080 | **67,6 tok/s** | 14,80 | 1,35 s |
| cross-rig PP=2, 20/12 Layer | 3080 + 2080 Ti | **55,1 tok/s** | 18,16 | 3,42 s |
| cross-rig TP=2 | 3080 + 2080 Ti | **bootet nicht** (siehe unten) | | |

Rauschboden A-vs-A, gleicher Arm zweimal: **0,2 % (2B), 1,1 % (solo 4B), 2,1 %
(cross-rig PP)**. Jede Differenz unter ~2 % ist auf diesem Rig nicht berichtbar.

**Urteil: die Pipeline kostet gegen die schnellere Karte allein 18 %** (55,1 gegen
67,6) und den 2,5-fachen Prefill-TTFT. Das ist **exakt** die Vorhersage aus Teil 1 §0.5
— PP schlaegt das schnelle Rig nicht, solange das Modell allein darauf passt; der Nutzen
ist Kapazitaet. Die 44,2 tok/s aus dem intra-Rig-Smoke (§9) sind ein anderes Modell
(27B-Q3-GGUF auf 5090+3080) und taugen nur als Groessenordnung, nicht als Vergleich;
deshalb ist der solo-3080-Arm ueberhaupt gebaut worden.

**Der TP-Vergleichsarm hat keine Zahl, und das ist selbst das Ergebnis.** Dieselben
zwei Karten als flache cross-rig-TP=2-Gruppe kommen auf reinem NCCL nicht hoch: erst
`AttributeError: 'str' object has no attribute 'local_reader_ranks'` auf Rig 2 (der
Message-Queue-Broadcaster, den Runbook §4.3 fuer cross-rig per
`SGLANG_USE_MESSAGE_QUEUE_BROADCASTER=0` abschaltet), danach stirbt Rang 0s Scheduler
still in `init_distributed`, waehrend Rang 1 dauerhaft in `all_reduce` steht
(py-spy: `distributed_c10d.py:3075`). Genau deshalb faehrt das cross-rig-TP=4-Rezept
aus §4.3 auf HTCCL/UCX — und HTCCL ist host-gestaged, also eager. **Die Pipeline
braucht weder den Broadcaster-Workaround noch HTCCL und behaelt ihre CUDA-Graphen.**
Der Grund ist strukturell: unter PP haelt jeder Knoten `tp_size=1`, also spannt nie
eine TP-Gruppe ueber beide Hosts. Damit ist Teil 1 §0.6 ("PP gewinnt gegen das heutige
cross-rig TP") auf diesem Rig nicht nur quantitativ, sondern qualitativ bestaetigt —
der TP-Arm existiert ohne Zusatzmaschinerie gar nicht.

Nach [[rig-ist-untergrenze]]: diese Zahlen sind der Boden dieses Rigs (kein NVLink,
kein P2P, GPU0 auf x4, 2080 Ti hinter B550-Gen3-x4), kein Urteil ueber cross-rig PP
als Feature.

## 12. Was die Grenze wirklich kostet — gemessen, nicht gerechnet

`scripts/pp/pp_link_pingpong.py`, NCCL-Sockets auf der 40G-Leitung, hidden 2560 fp16,
Einweg = halbe Rundreise. Tabelle im Runbook §4.9. Kernwerte:

- **bs=1-Decode: 10,0 KiB, 142 us Einweg** — die Nutzlast, die Teil 1 §3.2 aus den
  Bytes abgeleitet hat, jetzt auf dem Draht bestaetigt.
- 2048er-Prefill-Chunk: 20,0 MiB, 10,25 ms. Bei 8192 Token 80 MiB in 39,5 ms =
  **2,07 GB/s** — das ist die 40G-Strecke (die 1-GbE-Leitung liefert 0,105 GB/s),
  und es deckt sich mit den 15,45 Gbit/s iperf3-Einzelstrom aus Runbook §4.8.
- Damit ist der Decode-Grenzuebergang **~0,4 ms von 18,2 ms Rundenzeit, ~2 %.**
  Der Link ist nicht, was eine cross-rig-Pipeline bezahlt.

**Neuer Befund, und der einzige billige Hebel, der an der Grenze noch offen liegt:
bei bs=1 kostet die gepicklete Metadata MEHR als die Nutzlast** — 249 us gegen 142 us,
also 64 % des Grenzuebergangs. `send_tensor_dict` schickt vor jedem Crossing einen
Groessen-Tensor plus Pickle-Payload ueber gloo (Teil 1 §1.3 hat das als "cachebar;
heute nicht gecacht" notiert, ohne Zahl). Die Formen sind pro Batchgroesse statisch.
Ein Formen-Cache am Crossing ist damit **die groesste verbleibende Grenz-Ersparnis**
und gehoert nach vorn in Slice 3.

In-Server-Sicht (`SGLANG_PP_BOUNDARY_STATS`, 4B-Lauf): Stage 0 sendet doppelt so oft
wie sie empfaengt — zwei Mikrobatches in Flight (`pp_loop_size = pp_size`), von denen
bei bs=1 nur einer Arbeit traegt. Die 9,2 ms blockierendes `recv` auf Stage 1 sind die
Blase, nicht der Draht; der Zaehler beschriftet das selbst, weil die Verwechslung
sonst garantiert ist.

## 13. Befunde, die den Bau betreffen

**(a) NCCLs Verbs-Pfad ist auf dieser RoCE-Strecke kaputt, UCX auf denselben HCAs nicht.**
Mit `NCCL_IB_HCA=rocep*s0f1 NCCL_IB_GID_INDEX=3` stirbt der erste 5120-Byte-Proxy-Tensor
an `IBV_WC_REM_INV_REQ_ERR(9) ... req_type=Send`. Sockets auf derselben Karte laufen und
liefern 2,07 GB/s. Fremder Bug ([[eigene-bugs-nicht-fremde]]), nicht gejagt; Default im
Rank-Skript ist `NCCL_IB_DISABLE=1`, `NCCL_IB=1` schaltet ihn fuer eine spaetere Jagd
wieder scharf. Bei 10 KiB pro Mikrobatch waere Verbs ein Latenz-, kein Bandbreitengewinn.

**(b) Volle Decode-CUDA-Graphen laufen auf BEIDEN Stages, auch der sm75-Stage.**
Das ist der strukturelle Unterschied zum heutigen cross-rig TP=4, das host-gestaged ist
und deshalb per Design eager fahren muss (Runbook §6.3). Teil 1 §3.4 hat das als den
erwarteten Groessenordnungs-Gewinn benannt; er ist da.

**(c) `--rank-gpu-id` bleibt zu Recht Ein-Knoten.** Jeder Knoten waehlt seine Karte per
`CUDA_VISIBLE_DEVICES` + `--base-gpu-id 0`. Das laesst nebenbei genau ein sichtbares
Geraet pro Prozess uebrig — die Mixed-Arch-PDL-Falle aus §9.1(a) kann cross-rig gar
nicht zuenden.

**(d) Ein Hybrid-GDN-Modell teilt seinen KV nach VOLL-ATTENTION-Layern, nicht nach Layern.**
Beim 2B-Smoke (24 Layer, `full_attention_interval 4`) landeten bei `--pp-layer-ratio 14,10`
drei Voll-Attention-Layer auf jeder Stage — identische `K size: 1.17 GB` trotz 14:10.
Der 4.7-Befund "KV folgt dem Layer-Split exakt" gilt fuer ein dichtes Modell. **Jeder
Split-Planer, der `num_hidden_layers` allein liest, sizet jedes Hybrid falsch.** Neuer
Posten fuer Slice 3.

**(e) Die Welt-MIN auf `max_total_num_tokens` bleibt** (113671 auf beiden Stages).
Unveraendert der groesste offene Posten, cross-rig genauso wie intra-rig.

## 14. Slice 3 — Aufwand gegen den neuen Stand

Teil 2 §8.2 schaetzte 3-5 Tage. Der Stand nach Slice 2 praezisiert das: **4-6 Tage**,
weil zwei Posten dazugekommen sind und einer billiger wurde.

| # | Posten | Aufwand | Aenderung gegen Teil 2 |
|---|---|---|---|
| 1 | `_TP_PARTITION_RATIOS` je Stage | 0,5 T | unveraendert — mechanisch, `pp_rank` liegt vor |
| 2 | `max_total_num_tokens`-MIN Welt -> Gruppe | 2 T | unveraendert der grosse Posten; Falsifikator [[geteilter-puffer-familie]] davor |
| 3 | `mamba_cache_per_req` stage-lokal | 0,5 T | unveraendert |
| 4 | `--rank-tp-ratio auto` je Stage | 1 T | **teurer/spaeter**: braucht zusaetzlich (5) |
| 5 | **NEU** Split-Planer zaehlt Voll-Attention-Layer, nicht Layer | 0,5 T | §13(d); ohne das sizet der Planer jedes Hybrid falsch |
| 6 | **NEU** Formen-Cache fuer die Crossing-Metadata | 0,5 T | §12; groesster verbleibender Grenz-Hebel, 64 % des bs=1-Uebergangs |

Billiger geworden ist der Rahmen selbst: Slice 3 braucht **keinen** cross-rig-Bau mehr.
Die Grenze steht, das Rezept steht, die Messwerkzeuge stehen; jeder Slice-3-Posten laesst
sich intra-rig entwickeln und mit einem Boot ueber `scripts/pp/pp_crossrig_launch.sh`
gegenpruefen.

Weiterhin begruendet ausserhalb Slice 3: PP x Spec/MTP (Teil 1 §2D) und die
Weightless-KV-Lane (Teil 1 §2G2). Der HTCCL-P2P-Baustein bleibt an die **Vega 64**
gebunden und ist von PP entkoppelt — Slice 2 hat bestaetigt, dass eine NVIDIA-NVIDIA-
Pipeline ihn nicht braucht.

## Design-Kandidat (Nutzer-Diskussion 2026-07-28): Phasen-dualer MLP-Split (Delta-Duplikation)

Idee: je Karte ZWEI MLP-Shard-Saetze (prefill-optimaler + decode-optimaler Vektor),
Forward waehlt per Batch-Typ (Verify=M4 zaehlt als Decode). Schluessel: KV/Attention/
GDN-Geometrie bleibt FIX (entkoppelter kv-ratio #88/#210 + DCP-Token-Sharding) ->
kein Umsharden zwischen Phasen, all_reduce-Groesse unveraendert (hidden_size),
Decode-Graphen capturen die Decode-Partition. Kosten: nur das DELTA der Vektoren
dupliziert (~2-3 GiB beim 27B-FP8 fuer 62/37/37 vs ~50/43/43), bezahlt aus KV/max-
Sessions — als Ledger-Posten rechnen. Ertragsgrenze ehrlich: Kollektivboden 68-75 %
unberuehrt; erwartbar mittlere einstellige % Prefill bei 0 Decode-Verlust
(#264-Schere: +8,2 % Prefill vs -13,7 % Decode beim statischen Konzentrat).
Kontinuum: statischer Kompromiss (heute) <-> Delta-Dual (diese Idee) <-> volle
Duplikation = PD mit Rang-Reuse (DESIGN_107). Zweitnutzung benannt: Lastprofil-
Umschaltung ohne Reboot. VORAB-FALSIFIKATOR vor jedem Bau: beide Ziel-Vektoren
einzeln booten, Schere fuer UNSERE Vektoren nachmessen — Bau nur bei realer
Schere >=5 %. Aufwand ~2-4 Agententage. Wizard-Schalter, kein Default.
Praezisierung (Nutzer-Ruecksprache, gleiche Diskussion): (1) KV bestaetigt layout-
invariant (entkoppelter kv-ratio + DCP-Token-Shard). (2) GDN: Mixer INKL. qkvz-
Projektionen bleiben fix (State klebt am Rang; Umzug ~19 MiB/Request je Tick ist
toedlich) — switchbar ist NUR der FFN/MLP-Teil jedes Layers (~2/3 der FLOPs/Bytes,
Loewenanteil des Gewinns bleibt). (3) Draft: NEXTN/MTP ist solo auf der 5090,
phaseninvariant, EIN Draft fuer beide Partitionen; Verify (M=4) laeuft auf der
aktiven Decode-Partition. TRUMPF vs PD: ein Engine-Prozess -> Spekulation und volle
Decode-Graphen bleiben AN (PD erzwingt Spec aus) — die Phasen-Optimierung ohne
No-Spec-Malus.
Fremd-Hardware-Skalierung (Nutzer-Hinweis): Der Gewinn skaliert INVERS zum
Kollektivanteil. Hier frisst der Boden 68-75 % des Prefill-Fensters — die #264-
Schere (+8,2 %/-13,7 %) wirkte nur auf den Rest. Auf Rigs mit P2P/NVLink/vollen
Lanes schrumpft der Boden auf einen Bruchteil -> dieselbe Vektor-Umverteilung
wirkt auf den 2-3x groesseren Nicht-Kollektiv-Anteil, erwartbare Schere entsprechend
groesser ([[rig-ist-untergrenze]]: hiesige Zahl ist der BODEN des Features).
Ehrliche Zielgruppen-Grenze: Die Schere existiert NUR bei gemischten Karten —
homogene Rigs haben identische Optimal-Vektoren fuer beide Phasen, dort ist das
Feature ein No-op. Das ist exakt unsere Kern-Zielgruppe (mismatched consumer GPUs).

## Deduktion (Nutzerfrage 2026-07-28): Mixed-Karten + volle Lanes + NVLink zwischen ZWEIEN
Kein NVLink lokal — reine Deduktion ([[rig-ist-untergrenze]]), Validierung gehoert zu #205.
Drei tragfaehige Muster, je mit Gewinn- und exakter Kosten-Dimension:
1. INSEL-VORREDUKTION (hierarchische Kollektive): erst NVLink-intern reduzieren, dann
   EIN Austausch Insel<->Drittkarte via PCIe -> PCIe-Verkehr des all_reduce ~halbiert.
   Gewinn: Prefill (bandbreitengebundene 20-MiB-Chunks) deutlich, Decode-Latenz maessig.
   Kosten: KEINE Kapazitaet, nur Komplexitaet; Deckel: Drittkarte bleibt Taktgeber.
2. INSEL ALS SUPER-RANG (TP-in-TP oder PP-Grenze Insel|Drittkarte): NVLink-Paar als ein
   schneller logischer Rang (interne Kollektive quasi frei), aussen nur noch 2-Wege.
   Gewinn: Decode ms/Runde UND Prefill (weniger teure externe Kollektive) — die Insel-
   Version der heutigen Cross-Rig-PP-Erkenntnis. Kosten: max KV/parallele Sessions auf
   der Insel bei Gewichtskonzentration (gleiche Waehrung wie Phasen-dual); PP-Variante
   zusaetzlich no-spec + Kapazitaet-statt-Tempo.
3. NVLINK ALS DATENWEG: Partner-VRAM als Spill-/Offload-Tier (Experten/KV-Spill mit
   2-4x Host-RAM-Bandbreite ohne CPU), dedizierte Draft-Karte (#114 explizit NVLink-
   gated), Zero-Copy-PD-Handover/Session-Migration. Gewinn: TTFT unter Last, Offload-
   Durchsatz, Migrationszeit. Kosten EXAKT: max KV des Partners (VRAM wird Fremdlager).
Plus: Phasen-dualer MLP-Split gewinnt unter NVLink ueberproportional (Kollektivboden
schrumpft -> Vektor-Schere waechst), Kostendimension unveraendert (Delta-GiB aus KV).
Wizard-Anschluss: NVLink-Paar-Erkennung als Zeile der Faehigkeits-Tafel, Muster 1-3 als
gated Familien (v2).
Ergaenzung (Nutzer, gleiche Diskussion): Karten ggf. in MEHR als 2 teilen fuer
optimale Platzierung — mehrdimensional. Praezisierung/Zielbild:
- Zwei Granularitaets-Achsen: (1) KONTINUIERLICH innerhalb eines Rangs (rank-tp/
  mlp/vocab/kv-Ratios, quasi kostenfrei, GEBAUT; Phasen-dual ergaenzt die WANN-
  Dimension); (2) PROZESS-Sub-Raenge/Rollen pro Karte (Co-Location #82, ueberlappende
  PD-Gruppen, Solo-Draft, Zusatz-Solo) — je Zusatzprozess FIXPOSTEN ~1,5-3 GiB
  (CUDA-Kontext+Graphen+Aktivierungen; 28,5-GiB-Lektion) + NCCL>=2.30-Bedingung.
- Folgerung: Granularitaet OPTIMIEREN, nicht maximieren — Sub-Rang nur wo
  Platzierungsgewinn > Fixposten (Ledger bepreist).
- ZIELBILD Planner/Wizard v2: Karte = Bin mit Kapazitaetsvektor (Compute, Bandbreite,
  VRAM, Link-Nachbarschaft inkl. NVLink-Inseln); Rolle = Item mit Ressourcenvektor +
  Phasenprofil (WAS/WO/WANN); Ratios machen Items stufenlos, Co-Location erlaubt
  Mehrfachbelegung, Ledger bepreist jeden Prozess. Kleines Zuweisungsproblem statt
  Familienwahl — der Nutzer sieht die beste Rollenbelegung seiner Bins inkl. des
  Fixposten-Preises jeder weiteren Teilung.
FALSIFIKATOR-ERGEBNIS (2026-07-28, 10 GPU-min): Schere 5,3-5,7 % — aber KOMPLETT vom
Prefill getragen (P-Vektor 2,5,2, +9,6 Punkte MLP auf die 5090: Prefill +5,3-5,7 %
ueber Rauschboden, Decode-Nachteil 2,50 % = UNTER Rauschboden = 0). Damit ist die
Praemisse des Phasen-dual-Baus (Decode-Verlust vermeiden) bei moderatem Vektor NICHT
belegt — der statische Vektor allein koennte den Gewinn kostenlos liefern. Wichtige
Messfalle dokumentiert: --rank-gpu-id ist torch-CUDA-Ordnung; bei rank_gpu_id=[1,0,2]
sitzt RANG 1 auf der 5090 -> Vektor 2,5,2 (naives 5,2,2 haette eine 3080 konzentriert
und die Messung still entwertet). ENTSCHEIDUNGSWEG: Prefill-Bein 3x je Arm
replizieren (n=1 zu duenn); haelt der Gewinn UND bleibt Decode unsichtbar ->
statische 2,5,2-Empfehlung shippen (Planner/Wizard), Phasen-dual ZURUECKSTELLEN
("nicht bauen solange Decode-Kosten unsichtbar"); zeigt die Replikation doch
Decode-Kosten -> Phasen-dual-Praemisse lebt wieder.
Verallgemeinerung (Nutzer 2026-07-28): JE ZIEL EIN EIGENER Verteilungsschluessel
(Compute/Bandbreite/Hops) — Zweiteilung nach Wechselbarkeit:
- LAUFZEIT-wechselbar: zustandslose Gewichtsschluessel (MLP-Vektoren) — Phasen-dual
  verallgemeinert auf N Vektoren (Delta-Speicherung, Wahl je Batch-Typ/Modus).
- BOOT-strukturell: Pool-Geometrien (KV-Pools, Mamba-Slots = maxkv/session-Ziele) —
  wechselbar nur zwischen BETRIEBSMODI via Hibernate (#89)/Umsharder (Sekunden,
  kein Reboot), je Modus eigener Schluesselsatz.
Neue Ziel-Kandidaten (Briefing Q5/v2): ttft@N (parametrisiert nach erwarteter
Request-Zahl) und session-max via GDN-replicate (Slot-Multiplikation durch
replizierte GDN-Shards; Vorab-Klaerung Scheduler-Session-Affinitaet). Alle Ziele
erscheinen im Wizard als Matrix-Spalten mit Abwaegungszeile.
REPLIKATIONS-ENDVERDIKT (2026-07-28): Schere haelt NICHT — Prefill +3,6 % Median
(3,0-3,7) IM Rauschboden; Decode -10,8 % nur n=1 (Original-Mittel ~2,5 %). Phasen-
dual NICHT gebaut, statischer 2,5,2 NICHT geshippt (Register). Auf DIESEM Rig ist
der MLP-Vektor am Produktionspunkt damit zweifach als flach belegt (#264 extrem
negativ, moderat im Rauschen) — konsistent mit "der echte Hebel ist der
Kollektivboden". Fremd-Hardware-Erwartung (NVLink-Klasse: Boden schrumpft, Schere
waechst) bleibt als Kandidat mit Wiedereroeffnungs-Bedingung bestehen.
KERNSATZ zur PRIO (Nutzer-Klarstellung 2026-07-28): Die Dual-Gruppen-Runtime ist
der GLEICHZEITIGKEITS-Mechanismus des gesamten Zielbilds — ohne sie besetzt das
System pro Boot nur EINEN Punkt im Kontinuum (ein Schluessel, ein Ziel; mehrere
Ziele nur als Kompromissvektor oder Moduswechsel); MIT ihr besetzen mehrere Lanes
gleichzeitig verschiedene Punkte ueber denselben Gewichts-Bytes (PD-Lane =
prefill-optimal, Verband = decode-/kv-optimal, Zusatz-Lane = Session-Kapazitaet).
Erst dadurch wird "in mehrere Dimensionen gleichzeitig optimieren" real statt
sequenziell — der Solver wird vom Berater zum Rollen-Zuweiser. Deshalb Vorrang
vor Solver-Ausbau/Wizard-v2/allen Queue-Posten.
PRIO-Nachtrag (Nutzer): Pflichtumfang der Dual-Gruppen-Runtime ueber die Slices:
- CUDA-GRAPHEN JE LANE: Verband behaelt volle Decode-Graphen; die PD-/Zusatz-Lanes
  bekommen eigene Graph-Sets (Praezedenz: Weightless-Lane-Graph-Capture #133/#136 —
  symmetrische Capture, +385 % damals). Kein eager-Rueckfall als Dauerzustand.
- DRAFTER ALS TP=1 JE KARTE im Strang MITTESTEN (Replicated-Drafters-Kandidat wird
  Lane-Typ "Drafter-Lane"): Bautor bleibt (Gewinn-Dimension ueber Rauschboden,
  ms/Verify je Rang), aber die Messung gehoert in Slice B/C, nicht separat.
- RESSOURCEN-GRUNDSAETZE: (1) kein Compute verschenken — Lanes fuellen Leerlauf der
  Karten (Bin/Rollen-Modell); (2) kein VRAM doppelt, der nicht doppelt sein MUSS,
  um Geschwindigkeit zu kaufen — Duplikation ist immer eine BEWUSSTE, bezifferte
  Entscheidung (Delta im Ledger), nie Nebenwirkung; (3) Geschwindigkeit DURCH
  Verzicht (weniger gleichzeitige Sessions, kleinerer KV) muss als expliziter
  Regler moeglich sein — der Q3-Solo-Trade (2,94x Prefill fuer 4,21x KV) ist das
  Muster. Alles greift in alles: Lane-Zuschnitt, Schluessel, Graphen, Draft-Slots
  sind EIN gemeinsamer Abwaegungsraum, kein Feature-Stapel.
PRIO-Nachtrag 2 (Nutzer, Spreizung + Dispatcher): Verallgemeinerung der Dual-
Gruppen-Runtime — die Lanes muessen nicht fest "PD=Prefill, Verband=Decode" sein,
sondern koennen in UNTERSCHIEDLICHE RICHTUNGEN gespreizt werden (je Lane eigener
Schluessel/eigenes Ziel: Langkontext/max-KV vs Kurz-Request/Latenz, prefill- vs
decode-lastig). NEUES BAUTEIL: der DISPATCHER — Routing eingehender Arbeit nach
Request-Profil (Promptlaenge, erwartete Ausgabe, Latenz-Klasse #242, Praefix-
Wiederkehr) zur passend gespreizten Lane; Profilwechsel mitten in der Session via
Handover/Migration. Gates: (1) Solver/Ledger prueft, ob die Spreizung auf der
Hardware traegt ("genuegend compute/vram/bandbreite/latenz" = Klammer-Rechnung);
(2) Dispatcher-Bautor: Routing muss messbar besser sein als die beste Einzel-Lane
(sonst Komplexitaet ohne Gewinn). Einordnung: Slice-Reihenfolge unveraendert
(A Machbarkeit, B Dual-Lane+Pools, C Aggregat/Interferenz); Spreizung+Dispatcher
= Slice D, erst nach C-Belegen. Motto als Strang-Gesetz: alles greift in alles,
WO ES SINN ERGIBT — Sinn belegt der Falsifikator, nicht die Idee.
PRIO-Nachtrag 3 (Nutzer, Feature-Vollstaendigkeit je Lane): Eine gespreizte Lane
ist KEINE abgespeckte Lane. In JEDER Lane (PD, Verband, Drafter, kuenftige
Spreiz-Richtungen) muessen CUDA-Graphen, MTP/Spec UND alle Features funktionieren,
die es "schnell und/oder gross" machen: adaptive Draft-Laenge, DFLASH/NEXTN-Leiter,
uneven TP, uneven DCP, HiCache (RAM+Disk), Spill/Budget, Expert-Offload,
Suspend/Hibernate, fp8-KV. HARTES ARCHITEKTUR-KRITERIUM: ein Lane-Design, das
eines dieser Features strukturell ausschliesst (nicht bloss "noch nicht gebaut"),
ist verworfen — Ausschluss nur bei harter physikalisch/logischer Grenze, benannt
und begruendet (Regel "alles mit allem kombinierbar"). Praktisch heisst das fuer
die Slices: Feature-x-Lane-Matrix wird in Slice B/C mitgefuehrt (Status je Zelle:
geht / geht noch nicht / physikalisch begruendet ausgeschlossen), Byte-/Bautor-
Gates gelten je Zelle. Bekannte Vorarbeit: Chain-Spec auf der weightless Lane
(#143) und Graph-Capture dort (#133/#136) belegen, dass Spec+Graphen in einer
Zweitlane gehen — der Weg ist frei, nicht nur behauptet.
PRIO-Nachtrag 4 (Nutzer, elastische Lane-Belegung / "spilled offload"-Sicht):
PD<->Verband ist wie ein Spill-Offload zu sehen, der RECHNET statt nur Daten
wegzuschaufeln — die PD-Lane ist ein belegter Slot, der bei Leerlauf zurueckgegeben
wird. Wenn PD fertig ist und der Verband den freigewordenen Platz nutzen koennte,
soll ein SWITCH passieren — falls schnell genug; sonst zuerst MESSEN, ab wann es
sich lohnt (Amortisationsschwelle). Kostenmodell in drei Stufen, weil Bytes geteilt
sind und beim Switch KEINE Gewichte bewegt werden muessen:
  (1) Compute-Zeitanteil der 5090: praktisch kostenlos, sofort — reiner
      Scheduler-Entscheid (Verband bekommt die Slots zurueck). Immer machen.
  (2) VRAM-Rueckgabe (KV-Pool + Graph-Pools + Workspace der PD-Lane an den
      Verband verleihen): messbarer Umschaltpreis — Slice-B-Pool-Design MUSS
      leihbare/rueckforderbare Segmente vorsehen (nicht zwei statisch getrennte
      Pools). Vorarbeit: #102 taggbare Capture-Pools, #93 Aliasing/Remap,
      Suspend-to-RAM-Maschinerie, #140 "phasen-multiplexte VRAM-Slots" (dort
      als Design notiert — hier ist der Ort, wo es real wird).
  (3) Voll-Umspreizen (Geometrie-/Schluesselwechsel einer Lane): teuerste Stufe,
      via Handover-Maschinerie (#261, Round-Trip byte-identisch), lohnt nur bei
      langem Leerlauf/Profilwechsel — Dispatcher-Entscheid (Slice D).
PFLICHT-MESSUNG (Slice B/C): Umschaltzeit je Stufe + Gewinnrate danach =>
Amortisationsschwelle in Sekunden Leerlauf; unterhalb der Schwelle bleibt die
Belegung stehen (kein Flattern) — Hysterese wie beim Cross-Algo-Bandit #156.
PRIO-Nachtrag 5 (Nutzer, Prioritaetsordnung der Belegung): PD ist IMMER
priorisiert, bei Compute UND VRAM. Der Verband "schleicht hinterher" = arbeits-
erhaltender Nachnutzer (Scavenger-Klasse): er nutzt aus, was brach liegt, und
umgekehrt nutzt PD Brachliegendes des Verbands — nichts liegt je ungenutzt
(kein Compute verschenken, Nachtrag 1), aber im KONFLIKT gewinnt PD sofort.
Konsequenzen:
  - Scheduler (Slice B): zwei Klassen, PD=Vordergrund mit Garantie, Verband=
    Nachnutzer. Praeemptionspunkte sind die natuerlichen Korngrenzen: Chunk-
    Grenzen des chunked prefill + Decode-Schritt-Grenzen — der Verband gibt
    die 5090 an diesen Punkten ab, kein Mid-Kernel-Abbruch noetig.
  - VRAM: PDs Budget ist RESERVIERT (nie verliehen ohne Rueckholgarantie).
    Der Verband leiht nur, was schnell raeumbar ist: geliehene Segmente tragen
    ausschliesslich evakuierbare Inhalte (spillbare/retractbare Session-KV via
    #236/#242-Maschinerie, verwerfbarer Scratch) — NIE permanente Posten.
  - SCHLUESSELMETRIK: Rueckhol-Latenz (Verband gibt zurueck, wenn PD-Arbeit
    ankommt) — sie ist die eigentliche Garantiequalitaet der PD-Prioritaet und
    wird in Slice B/C je Stufe gemessen (Gegenstueck zur Amortisationsschwelle
    aus Nachtrag 4: Schwelle sagt WANN leihen, Rueckhol-Latenz sagt was es
    kostet, es zurueckzugeben).
  - Interferenz-Messung (Slice A) wird mit dieser Brille gelesen: die zu
    schuetzende Groesse ist PDs ms/Runde; Verband-Degradation unter PD-Last ist
    erwartet und akzeptabel, solange der Verband Brachliegendes produktiv macht.
PRIO-Nachtrag 6 (Nutzer, sharebarer KV-Pool via Spill-VRAM-Offload): Die KV-Pools
von PD und Verband koennen optional SHAREBAR gemacht werden, indem der Spill-
Offload-Mechanismus (#224 zielwaehlbar, #134 tiered Fabric) ein neues Tier
bekommt: VRAM der jeweils anderen Lane. Bewusst: dann ist nur EINE Lane schnell
(die, deren Layout kanonisch liegt; die andere zahlt Uebersetzung/Transport) —
aber in benannten Situationen gewinnt man. ZU EVALUIEREN (Slice B/C), je Szenario
eigenes Gate:
  (a) Zero-Copy-Handover: PD-Prefill -> Verband-Decode-Uebernahme ohne KV-Kopie
      fuer den 5090-residenten Anteil (Steigerung des 1,65x-Einzeljob-Werts;
      Geometrie-Uebersetzung via Handover-Maschinerie #261 bleibt noetig, aber
      Bytes bleiben liegen).
  (b) Kapazitaets-Pooling bei Leerlauf: idle Lane => andere Lane bekommt fast
      beide KV-Budgets KOPIERFREI (verschaerft Nachtrag-4-Stufe-2: leihen ohne
      umziehen).
  (c) Burst-Absorption: Verband unter Sessiondruck spillt kaelteste Sessions in
      PDs brachliegende Pool-Region statt in Host-RAM — VRAM-zu-VRAM-Restore
      schneller als Host-Restore. MESSPUNKT: Restore-Latenz VRAM-Tier vs
      Host-Tier vs Disk-Tier (erst diese Zahl entscheidet, ob das Tier lohnt).
  (d) Praefix-Dedupe ueber Lanes: gleicher Praefix in beiden Lanes = EINE
      KV-Kopie bedient beide (HiCache-Schluessel traegt Geometrie #241 —
      Uebersetzungsschicht noetig, sonst getrennte Welten).
RANDBEDINGUNGEN: PD-Prioritaet (Nachtrag 5) gilt unveraendert — Inhalte in PDs
Region sind IMMER evakuierbar, Rueckhol-Latenz ist das Gate; Geometrie-Mismatch
(PD tp1-Layout vs Verband-Shard-Layout) ist der eigentliche Preis, kanonisches
Layout + Owner-Rule-Zugriff (#115-Muster) vs Uebersetzung-bei-Umzug sind die zwei
Kandidaten. OPT-IN-Feature ("falls man das moechte"), kein Default. Architektur-
Voraussetzung (ein Allocator/Adressraum) ist bereits Slice-A-Auflage.
PRIO-Nachtrag 6b (Nutzer-Praezisierung): Das Sharing ist KARTENLOKAL — geteilt
wird nur der ANTEIL des Verband-KV-Pools, der physisch auf der/den PD-Karten
liegt (konkret: der 5090-residente KV-Shard des Verbands gegenueber PDs Pool auf
derselben 5090). Konsequenzen: (1) kein Cross-Card-Transport im Spiel — Sharing
ist reine Allocator-/Belegungsfrage im selben Device-VRAM, die Szenarien (a)/(b)
werden echt zero-copy; (2) "nur eine Lane schnell" bezieht sich damit allein auf
die LAYOUT-Kanonizitaet (welches Sharding liegt nativ), nicht auf Transportwege;
(3) die 3080-Anteile des Verband-KV bleiben unberuehrt ausserhalb des geteilten
Bereichs; (4) das VRAM-Spill-Tier aus (c) bedeutet praezise: Verband spillt in
den brachliegenden PD-Poolbereich DERSELBEN Karte — Restore ist ein Device-Copy
oder gar nur ein Ownership-Wechsel im Allocator.
PRIO-Nachtrag 7 (Nutzer, FP8-Kandidat tp2-in-tp3): Zu evaluieren (Solver, Desk):
Qwen3.6-27B-FP8 als Dual-Gruppe — PD-Verbund = uneven DCP TP=2 auf 5090+3080a
(die x8-angebundene; PD-Gruppe INTERN selbst schluessel-optimiert auf Bandbreite/
Compute/VRAM-Groesse), Haupt-Verbund = TP=3 uneven DCP ueber alle drei Karten,
mit REUSE von allem aus dem PD-TP2-Verbund. Nesting-Bedingung: TP3-Shards auf
5090 und 3080a muessen TEILMENGEN der TP2-Shards sein (u_A<=v_A, u_B<=v_B als
Box-Grenzen — genau die Rollen-als-Grenzen-Maschinerie des Solvers); 3080b
traegt das Komplement als einzige neue Bytes => volle Gewichte liegen weiterhin
genau EINMAL im Rig. Engste Karte: 3080a (20 GiB, traegt Fixposten BEIDER Lanes).
Kernfrage "mehrere Richtungen oder nur eine": die Nesting-Kopplung beschneidet
den TP3-Schluesselraum — zu beziffern ist der PREIS der Kopplung (bestes TP3-Ziel
mit vs ohne Nesting-Grenzen) je Zielpaar; erst diese Zahl entscheidet, ob
Mehrfach-Optimierung traegt oder die PD-Richtung dominiert.
PRIO-Nachtrag 8 (Nutzer, Mehrfach-Gruppen-Runtime): Es ist KEINE Dual-, sondern
eine MEHRFACH-Gruppen-Runtime. PD kann je nach Workload unterschiedlich verteilt
sein, Main ebenso, sogar kombiniert mit den PDs — N Lanes, jede mit eigenem
Schluessel/eigener Spreizung, ueber geteilten Bytes wo die Nesting-Algebra es
erlaubt. ANFORDERUNG "immer ein Optimum findbar": (a) KEINE Zweier-Hartcodierung
in Datenstrukturen/Signaturen — lane_id statt "die Lane", Gruppenlisten statt
Paar-Parameter; Slice B baut genau EINE PD-Lane, aber generisch benannt.
(b) Nesting-Algebra von paarweise auf mengenweise verallgemeinern (Huellbaum
ueber N Schluessel; die 65-von-497-Nichtnestbarkeits-Klasse gilt je Paar und
muss ueber die Menge geprueft werden). (c) Der Solver ist der Optimum-Finder:
Lane-Anzahl + Verteilungen + Rollen als Suchraum, Ledger/coresident_budgets als
Machbarkeits-Klammer, Pareto ueber die Zielrichtungen — die Kombinatorik gehoert
in den Solver, nie in Hand-Aufzaehlung. (d) Prioritaetsordnung (Nachtrag 5)
verallgemeinert zu Prioritaets-KLASSEN ueber N Lanes (nicht nur PD>Main);
Scavenging paarweise transitiv. (e) Elastische Belegung (Nachtrag 4) und
kartenlokales KV-Sharing (Nachtrag 6) gelten je Lane-PAAR auf gemeinsamen Karten.
Slice-Zuschnitt unveraendert: B baut die erste Zweitlane (generisch), C die
Nebenlaeufigkeit, D Spreizung+Dispatcher — N>2 ist dann Konfiguration, kein Umbau.
PRIO-Nachtrag 9 (Nutzer, Modellfamilien-Generalitaet): Die Mehrfach-Gruppen-
Runtime (Split-PD, Lanes, geteilte Bytes) muss fuer NORMALE Modelle OHNE GDN
(dense Full-Attention, Llama-Klasse) und fuer MoE-Modelle funktionieren —
nicht nur fuer das Qwen-Hybrid-Vehikel. Einordnung nach heutigem Stand:
(a) DENSE ohne GDN: strukturell einfacher (kein Mamba-Posten), aber UNGETESTET
    — und #196 (weightless auf dense Llama still falsch) verbietet den
    Analogieschluss. Braucht einen eigenen Test-Arm (Llama-8B-Vehikel liegt
    lokal, TP=2/3-Rezepte existieren).
(b) MoE: ECHTER BAU-Posten — die Slice-B-Schalen-Taxonomie (465 Schalen:
    column/row/embed/lm_head/conv) kennt KEINE Experten-Tensoren; Komplement-
    Loader muss Expert-Schalen lernen (Vorarbeit: Expert-Dim-Sharding #80,
    moe_wna16 #83); GGUF-MoE zusaetzlich mit der Materialisierungs-Falle
    (#123-Familie, Guard #268 faengt heute) und der Frage Expert-Offload x
    Lane-Komplement auf derselben Karte.
REGELN: gilt unter "alles mit allem kombinierbar" — Ausschluss nur bei harter
Grenze, benannt; die Feature-x-Lane-Matrix bekommt eine MODELLFAMILIEN-Achse
(hybrid-GDN / dense / MoE / MoE-GGUF je Zelle: geht / geht-noch-nicht /
begruendet-ausgeschlossen). Slice-Zuschnitt: aktueller Lane-NEXTN-Bau bleibt
auf dem Q3-Vehikel; dense-Arm + MoE-Schalen = eigener Folge-Slice (mit dem
FP8-Zweikarten-Arm buendeln — beides sind Loader-/Schalen-Erweiterungen).
Nachtrag-9-Ergaenzung (Nutzer, GGUF-Ebenen): GGUF ist im Kern der Runtime die
BESTGETESTETE Familie (das gesamte A/B/C-Vehikel ist 27B-Q3-GGUF). Fehlend
sind exakt zwei Ebenen, beide seit 2026-07-28 mit lauten Guards versehen:
(1) GGUF-MoE — Doppel-Luecke: keine Load-time-Offload-Haelfte (#123,
Materialisierung im Postprocess) UND keine Experten-Schalen im Komplement-
Loader; Guard #268 verhindert stilles Fehlverhalten. => MoE-GGUF-Zelle der
Familien-Achse, Folge-Slice.
(2) GGUF auf sm75 — Kernel-Floor sm_80, Guard #269 (frueh, vor Gewichts-Load).
Konsequenz fuer Lane-Planung: jede Lane/Satellit auf der 2080 Ti braucht
nicht-GGUF-Quant; die #214-Gate-Tabelle prueft das bereits (arch_gguf_sm75).

PRIO-Nachtrag 10 (Analyse auf Nutzerfrage, 2026-07-28): DUAL-LANE AUF N
GLEICHEN KARTEN vs. NORMALES TP-N — wann sind Lanes schneller, insbesondere
bei langsamem Interconnect?
(1) Kollektiv-Posten: TP4 zahlt pro Layer 2 All-Reduces ueber 4 Raenge; zwei
    TP2-Lanes zahlen nur 2-Rang-Kollektive und NULL Verkehr zwischen den
    Paaren. Posten waechst mit Interconnect-Langsamkeit (PCIe-PHB: typ.
    15-25 % Schrittzeit; Ethernet/Cross-Rig: dominant). Zusaetzlich
    Straggler-Halbierung: 2-Rang-Sync statt 4-Rang-Sync pro Kollektiv —
    zaehlt besonders bei Spec-Decode (viele kleine Verify-Kollektive).
(2) Ehrliche Gegenrechnung (Batching amortisiert Gewichts-Lesen):
    Einzelstream: TP2 verliert die halbe Aggregat-Bandbreite; gewinnt nur
    wenn Komm-Ersparnis > verdoppelte Gewichts-Lesezeit (NVLink: nie;
    PCIe: selten; Ethernet: ja). Zwei Streams — Kreuzungsbedingung:
      t_comm(TP4) - t_comm(TP2) > t_mem(Viertel-Shard)
    d.h. zwei nebenlaeufige TP2-Lanes schlagen TP4@bs=2 erst, wenn die
    4-Ring-Mehrkosten die Viertel-Shard-Lesezeit uebersteigen. NVLink nein,
    PHB knapp, Cross-Rig/NORDSTERN klar ja.
(3) Staerkster Gewinn = heterogene WORKLOADS (Slice D), nicht heterogene
    Karten: Prefill saettigt SM, Decode saettigt Bandbreite; unter TP4 mit
    chunked-prefill stoert jeder Chunk die Decode-Latenz aller Requests.
    Gespreizte Lanes (Prefill-Paar / Decode-Paar) entkoppeln das. Empirie
    vom Hetero-Rig uebertragbar: komplementaere Engpaesse -> 1.13/1.44
    Kartenaequivalente; einziger Stoerfall dir1 = zwei SM-saettigende
    Lasten gepaart. Auf gleichen Karten faellt die Hetero-Komplikation weg.
(4) PREIS der Byte-Teilung: Nist-Richtung ist fein-in-grob (TP4-Viertel ist
    Teilmenge der TP2-Haelfte, nie umgekehrt). Fuer TP4-Lane + TP2-Lanes
    ueber EINEM Byte-Satz muss die Ablage als 2x TP2-Haelften erfolgen:
    pro Karte halbes statt viertel Modell = ein Viertel Modellgroesse
    weniger KV-Raum. Das ist der Dispatcher-Trade Weight-Bytes gegen
    Konfigurationsfreiheit; auf VRAM-satten Rigs kaufbar, auf knappen nicht.
KONSEQUENZ: Dual/Mehrfach-Lane ist auch auf HOMOGENEN Rigs kein Hetero-
Sonderfall, sondern ein eigenstaendiger Perf-Hebel — Normal-Rig-Nutzen
(1/2/4/8 gleiche GPUs) fuer FEATURES_VS_UPSTREAM-Einordnung: hoch bei
langsamem Interconnect (kein NVLink) und bei PD-Mischlast; neutral bis
negativ bei NVLink + reiner Einzelstream-Decode-Last.

Nachtrag-10-STATUS (Nutzer-Einwand, 2026-07-28): UNBESTAETIGTE HYPOTHESE.
Punkte (1)-(2) sind Papier-Rechnung, Punkt (3) ist ein Transfer von Hetero-
Messungen auf homogene Karten — beides ungemessen. Vor jeder Verwendung in
FEATURES_VS_UPSTREAM/README als "Normal-Rig-Nutzen" gilt Messpflicht.
FALSIFIKATOR (auf diesem Rig moeglich, ohne 4 gleiche Karten):
(a) Kollektiv-Achse direkt messen: t_comm pro All-Reduce bei 2 vs 3 Raengen
    (NCCL-Floor-Methodik aus #199 wiederverwenden) + t_mem eines
    Viertel-Shard-Reads -> Zahlen in die Kreuzungsbedingung einsetzen.
    Kein Modell-Boot noetig, billig.
(b) Homogenes Paar real: die beiden 3080 sind identisch. TP2-Verband auf
    beiden vs zwei TP1-Lanes (eine je 3080) mit 2 Streams, gleiches Modell,
    ms/Decode-Schritt je Rang (CUDA-Events, versetzt gelesen), A-vs-A-
    Rauschboden zuerst. Testet die Paar-Split-Logik an der kleinsten Stufe
    N=2 -> N=1x2; Ergebnis skaliert NICHT automatisch auf 4x — nur die
    Kreuzungsbedingung selbst wird bestaetigt oder verworfen.
(c) Spreizungs-Punkt (3) homogen: Prefill-Lane auf 3080-A, Decode-Lane auf
    3080-B vs beide Lasten gemischt auf TP2 — prefill_wait_ms/Decode-Latenz
    als Metrik (dir1-Methodik).
Einreihung: NACH Lane-Spec-Runde 3 und GDR-Fenster; (a) ist GPU-leicht und
kann in ein bestehendes Fenster huckepack.

PRIO-Nachtrag 11 (Nutzer 2026-07-28, Pfad-Dispatcher x Mehrfach-Gruppen):
Der groessen-/lastbewusste Comm-Pfad-Dispatcher (Task #279, Folge aus der
GDR-Matrix #278) MUSS in der Mehrfach-Gruppen-Runtime funktionieren und mit
allem Bestand kompatibel sein (Nutzer-Wort). Architektur-Konsequenzen:
(1) Dispatcher-Zustand lane-keyed, nie prozessweit (GraphSharedOutput-Lehre
    der Geteilte-Puffer-Familie).
(2) Das Saettigungssignal ist GRUPPENWEIT: mehrere Lanes teilen dieselbe
    NIC und denselben RAM-Pfad — der Dispatcher aggregiert die Queue-Tiefen
    aller Lanes und wird damit der gemeinsame Comm-Arbiter der Gruppen.
    Das zahlt direkt auf die Slice-D-Zielfunktion ein (saettigende Lasten
    nicht paaren): Slice D entscheidet WO Lasten laufen, der Dispatcher
    entscheidet WORUEBER sie kommunizieren — gleiche Eingangsdaten
    (Paar-Matrix, Umschlagpunkte), zwei Entscheidungsebenen.
(3) Ueberlauf respektiert die Prioritaetsklassen aus Nachtrag 5 (PD-Verkehr
    vor Main-Verkehr, Summen-Invarianz bleibt Testkriterium).
(4) Graph-Safety als hartes Tor: Pfadwahl capture-stabil oder graph-sichere
    Indirektion — Nachtrag-3-Regel (Lanes behalten Graphs+MTP) gilt.
Eingangsdaten kommen aus #278 (Umschlagpunkte je Pfad x Richtung x Karte,
HOL-p99-Faktor aus dem Mix-Szenario, V5-Serialisierung); Baustart erst nach
deren Bericht.

PRIO-Nachtrag 12 (Analyse auf Nutzerfrage, 2026-07-28): DYNAMISCHES DUAL —
Lanes bekommen ihre Ressourcennutzung nicht einmalig beim Start, sondern ein
Regler verschiebt sie lastabhaengig zur Laufzeit. Schreibtisch-Analyse, kein
Bau; alle Zahlen aus Slice C (DESIGN_121 §11) und dem GDR-/Hibernate-Bestand.

(1) ABGRENZUNG GEGEN NACHTRAG 4/5 — was wirklich neu ist.
Nachtrag 4 regelt VRAM, getriggert von LEERLAUF, in drei diskreten Stufen,
mit Amortisationsschwelle und Hysterese. Nachtrag 5 ordnet den KONFLIKT
(PD gewinnt), das ist eine statische Prioritaet, kein Regler. Die Nutzer-Idee
faellt in drei Teile, und nur zwei davon sind neu:
  (a) VRAM-Achse: dasselbe wie Nachtrag 4, andere Worte. Kein neuer Inhalt.
  (b) TRIGGER-ACHSE: echte Verschaerfung. Nachtrag 4 leiht, wenn eine Lane
      IDLE ist; die Idee will umverteilen, wenn eine Lane BESCHAEFTIGT, aber
      marginal weniger wert ist. Aus "Leerlauf-Schwelle in Sekunden" wird
      "marginaler Beitrag zum Aggregat" — Nachtrag-4-Leerlauf ist dann der
      Sonderfall Grenzbeitrag=0. Das ist eine echte Verallgemeinerung.
  (c) RESSOURCEN-ACHSE (Compute und Bandbreite GETRENNT regelbar): als
      Hardware-Zuteilung existiert das auf diesem Rig NICHT — siehe (2).
      Als WORKLOAD-Komposition existiert es, und dann ist es Slice D.
Ehrliches Fazit zu Frage 1: (b) ist neu, (c) ist neu nur in der Lesart
"Lastmischung statt Ressourcenzuteilung", (a) ist Bestand.

(2) STELLHEBEL, ehrlich bewertet. Die Lanes sind seit C1/C2 THREADS EINES
PROZESSES mit eigenem Stream — das entscheidet die halbe Tabelle.
| Hebel | verfuegbar? | taugt zur Compute/Bandbreiten-Trennung? |
|---|---|---|
| Stream-Prioritaet (C2, -3 aus [0,-3]) | ja, gebaut | nein — ordnet nur die Reihenfolge freiwerdender Bloecke, teilt nichts zu |
| Admission-Yield / Duty an der Korngrenze (C2) | ja, gebaut | nein, aber wirksam: verschiebt ZEITanteil, byte-neutral |
| MPS `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` | ja, aber PROZESS-gebunden (Client liest es bei Kontext-Erzeugung) | UNBRAUCHBAR fuer Ein-Prozess-Lanes; erzwaenge Rueckbau auf Prozess-Lanes und damit Verlust der geteilten Bytes — der Kern des ganzen Strangs |
| Green Contexts / SM-Masken (libcuda) | Symbole im Treiber vorhanden (`cuGreenCtxCreate`, `cuDevSmResourceSplitByCount`, `cuCtxFromGreenCtx`, `cuGreenCtxStreamCreate`; Treiber-CUDA 13.2), GeForce-Tauglichkeit + Granularitaet UNGEPRUEFT | einziger echter SM-Zuteiler in EINEM Prozess — aber siehe Graph-Falle in (4) |
| Verleih-Primitive Stufe 2 (C2) | ja, gebaut, 0,76/2,49 ms | regelt VRAM, nicht Compute |
| Chunk-/Batch-Groessen-Regler (Scheduler) | billig, teils vorhanden (`--dual-group-lane-speed-dial`) | INDIREKT der staerkste Hebel: Batchgroesse ist der arithmetische-Intensitaets-Regler (Decode bandbreiten- -> rechengebunden), Chunkgroesse zerlegt die SM-saettigende Prefill-Spitze. Preis: nicht byte-neutral, siehe (4) |
KERNBEFUND: BANDBREITE IST AUF DIESER HARDWARE NICHT ZUTEILBAR. Es gibt kein
Speicher-QoS auf GeForce (MIG existiert dort nicht, und weder sm86 noch sm120
bieten Memory-Controller-Partitionierung). Die Bandbreiten-"Zuteilung" kann
ausschliesslich indirekt entstehen, indem man WAEHLT, welche Lastform wo
laeuft — Prefill saettigt SM, Decode saettigt Bandbreite (Nachtrag 10 (3),
gemessen als dir1 +9,7 % gegen Decode-Arm E 1,440). Damit ist die
interessanteste Lesart der Nutzer-Idee nicht als Ressourcen-Regler baubar,
sondern nur als LASTMISCHUNGS-Regler. Das ist keine Abwertung: der
Decode-Arm-Gewinn (+57,5 % gegen +16,0 %) ist genau der Betrag, den die
richtige Mischung schon heute kauft.

(2b) GEWICHTS-REDUNDANZ-BUDGET JE KARTE (Nutzer-Ergaenzung) — der Vorab-
Parameter, der den AKTIONSRAUM des Reglers festlegt: wieviel Modell darf eine
Karte ZUSAETZLICH zum notwendigen Minimum halten (bis zu einem ganzen Modell).
(a) STUFEN, aus der Nesting-Algebra statt aus dem Bauchgefuehl. Eine
    Layout-Menge ist ihre Schnittmenge (`partition_cuts` liefert das Cut-Set;
    Verfeinerung = Obermenge der Cuts). Daraus drei Korrekturen am
    Bauchgefuehl:
      - Budget 0 ist NICHT "ein Layout". Wer das FEINSTE je gewuenschte
        Cut-Set haelt, bekommt jede GROEBERE Aufteilung geschenkt (jeder
        grobe Shard ist eine Vereinigung feiner Shards) — der gesamte
        Down-Set der gehaltenen Cuts ist gratis erreichbar.
      - Bezahlt wird nur UNVERGLEICHBARKEIT: zwei Layouts, deren Cuts an
        verschiedenen Stellen liegen (das `nesting_hull`-Beispiel [6,2] gegen
        [7,1] auf [6,1,1] — beide nisten in der Gruppe, in EINANDER bei
        keinem Unit-Count). Und auch dann nicht ein zweites ganzes Modell:
        der Preis ist exakt
          delta = SUMME ueber Karten von ( |A_range VEREINIGT B_range| - |A_range| )
        also nur die Shards, die ueber die abweichende Schnittkante ragen.
        Gehalten wird die gemeinsame Verfeinerung (Vereinigung der Cut-Sets),
        dann sind BEIDE Layouts Vereinigungen gehaltener Shards.
      - Erst die Voll-Freiheit kostet voll: 1,0 Modell je Karte = jede Karte
        kann jede Rolle solo (DP-artig).
    Sinnvolle diskrete Leiter: R0 = feinstes Cut-Set (Down-Set gratis);
    R1 = R0 + straddle-delta fuer eine BENANNTE zweite Layout-Familie
    (typisch klein, weil nur Randshards); R2 = Nachtrag-10-Fall (TP2-Haelfte
    je Karte mit genisteten TP4-Vierteln = 2,0x gegen das TP4-Minimum, dort
    bereits als "ein Viertel Modellgroesse weniger KV-Raum" beziffert);
    R3 = 1,0 Modell je Karte. Kontinuierlich ist der Trade nur formal — real
    springt er an den Cut-Sets, und die Leiter IST die zulaessige Menge.
(b) UMSCHALTKOSTEN je Stufe:
      - Bytes gehalten, Layout im Down-Set: Plan-Flip (`_TP_PARTITION_RATIOS`
        + `segments`, seit C1 kontext-lokal) = Mikrosekunden. Der Preis ist
        nicht Zeit, sondern GRAPH-SPEICHER: jedes erreichbare Layout braucht
        sein eigenes Capture-Set und seinen eigenen Pool (C1 gibt Pools je
        Lane). Umschaltfreiheit kostet also zweimal VRAM — Gewichte plus
        Graph-Pools.
      - Bytes NICHT gehalten: Nachladen aus RAM/Disk. Groessenordnung aus
        Hibernate #89: 8-14 s fuer uneven TP=3 (von 50 s). Das sind drei bis
        vier Groessenordnungen ueber dem Verleih-Zyklus (3,25 ms) und fuenf
        ueber einem Plan-Flip. Amortisation bei ~10 s Umschaltpreis und
        plausiblen 20-50 % Gewinn verlangt MINUTEN stabiles Regime.
      HARTE FOLGERUNG: der Regler darf ausschliesslich innerhalb der
      gehaltenen Byte-Menge arbeiten. Alles, was Nachladen verlangt, ist
      Nachtrag-4-Stufe-3 (Handover #261) und gehoert nicht in die
      Regelschleife, sondern in den Moduswechsel.
(c) SOLVER-ANBINDUNG, damit es rechenbar statt gefuehlt ist: das Budget geht
    zweimal ein. Als BOX-GRENZE ueber `coresident_budgets` — die
    Redundanz-Bytes sind vorab reserviert (`reserve_mib` bzw. reduziertes
    `effective_vram_mib` in `PlanInputs`, genau der Parameter, den
    `_capacity_under` schon variiert), womit die KV-Kapazitaet automatisch
    gegen die Umschaltfreiheit rechnet. Und als AKTIONSRAUM ueber
    `nesting_hull(lanes, probes)`: die Kandidaten-Layouts werden als
    Lane-Keys eingespeist; was nistet, ist gratis erreichbar, was nicht
    nistet, bekommt das delta aus (a) aufgeschlagen oder faellt raus. Damit
    wird die Solver-Frage: maximiere E ueber (Lane-Menge, Redundanz-Budget)
    unter der `coresident_budgets`-Klammer. Das ist dieselbe Pareto-Front wie
    in Nachtrag 8 (c), nur mit einer zusaetzlichen Achse.

(3) REGLER-SKIZZE.
SIGNAL: `prefill_wait_ms` ist als Sensor untauglich und der Grund ist bekannt
— gemessen 0,01 ms, waehrend die Device-Zeit von 583 auf 627-638 ms stieg.
Es misst Einreihung, die Konkurrenz sitzt aber in der Rechenzeit. Das
richtige Signal steht schon in der C3-Formel: der ONLINE gemessene
Degradationsanteil
    share_c = Rate_c(gemeinsames Fenster) / Rate_c(solo-Boden)
je Lane, aus CUDA-Events auf dem jeweiligen Lane-Stream gegen die beim Boot
erhobenen Boeden. Damit ist das Sensorsignal IDENTISCH mit der
Slice-D-Zielfunktion E = SUMME share_c — der Regler misst genau das, was er
maximieren soll, und braucht keine Profiler-Zaehler.
UNTERSCHEIDUNG SM- gegen BANDBREITEN-KONKURRENZ ohne Hardware-Zaehler
(DCGM-Prof-Metriken sind auf GeForce nicht verlaesslich): die Lastform wird
ANALYTISCH etikettiert statt gemessen — der Kostenmodell-Pfad
(`build_cost_model` / `uneven_perf`) kennt aus der Batch-Form FLOPs und
gelesene Bytes und damit die arithmetische Intensitaet des Schritts. Regel:
degradieren zwei Schritte hoher Intensitaet gemeinsam -> SM-Konkurrenz
(dir1-Fall); degradieren zwei Schritte niedriger Intensitaet gemeinsam ->
Bandbreite. Der Regler braucht also Etikett (analytisch) x Degradation
(gemessen), beides vorhanden.
STELLGROESSE + ZEITKONSTANTEN, zwei Schleifen mit verschiedenen Takten:
  - SCHNELL (byte-neutral, gratis): Admission-Duty und Stream-Prioritaet an
    der Korngrenze. Kosten praktisch null (64 Yields/32 s = 0,35 % des
    Fensters, Max 2,17 ms), Takt bis Iterationsgrenze (~33,8 ms Verify) =
    bis ~30 Hz. Messfenster muss aber ueber dem Rauschboden liegen
    (A-vs-A-Spannweiten 0,25-0,39 %), also EMA ueber ~1 s.
  - LANGSAM (kostet, nicht byte-neutral): VRAM-Verleih (Zyklus 3,25 ms,
    Amortisation ~0,1 s -> hoechstens ~10 Hz sinnvoll, praktisch ~1 Hz) und
    Chunk-/Batch-Stufen.
HYSTERESE: wie bei #156 und wie die bestehende 5-s-Leerlaufschwelle —
Umschalten erst, wenn der geschaetzte E-Gewinn ueber mehrere Fenster stabil
groesser ist als der Rauschboden PLUS Umschaltkosten; Flaps zaehlen, nicht
verweigern (der bestehende `refused_min_hold`-Vertrag bleibt).

(4) GEFAHREN, konkret.
  - GRAPH-KOMPATIBILITAET ist die harte Grenze (Nachtrag-3-Regel). Green
    Contexts sind der einzige echte SM-Zuteiler, aber ein Graph ist an den
    Kontext gebunden, in dem er gecaptured wurde: eine SM-Aufteilung zur
    Laufzeit AENDERN hiesse re-capturen — verboten. Der einzige zulaessige
    Bau ist eine VORAB gecapturete Leiter weniger fester Splits, deren Preis
    Graph-Pools mal Sprossenzahl ist. Damit ist auch die Frage "kontinuier-
    licher Regler?" entschieden: fuer alle Form- und Kontext-Stellgroessen
    ist der Aktionsraum DISKRET und wird von der Capture-Leiter definiert;
    kontinuierlich sind nur die schwachen Zeitanteil-Hebel.
  - DETERMINISMUS: Prioritaet und Duty sind byte-neutral (sie aendern WANN
    Kernel laufen, nicht welche Formen). Chunk- und Batchgroesse sind es
    NICHT — andere Kachelung, andere Reduktionsreihenfolge. Daraus die
    Regel: Form-Stellgroessen duerfen nur an REQUEST-Grenzen wirken,
    Zeitanteil-Stellgroessen an jeder Korngrenze. Sonst ist die eigene
    Byte-Gate-Methodik nicht mehr anwendbar.
  - MESSBARKEIT: E ist nur je REGLER-ZUSTAND definiert. Jedes Fenster muss
    die Sprossen-Id mitfuehren; ein E ueber ein Fenster mit Sprossenwechsel
    ist unauswertbar und darf nicht berichtet werden. Das ist Buchfuehrung,
    kein Bau, aber es muss vor der ersten Messung stehen.
  - RUECKKOPPLUNG AUF SICH SELBST: der Regler aendert die Last, an der er
    misst (Selbstkonditionierungs-Falle aus #156). Die Solo-Boeden muessen
    deshalb aus einem regler-freien Boot stammen und duerfen nicht im
    laufenden Betrieb nachgezogen werden.

(5) VERDIKT.
JA, lohnender naechster Schritt — aber NEU GERAHMT: es ist kein neuer
Ressourcen-Zuteilungs-Mechanismus, sondern SLICE D MIT REGELSCHLEIFE. Der
Dispatcher steigt vom offenen Router zum geschlossenen Regler auf, E ist
seine gemessene Zielgroesse, und Nachtrag-4-Leerlauf wird Sonderfall des
Grenzbeitrag-Triggers. Slice D wird also weder ersetzt noch ergaenzt — er
bekommt seine Zielfunktion und seinen Aktionsraum praezisiert, plus die
Budget-Achse aus (2b).
KLEINSTER FALSIFIZIERBARER SLICE, in dieser Reihenfolge, jeweils Abbruch bei
Rot:
  S1 (Buchfuehrung, kein Mechanismus, GPU-leicht): E ONLINE aus den bereits
     vorhandenen CUDA-Event-Zeiten und den Boot-Boeden je 1-s-Fenster
     schaetzen und protokollieren. FALSIFIKATOR: wenn die Online-Schaetzung
     den Rauschboden (0,25-0,39 %) nicht unterbietet, ist KEIN Regler
     baubar, egal welche Stellhebel es gibt. Billigster Test, kommt zuerst.
  S2 (20-Zeilen-ctypes-Probe, ein freies Fenster): Green Context mit
     SM-Split auf sm86 UND sm120 erzeugbar? Welche Granularitaet? Und:
     ueberlebt ein in einem Green-Context-Stream gecaptureter Graph einen
     Wechsel der Aufteilung? Entscheidet, ob es ueberhaupt einen SM-Hebel
     gibt oder ob (2c) endgueltig auf Lastmischung zusammenfaellt.
  S3 (A/B auf dem bestehenden Slice-C-Stand, kein neuer Code): zwei feste
     Sprossen der schon gebauten Hebel (Duty/Speed-Dial/Prioritaet) gegen
     die Festeinstellung, gemessen an der SCHLECHTEN Paarung
     (Prefill x Prefill, dir1). FALSIFIKATOR: schlaegt keine Sprosse die
     Festeinstellung um mehr als den Rauschboden, gibt es nichts zu regeln —
     dann bleibt es beim statischen Dispatcher-Entscheid.
S1 und S3 brauchen keinen neuen Mechanismus, nur Messung; erst danach
lohnt Bau. Einreihung: nach der Lane-Spec-Kette, vor Slice-D-Bau.
