# DESIGN #121 — Dual-Gruppen-Runtime (Variant C), Slice A

Basis: `origin/integration/r3-probe-next2` (da80a9b80f), Branch
`feat/dual-group-runtime-a`, Worktree `/spinning/wt-dualgroup`. Stand 2026-07-28.

Vorgaenger-Dokumente: DESIGN_107 (PD-Topologie, insbesondere Nachtrag 2
"ueberlappende Gruppen — genestete Ratios"), DESIGN_201 (Bin/Rollen-Zielbild).

---

## 0. Was Slice A behauptet und was es belegt

Behauptung: EIN Prozess kann zwei ueberlappende TP-Gruppen ueber EINEM
Gewichtssatz fahren. Die 5090 haelt die vollen Gewichte GENAU EINMAL; diese
Bytes dienen gleichzeitig (a) dem Rang 0 des grossen Verbands (TP=3 uneven)
und (b) einer eigenstaendigen PD-Lane, die allein auf der 5090 prefillen kann.

Slice A belegt davon:

1. den Architektur-Entscheid (§1, Schreibtisch-Falsifikator),
2. die Nesting-Algebra samt Ablehnungen (§3, CPU-getestet),
3. die lokalen Kollektiv-Ersatzoperationen (§4, CPU-getestet, exakt gegen
   die TP-Semantik, NICHT gegen das Monolith-GEMM — Begruendung in §4.3),
4. den VRAM-Posten-Nachweis je Posten geteilt/genestet/dupliziert (§5),
5. den Graph-Weg beider Lanes (§6).

Was Slice A NICHT belegt, steht in §8 (Aufwand B/C) und im Rueckgabe-Bericht.

---

## 1. Falsifikator: (a) In-Prozess-Zweitlane vs (b) Zweitprozess-Co-Location

### 1.1 Die Entscheidungsfrage

Die zweite Gruppe (FAST/PD) braucht auf der geteilten Karte Zugriff auf
Gewichtsbytes, die der Verbandsrang dort schon haelt. Zwei Wege:

* **(a)** FAST laeuft als zweite Spur IM Prozess des Verbandsrangs. Beide
  Spuren sind derselbe CUDA-Kontext, dieselben Tensor-Objekte. Die
  gruppeninternen "Kollektive" der FAST-Gruppe sind dann keine
  Netzwerkoperationen, sondern lokale Tensor-Ops (`cat` statt all-gather,
  `add` statt all-reduce), weil beide Shards im selben Adressraum liegen.
* **(b)** FAST laeuft als zweiter Prozess auf derselben physischen Karte.
  Gewichtsteilung dann nur ueber CUDA-IPC; die Gruppe braucht echte
  NCCL-Kollektive.

### 1.2 Entscheid: (a). Fuenf Belege, jeder einzeln hinreichend.

**Beleg 1 — (b) ist auf diesem Host nicht bootbar, (a) schon.**
Zwei Raenge auf einer physischen GPU brauchen Laufzeit-NCCL >= 2.30. Gemessen
auf diesem Rig: 2.28.9 (`rigmon/capabilities.py`, `probe_nccl_colocation`),
kein MPS-Daemon (`/tmp/nvidia-mps` fehlt). (b) waere nur im Container
(`docker/htsglang.Dockerfile`, NCCL 2.30.7 gepinnt) validierbar. (a) ist EIN
NCCL-Rang je Karte — die Schwelle greift gar nicht.

**Beleg 2 — der CUDA-IPC-Weg fuer Gewichtsteilung ist bereits verworfen.**
DESIGN_107 §"Kosten Weg 1": kein einzelner Allokationspunkt (~10
`load_model`-Varianten, Postprocess schreibt Tensoren um), tausende
Einzel-Handles je quantisiertem Layout. Zusaetzlich: die Isolationseigenschaft,
die (b) erhalten soll, ist unter Gewichtsteilung ohnehin weg — geteilter
Speicher ist geteiltes Schicksal.

**Beleg 3 — die gruppeninternen Kollektive verschwinden bei (a) ersatzlos.**
Das ist der eigentliche strukturelle Gewinn, nicht nur eine Ersparnis. Eine
FAST-Gruppe mit zwei Shards, beide lokal, ersetzt
  all-gather -> `torch.cat`
  all-reduce -> `a + b`
Damit hat die FAST-Gruppe *keinen* Kommunikator, kann also auch nicht mit den
Kommunikatoren des Verbands kollidieren. Der dokumentierte S4b-Hang
(gleichzeitige Kollektive auf einem Kommunikator, `congruent_lane.py`
Modul-Docstring) existiert fuer diese Lane nicht. Bei (b) existiert er, und
zusaetzlich die Hang-Familie "Raenge in verschiedenen Branches".

**Beleg 4 — der Prozess-Fixkostenposten faellt weg.** Ein zweiter Prozess auf
der Karte kostet einen eigenen CUDA-Kontext (~0.5–1 GiB je Karte) plus einen
eigenen Allocator-Pool. Auf der 5090 ist genau dieser Betrag die Reserve, aus
der der Lane-KV bezahlt werden soll.

**Beleg 5 — die Bausteine fuer (a) existieren und sind belegt.** Kein
Neubau, sondern zweite Nutzung:
* `runtime_context.py:114` `ParallelContext.override(...)` — Konstruktionszeit-
  Override der TP-Geometrie. Genutzt von der Weightless-Lane und vom
  Draft-Solo-Host (`model_runner.py:1941-1966`): ein Rang baut sein Modul
  unter `tp_size=1, tp_rank=0`, obwohl die Prozessgruppe TP=3 ist. Linears
  cachen `tp_size` bei der Konstruktion (`linear.py:406`, `:1861`), der
  Override haelt also durch den Forward.
* `model_runner.py:1748` `_build_weightless_worker_meta_model` — ein
  vollstaendiger Modulbaum OHNE Gewichte auf `meta`. Genau das Vehikel fuer
  den Lane-Huellbaum (§4.2).
* `parallel_state.py:2792` `patch_tensor_parallel_group` und die zwei
  bestehenden Zweit-Kommunikator-Praezedenzen `_DCP_SPILL` (`:2014`,
  `set_dcp_spill_active` `:2070`) und `_PDMUX_PREFILL_TP_GROUP` (`:2018`).
* `congruent_lane.py` — Kadenz-Gate + Gewichtsteilungs-Invariante per
  `data_ptr()`-Vergleich, direkt wiederverwendbar.

### 1.3 Graph-Faehigkeit beider Lanes (Nutzer-Pflichtkriterium)

Der Entscheid muss die CUDA-Graph-Faehigkeit BEIDER Lanes tragen; eager als
Dauerzustand ist keine Endform.

**(a) traegt Graphen fuer beide Lanes, und dafuer gibt es Praezedenz.** Die
Weightless-Lane (#133/#136) hat In-Prozess-Zweitlane-Graph-Capture bewiesen:
`decode_cuda_graph_runner.py:1392 _capture_one_shape_weightless` captured ein
ZWEITES Graph-Set fuer die Zweitlane im selben Prozess, symmetrisch ueber die
Raenge. Die drei Bedingungen, die dort erfuellt sein mussten, sind fuer die
FAST-Lane leichter, nicht schwerer:
1. *Statische Adressen*: Die FAST-Lane hat einen eigenen, festen KV-Pool und
   feste Eingabepuffer — dieselbe Konstruktion wie beim Verband.
2. *Keine Kollektive im Graph*: Die Weightless-Capture musste den
   dcp-Dispatch im Graph halten (Kollektiv im Graph, der schwierige Fall).
   Die FAST-Lane hat gar keine Kollektive (Beleg 3) — ihr Graph ist rein
   lokal. Das ist der EINFACHERE Fall.
3. *Symmetrische Capture ueber die Raenge*: entfaellt, weil die FAST-Lane nur
   auf einem Rang existiert und keinen Kommunikator anfasst. Die Capture ist
   damit eine rein rang-lokale Operation. (Achtung: die *Auszeit* der
   Capture ist rang-lokal sichtbar — der Verband darf waehrenddessen kein
   Kollektiv erwarten. Loesung wie beim Spill-Tick: Capture in der
   Boot-Phase, vor dem ersten Verbands-Forward, nicht im Betrieb.)

Fuer Punkt 3 existiert die Ausnahmefahne bereits und ist gegen zwei
GEMESSENE Haenger gebaut: `spec_solo_rank_local_graphs` haengt
`_harmonize_cuda_graph_plan`s `all_gather_object` (`model_runner.py:1207`,
dokumentierter 8-Minuten-Haenger bei TP=5) und
`enter_capture_group_barrier`s `tp_group.barrier()`
(`runner/base_runner.py:91-93`, #194) fuer eine rang-lokale Capture aus. Die
FAST-Lane braucht genau diese Fahne, nicht eine neue Mechanik.

ZWEI Graph-Grenzen bleiben und gehoeren in Slice C, nicht A: `is_capture_mode`
ist ein blosses Modul-Global (`runner_utils/capture_mode.py:32`) und der
Graph-Memory-Pool ist geteilt (`runner_utils/pool.py:34-40`) unter der
ausdruecklichen Annahme, "die beiden Phasen replayen nie gleichzeitig".
Sequentielle Lane-Ticks (Slice A/B) verletzen das nicht; echte
Nebenlaeufigkeit (Slice C) tut es und braucht einen eigenen Pool je Lane.

Der Verband behaelt seine Graphen unveraendert: die FAST-Lane fasst weder
seine Puffer noch seine Kommunikatoren an; ihre Parameter sind dieselben
Tensor-OBJEKTE, also dieselben Adressen — ein Graph, der die Gewichte liest,
bleibt gueltig.

**(b) traegt Graphen nur unter MPS.** Zwei Prozesse ohne MPS
zeitscheiben-serialisieren auf der Karte; ein Graph-Replay des einen Prozesses
laeuft dann nicht neben dem des anderen, sondern zwischen dessen Scheiben.
Die Graph-Faehigkeit waere formal gegeben, die Nebenlaeufigkeit — der Zweck —
nicht. Damit ist (b) auch unter dem Graph-Kriterium schwaecher.

**Slice-A-Zwischenstand**: die FAST-Lane laeuft zunaechst eager. Der Weg zu
Graphen ist damit belegt (Punkt 1-3 oben) und nicht offen; er ist Slice B.

### 1.4 Feature-Paritaet je Lane (hartes Kriterium)

Eine Lane ist keine abgespeckte Lane. Geprueft wird nicht "gebaut", sondern
"strukturell offen". Ein struktureller Ausschluss waere ein K.o. fuer den
Entscheid; es gibt in (a) keinen.

| Feature | In-Prozess-Lane (a) | Praezedenz / benannte Arbeit |
|---|---|---|
| CUDA-Graphen | offen | #133/#136 Zweitlane-Capture; rang-lokale Fahne `spec_solo_rank_local_graphs` (`model_runner.py:1194`, `base_runner.py:91`). Bei ECHTER Nebenlaeufigkeit zusaetzlich ein eigener Graph-Memory-Pool je Lane (`runner_utils/pool.py:34-40` teilt heute einen) — Zusatz, keine Grenze. |
| MTP / Spekulation | offen | #143 Chain-Spec auf der weightless Lane. Mehrere `ModelRunner` je Prozess sind Bestand (`tp_worker.py:434-459`); Draft-Solo baut einen UNGESHARDETEN Draft unter genau dem Override, den die Lane benutzt (`eagle_worker_v2.py:664`). |
| Adaptive Draft-Laenge | offen | Controller-Zustand ist per-Runner, kein Prozess-Global. |
| Uneven TP | ist der Mechanismus | Die Segmentierung in §3 druckt die Lane-Ratio direkt aus; eine mehrkartige Lane (DESIGN_107 Nachtrag 4) ist derselbe Ausdruck mit mehr Segmenten. |
| Uneven DCP | offen | Der Lane-KV ist ein eigener Pool; bei einkartiger Lane ist DCP degeneriert (ein Rang) — ein Grenzfall, kein Ausschluss. Mehrkartige Lane: eigener DCP-Vektor, dieselben Globals brauchen dann das Scoping aus §3.4. |
| HiCache | offen | Haengt an einem Pool, nicht an der Gruppe; `_wl_attach_spill_host_pool` (mixin:2097) haengt bereits einen zweiten Tier an einen zweiten Pool. |
| KV-Spill | offen | dito (`_kv_sess_attach_host_pool`, mixin:2162). |
| Expert-Offload | offen, mit benannter Arbeit | `set_offloader` (`model_runner.py:760`) ist der EINZIGE unbewachte Prozess-Global dieser Familie. Der A-Baum bringt seinen Offload-Plan mit; der Komplementbaum muss sich beim selben Offloader registrieren. Arbeit, keine Grenze. |
| fp8-KV | offen | Per-ROLLE-KV-Praezision existiert bereits (#127, `weightless_kv_worker_cache_dtype`, mixin:2936-2949) — die Lane setzt ihren Pool-dtype selbst. |

Fuer (b) ist dieselbe Tabelle SCHLECHTER, nicht besser: Expert-Offload
braeuchte eine zweite Host-Kopie oder IPC, und die Graph-Nebenlaeufigkeit
haengt zusaetzlich am fehlenden MPS.

### 1.5 Elastische Belegung — darf nicht verbaut werden

Vorgabe: laeuft die Lane leer, geht ihr Slot zurueck. Drei Stufen, gegen
beide Wege bewertet.

**Stufe 1, Compute-Zeitanteil.** Reiner Scheduler-Entscheid. In (a) ist es
exakt das Kadenz-Gate aus `congruent_lane.py` — ein Scheduler entscheidet je
Iteration, wer rechnet, mit sofortiger Wirkung. In (b) kann ein Prozess nur
aufhoeren einzureichen; wie die Karte die verbleibende Zeit verteilt,
entscheidet ohne MPS der Treiber, nicht der Scheduler. Stufe 1 ist in beiden
moeglich, in (a) aber steuerbar statt bloss unterlassbar.

**Stufe 2, VRAM-Rueckgabe — hier trennt sich der Weg strukturell.** In (a)
gibt es EINEN Allocator und EINEN Adressraum: gibt die Lane ein
Pool-Segment frei, faellt es in denselben Caching-Allocator zurueck, aus dem
der Verband seinen Pool zieht. Verleihbare Segmente sind damit ein reines
Pool-Design (Slice B), keine Systemfrage. In (b) faellt ein freigegebenes
Segment in den Allocator DES ANDEREN Prozesses; um es dem Verband
verfuegbar zu machen, muesste der Lane-Prozess an den Treiber zurueckgeben
UND der Verbandsprozess seinen bereits dimensionierten, teils
graph-captureten Pool zur Laufzeit vergroessern. Das ist praktisch ein
Neustart. **Stufe 2 ist in (b) nicht nur ungebaut, sondern strukturell
verwehrt** — fuer sich genommen bereits ein hinreichender Grund fuer (a).

**Stufe 3, Voll-Umspreizen via Handover (#261).** Umsharder-Territorium,
von der Lane-Architektur unabhaengig; DESIGN_107 Nachtrag 2 beschreibt den
zero-copy-Anteil.

**Auflage an Slice A/B, damit Stufe 2 offen bleibt:** der Lane-Pool wird
NICHT als zweiter statischer Pool neben dem Verbandspool dimensioniert,
sondern als KARVE aus demselben profilierten Budget — das Muster gibt es
schon zweimal (`cap_tokens` mixin:4468, Staging-Karve mixin:2465-2537, wo
die physischen Tensoren ihre Groesse behalten und nur der logische
Slot-Raum wandert). Genau diese Form ist verleihbar; zwei getrennt
profilierte Pools waeren es nicht. Slice A legt sich damit nicht fest, aber
es schliesst die Form aus, die Stufe 2 verbauen wuerde.

Die Amortisations-MESSUNG (lohnt die Rueckgabe den Umzug?) ist Slice B/C.

### 1.6 Praeemption an natuerlichen Korngrenzen

Vorgabe: PD ist prioritaer, der Verband ist arbeitserhaltender Nachnutzer;
Praeemption soll an Chunk- und Decode-Schritt-Grenzen passieren, nicht
mitten im Kernel. Bewertung des Kandidaten:

(a) gibt diese Yield-Punkte HER, und zwar ohne Neubau. Beide Lanes werden
von EINEM Scheduler-Loop getaktet; der Uebergabepunkt ist die Iterations-
grenze, an der der Loop entscheidet, welche Lane den naechsten Batch baut —
genau das, was das Kadenz-Gate in `congruent_lane.py` heute schon tut, nur
mit umgekehrter Prioritaet. Die zwei natuerlichen Koerner existieren
bereits als Schleifengrenzen: die Chunk-Grenze des chunked prefill und die
Decode-Schritt-Grenze. Mid-Kernel-Abbruch kommt gar nicht erst in Frage,
weil nie zwei Batches gleichzeitig eingereicht werden (serielle Ticks,
Slice A/B). Bei echter Nebenlaeufigkeit (Slice C) bleibt die Korngrenze
dieselbe — sie wird dann zur Einreihungs-, nicht mehr zur
Ausfuehrungsgrenze.

(b) gibt sie NICHT her. Zwei Prozesse haben zwei Scheduler-Loops; ohne MPS
entscheidet der Treiber die Zuteilung, und ein Prozess kann den anderen
nicht an einer Korngrenze anhalten, sondern nur selbst aufhoeren
einzureichen. Eine PD-Prioritaet waere dort eine Bitte, keine Zusage.

Der Zwei-Klassen-Scheduler selbst ist Slice B; hier steht nur, dass (a) ihn
traegt und (b) ihn nicht traegt.

### 1.7 Was (a) kostet — ehrlich

* Absturz-Robustheit: ein Fehler in der Lane killt den Verbandsrang. Unter
  JEDER Gewichtsteilung ohnehin verloren (DESIGN_107), aber bei (a) auch fuer
  reine Rechenfehler.
* Die Lane-Forwards und die Verbands-Forwards konkurrieren um SMs desselben
  Kontexts. Das ist keine Schwaeche von (a) gegenueber (b) — ohne MPS ist (b)
  schlechter — aber es ist der Grund, warum §7 die Interferenz misst statt sie
  zu behaupten.
* `ParallelContext._overrides` und `_TP_PARTITION_RATIOS` sind
  Prozess-Globals, nicht thread-local. Zwei Lanes duerfen ihre Forwards
  deshalb nicht aus zwei Threads ueberlappen, solange die Geometrie ueber
  diese Globals gelesen wird. Slice A haelt die Ausfuehrung deshalb seriell
  (S1-Muster, wie der Spill-Tick und die congruent lane). Echte
  Stream-Nebenlaeufigkeit braucht die Geometrie am Modul statt am Global —
  benannter Posten in §8.

---

## 2. Die Topologie, konkret auf diesem Rig

```
Karte              NVML  CUDA  Verband (BIG, TP=3)      FAST-Gruppe (PD)
RTX 5090   32607M    1     0   Rang 0, Ratio-Anteil 6   v0 (=Rang 0, geteilt)
                                                        v1 (Komplement, 2/8)
RTX 3080   20480M    0     1   Rang 1, Anteil 1         —
RTX 3080   20480M    2     2   Rang 2, Anteil 1         —
```

BIG-Ratio `[6,1,1]` ueber 8 Einheiten. FAST-Ratio `[6,2]` — beide Shards
liegen auf der 5090, also ist die FAST-Gruppe eine ZWEI-Rang-Gruppe, deren
beide Raenge derselbe Prozess sind. v0 ist byte-identisch der Verbands-Shard
von Rang 0; v1 ist genau das, was die beiden 3080er zusammen halten.

Summe auf der 5090: 6/8 + 2/8 = 8/8 = die vollen Gewichte, GENAU EINMAL.

---

## 3. Nesting-Algebra (der load-bearing Teil)

### 3.1 Warum das nicht trivial ist

`partition_units` (`distributed/utils.py:689`) ist Largest-Remainder mit
Mindestens-1-Bumping. Rang 0 bekommt in BIG und FAST dieselbe reelle Quote
(`b0/sum(B) == f0/sum(F)` per Konstruktion), also dasselbe `int(quota)` —
aber die REST-Verteilung kann divergieren: bekommt Rang 0 in der einen
Gruppe einen Rest-Bump und in der anderen nicht, sind die Shards
verschieden und die Bytes NICHT teilbar. Ebenso kann die
kv-Gruppen-Ausrichtung (`_partition_units_kv_aligned`, `groups`-Argument)
die Grenzen verschieben.

Nesting ist deshalb eine EIGENSCHAFT, die je (Einheitenzahl, groups, Familie)
geprueft werden muss — keine Konstruktionsgarantie.

### 3.2 Die Bedingung

Die FAST-Gruppe ist durch eine Zerlegung der BIG-Raenge in zusammenhaengende
Segmente definiert; FAST-Rang `f` traegt das Segment `S_f`, sein Ratio-Eintrag
ist `sum(B[r] for r in S_f)`.

Nesting gilt fuer eine Probe `(units, groups)` genau dann, wenn

```
partition_units(units, F, groups)[f] == sum(partition_units(units, B, groups)[r] for r in S_f)   fuer alle f
```

Weil die Segmente zusammenhaengend und in Reihenfolge sind, folgt daraus die
Gleichheit der Praefixsummen und damit die Gleichheit der Unit-RANGES — die
geteilten Bytes sind dieselben Bytes, nicht nur dieselbe Menge.

### 3.3 Proben

Geprueft wird gegen die Einheitenzahlen, die das Modell wirklich benutzt:
Attention-q-Einheiten (mit `groups`), kv-Kopfzahl, MLP-Einheiten
(`intermediate // gcd(intermediate, 16)`), MoE-Experten, Vokabular. Der
Ablehnungstext nennt Familie, Einheitenzahl, beide Partitionen und das
Segment, an dem es bricht.

Implementierung: `python/sglang/srt/distributed/dual_group.py`
(`NestedGroupPlan`, `derive_nested_plan`, `transformer_nesting_probes`,
`check_nesting`, `nesting_failures`), Tests
`test/registered/unit/distributed/test_dual_group_nesting.py`.

**Der Check ist nicht kosmetisch — gemessen.** Fuer das Rig-Paar
`[6,1,1] -> [6,2]` nesten 65 von 497 geprueften Einheitenzahlen NICHT.
Beispiel `units=14`: BIG teilt `[10,2,2]`, FAST teilt `[11,3]` — Rang 0
haelt 10 Einheiten im Verband, aber 11 in der Lane. Waere das ungeprueft
durchgelaufen, haette die Lane auf einem Shard gerechnet, der dem
Verbandsrang gar nicht gehoert, und die Behauptung "geteilte Bytes" waere
still falsch gewesen. Eine Ratio, die jede vorkommende Einheitenzahl exakt
teilt, ist immer genestet; das ist die praktische Auswahlregel.

Zweiter Bruchfall, den derselbe Pfad faengt: die REPLICATED-KV-Schwelle
(`kv_heads < tp_size`) kann fuer die eine Gruppe greifen und fuer die andere
nicht — bei 2 kv-Koepfen ist BIG (TP=3) replicated-kv und FAST (TP=2) nicht.
Dann ist Nesting nicht verletzt, sondern UNDEFINIERT: kein Shard der einen
Geometrie ist ein Shard der anderen. Der Ablehnungstext sagt genau das.

### 3.4 Der fehlende Scoping-Primitiv

`_TP_PARTITION_RATIOS` ist ein Modul-Global mit blossem Setter und ohne
Save/Restore (`distributed/utils.py:89-131`). Der einzige Diskriminator, ob ein
Plan gilt, ist `len(ratios) == tp_size` (`utils.py:778`, `:829`). Fuer eine
zweite Gruppe anderer Groesse (hier 2 statt 3) faellt das stillschweigend auf
den GLEICHVERTEILTEN Split zurueck — das waere ein still falscher
Komplement-Shard, kein Fehler.

Deshalb: `scoped_tp_partition_ratios(...)` (Kontextmanager, Save/Restore) in
`distributed/utils.py`, nach dem einzigen bestehenden Idiom dieser Familie
(`model_runner_kv_cache_mixin.py:3994-4015` macht das von Hand fuer die
CP-Token-Ratios). Der Komplement-Shard wird unter

```
with scoped_tp_partition_ratios(F), get_parallel().override(tp_size=2, tp_rank=1, ...):
    build + load
```

gebaut — dann liefert der bestehende Loader ohne jede Aenderung genau die
Einheiten, die die 3080er halten.

---

## 4. Die lokalen Kollektiv-Ersatzoperationen

### 4.1 Column-parallel (all-gather -> cat)

Ein Column-Linear liefert je Rang seinen Ausgabe-Slice, bei
merged/QKV-Linears als Konkatenation ueber die Teilausgaben (`gate|up`
bzw. `q|k|v`). Der lokale Ersatz muss deshalb PRO TEILAUSGABE
konkatenieren:

```
voll = cat_s( cat_r( teil_r[..., off(r,s) : off(r,s)+size(r,s)] ) )
```

Nur so entsteht die kanonische Reihenfolge `[q_all, k_all, v_all]` statt
`[q0,k0,v0,q1,k1,v1]`. Das ist exakt (reine Umkopie, keine Arithmetik) und
damit bitweise identisch zum echten all-gather.

### 4.2 Row-parallel (all-reduce -> add) und der Huellbaum

Ein Row-Linear bekommt die volle Eingabebreite, teilt sie an der
Unit-Grenze, wendet je Shard dessen Gewicht an und ADDIERT. Das ist exakt
das, was ein 2-Rang-all-reduce tut.

Damit alles ZWISCHEN Column-Split und Row-Reduce (Attention-Geometrie,
GDN-Mixer, Aktivierung) die VOLLE Breite sieht, wird der Lane-Modulbaum
unter `override(tp_size=1, tp_rank=0)` gebaut — die exakte
Weightless-Head-Konstruktion — und zwar auf `meta` (kein Gewichtsabdruck,
Praezedenz `_build_weightless_worker_meta_model`). Die parallelen Linears
dieses Huellbaums werden anschliessend durch Schalen ersetzt, die auf
(A = Modul des Verbandsrangs, B = Modul des Komplementbaums) verweisen und
§4.1/§4.2 ausfuehren.

Drei Baeume, aber nur ZWEI Gewichtssaetze:
* A-Baum = der Verbands-Rang-0-Baum (existiert, 6/8, unveraendert),
* B-Baum = Komplementbaum (2/8, neu geladen),
* Huellbaum = `meta`, 0 Bytes.

### 4.3 Was das fuer das Byte-Gate heisst — vorab, ehrlich

`cat` und `split` sind reine Datenbewegung und damit bitweise identisch zu
einem echten all-gather DERSELBEN Rang-Ausgaben. Die Summe `a+b` ist
bitweise identisch zu einem echten 2-Rang-all-reduce (beides eine Addition).

Gemessen (nicht angenommen, Test
`test_column_gather_does_not_reproduce_a_monolithic_gemm`): schon der
Column-Pfad ist NICHT bitweise identisch zu einem monolithischen GEMM. Das
Aufteilen der Ausgabedimension aendert das Blocking des Kernels, also die
Akkumulationsreihenfolge ueber k — und das, ohne dass ueberhaupt eine
Reduktion im Spiel waere. Fuer den Row-Pfad kommt die andere
Summationsreihenfolge dazu; ob sie zufaellig zusammenfaellt, ist Glueck und
wird nicht als Eigenschaft getestet.

Folge fuer das Byte-Gate: "PD-Lane-Prefill == Verbands-Prefill" kann
strukturell nicht bitweise bestehen — der Verband reduziert ueber DREI
Summanden und rechnet mit anderen Shard-Formen. Die belastbaren
Formulierungen sind:
* bitweise gegen eine echte FAST-Gruppe (die Referenz, die die Lane ersetzt),
* numerisch-nah mit benannter Toleranz gegen den 3-Rang-Verband,
* und — das eigentliche Gate — die Gewichts-IDENTITAET der geteilten Shards
  per `data_ptr()`, die exakt und binaer pruefbar ist.
Das ist die Vorhersage vor der Messung, keine nachtraegliche Einraeumung.

---

## 5. VRAM-Posten der Minimal-Runtime (Nutzer-Pflichtangabe)

Je Posten auf der geteilten Karte: geteilt / genestet / dupliziert, mit Grund.

| Posten | Status | Grund |
|---|---|---|
| Gewichte Verbandsrang 0 (6/8) | **geteilt** | dieselben Tensor-Objekte; A-Baum ist der Verbandsbaum. Per `data_ptr()` verifiziert (congruent-lane-Invariante). |
| Gewichte Komplement (2/8) | **genestet, einmalig** | Die 5090 haelt 8/8 statt 6/8. Der Zuwachs 2/8 ist KEINE Duplikation: diese Bytes existieren sonst nur auf den 3080ern. Ohne sie kann die Lane nicht solo prefillen — das ist der Kaufpreis des Features. |
| Lane-Huellbaum | **0 Bytes** | `meta`-Device, Praezedenz `_build_weightless_worker_meta_model`. |
| Lane-KV-Pool | **dupliziert (bewusst, klein)** | Die Lane muss request-disjunkte Slots haben; DESIGN_107 Nachtrag 3 macht die Groesse zur Planner-Stellschraube. Jedes MiB fehlt dem grossen Pool. |
| Lane-GDN/Mamba-State-Pool | **dupliziert (klein)** | DESIGN_107 Nachtrag 2: ~1,5 MiB x Kopfanteil x Sessions je Layer-Satz. Zwei Pools auf tp1 sind Budget, kein Korrektheitsproblem. |
| Lane-Aktivierungen/Scratch | **dupliziert waehrend des Ticks** | Per Forward, nie geteilt. Bei serieller Ausfuehrung (Slice A) ueberlappt der Peak NICHT mit dem Verbands-Peak. |
| CUDA-Kontext | **geteilt** | derselbe Prozess — der Posten, den Weg (b) zusaetzlich zahlen wuerde. |
| GGUF-Dequant-Scratch (#257) | **geteilt (Workspace) / einmal reserviert** | Der Lane-Forward benutzt denselben persistenten Dequant-Workspace. Die Reservierung ist bereits im KV-Profiler; das groesste Ziel der Lane (voller lm_head statt 6/8-Shard) ist groesser als das des Verbandsrangs und muss deshalb das reservierte Maximum stellen. |
| Verbands-Decode-Graphen | **unveraendert** | Die Lane fasst weder Puffer noch Kommunikatoren an. |
| Lane-Graphen | **Slice B** | in Slice A eager; §1.3. |

---

## 6. Ausfuehrungsmodell Slice A

Seriell, nach dem belegten S1-Muster: ein Lane-Tick laeuft ANSTELLE einer
Verbands-Decode-Iteration, gegatet durch die Kadenz aus `congruent_lane.py`
(Decode-Prioritaet). Rang-Uniformitaet ist hier leichter als dort: der Tick
ist rein rang-lokal auf Rang 0 und fasst kein Kollektiv an — es gibt keine
Branch-Divergenz ueber die Raenge, weil die anderen Raenge von der Lane
nichts sehen. Genau deswegen kostet ein Lane-Tick den Verband aber eine
Iteration Wartezeit; das ist die Groesse, die §7 misst.

---

## 7. Messplan Interferenz (Falsifikator fuer Slice B/C)

Verband MIT Graphen und Spekulation (full-perf-Regel), nicht eager.

Die Leserichtung folgt der Prioritaetsvorgabe (PD zuerst, Verband
arbeitserhaltender Nachnutzer): die zu SCHUETZENDE Groesse ist die ms/Runde
der PD-Lane. Deshalb werden beide Richtungen getrennt erhoben, und nur eine
davon darf degradieren:

* **Richtung 1 (muss klein bleiben): PD-Degradation unter Verbandslast.**
  ms/Prefill der Lane solo vs. ms/Prefill der Lane waehrend der Verband
  dekodiert. Ein grosses Delta hier falsifiziert die Prioritaetszusage.
* **Richtung 2 (darf degradieren): Verbands-Degradation unter PD-Last.**
  ms/Verify je Rang solo vs. ms/Verify je Rang waehrend die Lane prefillt.
  Das ist der akzeptierte Preis, nicht der Befund.

Je Richtung 15 s, ms/Runde je Rang, Delta beziffert und gegen den
A-vs-A-Rauschboden begruendet (erst A-vs-A messen, dann verschraenkt).

---

## 8. Aufwand Slice B / C (Schaetzung, siehe Rueckgabe-Bericht fuer den Stand)

* **B1 Komplement-Loader + Huellbaum-Schalen**: 1,5–2 Tage. Die Schaerfe
  liegt bei den v2-Parameter-Loadern (`layers/parameter.py:157/210/280/342`),
  die `tp_size=None` an `tp_loaded_shard_start` geben, also "was gerade global
  installiert ist" lesen. Mit dem Scoping-Primitiv aus §3.4 ist das
  beherrschbar, aber jeder quantisierte Pfad (GGUF, FP8, AWQ, GPTQ) braucht
  einen eigenen Sichtungstest.
* **B2 Lane-Pools + Lane-Prefill**: 1–1,5 Tage. Eigener kleiner KV- und
  GDN-State-Pool mit voller Kopfzahl. Die Bau-Seite ist billig — die Pools
  sind reine Datenobjekte ohne Registry, und es gibt drei Praezedenzen fuer
  einen zweiten Pool im selben Prozess (`_wl_attach_spill_host_pool`
  mixin:2097, `_kv_sess_attach_host_pool` mixin:2162,
  `init_unified_*_pools` mixin:1937/2013) sowie zwei fertige Karve-Muster
  (`cap_tokens` mixin:4468, Staging-Karve mixin:2465-2537). Die Kosten
  liegen in den Prozess-Globals, die die Sizing-Pfade lesen
  (`get_cp_token_ratios`, `_WEIGHTLESS_KV_HEAD_RANK`) und in einer Falle:
  `_apply_token_constraints` (mixin:3656) enthaelt ein `ReduceOp.MIN`
  all-reduce (`:3700`) — die Lane MUSS rang-lokal sizen oder sie haengt die
  Gruppe.
* **B3 Lane-Graphen**: 1 Tag. §1.3 Punkt 1-3; der leichte Fall der
  Weightless-Capture.
* **C1 Echte Nebenlaeufigkeit (zweiter Stream)**: 2–3 Tage, und der
  Blocker ist NICHT NCCL (die Lane hat keinen Kommunikator), sondern die
  Prozess-Globals `_TP_PARTITION_RATIOS` / `ParallelContext._overrides`. Die
  Geometrie muss ans Modul, nicht ans Global — das ist eine breite, aber
  mechanische Aenderung.
* **C2 Handover FAST->BIG zero-copy**: 1–2 Tage auf der bestehenden
  Umsharder-/Handover-Maschinerie; DESIGN_107 Nachtrag 2 beschreibt die
  Geometrie (tp1s 6/8-Anteil bleibt liegen, nur 2/8 wandert).
* **C3 Gemeinsamer Tuner (Planner-Front)**: DESIGN_107 Nachtrag 3/3a,
  ausdruecklich Planner-Territorium, nicht hier.

---

## 9. Ausfuehrungsstand Slice A (2026-07-28, Commit 0d155ba32d)

GEBAUT und gruen (34 Unit-Tests, `CUDA_VISIBLE_DEVICES=99`):
* `distributed/dual_group.py` — Segmentierungs-Plan, Nesting-Check,
  modellabgeleitete Proben, lokale Kollektiv-Ersatzoperationen,
  Posten-Tabelle mit Sharing-Status.
* `distributed/utils.py::scoped_tp_partition_ratios` — der fehlende
  Scoping-Primitiv aus §3.4.
* Runbook-Abschnitt 4.10 (Auswahlregel fuer eine nestbare Ratio).
* Gesamte distributed-Suite unveraendert: vorher 456 gruen / 16 rot,
  nachher 490 gruen / dieselben 16 rot (LOCAL_RANK, NVML-Retry —
  vorbestehend).

GEMESSEN auf dem Rig (Verband MIT Graphen + NEXTN, nicht eager),
Qwen3.6-27B-Q3_K_M-GGUF, TP=3 `--rank-tp-ratio 6,1,1`,
`--rank-gpu-memory-mib 28100,17780,17780`:
* Boot gruen, `max_total_num_tokens=81960`, Ausgabe ueber 400 Token kohaerent.
* Decode solo, 3 identische Laeufe: 31,32 / 31,03 / 31,20 ms je
  Verify-Runde (verify_ct 305 und Accept 1,311 in allen drei identisch)
  -> **A-vs-A-Rauschboden 0,9 % Spannweite**, 41,9 tok/s.
* Kaltes Prefill, 6847 Token: 806 / 798 ms je 1k Token (1,0 % Spannweite).
* 5090: 19933 von 32607 MiB belegt, also 12,4 GiB frei bei einem Modell von
  12,87 GiB. Das Komplement (2/8) sind 3,22 GiB — es passt mit ~9 GiB Rest
  fuer die Lane-Pools. **Machbarkeitsblock erfuellt.**

NICHT gebaut, daher nicht gemessen: die Lane selbst (Komplement-Loader,
Huellbaum-Schalen, Lane-Pools). Damit gibt es weder Byte-Gate noch
Interferenz-Arme. Beide Messrichtungen aus §7 sind spezifiziert und haben
ihren A-Arm; sie sind der erste Punkt von Slice B.

---

## 10. Slice B (2026-07-28, feat/dual-group-runtime-b, Basis f8b528dc36)

### 10.1 Befund VOR dem ersten Edit: die Slice-A-Beispielratio nestet fuer das
Vehikel NICHT

Der Nesting-Check mit der ECHTEN Qwen3.6-27B-Geometrie (24 q-Koepfe, 4
kv-Koepfe, intermediate 17408, GDN 16 k-Koepfe, vocab 248320) lehnt
`[6,1,1] -> [6,2]` ab: die Attention-Achse hat 4 kv-Gruppen-Einheiten, der
Min-1-Bump teilt sie im Verband `[2,1,1]`, die Lane wollte `[3,1]` — Rang 0
haette 2 Einheiten im Verband, aber 3 in der Lane. Slice A hat mit einer
synthetischen Dense-Geometrie getestet; der Check hat genau den Fall
gefangen, fuer den er gebaut wurde.

GEWAEHLTE RATIO fuer das Vehikel (0 Nesting-Fehler, alle Achsen exakt
teilbar, also immer genestet):

    BIG:  --rank-tp-ratio 2,1,1  --rank-mlp-ratio 6,1,1  --rank-vocab-ratio 6,1,1
    FAST: base (2,2), mlp (6,2), vocab (6,2)   (Segment-Summen, NestedGroupPlan.family_ratios)

Attention/GDN folgen der Basis (`[2,1,1]`: 4 kv-Gruppen -> [2,1,1] exakt,
16 GDN-k-Einheiten -> [8,4,4] exakt), MLP/Vocab behalten die 6,1,1-Spreizung
(1088 -> [816,136,136], 3880 -> [2910,485,485], beide exakt).

KONSEQUENZ fuer die Interferenz-Messung: der A-Arm aus Slice A (reines
6,1,1) ist fuer den B-Arm-Vergleich nicht die richtige Basis — der A-Arm
wird unter der nestbaren Konfiguration (2,1,1 + Familien) neu erhoben
(3 Laeufe, 10-20 s, Graphen+NEXTN), bevor der B-Arm startet.

KOMPLEMENT-GROESSE aendert sich gegenueber der 3,22-GiB-Schaetzung:
Attention+GDN-Komplement ist 1/2 statt 1/4 der jeweiligen Familienmasse,
MLP/Vocab bleiben 1/4. Der Posten wird beim Boot gemessen und im
Posten-Block geloggt (format_vram_posts), nicht vorab behauptet.

### 10.2 B1-Bau-Befunde (waehrend der Implementierung, nicht vorab gewusst)

1. **Loader liest die Ratios zur LADE-Zeit, nicht nur zur Konstruktion.**
   Der v1-`weight_loader` (linear.py) ruft `tp_loaded_shard_start` mit den
   layer-gecachten tp-Werten, aber gegen den GLOBAL installierten Vektor.
   Konsequenz: Konstruktion UND Laden des Komplementbaums muessen beide im
   `scoped_tp_partition_ratios`-Scope stehen. Als CPU-Test festgehalten
   (test_dual_group_lane.py, inkl. Negativkontrolle: ohne Scope faellt der
   2-Rang-Load still auf den Even-Split zurueck).
2. **conv1d ist kein GEMM-Shell-Fall.** Die 3-Gruppen-Struktur (q|k|v-
   Conv-Kanaele) steht NICHT in output_partition_sizes (eine flache
   Partition), sondern im mamba_v2-Loader-Layout. Die Komposition der
   vollen Conv-Gewichte braucht die per-Part-Gruppenbreiten aus dem
   GDN-Mixer (local_num_k/v_heads x head_dims); in-place copy_ Pflicht,
   weil RadixLinearAttention bei Konstruktion VIEWS captured.
3. **Huelle wird REAL auf cuda gebaut, nicht auf meta** (Abweichung von
   §4.2, begruendet): GGUF-Grossgewichte sind lazy (UninitializedParameter,
   0 Bytes) — meta braechte nichts; die kleinen realen Tensoren (conv,
   dt_bias, A_log, Normen) sind genau die, die captured Views tragen und
   in-place gefuellt werden muessen. Replizierte Params werden per
   .data-Alias auf den A-Baum geteilt (0 Bytes, data_ptr-Gate prueft sie).
4. **Sampler-Kollektiv nur bei SYNC_TOKEN_IDS_ACROSS_TP/Grammars** —
   Lane-Jobs greedy ohne Grammar => kein Kollektiv im Lane-Sample-Pfad.
5. **Lane-Runner = ModelRunner(is_draft_worker=True, is_dual_group_lane=True)**
   mit Lane-server_args-VIEW (shallow copy: spec aus, dcp 1, eigene
   mrr/mamba-Slots, Rank-Plaene None). Vier rank-lokale Gates gesetzt:
   Speicher-Probe distributed aus, _apply_token_constraints-Kurzschluss,
   Load-Ende-monitored_barrier uebersprungen, set_offloader bewacht
   (der bekannte unbewachte Prozess-Global hat uns real getroffen).

### 10.3 Slice-B-Ausfuehrungsstand (2026-07-28, Zweig feat/dual-group-runtime-b)

GEBAUT UND AUF DEM RIG VALIDIERT (27B-Q3-GGUF, Verband TP=3 Basis 2,1,1 +
mlp/vocab 6,1,1, NEXTN+Graphen; Boot-Rezept Runbook 4.11):

* B1 Komplement-Loader + Huellbaum-Schalen: 465 Schalen (231 column, 183
  row, 2 embed, 1 lm_head, 48 conv-komponiert), 321 replizierte Params per
  .data-Alias geteilt, 96 GDN-Vektoren komponiert. GATE data_ptr-IDENTITAET:
  BESTANDEN, 1058 Identitaeten. Posten gemessen: Komplement 5780 MiB nested,
  Huellen-Residuum 914 MiB, geteilter Shard 0 MiB.
* B2 Lane-Pools + Lane-Prefill: rang-lokales Sizing aus 1600-MiB-Budget
  (25600 Token, mrr 1, Mamba-Slots eigenstaendig), Lane generiert ALLEIN
  kohaerent (Fortsetzung deckungsgleich mit der Verband-Fortsetzung
  desselben Prompts; eager- und Graph-Pfad byte-identische Output-IDs).
* B3 Lane-Graphen: Decode-Graph (bs=1) + Prefill-Breakable-Graphen
  (gedünnte Tier-Leiter bis 2048) rang-lokal erfasst
  (spec_solo_rank_local_graphs-Muster, keine Kollektive im Graph).
  Decode 61,7 -> 16,6 ms/Schritt (3,7x), Prefill 440 -> 283 ms/1k.

B-ARM INTERFERENZ (im selben Boot, A-vs-A-Boeden zuerst):
* Boeden: Verband solo 33,38/33,80 ms/Verify (1,3 %), Lane solo 580,0/
  580,0/580,9 ms je 2048er-Prefill (0,16 %).
* Richtung 1 (GESCHUETZT): Lane-Prefill unter Verband-Decode-Last
  585,3/585,6 ms = +0,9 % — die PD-Prioritaetszusage haelt (Tick laeuft vor
  dem Verbands-Batch, Wartezeit maximal eine Verify-Runde).
* Richtung 2 (erlaubt): Verband 50,6 ms/Verify (wall) unter DAUER-Lane-Last
  vs 33,4-33,8 solo = ~+50 % bei vollem Lane-Duty-Cycle. Accept-Laenge
  unveraendert (1,38) => reine serielle Wartezeit (S1-Preis), keine
  Rechenzeit-Degradation. SM-Nebenlaeufigkeit ist Slice C.

FEATURE-x-LANE-MATRIX (Nachtrag 3, Stand Slice B):
| Feature | Lane-Status | Anmerkung |
|---|---|---|
| CUDA-Graphen (Decode) | GEHT | rang-lokal, validiert |
| CUDA-Graphen (Prefill/breakable) | GEHT | Tier-Leiter gedünnt, validiert |
| Greedy-Generation | GEHT | Kohaerenz-Gate |
| GGUF-Quant-Pfad | GEHT | apply liest keine TP-Globals zur Forward-Zeit |
| GDN/Mamba-Zustand | GEHT | eigener Pool, volle Kopfzahl, free via free_mamba_cache |
| MTP/Spec auf der Lane | geht-noch-nicht | Lane-View spec=None; Praezedenz #143 vorhanden |
| Radix/HiCache auf der Lane | geht-noch-nicht | Lane bewusst radix-frei (Mamba-Ratio-5-Falle); Karve/Verleih Slice C |
| FP8-Quant-Pfad | geht-noch-nicht (einkartig: physikalisch begrenzt fuer 27B) | s.u. |
| Nebenlaeufigkeit (2. Stream) | AUSGESCHLOSSEN in B | Prozess-Globals (set_server_args-Swap!, _TP_PARTITION_RATIOS, is_capture_mode, Graph-Pool, _DEQUANT_WS seriell) |

BENANNTE ZWEIER-/SONDER-ANNAHMEN (Nachtrag 8, ehrlich):
1. derive_lane_plan nutzt derive_nested_plan(base) = die Zwei-Segment-Form
   und shared rank fest = 0; NestedGroupPlan/Schalen/Segmente sind N-aer.
2. build_dual_group_lanes baut genau EINE Lane (Liste vorhanden, lane_id
   ueberall); N Lanes = Konfigurationsschleife, kein Umbau.
3. Der set_server_args-Swap je Tick ist die groesste Slice-C-Huerde: die
   Forward-Maschinerie liest get_server_args() prozess-global
   (prepare_for_extend-Mamba-Gate, check_cuda_graph_backend). Fuer echte
   Nebenlaeufigkeit muessen diese Reads an den Runner/Batch.

FP8-ARM (Nutzer-Auflage, Klammer statt Messung): Der EINKARTIGEN
In-Prozess-Lane fehlt fuer Qwen3.6-27B-FP8 der Platz — volle Gewichte
einmal ~25 GiB + Rang-0-Nicht-Gewichts-Floor ~4,7 GiB (Runbook 7.1,
gemessen) + Lane-Pools/-Graphen/Huelle ~3 GiB > 32,6 GiB 5090-Total; der
400-MiB-Korridor ist unerreichbar. Die machbare FP8-Form (EVAL_272
Kandidat A: PD 59,9@mrr1 + Haupt 69,18,49@mrr4, Polster 2313 MiB) ist die
ZWEIKARTIGE Lane (5090+x8-3080, cuda:2!) mit echten Lane-Kollektiven —
benannte Slice-C-Uebergabe "FP8+kv-fp8-Arm ausstehend", inkl. des offenen
Sichtungstests fuer den FP8-Parameter-Loader-Pfad im Lane-Scope. Die
Q3-only-Validierung ist damit ausdruecklich NICHT als vollstaendige
Quant-Pfad-Abdeckung berichtet.

SLICE-C-UEBERGABEN (gesammelt):
* Nebenlaeufigkeit: server-args-Swap + ParallelContext-Override sind
  tick-seriell; Geometrie/Config an Modul/Batch statt Global.
* Graph-Pool je Lane (runner_utils/pool.py teilt einen), is_capture_mode.
* _DEQUANT_WS: seriell geteilt ok, nebenlaeufig aliasiert (Geteilte-
  Puffer-Familie).
* Verleihbare Pool-Segmente: Lane-Pools sind eigene Objekte hinter dem
  gemeinsamen Allocator (Stufe-2-offen); Verleih-Logik ungebaut.
* Zweikartige Lane (FP8-Form), Handover/Zero-Copy-Uebernahme (#261),
  Dispatcher (Slice D).
* stats()-Ergebnisliste ist auf die letzten 8 Jobs gekappt (Messfenster-
  Zaehlung im Interferenz-Skript beachtet das).

---

## 11. Slice C (2026-07-28, feat/dual-group-runtime-c, Basis a2c8f76c42)

### 11.1 C1 — Entglobalisierung: Ueberlagerung statt Tausch

Slice B TAUSCHTE die prozessglobale Config je Lane-Tick (`set_server_args`
rein, restore raus). Genau dieser Tausch verbietet Nebenlaeufigkeit: waehrend
die Lane-Args publiziert sind, laese ein gleichzeitiger Verbands-Forward auf
einem anderen Thread die Lane-Config.

Der Ersatz ist eine UEBERLAGERUNG statt eines Tauschs. Eine Lane forwardet
auf ihrem eigenen Thread; `lane_scope` legt Identitaet und Config in eine
Kontextvariable, die per Konstruktion thread-lokal ist (ein frischer Thread
startet beim Default). Damit loesen Lesungen so auf:

    Lane-Thread     -> Ueberlagerung gesetzt  -> ServerArgs der Lane
    Verbands-Thread -> Ueberlagerung leer     -> prozessweit publizierte Args

Die ~370 `get_server_args()`-Stellen der Forward-Maschinerie werden dadurch
lane-korrekt, OHNE angefasst zu werden. Das ist bewusst kein Ausweichen vor
dem "Reads an Runner/Batch ziehen": eine Kontextvariable reist nicht mit
einem OBJEKT mit, deshalb gilt die Invariante, dass ein Batch auf demselben
Thread gebaut UND geforwardet wird, auf dem er entstanden ist — der Lane-
Worker tut beides.

Umgestellte Globals (jeweils: innerhalb eines Threads identisches Verhalten,
ueber Threads hinweg isoliert):

| Global | Warum es nebenlaeufig bricht |
|---|---|
| `RuntimeContext._server_args` | der benannte Slice-C-Blocker |
| `ParallelContext._overrides` | `tp_size=1` der Lane darf die Kollektive des Verbands nicht dimensionieren |
| `_TP_PARTITION_RATIOS` | 2-Eintrag-Vektor der Lane laesst den Verband still auf Even-Split zurueckfallen (Sentinel noetig: `None` ist ein GUELTIGER Planwert) |
| `is_capture_mode` | eine capturende Lane darf den Verbands-Forward keine Capture-Zweige nehmen lassen |
| Graph-Memory-Pool | Graphen mit gemeinsamem Pool teilen die Puffer ihrer Zwischenwerte — gleichzeitiges Replay ueberschreibt |
| GGUF `_DEQUANT_WS` | Geteilte-Puffer-Familie: die Sicherheitsbegruendung des Puffers ist ausdruecklich seriell (dequant->GEMM back-to-back auf EINEM Stream) |

Die Aufteilung der RESSOURCEN (Pool, Workspace) wird nur im nebenlaeufigen
Modus vorgenommen; seriell bleibt die Lane-Scope-Id `None`, also dieselben
VRAM-Posten wie Slice B. Das ist die Bedingung des harten Tors.

### 11.2 Die Globals, die erst der ERSTE nebenlaeufige BOOT gefunden hat

Nicht durch Lesen gefunden, sondern durch Laufenlassen. Der erste
nebenlaeufige Boot starb auf Rang 0 mit `AssertionError` in
`GDNAttnBackend.forward_extend`, erreicht aus dem MTP-Draft-Extend des
VERBANDS — weil `get_attn_backend()` ueber `model_executor.forward_context`
aufloest, dessen Modul-Docstring den Fall bereits benannte: *"_current is a
plain module-level global, not thread-local ... if worker threads ever share
a process, migrate to contextvars.ContextVar"*. Die Lane publizierte ihr
GDN-Backend; der gleichzeitige Draft-Forward des Verbands las es; ein
Full-Attention-Aufruf landete im GDN-Backend. Rang 0 brach ab und riss die
Raenge 1 und 2 ueber gloo mit.

Ebenfalls umgestellt, gleiche Familie:
* `tc_piecewise`: `_tc_piecewise_forward_context` (forward_batch +
  attention_layers, an der Split-Op-Grenze aufgeloest) und
  `_in_tc_piecewise_cuda_graph`,
* `breakable_cuda_graph._in_breakable_cuda_graph` — wird von
  `get_is_capture_mode()` gelesen,
* `dcp/collective_guard._ENABLED` / `_STEP` — ein Lane-Decode-Replay schaltet
  den Guard aus und wieder an; leckt das mitten in den Verbands-Forward,
  ueberspringt Rang 0 einen Handshake, den die anderen Raenge noch machen:
  genau der Rang-Divergenz-Haenger, den der Guard fangen SOLL, verursacht
  durch den Guard.

LEHRE (Rank-lokaler-Test-vor-Kollektiv-Familie, neue Auspraegung): eine
Global-Inventur per grep findet die Globals, die man SUCHT. Diese vier fand
erst der Boot, weil sie nicht ueber `get_server_args` laufen, sondern ueber
per-Forward-Kontexte. Der Falsifikator (zwei Threads nachweislich
gleichzeitig im Scope) ist billig; die Inventur allein ist nicht hinreichend.

### 11.3 C2 — Zwei-Klassen-Scheduler

Die Lane bekommt einen eigenen Thread — der EINMAL in den Lane-Scope geht und
darin bleibt — und einen eigenen CUDA-Stream mit HOHER Prioritaet (gemessen:
Prioritaet -3 aus dem Bereich [0,-3]). Der Verband behaelt den Default-Stream
und ist der arbeitserhaltende Nachnutzer. Die PD-Prioritaet landet damit
dort, wo die Hardware sie honoriert: die Bloecke eines hochprioren Streams
werden vor denen des Nachnutzers eingeplant, SOBALD BLOECKE FREI WERDEN —
Praeemption an der natuerlichen Korngrenze, nie mid-Kernel.

An jeder Korngrenze (Iterationsgrenze = Decode-Schritt-Korn; chunked prefill
trifft sie je Chunk) tut der Scheduler drei Dinge: (1) neu eingetroffene
Lane-Arbeit darf zuerst EINREICHEN (`--dual-group-lane-admission-ms`,
gemessen je Vorkommen: Mittel 0,71-1,80 ms, Max 2,17 ms, 64 Vorkommen je
32-s-Fenster — 0,35 % des Fensters); (2) laenger untaetige Lane verleiht ihr
Segment; (3) niemals auf den Lane-Abschluss blockieren.

Lane-Timings wandern von geraetweitem `synchronize` auf CUDA-Events des
Lane-Streams — sonst enthielte jede Lane-Messung die Kernel des Verbands.

### 11.4 Verleih-Stufe 2, gemessen

Vertrag: das Lane-Budget ist RESERVIERT und wird nie ohne Rueckholgarantie
verliehen; geliehene Segmente tragen nur Evakuierbares. Gebaut sind der
Mechanismus, seine Instrumentierung und der Verwerfbarer-Scratch-Borger.

| Groesse | Messung (1024 MiB Segment) |
|---|---|
| Verleih-Latenz | 0,76 ms Mittel, 0,82 ms Max |
| **Rueckhol-Latenz** | **2,49 ms Mittel, 2,71 ms Max** |
| Amortisationsschwelle (konfiguriert) | 3 s Leerlauf; direkt nach Lane-Arbeit wird NICHT verliehen (geprueft) |
| Hysterese | Rueckholen wird nie verweigert, nur ein Flap gezaehlt (`refused_min_hold`) |

Aus den Zahlen abgeleitet: ein Verleih-/Rueckhol-Zyklus kostet die
geschuetzte Klasse 3,25 ms. Physikalisch amortisiert das schon eine
Haltezeit von ~0,1 s (30:1). Die Default-Schwelle von 5 s ist daher NICHT
kostengetrieben, sondern flatterngetrieben — die ehrliche Lesart.

### 11.5 C3 — Aggregat-Beweis: die Kartenaequivalent-Rechnung

Serielle Tick-Teilung ist per Konstruktion eine Nullsummen-Teilung EINER
Wanduhr. Das macht die richtige, duty-unabhaengige Messlatte sichtbar:

    share_c = Rate_c(gemeinsames Fenster) / Rate_c(solo)          je Klasse c
    E       = share_Verband + share_Lane                          Kartenaequivalente

    Rate_Verband  = Ausgabe-Token/s des /generate
                  = accept_len / (ms_per_verify / 1000)
    Rate_Lane     = Prefill-Token/s ueber das Fenster   (Prefill-Arm)
                  = Decode-Schritte/s ueber das Fenster (Decode-Arm)

E = 1,0 heisst: die zwei Klassen haben eine Karte exakt aufgeteilt. E > 1,0
heisst: es gab echte Ueberlappung. Ein Boot je Modus, Boeden zuerst.

BOEDEN (A-vs-A, 3 Laeufe): Verband 33,65 ms/Verify seriell / 33,80
nebenlaeufig (Spannweite 0,39 %); Lane 583,75 / 583,09 ms je 2048er-Prefill
(0,25 %); Lane-Decode 55,25 / 56,18 Schritte/s. Die Boeden sind zwischen den
Modi gleich — die Nebenlaeufigkeitsmaschinerie kostet nichts, solange nur
eine Klasse laeuft.

| Arm | Modus | Verband | Lane | **E** |
|---|---|---|---|---|
| Prefill (2048er) | seriell | 40,58 -> 13,87 tok/s (0,342) | 3508 -> 2219 tok/s (0,632) | **0,974** |
| Prefill (2048er) | nebenlaeufig | 39,14 -> 15,90 tok/s (0,406) | 3512 -> 2544 tok/s (0,724) | **1,130** |
| Decode (96er Prompt) | seriell | 41,58 -> 26,45 tok/s (0,636) | 55,25 -> 15,34 Schr./s (0,278) | **0,914** |
| Decode (96er Prompt) | nebenlaeufig | 40,50 -> 34,36 tok/s (0,848) | 56,18 -> 33,22 Schr./s (0,591) | **1,440** |

    Prefill-Arm: E 0,974 -> 1,130  = +16,0 % Aggregat
    Decode-Arm:  E 0,914 -> 1,440  = +57,5 % Aggregat

Die serielle Messung bestaetigt die Nullsummen-Vorhersage (0,91-0,97), was
die Rechnung selbst validiert.

DIE ERKLAERUNG DES UNTERSCHIEDS ZWISCHEN DEN ARMEN ist der eigentliche
Befund und keine Ausrede: ein 2048er-Prefill ist eine SM-saettigende
GEMM-Last. Zwei saettigende Lasten auf einer Karte koennen nicht beide voll
laufen — Nebenlaeufigkeit kann dort nur die Luecken einsammeln, und genau
+16 % Luecken gibt es. Eine Decode-foermige Lane stellt kleine, latenz-
gebundene Kernel wie der Verband selbst, und dann zahlt die Ueberlappung mit
+57,5 %. Die Groesse, die die Nebenlaeufigkeit kauft, ist also nicht
konstant, sondern eine Funktion der Lane-Lastform — das gehoert in den
Dispatcher-Entscheid (Slice D).

### 11.6 Die beiden Interferenz-Richtungen

| Richtung | seriell | nebenlaeufig |
|---|---|---|
| 1 (GESCHUETZT): Lane-Prefill unter Verbands-Decode | 585,0-588,5 ms = **+0,4 %** | 629,0-639,4 ms = **+9,7 %** |
| 2 (erlaubt): Verband unter voller Lane-Last | 70,25 ms/Verify = +109 % | 63,32 ms/Verify = **+88 %** |
| 2 (erlaubt, volles Duty, Aggregat-Fenster) | 98,46 ms/Verify = +193 % | 83,21 ms/Verify = **+146 %** |

EHRLICH: die geschuetzte Klasse zahlt unter echter Gleichzeitigkeit MEHR als
unter Tick-Teilung (+9,7 % statt +0,4 %). Das ist kein Bruch der
Prioritaetszusage, sondern ihr Preis in einer anderen Waehrung: seriell
wartet die Lane nur, RECHNET aber allein; nebenlaeufig rechnet sie
gleichzeitig und teilt die SMs. Der Verband gewinnt dabei mehr, als die Lane
verliert (E steigt), und die Lane-Degradation bleibt einstellig.

Herkunftsbefund (aus den vorhandenen Daten, ohne eigenen Boot):
* Die dir1-Messung ist DEVICE-Zeit auf dem Lane-Stream (CUDA-Events), nicht
  Wanduhr — sie enthaelt keine Einreih-Wartezeit.
* Die Werte sind glatt (629,0-639,4 ms, Spannweite 1,6 % gegen 0,27 % solo).
  Eine Granularitaets-/Einreihungsursache muesste Spruenge in der
  Groessenordnung einer blockierenden Verify-Runde (33,8 ms) zeigen.
* Beide Seiten degradieren gleichzeitig und aehnlich glatt (Lane +9,7 %,
  Verband +12,2 % im selben Lauf). Bei einer Praeemptionsgrenze wuerde der
  Nachnutzer NICHT degradieren, weil er nichts abgibt.
* Die Admission-Statistik zeigt 64 Yields je 32-s-Fenster mit Max 2,17 ms —
  der Einreichpfad ist nicht der Engpass.
Damit: SM-Konkurrenz im Compute, nicht Praeemptions-Granularitaet. Der
direkte Beleg (Wall minus Device je Prefill = reine Wartezeit) ist jetzt
instrumentiert (`prefill_wait_ms`) und wird im naechsten Boot mitgelesen.

### 11.7 Korrektheits-Tor

* data_ptr-Gate: 1058 Identitaeten, in JEDEM der 7 Boots bestanden.
* Boeden seriell reproduzieren Slice B (33,4-33,8 / 580-584) — C1 hat die
  Zahlen nicht bewegt.
* BYTE-GATE seriell vs nebenlaeufig: identische Lane-Output-IDs bei 96-Token-
  Prompt, in beiden Modi selbst-deterministisch. Der 2048er-Prompt taugt
  NICHT als Byte-Gate — Qwen-GDN-Prefill ist ab ~109 Token dokumentiert
  nicht reproduzierbar (upstream, alle Backends); zwischen zwei Boots
  divergierte er entsprechend, was nichts ueber diesen Slice aussagt.

### 11.8 Speed-durch-Verzicht-Regler

`--dual-group-lane-speed-dial` loest auf beide Kapazitaets-Posten auf,
reduziert nur und loggt die Aufloesung.

| Dial | Budget | Lane-Token | 5090 belegt | Lane-Prefill 2048 | Lane-Decode |
|---|---|---|---|---|---|
| aus | 1600 MiB | 25600 | 30423 MiB | 583,1 ms | 56,18 Schr./s |
| 1.0 | 200 MiB | 3200 | 28979 MiB | 583,7 ms | 52,43 Schr./s |

BEFUND, negativ und nuetzlich: der Regler kauft an diesem Arbeitspunkt VRAM,
kein Tempo — bei mrr=1 begrenzt die Poolgroesse die Kernel nicht. Er gibt
1444 MiB frei. Die Gegenprobe (dieselben 1444 MiB auf Rang 0 zurueckgegeben)
aendert `max_total_num_tokens` NICHT (81960 in beiden Faellen): der Verbands-
KV wird vom KNAPPSTEN Rang dimensioniert, und das sind die 3080er. Auf
diesem Rig sind die freigewordenen Bytes der geteilten Karte also nur fuer
die Lane-Seite oder fuer den Verleih verwertbar, nicht fuer den Verbands-KV.
Auf einem Rig ohne diese Asymmetrie faellt der Befund anders aus.

### 11.9 Feature-x-Lane-Matrix (Stand Slice C)

| Feature | Lane-Status | Anmerkung |
|---|---|---|
| CUDA-Graphen (Decode/Prefill) | GEHT | unveraendert; im nebenlaeufigen Modus eigener Graph-Pool je Lane |
| Greedy-Generation | GEHT | Byte-Gate seriell==nebenlaeufig |
| GGUF-Quant-Pfad | GEHT | Dequant-Workspace jetzt lane-gekeyt |
| GDN/Mamba-Zustand | GEHT | eigener Pool |
| **Nebenlaeufigkeit (2. Stream)** | **GEHT** | eigener Thread + hochpriorer Stream, E 1,13-1,44 |
| **Verleih-Stufe 2 (VRAM)** | **GEHT** (Scratch-Borger) | Rueckhol-Latenz 2,49 ms; Session-KV-Borger = Folge-Slice |
| Speed-durch-Verzicht-Regler | GEHT | s. 11.8 |
| MTP/Spec auf der Lane | geht-noch-nicht, Weg benannt | KEIN Config-Flip: der Lane-Runner ist `is_draft_worker=True` und baut gar keinen EAGLE-Worker; der formgleiche Vorlaeufer `--speculative-draft-placement solo` braucht in `_solo_init_lm_head` ein GRUPPEN-KOLLEKTIV, das die rang-lokale Lane-Bringup per Vertrag nicht darf. Machbarer Weg: NEXTN-Kopf ueber denselben Komplement-/Schalen-Mechanismus rang-lokal zusammensetzen (Nesting gilt, gleiche Einheitenzahlen wie das Hauptmodell, MTP-Kopf steckt im selben `Qwen3_5ForCausalLMMTP`-Gewichtssatz) + Draft-KV/Spec-Metadaten ins Lane-Budget + Lane-Tick wird Verify-Schleife. |
| Radix/HiCache auf der Lane | geht-noch-nicht | unveraendert seit B |
| FP8-Quant-Pfad (zweikartige Lane) | geht-noch-nicht, Schreibtisch fertig | s. 11.10 |

### 11.10 FP8-Arm: die Schreibtisch-Haelfte, und sie faellt guenstiger aus

Beide offenen Fragen sind CPU-beantwortet (9 Tests), und beide fallen besser
aus als befuerchtet:
* Die `tp_size=None`-Falle ist auf dem v2-Parameter-Pfad LAUT, nicht still:
  der Parameter traegt eine anderswo berechnete Shard-Groesse, also wird der
  Widerspruch zum installierten Vektor benannt ("uneven-TP shard mismatch").
  Still ist der v1-`linear.py`-Pfad, weil die Schicht dort even-split
  konstruiert wird und mit ihrem eigenen falschen Offset uebereinstimmt.
* Block-quantisiertes FP8 bringt eine ZWEITE geshardete Achse mit
  (`ceil(out/block_n)`), die GGUF nicht hat. Auch sie kann nicht still
  falsch laden: beide Achsen werden mit derselben Einheitenzahl partitioniert
  und eine Skalenachse, die kein Vielfaches davon ist, wird VERWEIGERT.

Damit ist die FP8-Bedingung eine PLANUNGSREGEL, keine Laufzeitgefahr: die
Lane-Ratio muss jeden Rang-Ausgabeanteil auf ein ganzes Vielfaches der
Quantisierungsbloecke legen. Als Praedikat vor dem Boot pruefbar.

Der zweikartige Lauf selbst (EVAL_272 Kandidat A, 5090 + x8-3080 = cuda:2)
braucht ECHTE Lane-Kollektive — die heutige Lane hat per Konstruktion keinen
Kommunikator. Das ist ein eigener Bau und ist benannte Uebergabe.

### 11.11 Slice-D-Uebergaben (Dispatcher-Anschlusspunkte)

1. **Lastform bestimmt den Gewinn** (11.5): +16 % bei SM-saettigender
   Lane-Last, +57,5 % bei latenzgebundener. Der Dispatcher hat damit eine
   messbare Zielfunktion statt einer Heuristik — Routing soll Lasten so
   mischen, dass E maximal wird, nicht bloss "verteilen".
2. **Verleih-Interface** steht (lend/reclaim + Borger-Protokoll); der
   Session-KV-Borger (#236/#242 als VRAM-Tier in die Lane-Region) haengt
   genau daran.
3. **Zweikartige Lane / echte Lane-Kollektive** — Voraussetzung fuer FP8
   und fuer jede Lane, die groesser ist als eine Karte.
4. **Lane-MTP** ueber den Komplement-Mechanismus (11.9).
5. Benannte Zweier-Annahmen aus Slice B stehen UNVERAENDERT
   (`derive_lane_plan` Zwei-Segment, shared rank 0, eine Lane); der
   Umfang von C hat sie nicht erzwungen.
6. `--dual-group-lane-admission-ms` und die Stream-Prioritaet
   (`SGLANG_DUAL_GROUP_LANE_STREAM_PRIORITY`) sind die zwei Stellschrauben
   der Zwei-Klassen-Politik, beide instrumentiert.

### 11.12 Modellfamilien-Achse (PRIO-Nachtrag 9) — ehrliche Buchfuehrung

Bisher ist ALLES an dieser Runtime auf genau einer Familie gemessen:
hybrid-GDN dense GGUF (Qwen3.6-27B-Q3_K_M). Die Achse unten trennt, was
familienneutral GEBAUT ist, von dem, was nur auf dieser Familie GEPRUEFT ist
— das sind zwei verschiedene Aussagen, und die zweite ist ueberall die
schwaechere.

#### Welche Bausteine sind GDN-spezifisch, welche familienneutral?

| Baustein | Familienbindung | Begruendung |
|---|---|---|
| `LaneColumnParallelShell` / `LaneRowParallelShell` | **neutral** | arbeiten ueber `output_partition_sizes` / `input_size_per_partition`, also ueber die Linear-Schnittstelle; kein Modellwissen |
| `LaneVocabEmbeddingShell` / `LaneLmHeadShell` | **neutral** | Vokabular-Achse, jede Familie hat sie |
| `assemble_lane_shells` Taxonomie | **neutral im Kern, LUECKENHAFT** | erkennt Column/Row/Vocab/LMHead. Kennt KEINE Experten-Tensoren (s.u.) |
| `_COMPOSED_LINEAR_SUFFIXES = ("conv1d",)` + `_gdn_conv_sub_sizes` | **GDN-spezifisch** | die 3-Gruppen-Conv-Kanalstruktur des mamba_v2-Layouts; eine dense-Llama-Familie hat diesen Zweig gar nicht (er greift schlicht nie) |
| `_finalize_hull_params` Sonderfall `dt_bias` / `A_log` | **GDN-spezifisch** | per-Kopf-GDN-Vektoren; bei dense/MoE laeuft nur der Alias-Zweig |
| `derive_lane_plan` / Nesting-Proben | **neutral, MoE bereits vorgesehen** | `transformer_nesting_probes` zieht `num_experts` aus der Config und prueft die Experten-Achse mit; das ist gebaut, aber nie an einem MoE gemessen |
| Lane-Pool-Sizing (`_resolve_dual_group_lane_pool_config`) | **neutral** | benutzt den Stock-Configurator; die Mamba-/GDN-Slots entstehen dort nur, wenn das Modell hybrid ist |
| `max_mamba_cache_size` im Lane-Args-View | **GDN-spezifisch, aber harmlos** | ein Feld, das eine dense Familie ignoriert |
| Lane-Scope / Nebenlaeufigkeit / Verleih / Speed-Dial (C1/C2) | **neutral** | Kontextvariablen, Threads, Streams, Pool-Segmente — kein Modellwissen |
| NEXTN-Kopf-Weg (11.9/11.13) | **neutral im Mechanismus** | der Kopf wird als "ein weiterer Layer in der Geometrie des Verbands" behandelt; das gilt fuer jede Familie mit MTP-Kopf. Das Vokabular-Teilen haengt an `set_embed_and_head_modules`, einer Konvention der MTP-Modellklassen, nicht an GDN |

#### Feature-x-Lane-x-FAMILIE

Legende: GEHT = auf dem Rig gemessen; gebaut-ungemessen = Code traegt es,
keine Messung; LUECKE = Baustein fehlt nachweislich.

| Feature | hybrid-GDN dense GGUF | dense ohne GDN (Llama-Klasse) | MoE | MoE-GGUF |
|---|---|---|---|---|
| Komplement-Loader + data_ptr-Gate | **GEHT** (1058 Ids) | gebaut-ungemessen (nur der GDN-Zweig entfaellt) | **LUECKE** (Experten-Schalen) | **LUECKE** |
| Huellbaum + Schalen | **GEHT** (465 Schalen) | gebaut-ungemessen | **LUECKE** | **LUECKE** |
| Lane-Pools / Lane-Prefill | **GEHT** | gebaut-ungemessen (einfacher: keine Mamba-Slots) | gebaut-ungemessen | gebaut-ungemessen |
| Lane-Graphen | **GEHT** | gebaut-ungemessen | gebaut-ungemessen | gebaut-ungemessen |
| Nebenlaeufigkeit (C2) | **GEHT** (E 1,13/1,44) | gebaut-ungemessen | gebaut-ungemessen | gebaut-ungemessen |
| Verleih-Stufe 2 | **GEHT** (2,49 ms) | gebaut-ungemessen | gebaut-ungemessen | gebaut-ungemessen |
| NEXTN-Kopf auf der Lane | s. 11.13 | gebaut-ungemessen | **LUECKE** (Experten im Kopf) | **LUECKE** |
| Nesting-Check der Experten-Achse | n/a | n/a | gebaut-ungemessen | gebaut-ungemessen |

#### Die benannte LUECKE: Experten-Schalen im Komplement-Loader

`assemble_lane_shells` kennt vier Modulklassen — `ParallelLMHead`,
`VocabParallelEmbedding`, `RowParallelLinear`, `ColumnParallelLinear` — und
wirft fuer jedes Huellmodul, das keine Entsprechung im Teilbaum hat, einen
harten Fehler. Ein MoE-Modell bringt `FusedMoE`-Module mit, deren Gewichte
(`w13_weight`, `w2_weight`, Skalen, Experten-Maps) NICHT ueber die
Linear-Schnittstelle laufen. Konsequenzen, ehrlich getrennt:

1. Die Experten-Tensoren wuerden von der Taxonomie nicht erkannt und
   landeten in `_finalize_hull_params` im "weder repliziert noch bekannt
   komponiert"-Zweig — also ein LAUTER Fehler beim Bau, kein stilles falsches
   Ergebnis. Das ist die gute Nachricht: die Luecke ist gebaut, um sich zu
   melden.
2. Gebraucht wird eine fuenfte Schalenklasse, die die Experten-Achse
   zusammensetzt. Die Achse ist NICHT die Ausgabedimension eines GEMM,
   sondern die Experten-Nummerierung (EP) bzw. die Intermediate-Dimension je
   Experte (TP-in-MoE) — je nachdem, was der Verband gerade sharded. Das ist
   dieselbe Art Arbeit wie `_gdn_conv_sub_sizes` (Gruppenbreiten aus dem
   besitzenden Modul ziehen), aber gegen das MoE-Layout.
3. Der Nesting-Check ist dafuer bereits vorbereitet
   (`transformer_nesting_probes` prueft `num_experts` als eigene
   Einheitenzahl) — die ALGEBRA fehlt also nicht, nur die Schale.
4. Expert-Offload (#77) ist ein zusaetzlicher Posten: `set_offloader` ist
   der bekannte unbewachte Prozess-Global, den Slice B fuer die Lane bereits
   bewacht hat; ein MoE-Lane-Bau muesste sich beim selben Offloader
   registrieren statt einen zweiten zu installieren.

Aufwandsklasse: eine Schalenklasse plus Sichtungstest je MoE-Quantpfad —
vergleichbar mit B1, nicht groesser.

### 11.13 NEXTN auf der Lane — Stand: Kopf assembliert, Pool-Verdrahtung offen

BELEGT (Rig, `--dual-group-lane-spec`, 6 Boots):

    dual-group lane draft model assembled: shells column=2 row=2 embed=1
    lm_head=1 composed=0; params aliased=8 composed_vec=0;
    shared-byte gate PASSED (16 data_ptr identities).
    dual-group lane draft: embed/lm_head pointed at the lane target's
    full-vocabulary shells (one set of tables, not two).

Damit ist die eigentliche Frage beantwortet: der NEXTN-Kopf laesst sich mit
dem Komplement-/Schalen-Mechanismus RANG-LOKAL auf der Lane zusammensetzen,
mit data_ptr-Identitaet auf dem geteilten Segment und ohne ein einziges
Gruppen-Kollektiv — genau dort, wo `--speculative-draft-placement solo`
scheitert (dessen `_solo_init_lm_head` ist ein Kollektiv und verweigert
GGUF-Packed-Vokabular ueberhaupt). Der Kopf ist ein weiterer Layer in der
Geometrie des Verbands; die Algebra brauchte keine Erweiterung.

DREI BEFUNDE AUS DEM BAU, jeder hat einen Boot gekostet:

1. **Reihenfolge**: der Kopf muss VOR den Pools beider Lane-Runner
   assembliert werden. Sein Komplement-Load konstruiert einen eigenen
   `ParallelLMHead` (1,19 GiB), bevor ihm jemand den des Ziels geben kann;
   nach der Pool-Allokation sind noch ~600 MiB frei und das OOMt. Das ist
   dieselbe Begruendung, aus der der Verband seinen embed/head-Share FRUEH
   macht.
2. **Huelle des Kopfes auf `meta`**: anders als das Zielmodell hat der Kopf
   keine GDN-Conv-Views, die bei der Konstruktion gecaptured werden (seine
   eine Schicht ist volle Attention), und seine eigenen Vokabular-Tabellen
   waeren 2,37 GiB, die sofort ersetzt werden. `_finalize_hull_params`
   aliast jetzt ZUERST und urteilt danach, damit ein Meta-Platzhalter, der
   die geteilte Storage bekommt, als gefuellt gilt (Parameter-OBJEKT-Tausch
   statt `.data`, weil `set_data` den Typwechsel meta->cuda ablehnt).
3. **EIN Budget, geteilt**: beide Lane-Runner lesen
   `--dual-group-lane-budget-mib`; ohne expliziten Split haette der Kopf ein
   ZWEITES volles Budget genommen und der Kapazitaets-Posten der Lane sich
   hinter dem Ruecken des Betreibers verdoppelt (`split_lane_budget`,
   4 CPU-Tests). Ressourcen-Grundsatz 2, wortwoertlich.

OFFEN, praezise benannt: die POOL-VERDRAHTUNG des Kopfes. Der letzte Boot
stirbt in `pool_configurator.calculate_pool_sizes` an einer Division durch
Null, und die Ursache ist ein Entwurfsfehler von mir, nicht ein Bug dort:
ein NEXTN-Draft dimensioniert seinen KV in sglang NICHT aus einem eigenen
Budget, sondern bekommt die `memory_pool_config` des Ziels durchgereicht
(`TpModelWorker` tut genau das fuer den Verbands-Draft). Mein
`_resolve_dual_group_lane_pool_config`-Pfad rechnet stattdessen Token aus
MiB, und fuer den Kopf ist die Byte-je-Token-Zahl 0. RICHTIGE FORM: der
Lane-Kopf bekommt die `memory_pool_config` und den Allocator des
LANE-ZIELS; der Budget-Split entfaellt dann fuer den KV-Teil und bleibt nur
fuer die Graphen des Kopfes noetig. Aufwand: klein, aber es ist eine
Aenderung am Bau, keine Konfiguration.

DANACH erst kommt die Kette selbst (Prefill -> k Draft-Schritte -> ein
Verify-Forward, greedy, topk 1), samt `capture_hidden_mode` fuer die
Hidden-States und der KV-Ruecknahme verworfener Token. Sie ist NICHT gebaut.

`--dual-group-lane-spec` ist per Default AUS; alle Slice-C-Belege oben sind
davon unberuehrt.

VORHERSAGE-KORREKTUR, ehrlich: der Machbarkeitsblock schaetzte den Kopf auf
~120 MiB Komplement. GEMESSEN: **2684 MiB**. Der Fehler war die Annahme, das
Komplement bestehe nur aus der Decoder-Schicht — der Stock-Loader laedt den
kompletten Kopf-Checkpoint inklusive seines Vokabular-Anteils. Der Versuch,
diesen Anteil nach dem Zeigen auf die Ziel-Schalen freizugeben, brachte
gemessen 0 MiB (die Tabellen liegen nicht dort, wo der Modulname sie
vermuten laesst) — auch das ist offen und gehoert zur Pool-Verdrahtung.

### 11.14 Lane-Spec (feat/dual-group-lane-spec) — Pool-Verdrahtung gruen, Kopf-Dispatch offen

#### Punkt 1 (Draft-KV-Pool): GEBAUT UND AUF DEM RIG BELEGT

    dual-group lane 0 HEAD pool sizing (rank-local): budget 400 MiB,
    1 KV-bearing layer(s), 4096 B/token -> max_total_num_tokens=19200,
    max_running_requests=1.

Der Weg dorthin und die Rechnung:

* URSACHE der Division durch Null, genau: der Configurator leitet die
  Schichtzahl aus der Modell-CONFIG ab. Die Draft-Config wird aus demselben
  Checkpoint gebaut und meldet daher die 64 Schichten UND die
  Full-Attention-Layer-Ids des ZIELS; geschnitten mit dem Layer-Bereich des
  Kopfes ergibt das 0 Schichten, also 0 Byte je Token.
* FIX: `_lane_kv_bearing_layer_count` zaehlt die `RadixAttention`-Module im
  ASSEMBLIERTEN Baum. Familienneutral, und es kann nicht mit dem Modell
  uneins sein, das gleich laeuft. Daraus rang-lokal:
  `Token = Budget // (kv_heads x (head_dim + v_head_dim) x n_layers x
  kv_elem)`, seitenausgerichtet, ohne Profiling, ohne Kollektiv.
* RECHNUNG auf dem Vehikel: 4 kv-Koepfe x (128+128) x 1 Schicht x 4 B (fp8
  kv) = **4096 B/Token**; das Ziel liegt bei 1200 MiB / 19200 Token =
  65536 B/Token, der Kopf kostet also **1/16 des Ziels je Token**.
* DECKEL: aus 400 MiB folgen 102400 Kopf-Token — Plaetze, die der Kopf nie
  nutzen kann, weil er den Sequenzen des Ziels folgt. Gedeckelt auf dessen
  19200 (`dual_group_lane_token_cap`): **~325 MiB nicht allokiert**.
* KOSTENSTAND gesamt: der Kopf kostet **2684 MiB Gewichte** (gemessen, s.
  11.13) + 75 MiB Pool. Auf der 5090 heisst das, Rang 0 muss zurueckgeben;
  22800 -> 21000 MiB traegt (1418 MiB frei nach dem Boot, ueber dem
  400-MiB-Korridor), 19800 kippt bereits an der KV-Untergrenze von Rang 0.

#### Punkt 2 (Kette): GEBAUT, NICHT GRUEN

Geschrieben und im Zweig: Vorschlagsschleife (`_propose`, K Schritte, topk 1,
greedy), Verify mit Greedy-Accept und KV-Ruecknahme der verworfenen Kandidaten
(`_verify`), Runden-Buchfuehrung (ms/Runde + Accept-Laenge statt eines
Token-Mittels, das beide versteckt), Hidden-State-Capture im Lane-Prefill
(`CaptureHiddenMode.FULL`) und das Vorfuellen des Kopf-KV ueber denselben
Prompt.

NICHT gruen: der Kopf kommt im Bring-up nicht durch. Vier Dispatch-Kontrakte,
jeder von einem Boot gefunden, drei davon geloest:

1. eigenes KV-Sizing (oben) — GELOEST,
2. `init_cuda_graphs` ist auch fuer den EAGER-Runner noetig; ohne den Aufruf
   hat der Forward-Dispatch gar keinen `eager_runner` — GELOEST,
3. der Kopf darf die `req_to_token_pool` des Ziels NICHT teilen: er faehrt
   eine eigene `ScheduleBatch`, die einen eigenen Slot zieht, und bei mrr 1
   streiten beide um den einzigen (`alloc_req_slots runs out of memory,
   available_size()=0`) — GELOEST, eigene Tabelle (~128 KiB, kein Posten,
   der dieses Fehlerbild wert waere),
4. OFFEN: `init_cuda_graphs` betritt `init_decode_cuda_graph` auch bei
   durchgehend DISABLED-Phasen und dereferenziert dort
   `model_runner.graph_shared_output`, das nur die Ziel-Bringup fuellt ->
   `'NoneType' has no get_logits_buffer`. Der Kopf muss diesen Puffer
   bekommen oder ihn selbst bauen.

Ausserdem benannt: der Kopf laeuft EAGER. Sein Decode-Graph wuerde vom
generischen `DecodeCudaGraphRunner` erfasst, der eine Dummy-Batch mit
`spec_info=None` baut — ein MTP-Forward dereferenziert das. Der Verband
umgeht das ueber den EAGLE-spezifischen Draft-Graph-Runner; den Kopf dorthin
zu routen ist die Folgearbeit.

#### Punkte 3 und 4 (Gates, Messung): NICHT ERREICHT

Das entscheidende Gate — Lane-Solo-Kohaerenz MIT Spec byte-identisch zur
No-Spec-Lane (unter greedy muss Spekulation exakt dieselben Token liefern) —
ist geschrieben und lief gegen einen Kopf, der noch nicht forwarded. Die
Referenz steht bereit: `[1248, 1518, 29496, 13, 22044, 7370, 1680, 430, 279,
650, 92016, 11]` (96-Token-Prompt, in beiden Slice-C-Modi identisch).
Das data_ptr-Gate laeuft in JEDEM Boot mit und ist gruen: 1058 Identitaeten
fuer das Ziel, **16 fuer den Kopf**.

#### Zwei eigene Fehler, korrigiert und benannt

* `_drop_draft_complement_vocab` (aus 11.13) ist ENTFERNT. Es sollte den
  Vokabular-Anteil des Kopf-Komplements freigeben, hat gemessen 0 MiB
  gebracht — und es lief ueber ALLE Teilmodelle, also auch ueber das
  resident geteilte Modell des VERBANDS. Ein `embed_tokens = None` dort
  waere ein Eingriff in den laufenden Verband gewesen. Der richtige
  Ansatzpunkt ist ohnehin die Konstruktion, nicht das Nachtraegliche.
* Die Vokabular-Schalen des Ziels werden jetzt per TYP gesucht
  (`_find_lane_vocab_shells`), nicht per Attributpfad. Der Pfad
  unterscheidet sich je Modellfamilie (`model.embed_tokens` vs
  `model.model.embed_tokens` hinter einem Conditional-Generation-Wrapper),
  und danebenzugreifen ist STILL: `set_embed_and_head_modules` ueberspringt
  ein `None`-Argument, der Kopf behaelt seins — und stirbt erst beim ersten
  Forward. Jetzt ist es ein Bring-up-Fehler mit Satz.

### 11.15 Lane-Spec-Kette Runde 2 (feat/dual-group-lane-spec-r2, Basis b662cc98a3)

Runde 1 endete an Kontrakt 4. Runde 2 hat ihn und drei weitere geloest, die
Kette laeuft jetzt MECHANISCH durch (5 Boots, 0 Tick-Fehler im letzten),
liefert aber noch NICHT die richtigen Tokens. Das Kohaerenz-Tor ist ROT und
das ist der Stand — kein "fast fertig".

#### Die geloesten Kontrakte

**Kontrakt 4 — der Kopf las die Args des ZIELS.** Kein fehlender Puffer,
sondern ein Zustaendigkeits-Bruch: `check_cuda_graph_backend` fragt die
AKTIVEN Args (der Bring-up-Scope publiziert `lane_args`, Graphen AN), waehrend
`GraphSharedOutput.create_for_model_runner` `model_runner.server_args` fragt
(die Args des Kopfes, Graphen AUS). `init_decode_cuda_graph` lief also an
seiner eigenen DISABLED-Schranke vorbei und dereferenzierte den None-Puffer.
Fix ist die Slice-C-Maschinerie selbst: der Bring-up des Kopfes laeuft in
einem eigenen `lane_scope` mit den Args DES KOPFES — dann stimmen beide
Fragen ueberein, Decode UND Prefill nehmen ihren Frueh-Ausstieg. Wichtig:
`init_prefill_cuda_graph` nimmt fuer eine Lane bewusst NICHT den
Draft-Skip, der Kopf haette sonst auch noch Prefill-Graphen aufgenommen.

**Kontrakt 5 — Typ ist kein eindeutiger Selektor.** Runde 1 stellte
`_find_lane_vocab_shells` von Attributpfad auf TYP um. Der Zielbaum meldet
aber `embed=2`: ein multimodales Ziel traegt mehr als eine Vokabular-Tabelle,
und `modules()` liefert sie in Registrierungsreihenfolge. "Die erste vom
richtigen Typ" war ein Muenzwurf, und er fiel auf die Begleit-Tuermchen-
Tabelle (Breite 1152) statt auf die Sprach-Tabelle (5120). Still beim
Bring-up, ein Forward spaeter ein Cutlass-Signatur-Dump aus
`pre_fc_norm_embedding`. Jetzt entscheidet die BREITE: der Kopf kann nur die
Tabelle benutzen, deren `embedding_dim` die Hidden-Groesse ist (er
konkateniert ihre Ausgabe mit Hidden-States vor `fc`). Kein Treffer oder zwei
Treffer = Bring-up-Fehler mit Satz, nicht geraten.

**Kontrakte 6 und 7 — eine Wurzel, zwei Boots.** Die Ausnahme fuer die Lane
war als `is_draft_worker and not is_dual_group_lane` geschrieben. Das ist auf
BEIDE Lane-Runner wahr. Gemeint war nur das ZIEL (Draft-Worker nur dem Namen
nach, faehrt das volle Modell); der KOPF ist ein Draft-Modell im gewoehnlichen
Sinn und muss die Ausnahme gerade NICHT nehmen. Folge: der Kopf erbte die
64-Schichten-Geometrie des Ziels.
* Kontrakt 6: sein Full-Attention-Aufruf landete im GDN-Backend
  (`assert isinstance(mixed_qkv, torch.Tensor)`), weil seine Schicht-Id nicht
  in `full_attention_layer_ids` steht.
* Kontrakt 7: danach hatte sein KV-Pool eine LEERE Full-Attention-Abbildung
  (`layer_id=0 not in full attention layers: dict_keys([])`).
Beide sind dieselbe Fehl-Einordnung an zwei Stellen. Deshalb ist die
Unterscheidung jetzt EINMAL benannt — `ModelRunner.is_dual_group_lane_target`
— und alle drei Stellen fragen sie (Attention-Registry, KV-Pool,
Prefill-Graph-Skip). Ein Test verbietet das Wieder-Ausbuchstabieren.

#### Zwei eigene Funde nebenbei

* **Die Kette war im SERIELLEN Modus gar nicht erreichbar.** Runde 1 hat
  `_spec_step` nur in den nebenlaeufigen Worker verdrahtet; `tick()` rief
  immer `_decode_step`. Seriell ist der Default — eine Lane mit
  `--dual-group-lane-spec` baute den Kopf und fragte ihn nie.
* **Der Slot-Leck-Verstaerker.** Ein abgebrochener Tick gab die Pool-Slots
  nicht zurueck. Bei `max_running_requests` 1 hat der Kopf genau einen Slot,
  also machte EIN echter Fehler jeden spaeteren Job an
  `alloc_req_slots ... available_size()=0` scheitern — ein Lebensdauer-Problem,
  das wie ein Dimensionierungsproblem aussieht. `drop_active()` gibt jetzt
  beide Batches frei.

#### Kontrakt 8 — teilweise geloest, und der Grund fuer das rote Tor

`_verify` las `out.next_token_logits` POSITIONSWEISE. Dieses Feld ist
`[#SEQUENZ, vocab]` — der Logits-Processor waehlt EINE Zeile je Request (die
letzte Extend-Position). `preds[0]` war damit die Fortsetzung des LETZTEN
Vorschlags statt der Vorhersage nach `cand[0]`.

Der Beweis stand in den Zahlen, bevor er im Code stand: Accept-Laenge exakt
1,000 ueber alle 63 Runden. n_accept war immer 0, also wurde `preds[1:]` nie
indiziert und die Form-Diskrepanz blieb stumm; das emittierte Token setzte
einen verworfenen 3-Token-Schwanz fort, daher die Wiederholungsschleife.

Halber Fix drin: `CaptureHiddenMode.FULL` liefert die Post-Norm-Hidden-States
JEDER Extend-Position, also sind die Kandidaten-Logits eine lm_head-Anwendung
entfernt (`_candidate_logits`, die rang-lokale Reduktion von
`_get_logits`; sie VERWEIGERT sich laut, falls der Processor ein
TP-All-Gather wollte). Wirkung gemessen: Accept-Laenge 1,000 -> **1,312**,
also wird jetzt tatsaechlich akzeptiert.

**OFFEN, und das ist der Uebergabepunkt:** die emittierten Tokens stimmen ab
Index 1 immer noch nicht. `output_ids[1]` ist gleich `output_ids[0]`, d.h.
`preds[0]` sagt nach `" long"` wieder `" long"` voraus statt `" before"`.
Das deutet auf einen Versatz zwischen den Hidden-State-ZEILEN und den
Kandidaten-POSITIONEN, nicht mehr auf die Logits-Quelle. Naechster Schritt
ist eine einmalige Form-/Positions-Instrumentierung in `_verify`
(`hidden_states.shape[0]` gegen `len(cand)`, dazu `extend_start_loc` und
`positions`) — NICHT weiter raten.

#### Tore

| Tor | Ergebnis |
|---|---|
| data_ptr Ziel | GRUEN, 1058 Identitaeten, in jedem Boot |
| data_ptr Kopf | GRUEN, 16 Identitaeten, in jedem Boot |
| Vokabular-Schale nach Breite | GRUEN, 5120 protokolliert |
| Kette laeuft ohne Abbruch | GRUEN (Boot 5: 0 Tick-Fehler, 48 Runden) |
| **Kohaerenz mit Spec** | **ROT, erste Abweichung an Index 1** |

Die Referenz ist unabhaengig bestaetigt: `[1248, 1518, 29496, 13, ...]`
detokenisiert zu `" long before electronics. Early devices such as the
abacus,"` und der VERBAND liefert auf denselben 96 Prompt-Tokens genau das.
Die Spec-Lane liefert `" long long Brennan, a!! !..."`.

#### Zahlen (Boot 5, seriell, Kopf EAGER) — Mechanik-Kosten, KEIN Feature-Ergebnis

| Groesse | Wert |
|---|---|
| Prefill (96 Tokens) | 149,07 ms |
| ms je Spec-RUNDE | 113,61 ms |
| Accept-Laenge | 1,312 |
| abgeleitet: ms je Token | 86,6 ms |
| Basis ohne Spec (dokumentiert) | 16,6 ms/Schritt mit Graphen, 61,7 ms eager |

Ehrlich gerechnet gegen die EAGER-Basis ist die Kette an diesem Arbeitspunkt
ein VERLUST (86,6 gegen 61,7 ms/Token), gegen die Graph-Basis ein grosser.
Zwei benannte Gruende, beide strukturell und beide noch nicht adressiert:
der Kopf laeuft eager (der Graph-Weg braucht den EAGLE-Draft-Graph-Runner),
und der Verify ist ein EXTEND-Forward des Ziels, waehrend die No-Spec-Basis
ein graph-gefangener DECODE ist. Diese Zahlen sind ausserdem an einer
NICHT-kohaerenten Kette erhoben und taugen nur als Groessenordnung der
Mechanik-Kosten.

**Nebenlaeufiger Messpunkt (Verband + Lane gleichzeitig) NICHT erhoben** —
bewusst. Eine Kartenaequivalent-Zahl an einer nachweislich falsch
rechnenden Kette waere eine Zahl, die niemand benutzen darf
(Einzelteil-vor-Verbund).

### 11.16 Lane-Spec-Kette Runde 3 (feat/dual-group-lane-spec-r3, Basis 5ee0b58810)

Nachtrag: Runde 3 lief, bevor diese Datei im Repo lag, und ist deshalb
zunaechst nur in `INTEGRATION_R3_VALIDATION.md` protokolliert worden. Die
beiden Befunde, auf denen Runde 4 aufsetzt, gehoeren hierher:

**Die Wurzel ist der REKURRENTE ZUSTAND, nicht die Indizierung.** Die
uebergebene Hypothese (Versatz zwischen Hidden-State-Zeilen und
Kandidaten-Positionen) wurde in einem Boot widerlegt: `hidden_rows=4` gegen
`len(cand)=4`, `positions=[96..99]`, `extend_prefix_lens=[96]`, vier frische
`out_cache_loc`-Slots — alle Achsen stimmen. Das Ziel ist ein GDN-Hybrid und
fuehrt neben der KV einen laufenden Zustand (Conv-Fenster + SSM). Ein Verify
ueber K+1 Kandidaten in EINEM fortgesetzten Extend schiebt diesen Zustand
ueber jeden Kandidaten weiter, angenommen oder nicht; es gibt keinen Slot zum
Freigeben. Ab der ersten Ablehnung sagt die KV "n angenommen" und der Zustand
"alle K+1".

**Der Falsifikator statt des Arguments:** der Verify konsumiert die Kandidaten
als EINZELNE DECODES (`_verify_by_decode`). Gleiche Accept-Regel, gleiche
emittierte Tokens, nur der bestrittene Forward getauscht. Ergebnis:
byte-identisch zur eigenen No-Spec-Lane, Accept 1,100 -> 1,383. Das ist eine
KORREKTHEITS-BRUECKE, kein fertiges Feature: sie kostet strukturell einen
Forward je emittiertem Token.

**Harness-Befund, der jedes spaetere Urteil bindet:** das Tor hatte nie einen
RAUSCHBODEN. Der Boden ist INHALTS-, nicht positionsgetrieben — zwei
No-Spec-Laeufe derselben Anfrage weichen bei einem offenen Fortsetzungstext
schon nach wenigen Tokens voneinander ab. Jedes Kohaerenz-Urteil ohne
mitgefuehrten A-gegen-A-Boden ist kein Urteil.

### 11.17 Lane-Spec-Kette Runde 4 (feat/dual-group-lane-spec-r4, Basis 640e4d7085)

Auftrag war der Umbau des Lane-Verifys von der seqdecode-Bruecke auf einen
echten `ForwardMode.TARGET_VERIFY`. Der Umbau ist gebaut, und er ist NICHT
uebernommen worden — die Grenze ist gemessen, nicht geschaetzt.

#### Wie der TARGET_VERIFY-Input gebaut wird

`build_lane_chain_verify_input` erzeugt einen `EagleVerifyInput`, der die
Kette der Lane als Verify-BAUM beschreibt: `draft_token` (die K+1
Kandidaten), `positions` (`n_cached .. n_cached+K`), `custom_mask` im
FULL_MASK-Layout, `draft_token_num` (die Schrittweite, mit der das
GDN-Backend den Verify je Request zerlegt und mit der die Zwischenspeicher
indiziert werden), `topk=1`, `spec_steps=K`, sowie die `retrieve_*`-Felder,
die die Kette topologisch beschreiben (das GDN-Backend liest sie nur bei
`topk > 1`, die Tree-Verify-Sampling-Kernel liegen nicht auf diesem Pfad —
die Accept-Regel der Lane ist ihre eigene). Die Maske wird von Hand gebaut
statt aus `build_tree_kernel_efficient` geholt, weil dieser Kernel
`parent_list`/`top_scores_index` eines EAGLE-Drafts braucht, die die
handgerollte Kette nie erzeugt; das Layout ist der Vertrag und steht als
solcher im Docstring plus CPU-Test. Eingehaengt wird in `_verify`: K+1
frische KV-Slots werden VOR dem Forward alloziert und in
`req_to_token[idx, n_cached : n_cached+D]` veroeffentlicht (der
Attention-Plan liest genau diese Zeile), dann `forward_mode`,
`spec_info`, `input_ids`, `out_cache_loc`, `seq_lens_sum` und
`capture_hidden_mode` gesetzt; nach dem Forward committet
`update_mamba_state_after_mtp_verify` den Zustand des letzten ANGENOMMENEN
Schritts, der Rest der Kandidaten-KV wird freigegeben, und die Buchfuehrung,
die sonst `prepare_for_decode` macht (`seq_lens`, `orig_seq_lens`,
`kv_committed_len`), wird explizit nachgezogen.

#### Was bewiesen ist und was nicht — der Accept-Cap-Falsifikator

Das Kohaerenz-Tor lief mit gemessenem A-gegen-A-Boden auf drei Prompts, die
VORHER nach ihrem Boden ausgewaehlt wurden (der Boden ist inhaltsgetrieben:
von vier Kandidaten hatten drei einen gruenen Boden ueber 12 Tokens, einer
nicht — die Prompt-Wahl ist Teil des Instruments und keine Nebensache).

| Arm | alphabet | squares | repeat |
|---|---|---|---|
| Boden No-Spec gegen No-Spec | gruen | gruen | gruen |
| Bruecke `seqdecode` gegen No-Spec | gruen | gruen | gruen |
| `target_verify`, Accept auf 0 gedeckelt | **gruen** | **gruen** | **gruen** |
| `target_verify`, ungedeckelt | rot @1 | rot @5 | rot @5 |
| Falsifikator `extend` | rot | rot | rot |

Der gedeckelte Arm ist ausserdem lauf-zu-lauf reproduzierbar, der
ungedeckelte nicht. Damit ist die Grenze scharf: mit Deckel wird nur Zeile 0
emittiert und nur Schritt 0 committet — also sind der Verify-INPUT (Maske,
Positionen, Kandidaten-Slots, `spec_info`) und der Zustands-Commit fuer den
Ein-Zeilen-Fall RICHTIG. Ohne Deckel ist es rot, und die Runden-Spur sagt wo:
Zeile 0 von `preds` folgt der No-Spec-Fortsetzung, die Zeilen >= 1 tun es
nicht und sind kaum eingabeabhaengig (ueber 96 Runden nahm Zeile 2 nur 15
verschiedene Werte an, einer davon in 39 Runden, gegen 83 verschiedene Werte
auf Zeile 1). Nicht die Wahl des Modus ist offen, sondern die Verkettung
UEBER die Draft-Schritte INNERHALB eines Forwards.

Deshalb bleibt `seqdecode` der Default. Ein Default ist eine Aussage ueber
Korrektheit, nicht ueber Ambition.

#### Der Befund, der die Prioritaet von Runde 5 aendert

Der Umbau war als Kostenfix gedacht — und die Messung zeigt, dass der
VERIFY-MODUS nie der teure Posten war:

| Arm | ms je Runde | Accept | abgeleitet ms/Token |
|---|---|---|---|
| No-Spec, graph-gefangen | 16,17 (je Schritt) | — | **16,17** |
| `target_verify` (1 Forward/Runde) | 67,65 | 1,189 | 56,90 |
| `seqdecode`-Bruecke | 68,95 | 1,105 | 62,40 |
| `extend` (falsch) | 90,94 | 1,103 | 82,44 |

Ein TARGET_VERIFY-Forward kostet ~67 ms gegen einen graph-gefangenen Decode
von 16,17 ms. Bei der gemessenen Accept-Laenge von ~1,19 spart der Wechsel
von der Bruecke auf den Ein-Forward-Verify also bestenfalls ~10 % — der
Faktor-3,5-Verlust gegen die Graph-Basis bleibt praktisch unveraendert
stehen. Um ihn ueber den Wechsel des Verify-Modus zu schliessen, muesste die
Accept-Laenge ueber ~4,2 liegen, was eine Kette mit K=3 nicht kann.

**Die Konsequenz fuer Runde 5, ausdruecklich:** der Hebel ist die
GRAPH-AUFNAHME des Verifys (und des Kopfes), nicht die Wahl des
Verify-Modus. Die Lane nimmt heute `ForwardMode.DECODE` auf, weil ihre
Args-Sicht `speculative_algorithm=None` setzt; ihr `capture_hidden_mode`
passt zudem nicht, was den Verify auch dann aus dem gefangenen Graphen
draengen wuerde. Beides ist benannt und keins davon ist in Runde 4 angefasst
worden — ein Umbau-Marathon war ausdruecklich nicht der Auftrag.

#### Zwei Nebenbefunde, die zur Buchfuehrung gehoeren

* **Die ms/Runde der Runden 1-3 waren zu klein.** Gemessen wurde nur der
  Verify; die K Draft-Forwards des Kopfes — die andere Haelfte dessen, was
  Spekulation kostet — standen in keiner Zahl. `_spec_step` misst jetzt die
  ganze Runde und meldet `verify_ms_mean` und `propose_ms_mean` getrennt,
  damit die beiden strukturellen Posten unterscheidbar bleiben.
* **Der Vergleich No-Spec gegen Spec traegt zwei Unterschiede, nicht einen.**
  Beide Spec-Arme teilen sich den Spec-PREFILL, der wegen
  `CaptureHiddenMode.FULL` eager laeuft, waehrend der No-Spec-Prefill
  graph-gefangen ist. Auf 24 Tokens verlassen `seqdecode` UND der gedeckelte
  `target_verify` die No-Spec-Bahn an derselben Stelle (Index 16) und stimmen
  untereinander weiter ueberein — der Rest ist dieser Prefill-Unterschied und
  nicht der Verify. Der nutzbare Horizont des Tors ist damit eine Eigenschaft
  des Prefills, nicht des Verifys, und das gehoert vor jedes weitere
  Kohaerenz-Urteil.

#### Der benannte erste Schritt von Runde 5 (Lokalisierer, ein Boot)

Offen ist genau eine Frage: liegen die falschen Zeilen >= 1 an der
VOLL-ATTENTION (der handgebauten `custom_mask` / dem `qo_indptr`-Pfad des
Verify-Wrappers) oder am GDN (der Verkettung ueber die Draft-Schritte in
`causal_conv1d_update` + Verify-Kernel)? Beide Teile sind im VERBAND korrekt,
also ist Lesen wenig wert — der Falsifikator ist billiger als die Analyse:

Der Verify-Forward ist zustandsREIN bis zum Commit (der SSM-Kernel laeuft mit
`disable_state_update=True`, das Conv-Fenster wird erst von
`fused_conv_window_scatter_with_mask` zurueckgeschrieben). Also: Conv-/SSM-
Zustand des Slots einmal sichern, den TARGET_VERIFY-Forward laufen lassen und
`preds` merken, Zustand zurueckspielen und die Kandidaten als EINZELNE DECODES
durchschieben (`_verify_by_decode` rechnet genau die Wahrheit fuer jede
Zeile). Zeile fuer Zeile vergleichen. Stimmt Zeile 1 ueberein und Zeile 2
nicht, ist es die Rekurrenz; sind alle Zeilen >= 1 gleichzeitig falsch, ist es
die Maske. Danach erst bauen. Der Deckel `tv_max_accept` bleibt als
Dauer-Falsifikator liegen.

Und unabhaengig davon, nach der Messlage die groessere Sache: die
Graph-Aufnahme des Verifys. Solange ein Verify-Forward 67 ms gegen 16 ms
gefangenen Decode kostet, entscheidet nicht der Verify-Modus ueber Gewinn
oder Verlust der Spekulation auf der Lane.
