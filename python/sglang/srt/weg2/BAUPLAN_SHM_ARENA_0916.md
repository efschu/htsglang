
## Stufe 4 (#1427, 16.09. abends): die Schreibseite ist gemappt

Stufe 3 hatte nur die Leseseite in place gebracht; Schreiben lief weiter
Karte -> 1-GB-Staging-Ring -> Store-Thread -> Arena -> Umbindung, und die
Mamba-Zustaende durch einen 13-Slot-Host-Anker-Pool. Der Park-Test (6 x 100k)
starb an genau diesem Transit (xsn184-187).

* KV/Draft (`pool_host/arena_pool.py`): `write_backup` beansprucht je Seiten-
  Hash einen Arena-Slot (`arena_claim`: frisch / einem frueheren Shard
  beitreten / schon COMPLETE), der Backup-Thread DMA't die K- und V-Extents
  dieses Rangs mit dem All-Layer-Kernel direkt in den Slot (Arena-Datenregion
  als Host-Puffer, Seitenschritt), der Ack merged die Extents in die
  Abdeckung (`arena_complete`, COMPLETE sobald jeder Layer-Shard drin ist),
  nimmt die Leser-Referenz und markiert den Knoten store-praesent. Kein
  Staging, kein Ring, kein Store-Write, keine Umbindung.
* Mamba (`pool_host/arena_mamba_pool.py`): Ansichten je lokalem Layer
  (temporal) und je Conv-Segment (q|k|v) auf die Mamba-Arena, exakt der Schnitt
  aus `temporal_extents`/`conv_extents`; Backup = Index-Kopie Karte -> Ansicht,
  Load = Rueckweg, Prefetch loest COMPLETE-Blobs in place auf
  (`arena_resolve_reads`, Haken in `_batch_io_v2`). Fensterteile kommen aus
  `_canonical_mamba_window`.
* Refs auf CLAIMED-Slots sind erlaubt (der Knoten des Schreibers haelt eine ab
  dem Claim); `free_slots` bumpt die Generation, eine spaete Vervollstaendigung
  wird abgewiesen.
* Noch Transit: nichts fuer KV/Draft/Mamba; Sidecar-Pools (SWA/Indexer) gibt es
  auf diesem Modell nicht.

### Stufe 4, Metall-Lehren (xsn188-192, 16.09. abends)

* Bindung VOR der Pruefung (#1427e): `pool.arena` ist erst nach `ensure_bound`
  gesetzt; wer erst schaut und dann bindet, laesst jeden Rang ohne Prefetch
  (PP1/PP2 lesen den Store nie) auf dem Staging-Ring. Gleiches fuer den
  Draft-Pool (#1427g, bindet sonst erst beim ersten Draft-Lesen, das P nie tut)
  und den Mamba-Pool der neu gebauten D-Gruppe (#1427c, Fensterteile nach dem
  Rebind an den AKTUELLEN Pool).
* Ein ungebundener Pool gibt keine Platzhalter aus und schreibt keine fremden
  Arena-Ids in seinen Staging-Puffer (#1427c/#1427g).
* Nicht in der Arena = Miss, nie Platten-Lesen in einen Platzhalter (#1427d);
  ein Raise im Storage-Thread liess die Anfrage ewig auf
  `pool_transfers_done` warten (#1033e: Flag bei jedem Ausgang).
* `op_fn is self._read_page` vergleicht zwei frische Bound-Method-Objekte und
  ist immer False (#1427e) -- per Name vergleichen.
* KV-Legs des Front brauchen denselben Epoch-Retry wie die Gewichts-Legs
  (#1428); ein abgerissener Keep-alive hat sonst den Boot beim ersten Wake
  gekillt, obwohl P 200 antwortete.

## Nachtrag 16.09. spät: Layout-Leiter je Größenklasse, YaRN je Klasse (Design, Nutzer-Order)

**Ist (nach #1447):** Der P-Cut-Solver preist jeden Frontier-Kandidaten mit seinem
Host-Bounce (`PP-CUT HOST-PRICE`), der Cap-Floor (262144 + Chunk) ist auf dem
Austausch-Arm Standard, und es schifft der schnellste fundierbare Schnitt. Fundierbar
ist heute nur 39,13,12, weil nur dessen Lane-Satz vermessen ist (5 Lanes, 15,75 GiB);
jeder unvermessene Schnitt wird mit dem Worst-Case (9 Lanes, 27,75 GiB) bepreist und
fällt am Ledger (Excess −1,3 GiB). Das `WEG2-XCHG-HOST-SLOT`-Instrument, aus dem die
Lanes gelernt werden, liefert seit Ring-off keine Zeilen mehr (0 in xsn203–208) —
das ist die eine Wand vor der Leiter: **erst Lane-Messung wieder verdrahten, dann
kann ein zweiter Schnitt überhaupt vermessen werden** (Boot mit
`SGLANG_WEG2_PCUT_BOUNCE_SLACK_GIB` und Ledger-Spielraum, danach ist er in
`XCHG_LANES_BY_CUT` bzw. dem Record).

**Leiter (Design):** Größenklassen nach geschätzter Prompt-Länge des Frontends
(`est_prompt`, schon vorhanden): ≤32k, ≤64k, ≤128k, ≤262k. Je Klasse der schnellste
Schnitt der Frontier, dessen Pool `Klasse + Chunk` hält (bs1: der Pool muss nur den
einen Prefill halten). Der Solver liefert die ganze Frontier bereits je Boot; die
Leiter ist eine Tabelle `Klasse → (Schnitt, Pool, ms/Chunk)` aus derselben Rechnung.

**Re-Cut zur Laufzeit:** ein P-interner Layout-Flip (PP-Stufen tauschen Layer über
dieselben Host-Bounce-Lanes wie der P↔D-Austausch, kein D-Beteiligter), ausgelöst vom
Frontend, wenn die nächste Batch-Klasse einen anderen Schnitt verlangt als der
installierte; Kosten ~1 Flip (1,6–2,0 s). Nur wenn der Backlog der Klasse den Flip
amortisiert (dieselbe Break-even-Regel wie `FLIP-ECONOMICS`).

**YaRN je Klasse:** RoPE-Skalierung ist Ladezeit-Konfiguration des Modells; ein
Faktorwechsel zur Laufzeit hieße Neu-Laden. Deshalb: Faktor pro Boot fest
(`--json-model-override-args rope_scaling`), Klassen >262k (Faktor 2/3/4 → 524k/786k/
1M) nur mit einem Boot dieses Faktors; D-Pool 697856 hält ~2,6×262k, P braucht dafür
die Pool-lastigen Schnitte (33,18,13 ff.). Automatik = Boot-Wahl, nicht Laufzeit-Wechsel.
