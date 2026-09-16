
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
