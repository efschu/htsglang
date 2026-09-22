# #113 — der Erstboot ohne ein einziges Byte von Platte auf D

**Nutzer-Order 22.09.:** „kein boot bei dem D irgendwas von platte lädt. muss
es nicht" / „es ist ja egal ob da P noch lebt, die bytes liegen ja immernoch
im vram" / „die dürfen nur nicht ‚zerstört' werden" / „dazu gibts ja die karte
wo was liegt".

## Die zwei Zeilen, an denen die Bytes heute sterben

`python/sglang/srt/weg2/tms_csrc/core.cpp:69-70`

```cpp
CURESULT_CHECK(cuMemUnmap((CUdeviceptr) ptr, metadata.size));  // VA loslassen
CURESULT_CHECK(cuMemRelease(metadata.allocHandle));            // ZERSTOERT die Seiten
```

`cuMemUnmap` gibt die virtuelle Adresse her — das ist genau das, was P tun
MUSS, damit D den Platz bekommt. `cuMemRelease` gibt das PHYSISCHE Handle
frei, und damit sind die Bytes weg. Nur die zweite Zeile ist das Problem.

## Die Zeitfolge, gemessen (fnFL2w54)

```
08:16:58  P fertig geladen (PP0, 110,16 s)
08:17:08  P pausiert: WEG2-SLEEP-TAG-TIME tag=weights_0 deposit_ms=2 pause
08:17:19  D-Prozess startet
08:17:40  D: Load weight begin, avail mem=29,15 GB   <- P hat alles hergegeben
```

`deposit_ms=2` fuer 27 GB ist eine BUCHUNG, kein Kopieren. Es wird nichts
hinterlegt, was D lesen koennte.

## Warum die naheliegenden Wege nicht gehen

| Weg | Preis | |
|---|---|---|
| P bleibt wach, D laedt daneben | 27080 + 27000 > 32607 MiB auf der 5090 | nein |
| P spillt seine 188 residenten in den Store | +21,3 GiB shmem | moeglich, aber eine KOPIE |
| P legt sein Bild ins Host-Backup | +38,6 GiB (`--weg2-weights-cpu-backup off` verhindert es bewusst) | nein |
| D laedt die 188 selbst von Platte | 37 % Plattenlast | **vom Nutzer verworfen** |
| **Handle ueberlebt, D mappt es** | **null Bytes bewegt** | **das hier** |

## KORREKTUR NACH DEM PRIOR-ART-GATE (devindex, 22.09.)

**Der Mechanismus existiert bereits und ist am Metall bewiesen.** Commit
`500c984795` (21.09.), "union arena slice 3b: ONE weight image per card,
both phase groups run off it":

> Owner packs its checkpoint tensors into the exportable VMM arena and
> publishes; peer rebinds every tensor it can PROVE identical (name+shape+
> stride+dtype+content checksum) and drops its own copy, keeping what it
> cannot prove. **METAL-PROVEN on the 5090: peer's parameters land at the
> owner's arena base, values bit-identical, card freed +1.50 GiB of a 1.50
> GiB shared set.**

Drei Module (`union_arena.py`, `union_arena_bind.py`, `union_arena_vmm.py`),
verdrahtet in `model_runner.py:2953` und `:2970`. Damit ist #113 NICHT ein
neuer Mechanismus, sondern eine EINZIGE fehlende Eigenschaft:

`tms_csrc/utils.h:182 cu_mem_create` legt jede TMS-Allokation ohne
`prop.requestedHandleTypes` an. Ohne
`CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR` kann
`cuMemExportToShareableHandle` auf ihr per Definition nicht gelingen
(`understand_prior-art.md` Sec. 5, gemessen: `grep
FILE_DESCRIPTOR|ExportToShareable tms_csrc/` -> 0 Treffer). Deshalb erreicht
die Union-Arena heute nur ihre EIGENE Arena (1,50 GiB) und nicht die ~27 GB,
die TMS haelt.

GEBAUT (diese Sitzung): der exportierbare Typ hinter
`SGLANG_WEG2_VMM_EXPORTABLE=1`, mit Rueckfall auf die alte Form, wenn der
Treiber ihn verweigert -- und der Schreiber im Launcher, der die Env an
BEIDE Gruppen gibt. Tests binden beide Seiten der Naht.

## Der Weg

1. **P laedt** wie heute und fuellt den geteilten Store mit seinen KALTEN
   Experten (324 von 512, #107/#109 — steht und ist am Metall belegt).
2. **P pausiert unter Adoption ANDERS:** `cuMemUnmap` ja, `cuMemRelease`
   NEIN. Das Handle wandert in eine Export-Tabelle, geschluesselt nach dem,
   was die Karte fuehrt: (Gruppe, Rang, Layer, Attribut, globale Experten-Id
   -> Handle + Offset).
3. **D laedt mit `--load-format dummy`** (#108/3, am Metall belegt: w53/w54
   zeigen `load_format='dummy'`), also ohne Platte, und OHNE Repack auf den
   Zufalls-Indizes (#112 — das war die Wurzel von w53 UND w54).
4. **D mappt** die exportierten Handles per `cuMemMap` an seine eigene VA.
   Die Mechanik ueber die Prozessgrenze existiert im Fork:
   `cuMemExportToShareableHandle` -> fd -> SCM_RIGHTS ->
   `cuMemImportFromShareableHandle` (die BAR1-Lanes, Task #32).
5. **Was D an anderer Stelle braucht als P es haelt**, geht ueber die Legs
   (BAR1, 13-14 GB/s x8) — das ist der normale Flip-Pfad, der steht.
6. **Die Export-Tabelle wird nach der Adoption geleert**, sonst haelt sie die
   Seiten fuer immer. Ein Rang, der exportiert und nie freigibt, ist ein Leck
   mit Ansage.

## Was die Karte dafuer fuehren muss (sie fuehrt es fast)

`expert_map.build()` liefert heute je Phase `resident` (je Rang die globalen
Ids) und `slot_of` (globale Id -> Store-Platz). Fuer #113 fehlt die dritte
Spalte: **Handle + Offset je residenter Id**, gefuellt von P beim Pausieren,
gelesen von D beim Mappen. Das ist dieselbe Karte, eine Spalte breiter —
nicht eine zweite Karte (Nutzer-Gesetz: „alles was geshardet wird braucht ne
karte", und zwar EINE).

## Reihenfolge und Riegel

* Der Export darf NUR unter armierter Adoption passieren. Ohne sie bleibt
  `cuMemRelease` genau da, wo es ist — byte-identisch zu heute.
* Ein Handle ohne Abnehmer MUSS freigegeben werden. Der Riegel dafuer gehoert
  an dieselbe Stelle wie `adopt.mark_adopted`: deckt der Import nicht alles,
  was die Karte nennt, wird nichts geglaubt und alles freigegeben.
* Der Test, der zaehlt, ist nicht „der Export lief", sondern: D's Tensor
  traegt nach dem Mappen DIESELBEN Bytes wie P's — an einer Stichprobe aus
  der Karte, nicht an der Absicht.

## Stand

Gebaut und committet (b60dd13f75 und davor): #108/3, #109, #110, #111, #112,
W100. Offen: dieses Dokument, Punkte 2/4/6 sowie #68a/#68b (Praedikat-
Gleichheit und paralleles `post_load`).
