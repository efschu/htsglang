# Catalog glossary DE -> EN (part 3: catalog texts)

Same term list as the dashboard glossary of parts 1/2 (card, rank, run, boot, dry run, verdict, refusal/refused, forceable,
dependency, edge, operating mode, hardware/model/server profile, planner, proposal, unverified, verified, borrowed, datasheet,
weights, experts, seats, context, reserve, bar, overflow, cut, value, row, switch, explanation, reason, note, path, driver,
tree, state, fit), plus the catalog-specific terms below. Machine-readable names stay as they are: flags, env names,
refusal/register codes (W40, HW-COUNT ...), log line wordings quoted from the source, `satz_quelle`, `evidence.anchor`
(verbatim source quotes), and the key vocabulary of the data (`rel`, `status`, `level`, `kind`).

| German | English |
|---|---|
| Aufteilung (group) | Split |
| Speicher (group) | Memory |
| Kontext (group) | Context |
| Fehlersuche (group) | Debugging |
| Weitere (group) | Other |
| Diagnose (group, text prefix) | Diagnostics |
| Watchers (group) | Watchdogs |
| Profil (group) | Profile |
| Cut (P-cut, layer cut) | cut (layer cut) |
| Stufe | stage (pipeline) / rung (ladder, share controller) |
| Gruppe P / D | group P / D |
| Seat, seats | seat, seats |
| Schlafrest | sleep remainder |
| Guard | latch (the `--host-riegel-gib` flag name is unchanged) |
| Anchor (Mamba anchor) | anchor (Mamba anchor; glossary key "Mamba anchor") |
| Accepted / prepared / planned | accepted / prepared / planned |
| Clamp (starvation clamp) | clamp (starvation clamp) |
| Envelope | envelope |
| Evidence / occupied / unoccupied | citation / verified / unverified |
| Path (path, code path) | path |
| Zustand | state |
| Bahn (Gewichtsbahn) | lane (weight lane) |

Display wording of the key vocabulary (shipped in `catalog.json` under `anzeige`, source `profile_catalog.REL_ANZEIGE` / `STATUS_ANZEIGE`):

| key | display |
|---|---|
| tauscht | trades |
| braucht | requires |
| schliesst_aus | excludes |
| abgeleitet_von | derived_from |
| skaliert_mit | scales_with |
| kuratiert | curated |
| erklaert | explained |
| geerntet | harvested |
| unerklaert | unexplained |
