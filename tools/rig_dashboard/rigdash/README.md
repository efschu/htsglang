# rigdash — maintenance notes

## Time-series DB, tabs, expert view (user 01.10., concept `/spinning/gpu-arb/docs/DASHBOARD-REDESIGN-1001.md`)

**VictoriaMetrics** (order 01.10. ~07:40Z) is the time-series DB of the rig. It runs on CT999 at `:8428`, together with
`node_exporter` (CT999 and Proxmox host 192.168.0.11:9100) and `nvidia_gpu_exporter` (NVML). Grafana runs next to it
at `:3000` with the board "Rig – history". It is installed with `deploy/vm/install_vm.sh` and
`deploy/grafana/install_grafana.sh`, both idempotent. Units and `scrape.yml` are in the repo.

- The sampler writes the IPC to VM every 5 s (`vmpush.Bridge`): `pdflip_front_*`, `pdflip_rank_*` and
  `pdflip_flip_*`, the flips at the time of the flip. Labels are `model`, `boot`, `group`, `rank`, `dir`, `kind`. **Never a rid as a label.**
- rigdash reads via PromQL (`vmpush.VmClient`):
  - the TTFT tiles (`/api/live` → `vm`),
  - the TTFT history (`/api/history` → `ttft`).
- New quantities go to VM as a metric, not as a new column in `history.sqlite` and never from a log.
- The Grafana board is changed in `deploy/grafana/make_dashboard.py` and then with `install_grafana.sh`.
  It cannot be changed in the Grafana UI (`allowUiUpdates: false`).

**`/api/live` once per second for everyone** (`server.LiveCache`):

- `?lean=1` returns boots that are only a table row, without curves.
- `?dev=0` leaves out features and image changes.
- The response is gzip-compressed.
- A zoom request is calculated on its own.

**Page:**

- Tabs: Overview (default), Boots, History, Cards, Development (rig edition only).
- Only the visible tab is drawn. A section with unchanged HTML is not touched (`put()`).
- The **expert view** (`body.expert`) shows everything with class `x`, all kv rows of the tiles and the source badges. The standard view shows only kv rows with class `keep`.
- **Rule for new measured values:** they enter the expert view with `x` (or without `keep`), never the standard view unasked (memory `dashboard-redesign-ttft-tabs-expert-1001`).
- The overview shows the key figure row per model: TTFT, decode, prefill P/D, flip time, queue, deaths/hangs.

**Keeping the feature list current:** the tab "Development" shows the state of `features.json`. It turns yellow after 12 h or
if a boot happened after that, red after 24 h. Below it are the **commits of the image line (48 h) without a building block**,
calculated from git (`features.new_commits`). The counter is in the tab title. Enter them with `features_update.py` as below.

## Flip time (ONE definition, user 06.10.2026)

P→D = last P chunk done → first decode token produced; D→P = last decode token produced → first
prefill chunk starts computing (first forward on PP0, `flip_user_time.prefill_start_source=pp_first_forward`,
never the leg-1 dispatch). Exception: no flip counts if no prefill or decode is pending (idle flip,
D→P: `flip_user_time.idle_flip`). Layer swap, lead-in and lead-out are only the BREAKDOWN of the one number, never a flip
time of their own. There is ONE calculation, `flipzeit.py`: `ipcboot.flip_views` measures per flip, the history writes every
counted flip once as the mark `flip_t2t` (open, provisional, without end point, idle: never), and every number on the page
(overview tile, history tile, chart, boot list) is `flipzeit.tile` over these marks. Only the WINDOW
differs and is stated in the caption: overview = last 60 min of the model, history = selected
range/zoom, boot list = whole boot. Tests: `tests/test_flipzeit_1006.py`.

**Phase bar with real work time (WACH-OHNE-ARBEIT-0929):** the rank rows arrive too late
(P after the pipeline pass, D one pass later). P work is therefore drawn from
`TIMING-FLUSH-WAIT t_unix_ms` at the head of the forward to + `FWD-TIMING-PREFILL total_ms` (paired
via #new-token and gpu-ms; the timing line comes only at the head of the next forward and then corrects
retroactively), D extends from `HOST-ANON-PASS phase=EXTEND wall_ms` (discarded below 50 ms).
Without an anchor the old drawing stays. Between `PDFLIP-FLIP done woke=D` and the first TP0
decode token the bar shows **flip lead-out D** instead of "awake, no work", split by
`PDFLIP-POST-WAKE-PASS n=0` (readings before pass 0, prepare = park resume, run = re-extend).
Applies to NF and 27B.

**TODO (27B review 29.09.):** this too is a pure display from log lines. Switch to
`events.jsonl` as soon as the front writes flip and pass events there (FLIP done, first decode, POST-WAKE-PASS,
forward start/end per rank); then the FWD/FLUSH/ANON pairing and the log scan are dropped.

## Deploy line (order 29.09.): `desk/dashboard-ipc-0929`

rigdash is deployed only from the line `desk/dashboard-ipc-0929`. Whoever changes it (features seat, dashboard seat, 27B) builds on it or takes over the line fast-forward and pushes there.

`deploy/install.sh` refuses with rc 3 if one of the two conditions is not met:
1. The revision is a descendant of the line tip on origin.
2. The revision contains the running release (`/opt/rigdash/current`).

A deliberate step back works only with `RIGDASH_DEPLOY_ROLLBACK=1`, the reason belongs in the decision log. Check only, without changing anything: `deploy/install.sh --check <rev>`. `RIGDASH_REPO=<worktree>` selects the repo if the script lies outside a checkout.

Occasion: the features seat deployed from `desk/dashboard-features-0929`. A deploy from there would have silently removed stage 1 of DASHBOARD-AUS-IPC again.

## Work at the time it happened (user 30.09. ~16:45Z: "all the time 16k/s prefill")

`activity.py` puts every piece of work at its real time. A prefill chunk runs via `prefill.last {t, gpu_ms}`
(PP0 start to end on the last PP stage), decode is Δ`decode.tokens` between two rank clocks without flip/extend windows,
flips come from the events. Reason: the cumulative counter jumps by a whole chunk; Δ counter / Δ sample gave 16,384 tok/s
and P/D overlap. Tiles: "rate of running burst", "best burst rate", "rate of bursts 60 s" (each tokens / wall clock of the burst, P/D), decode only over steady samples, per stream
only with decode before and after. History: 1-s model rows under the prefix `mi.`; the old `m.` rows from the wrong instrument
are no longer shown. Flip time: see the section "Flip time" above (one definition, `flipzeit.py`).
Audit of all values: `/spinning/gpu-arb/docs/DASHBOARD-PLAUSI-AUDIT-0930.md`; tests `tests/test_activity_0930.py`.

### D prefill: admit extends are not a prefill rate (order 880, user 03.10. "6 token/s prefill in D???")

On D every request taken over from P begins with an extend of mostly **1 new token** (rest from the cache). y8vb `D.log`:
3029 of 3051 `Prefill batch` lines have `#new-token: 1`; the stock value `input throughput (token/s)` of such a line is
1 token / wall clock since the last line = 5.8 tok/s. In the dashboard the number was not shown as a stock value, but the same error was in the
D key figures: the 1-token chunks (spacing < 1.5 s) join into a "burst" over minutes (tokens / wall clock = a few tok/s) and
the tile "D prefill since boot" divided all D tokens by boot wall clock or by all compute time. Now (`activity.WIDE_MIN_TOK = 64`):

* `activity.chunks(..., kind="wide"|"admit")` separates BEFORE the burst is formed; `Model.dwide` / `Model.dadmit`. The phase bar keeps all chunks.
* D rate (tile "D prefill", KPI, curve `D_prefill_rate`, phase segment) only from chunks with at least 64 new tokens; without such a chunk "–".
* Named separately: "D admit/extend (tokens per request)" = number of extends, tokens, avg tokens per request (`prefill.D.admit`, segment `admit_n/admit_tok`).
* "D prefill + admit since boot": the sum of all D tokens stays, the rates (GPU time and wall time) apply to the chunks ≥ 64 tokens in the 16-min ring
  (`totals.d_split`), because the cumulative counters carry no chunk width from before the ring.
* Log path (`live.py`): `wall_confounded_tps` and the D rates no longer include lines with < 64 tokens, `prefill.D.now.admit` counts them.
* Limit: the separation happens per 1-s sample step (avg new tokens per chunk in the step); an admit in the same step as a wide chunk counts with the wide one (±1 token).
* P is not affected (chunks of 1–16k tokens, every line counts).

## Smooth and honest, seats, zoom (user 30.09. ~21:05Z / ~21:10Z)

"the decode throughput … not continuous but extremely jumpy". Measured on the service (NF y5i): two causes.
1. **Sample slip.** The 1-s sampler takes longer than 1 s under load. About 28 % of the 1-s rows had no sample of their own
   and were written as a gap. Now a second counts as observed if it lies between two samples that are at most
   3 s apart (`Model.coverage`). The work in it is fixed by the rank clocks. Levels (KV) hold the last sample,
   NVML series in the history at most 2 s. The sum of power applies only over all cards, never over a part of them.
2. **Wall-clock denominator.** A 5-s bucket with 2 s of D extend showed half the decode throughput. Now the rate
   *while* the phase was computing is drawn (`*_rate` = tokens / `*_busy`). Decode time: steady samples in full; start and end of a stretch
   by Δ`decode.gpu_ms` × measured wall/GPU ratio. A bucket without this work is a gap. The wall-clock rate stays
   for tok/s/W, energy and token counting and is attached in the history as a hidden series.
   The history holds the shares (`dec_busy`, `p_busy`, `d_busy`, `dec_seat`) per 1-s row. Division happens only per shown
   bucket, so it is right at every level. `dec_bs_min/_max` are combined by MIN/MAX.

**Seats** (addendum 21:10Z): batch size of the decode rounds from Δ`decode.gpu_ms_by_bs`, weighted by round time (otherwise `decode.running`).
Seats are averaged only over the decode time, so sleep/flip/extend do not count as 0 seats. Per stream = tokens / seat seconds,
so total = per stream × avg seats holds exactly.

**Zoom** (`static/zoom.js`, without a library). Dragging in a time chart (history, card history, curves of the boot card,
phase bar) zooms all charts to this range. Back by double-click, by the button "Zoom back"/"all the way out" in the
zoom bar or by Esc. On a focused chart: + / − / ← / → / 0. Touch: drag horizontally.
The history reloads `api/history?from=&to=` in the matching grid (from ≤ 24 min 1 s). The boot card loads `api/live?zoom=t0,t1`
with 1/2/5-s buckets (`series_zoom`); the 60-s tiles stay on "now". The card bars at the top are instantaneous values without a time axis.
Tests: `tests/test_rates_glatt_0930.py`.

## Depth on hover: "token x–y (n new)" (user 02.10. ~12:04Z / ~12:15Z)

Prefill throughput falls with context depth, so every prefill segment of the phase bar (and the last burst
of the prefill tile, the P→D flip row "after prefill …") shows x = start depth (prefix), y = end depth, n = newly computed tokens,
plus tok/s over the first and last 15 % of the tokens (`activity.prefill_depth`). Sources, best first: rankstats
`prefill.last.ext` [[rid, start, end]] per chunk; events `request_done` `prefill.{P,D}` (exact per rid, but only at the end of the request);
otherwise Δ`prefill.cached_tokens` of the burst (#cached-token is only at the first chunk of a request). A decode segment shows
tok/s per batch size (samples with only one bs, Δ`gpu_ms_by_bs`) and per request token x–y (n new) with tok/s: from rankstats
`decode.reqs` [[rid, prompt, out]] per sample, otherwise estimated from `request_done` (avg of the request, linear). `prefill.last.ext`
and `decode.reqs` are built by the port seat (02.10.); until then the fallbacks apply. Texts DE/EN in the block `DEPTH-BEGIN` of
`static/index.html`. Tests: `tests/test_depth_1002.py`.

## Sampler in its own process, counters instead of instantaneous samples (user 30.09. ~21:40Z)

"shouldn't the sampler not fail under too much load? it should be able to do that in parallel somehow?"
The readings used to run as threads in the web server process. Under 5 clients polling `/api/live` continuously,
the NVML thread wrote no row for 287 s (GIL).
- With `--state-dir` the web server starts `python -m rigdash.sampler` as a child process (`sampler.Supervisor`) and
  supervises it. The sampler reads the IPC ring, NVML, the host and the history; it writes `history.sqlite` and `ring.sqlite`.
  The web server only reads (`IpcBoots(role="reader")`). If the sampler is dead or silent, the page shows a red banner,
  and `/api/health` reports `sampler.ok=false`. The supervisor restarts it. If the web server dies, the
  sampler ends itself (parent PID). The systemd unit stays the same, so a rollback never gives a second writer.
- Counters instead of instantaneous samples: the Δ of the rank counters between two readings applies over the rank clock (`activity.coverage`,
  up to 30 s). Power comes from `nvmlDeviceGetTotalEnergyConsumption`: energy Δ spread over the covered seconds
  (`history.spread_counter`). The host CPU comes from the /proc/stat Δ over all seconds since the last reading.
  A late reading thus loses nothing.
- Levels without counters (temperature, clock, load, memory, KV) are held for a skipped second and
  counted: `held` in `/api/health` (target 0). The `_hold` of the view is only an emergency measure and is counted as `view_filled`.
Tests: `tests/test_sampler_prozess_0930.py`, `tests/test_rates_glatt_0930.py::TestCounterBooking`.

Follow-up order 30.09. ~22Z: the sampler also reads `sources.py` (cards, PCIe, docker, gpuq, fronts) and the
rank files for the fields, and it runs the energy loop. The cards now come via NVML in the same process
instead of an `nvidia-smi` fork, every second. `power.draw` is the Δ of the energy counter over the interval.
The web server reads only `ring.sqlite` (`SourcesReader`, `EnergyReader`, rank table) and `history.sqlite`;
it measures nothing itself any more.

## IPC only, no boot log (user 29.09. via 27B; reprimand and order 30.09.)

"the dashboard should also be fed from the inter-process communication, not from logs" -- since 30.09. without exception:

- `ipcboot.py` (`IpcBoots`) replaces `live.LiveLogs` as the source of the boot cards. Every second a sample per boot state folder
  `/spinning/docker-acceptance/<line>/state/<boot_id>/` (`state.json`, `events.jsonl`, `rankstate/<G>/*.rankstats`) into a
  16-min ring; rates, windows, bursts, curves, phase bar and energy assignment are deltas of these samples.
- `ipcfields.field()` knows no log fallback: a field is `ipc`, `fehlt` (missing: the page shows "missing in IPC" and in the title the
  writer from `ipcfields.MISSING_WRITER`) or `leer` (empty: "no flip in this boot yet", "no rates: boot ended").
- `live.py`, `parse.py`, `stops.py:HarnessLogs` are no longer started by the server (legacy for tests).
  `tests/test_ipcboot_0930.py` turns red as soon as a wired module opens a `*.log` or the server starts a log reader.
- Inventory with IPC/MISSING per display and the MISSING list: `/spinning/gpu-arb/docs/DASHBOARD-AUS-IPC-INVENTAR-0929.md`.
- `tests/test_no_new_log_parsers.py` stays: no new regex.

## Calm display (user 29.09.: "it keeps shifting up/down")

`refresh()` does not rebuild sections via `innerHTML`, but works the new HTML in node by node via
`morph()` (children with an `id` are paired by the id — new cards, banners
and tables therefore need a stable `id`). The element at reading height is at the same place after
redrawing (`holdAnchor`, browser `overflow-anchor` is off for this), sections
do not shrink on refresh (min-height ratchets, opening/closing releases it), `<details open>`
stays open. 10 s after scrolling/typing/clicking and as long as something is selected, no redraw happens;
plus the switch "Pause live update" in the header. Event handlers in redrawn
parts as a property (`el.onmousemove = …`), not `addEventListener` — the node stays, after all.
Headless check: `python3 deploy/scroll_hold.py http://127.0.0.1:8891/ 300 --stress`
(the jump of the reading position must be 0 px).

The service itself is described in `server.py` and `deploy/rig-dashboard.service`
(LAN :8890, runs from `/opt/rigdash/current`, deploy via `deploy/install.sh <rev>`).
This file describes only what agents and operators have to **maintain**.

## Feature map (user order 29.09.: "this must always be kept current")

Source: `/spinning/gpu-arb/docs/features.json`. One row per feature with
`id, modell (27B|NF|beide), titel, fertig, zweige[{branch, sha}], schalter[…], gewinn[…],
aus_begruendung, verantwortlich`.

**Do not enter, rigdash calculates it itself:**

* **in image**: a commit from `zweige` is an ancestor of the image REV of the model
  (`state.json.rev`), otherwise the same `git patch-id` on the image line (27B picks under
  a new sha), otherwise the same subject. `zweige` are *alternatives* of the same change
  (original + picks); for several commits enter the last one of the feature.
* **active**: from `groups.<P|D>.launch` (argv + switch env) in the `state.json` of the running
  or last boot of the model — never from log text. If the snapshot is missing (image before
  the launcher commit `groups.<G>.launch`), from the profile file (comments do not count).
  A switch that is set nowhere has its `default`.

**When to enter** (with the CLI, never by hand edit without `check`):

```bash
U=/opt/rigdash/current/rigdash/features_update.py
# feature done/built (upsert by id; --zweig/--schalter replace the lists)
python3 $U set --id H106 --modell NF --titel "…" --fertig ja \
    --zweig desk/nf-…=<sha> --schalter FLLIPER_X=env:D:1:aus --verantwortlich NF-Implementierer
# 27B has picked it (new sha, old ones stay)
python3 $U add-zweig --id H106 --zweig desk/27b-unified-0926=<sha>
# gain -- measured (boot + source), calculated (planner/model) or unverified
python3 $U gewinn --id H106 --modell NF --metrik Flipzeit --vorher 3.1 --nachher 2.4 --einheit s \
    --art gemessen --quelle "fliptimes" --boot <boot_id>
# in the image, but off: why
python3 $U begruendung --id H63 --text "…"
python3 $U check
```

Rules: gains strictly per model — with `modell=beide` every gain carries `--modell`
(otherwise the CLI refuses). Switch: `NAME=art:gruppe:an_wert:default`, `art` env|flag,
`gruppe` P|D|beide|front|launcher (front/launcher: not in the group snapshot, rigdash
reads the profile), `default` an|aus. 27B enters its rows itself.

## Features target/actual (user reprimand 29.09.: "bug fixes are not features")

At the top the **product features** F1–F24 (`features.json` → `produkt`), below them the commits/fixes
as **building blocks** (`features`, card at the top). Per product feature: `id, nr, titel, soll`
(one measurable sentence), `ist.{27B,NF}` = `{status, wert, grund, beleg, belegt_am, quelle}`,
`bausteine` (ids from `features`), optional `kreuztabelle` (F2), `untertabelle` (F12 formats,
F23 prompt lengths), `matrix` (F24 form × bs × depth × text), `marker` (instrument per format).

* **Actual only with evidence**, otherwise `unbelegt`. Each seat writes only its column: NF the NF cells,
  the 27B seat the 27B cells (`import-27b` reads `features_27b_ist_0929.md`, source per row).
* **last verified** (`belegt_am`, ISO): if the evidence is older than the start of the last boot
  of this model, the dashboard shows the cell yellow "Actual outdated, measure again" (27B/user
  29.09.: new findings do not fall behind). `produkt-ist` without `--belegt-am` = now.
* **Matrix cells** are only measurements (`wert` or `ungültig` = EOS below 500 tokens);
  a missing cell is "unmeasured", never interpolated. Values without depth (agent load)
  are shown as the edge value `gemischt`.
* **Value in the current boot** is calculated by rigdash itself (live.py/state.json/profile) and shows
  the instrument for it; where none exists, the missing marker is shown (`MISSING` in
  features.py). Per format: NF INT4/NVFP4, 27B INT8/NVFP4/W4A8 (W4A8 = 3080 rank of an
  NVFP4 boot). The format comes from `PROFILE_FORMAT` of the profile (incl. the `source` base).
* A new building block without a product feature is only a note, not a write ban;
  `set --produkt F8` attaches it in the same step.

```bash
python3 $U produkt-ist --id F1 --modell NF --status fertig+aktiv --wert "…" --beleg "Boot …" [--belegt-am 2026-09-29T06:00Z]
python3 $U kreuz --modell NF --a kvonly --b dcp --status "nur Desk" --note "F15"
python3 $U matrix --id F24 --modell NF --form "Form A" --bs 1 --tiefe kurz --text code --wert "131,9 tok/s" --boot x177 --beleg "…" [--ungueltig]
python3 $U zeile-ist --id F23 --zeile 97k --modell NF --status fertig+aktiv --wert "24,18 s" --beleg x175
python3 $U import-27b [--md /spinning/gpu-arb/docs/features_27b_ist_0929.md]
python3 $U boot-override --boot <boot_id> --lifecycle "stopped (geplant)" --beleg "…"   # state.json stays untouched
python3 $U md --out /spinning/gpu-arb/docs/FEATURES-SOLL-IST-0929.md   # table as Markdown, values from the running rigdash
```

## History in our own part (user 30.09. ~15:30Z, replaces "history like Grafana" of 29.09.)

The Grafana template (`/spinning/gpu-arb/docs/vorlagen/dashboard-vorlage-grafana-0929.png`) was a
style example, not a copy template. The dark replica at the top of the page is gone; its contents are
spread through our part, in the page design (same tokens, font, cards, light and dark):

- **History** (`#verlauf`, below the boot cards): tiles decode per stream p50/p90, prefix cache hits,
  flip time P→D; charts prefill throughput (P, D), decode throughput (all streams + per stream),
  input tokens from cache / newly computed / handoff, KV occupancy (D, P), flip time per flip.
  Model 27B|NF and range 15m…7d at the top of the card.
- **Cards** (`#gpus-card`): tiles power draw (sum of all cards), hottest GPU, host CPU;
  charts **power draw** as the sum of all cards with the single cards thin
  below, temperature, SM clock, host CPU & memory. The 15-min sparklines per card are gone.
- Style: one y axis per chart (no double axis: prefill and decode separate), unit at the axis,
  area under the line, fine grid, end value as a dot, legend with the last value, cursor coupled across all
  charts. Colors: P blue (`--s1`), D orange (`--s2`), decode green (`--s3`), cards violet/
  yellow/magenta (`--s7/--s4/--s5`, dataviz reference palette, validated), sum in text color.
- The boot tile sparklines (`spark()`) carry the same style: area, quarter grid, −15/−10/−5/now,
  last value as a dot and number.

### Source of the model series: IPC, not log (user 30.09.: "no IPC via logs")

`history.Recorder.ingest_ipc` reads every 5 s with its **own** `IpcStates` (not the one of the
log collector, which polls only after a round over all logs) every boot state store and in it
`rankstate/<G>/*.rankstats` (pdflip.rankstats/1, timer-written). Per group the first rank (TP0/PP0) counts:

| Series | Calculation |
|---|---|
| P/D prefill tok/s | Δ`prefill.new_tokens` / Δ`ts` |
| Decode tok/s | Δ`decode.tokens` / Δ`ts` |
| Decode per stream | that / `decode.running`, only if Δ`decode.gpu_ms` ≥ 50 % of the wall clock (otherwise flip/idle in the interval) |
| KV occupancy D / P | `sched.full_token_usage` (level) |
| Input tokens | P: `cached_tokens` = from cache, `new_tokens` = newly computed P; D: `new_tokens` = newly computed D, `cached_tokens` = handoff P→D |
| Cache tiers | `state.json front.served_tokens.*.cached_tier` |
| Flip time | `flipzeit.py` over marks `flip_t2t` (from `events.jsonl` flip_begin/flip_done/flip_user_time + rankstats) |

A living but resting group delivers 0 (continuous line), a rank with a file older than
20 s delivers nothing (gap). `m.<model>.ipc` = 1 marks every interval in which the sampler read a
living boot; only within such intervals does the page bridge a gap of the per-stream line.
A boot that rigdash did not see live (service was off) is added once from its logs,
only up to the first IPC sample; the label then says "rankstats (IPC) · older sections from log (transition)".

**Open gap (proposal to the implementer seat, not built here):** at D the rank cannot
tell whether `prefill.cached_tokens` is the handoff P→D or a real prefix hit of a
D-direct request. rigdash therefore counts D cached conservatively as handoff (never as cache). A counter
`prefill.cached_tokens_handoff` in rankstats (the share of `cached_tokens` whose rid had a P leg 1,
the knowledge `Pending.leg1_ran` is in the front and would have to go with the leg-2 request to D) makes it exact.

### Two editions from one code: `--edition rig|release`

`rig` (default) delivers the whole page. `release` (env `RIGDASH_EDITION=release`) is the published
fLLiper edition: `server.edition_page` cuts out every block `<!--DEV:BEGIN-->…<!--DEV:END-->`
(HTML, CSS `/* … */` and JS `// …`), i.e. development state, start flags, features target/actual, building blocks,
image changes, containers, GPU window plan, last boots and the LAN links. No empty shell remains.
`/api/live` answers without `features`, `image_changes`, `gpuq`; `/api/launch`, `/pdflip` and `/api/pdflip/*`
return 404. New development parts belong in a DEV block (`tests/test_edition_0930.py` checks this).
**Exception profile tab (user decision 05.10., order 1984):** tab, panel, CSS and the modules `profil.js`, `profil_balken.js`,
`hwprofil.js`, `modellprofil.js` are NOT in a DEV block; `/api/profil/*` (load, edit, save, export, dry run,
bar `recompute`), `GET /api/hwprofil` (display only, does not talk to gpuq) and `/api/modellprofil/*` (reads only `config.json` and headers
under the model roots) also answer in `release`, still only in the LAN (proxy 403). Rig operation stays closed:
`POST /api/hwprofil/measure|cancel` (books gpuq) 403 with plain text, card planner, start line, `/api/launch`, `/api/pdflip/*` 404
(`tests/test_profil_release_edition_1984.py`).

### Deploying the profile editor: staging the planner modules and unit flags (order 1984)

`install.sh` puts `python/flliper` of the DASHBOARD revision to `/opt/rigdash/planner`; that is a dashboard line and does not carry the calculation modules of the
editor (`pdflip/profile_json, refusals, profile_catalog(+_curated), model_profile, card_identity, topology`, `rigmon/hardware_profile`,
`planner/profile_couplings, expert_residency, pp_cut`). They are on the Python release branches (`desk/profil-editor-release-27b-1005`,
`-nf-1005`). `deploy/stage_profil_modules.sh <rev>` puts the full tree `python/flliper` of the revision under
`/opt/rigdash/kartenplan/profil/releases/<sha>` and switches `profil/current` (the card planner tree `kartenplan/current` stays untouched):

    deploy/stage_profil_modules.sh --check   fa9e7d5c4c    # check, write nothing (default); dashboard revision -> REFUSED, exit 3
    deploy/stage_profil_modules.sh --dry-run fa9e7d5c4c    # plus the actions that --apply performs
    deploy/stage_profil_modules.sh --apply   fa9e7d5c4c    # writes, only under --root (default /opt/rigdash/kartenplan), idempotent (lead)
    deploy/stage_profil_modules.sh --unit-flags            # the unit lines

Afterwards the lead sets in the unit `RIGDASH_PROFIL_TREE` (= `--profil-tree`, ONE tree for editor, model profile, hardware profile and the couplings worker;
`--hw-tree` overrides it only for the hardware), `--couplings-python`/`RIGDASH_COUPLINGS_PYTHON` (Python of the flliper environment), if needed
`--profiles-release-dir`, `--profile-dir`, `--model-root`, `--edition release` and, only in the rig edition, `--hw-measure-tree/--hw-python/--hw-prefix`.
**MemoryMax:** the worker (`import flliper`) is in the cgroup of the unit (measured RSS 612 MiB), the unit stood at MemoryCurrent 487 MiB / peak 715 MiB
against `MemoryMax=1G`: raise to 2G. The worker also calculates the topology verdict of the dry run (op `topology`); if it is missing, the note
"Topology for N card(s) not checked" stays (tests: `tests/test_profil_staging_1984.py`, `tests/test_profil_topology_child_1984.py`).

### Oracle and proposal (AP-D, plan profile planner 06.10.)

Since AP-D the dry run (`POST /api/profil/dry`) asks the LAUNCHER itself: `profil_oracle.OracleService` holds a child process of its own
(`kartenplan_build/oracle_worker.py`, Python of the flliper environment like `--couplings-python`) that runs `launcher.main(--dry-run)` on an NVML replay of the
selected cards (`pdflip/propose_oracle`), first without and, on a refusal, once more with `--force`, and builds from it the document
`flliper.verdikt/1` (`pdflip/propose_verdict`): per item a verdict `{code, forcebar (from refusals.by_code), force_state, grund, konsequenz}`, plus
`PROFILE-VECTORS`, `RECORDS-NVEC`, `METAL-UNPROVEN` (the blockers in the text of HW-COUNT), `FIT` (hw_fit), `HW-BORROWED`, `HW-UNCALIBRATED` and a crash of the
launcher as `ORAKEL-ABSTURZ`. The return format of the dry run stays; new are `quelle` (`orakel` | `gate`), `orakel` (outcome, profile hash, cache) and
`verdikte`. If the oracle cannot be asked (child process, Python, model paths), the partial check of the planner gate applies WITH a note. A run to the end takes
16-18 s (measured 06.10.), hence the cache per (inventory, form, argv hash, state of the sources); live profiles drift: the profile hash (file and launch input)
is in every verdict and in the key.

`POST /api/profil/propose` ({basis: {kind, name}, form: flip|tp|dual|single, inventar: "rig" | [{card, pcie}], karte?, ziele?, model_path?, draft_path?}) calls
`propose()` (AP-C) and returns the server profile `flliper.server/1` (base profile + the values of the proposal, origin `planer`) with origin, verdict and edges per
value, the verdicts and the request for the bars (`what=phase_bars`, `form` flip|d_only|dual|single, contract `flliper.balken/1` of AP-H2). All four forms:
`flip`, `tp` and `dual` ask the launcher dry run (oracle); dual (AP-E) additionally carries the dual fit as the verdict "Planner calculation, not hw_fit"
(`DUAL-PASSUNG`, `DUAL-PFLICHT`). `single` (single card, AP-F; `einzel` is a name for it) has no launcher: exactly ONE card (`karte` = ordinal in the
hardware profile, default 0), a model path (`model_path` or the `PROFILE_MODEL` of the base profile; without a base profile it works too), the verdict is a
planner calculation (`ausgang` passt | passt_nicht | unbelegt, `art` planner calculation, no force) plus ServerArgs parse, the server profile a new profile from the
arguments of the normal server (`launch.argv` for `python -m flliper.launch_server`).
The oracle child process imports the launcher: the planner tree of the unit must carry `pdflip/launcher.py`, `propose*.py`, `hw_fit.py` and `fit_profiles_data`
(the full tree `python/flliper` of the revision, as with `stage_profil_modules.sh`).

**Memory of the oracle child process (MEASURED 06.10., review AP-D):** a full NF dry run on the reference rig (`nf-int4-h6-abl`, NVML replay) needs
a peak of **1.75 GiB RSS** (`/usr/bin/time -v`: maximum resident set size 1789432 kB; 47.98 s under CPUQuota 200%). Against `MemoryMax=2G` of the unit
(MemoryCurrent there 409 MB) that would not work. Therefore the service starts the child process in its OWN scope: `--oracle-prefix auto` (default; env
`RIGDASH_ORACLE_PREFIX`) = `systemd-run --scope -q -p MemoryMax=4G`, the scope does not count against the unit. `none` starts without a frame, any other value is
the command prefix itself. If the prefix fails at once (no `systemd-run`, no D-Bus), the child process runs once without it
(`OracleService.prefix_fallback`); then the unit limit applies again. The unit file carries `MemoryMax=2G` (couplings worker, 612 MiB RSS, see above).

### Storage and tiers

- `history.py`: `history.sqlite` in `--state-dir`. Tiers p0 (1 s NVML, 5 s host and model), p1 (10 s)
  and p2 (60 s). Retention 3 h / 3 d / 30 d, cap 256 MB. There is no additional service.
  Every series is a rate or a level, never a counter. Therefore the mean stays right over every tier.
- `cacheacct.py` calculates the cache trap: "from cache" = prefix hit at admission. The handoff P→D
  is a series of its own and never cache. For boots without an IPC sample the pairing of the `PDFLIP-SERVED` lines
  per rid stays (label "from log (transition)").
- New series: add the name in `history.view`. Put the writer in `Recorder`, never from
  a new log regex (`tests/test_no_new_log_parsers.py`).
- Charts: `static/grafik.js` with uPlot 1.6.32, embedded locally (`static/uplot.*`, MIT). There is no CDN.


## Card planner (item 510, tab "Card planner", rig edition only)

Optimal start configuration for a selected model/profile (27B INT8, NF INT4 abl, 27B NVFP4 dual, 27B FP8, 27B GGUF UD-IQ4_XS)
on 1..6 selected cards (catalog `kartenplan_catalog.py`, PCIe per card: gen, lanes, resizable BAR, via chipset).

* Verdict "works / does not work": the original planner functions `card_identity.arch_gate/order_cards/uncalibrated_message` and
  `topology.plan_topology` (`kartenplan_gate.py`, loaded by file path, synthetic cards, no GPU, no launcher).
* Plan: recording of the planner at the real boot (`kartenplan_data/*.json`: vram_plan.json, budget lines, argv/env, rank log items,
  flag explanations, `planer_nachrechnung` = `launcher.budgets_from_dc` in a child process against the boot numbers).
* Other cards/numbers: the planner refuses (HW-COUNT/HW-ARCH/HW-UNCALIBRATED/HW-TOPOLOGY); then there is only a labelled APPROXIMATION.
* Renew records (desk; reads logs, hence outside this package): `cd tools/rig_dashboard; python3 -m kartenplan_build.records;
  python3 -m kartenplan_build.bridge --trees-root <folder with <rev>/python/flliper>`.
* VRAM bar per card and phase (order 880): one bar = the card in the state in which the row group (P or D) is awake. The items lie as
  contiguous blocks in a fixed order **shared (driver) → P → D** (server: `kartenplan._annotate_segments`, stable by `SEG_ORDER`;
  the bracket below the bar shows the blocks). Mouse pointer/tap on an item: name, MiB/GiB, share of the card, phase, origin
  (**measured** = rank log/NVML, **planner value** = vram_plan or budget line) and a one-sentence explanation (`SEG_WHAT`).
  Overflow: if the sum is larger than the card, the bar grows (scale = sum), the card edge stays marked, the excess is hatched and a
  note "Card N: X MiB over the VRAM – profile does not fit" names the largest items. A negative rest in the rank budget (dual record: items
  overlap, the planner closes the card anyway) is **not** a misfit: hatched yellow, note "Rank budget overbooked by X MiB"
  (`overlap_mib` / `hard_over_mib`). Live NVML bars cannot overflow and stay unchanged.
* View without the service: `python3 -m rigdash.kartenplan_preview --port 18890 --tree <tree>/python`, then `http://127.0.0.1:18890/#t=kartenplan`.
* Deploy proposal: `deploy/install_510.sh --check` / `deploy/install_510.sh` (lead).


## Estimate model profile (PROFILE EDITOR S3, order 960, user 03.10.: "button to create the model profile and estimate values from the model at the desk")

`POST /api/modellprofil/schaetzen` with `{"path": "<model directory or .gguf>", "draft_path"?, "kv_dtype"?: "auto|fp8_e4m3",
"mamba_ssm_dtype"?: "float32|bfloat16", "gguf_file"?, "registry"?: true|"identifier"}` returns `{ok, profile, registry_fields?, elapsed_s, cached}`;
`profile` is `flliper.model/1` (every value `{v, src}` with source `config|Index|geschätzt|stat`), `registry_fields` the fields of a `form.ModelProfile` row
derived from the estimated profile.  `GET /api/modellprofil/modelle` lists the model directories under the roots (only `stat`).

* **The estimator is not here.**  `flliper/srt/pdflip/model_profile.py` (pure standard library) is loaded by file path from the planner tree like the gate of the card planner
  (`MODELLPROFIL_TREE`, otherwise `KARTENPLAN_TREE`, otherwise the candidates in `kartenplan.TREE_CANDIDATES` and `<repo>/python`).
  If the file is missing in the tree, the route answers 503 with the name of the file.  The tree under `tests/fixtures/modellprofil/` is a copy of
  `desk/profil-s3-modell-1003`.
* **Only `config.json` and the header lines are read** (8 bytes + JSON per shard or the GGUF head), never a weight.  The path must lie under a
  model root (`--model-root`, repeatable, or `RIGDASH_MODEL_ROOTS`; default `/spinning/llm_stuff/club-3090/models-cache`); relative
  paths, `..`, NUL and symlinks out of the root are rejected with 400.  Answers are remembered per path and file state (size, mtime).
* **LAN only** (also in the edition `release`, since 05.10.): 403 via the proxy (the route reads files under the model roots).  Body at most 64 KiB.
* `static/modellprofil.js` (`window.ModellProfil`): `liste()`, `schaetzen(path, opts)`, `zeilen(profil)` (rows `{gruppe, label, wert, roh, src, hinweis}`),
  `tabelle(profil)` (HTML building block, escaped), `bytes(n)`.  The UI is built by order 930; this module draws nothing itself.

## Measure and display the hardware profile (order 950, profile editor S2; rig edition only, LAN only)

Routes and JSON, no UI (that is built by the profile editor, order 930; `static/hwprofil.js` is the display module to mount).

* `GET /api/hwprofil` → `{profile, problems, window, job, gpuq, owner, window_len}`. `profile` is `flliper.hardware/1`: a **view** (not a fourth
  measurement file) over the card probe cache (`card_probe-*.json`), the stage-0 profile (`hw_profile-*.json`) and NVML. Every numeric value is
  `{v, src, at, probe, note}` with `src` = `gemessen` | `NVML` | `Datenblatt` | `geschätzt` | `nicht gemessen` (then `v: null` and `note` = reason).
  It is built in `flliper/srt/rigmon/hardware_profile.py` of the planner tree (loaded by file path, no `import flliper` in this process).
* `POST /api/hwprofil/measure` `{"cards": [<NVML index>, ...]}` books a gpuq window **itself** (owner `profil-editor`, only these cards,
  15 min (order 1006: all compute formats incl. native W4A4 + BAR1 link per pair in child processes), without `not_before`, exclusive; `mib` only if the body asks for it) and measures in it. Answer `action`:
  `messung_gestartet` (measurement started: child process runs, the window goes back AT ONCE afterwards, also after an error) · `wartet` (waiting: window `pending`: status, **nothing
  measured**, booking stays; pressing again takes it up again) · `abgelehnt` (refused: unplannable, card occupied despite the window, too little remaining time, gpuq gone;
  HTTP 409) · `laeuft_bereits` (already running). The gpuq token never leaves the process; the booking is also in `<state-dir>/hwprofil_window.json`, so that
  a restart returns an orphaned window.
* `POST /api/hwprofil/cancel` returns a waiting window.
* Measurement scope (order 1006, `card_probe --run`): per card SM count, L2, membw/GEMV, bf16, fp8, int8 W8A8, NVFP4 W4A8 (sm_8x only), W4A16 Marlin, W4A4 native
  (sm_12x only; older cards carry the reason), H2D/D2H bandwidth and latency (median 4 kB, minimum in the hover); per ordered pair host staging/p2p and the
  **BAR1 link** (`rigmon/bar1_probe.py`: one child process per card, production transport with byte proof, `--no-bar1` switches it off). A value that a
  microbenchmark cannot deliver stays "not measured" with a reason. `profile.bar1` = `{measured, complete, pairs_measured, pairs_total, note}`.
* **Saved (AP-A, profile planner 06.10.).** On the first call the service writes the profile to `--hw-profile-file` (env `FLLIPER_HARDWARE_PROFILE`,
  default `/var/lib/flliper/hardware.json`; rig and release alike; a write error is only a state, not a crash). `GET /api/hwprofil`
  also carries `persist` = `{enabled, state, label, captured_at, reason, id, drift, error, from_persisted, file}`; `state` = `erst_erfasst` | `neu_erfasst` |
  `vorhanden` | `abweichend` (file stays, `drift.changes` names the difference) | `nur_gespeichert` (NVML is silent: the file applies) | `keine_karten` |
  `nicht_schreibbar`. `POST /api/hwprofil/recapture` ("Capture again") reads NVML again and replaces the file: no gpuq window, also in release; a
  successful measurement also captures again. SM count (`pdflip/hw_sim.py`) and nominal bandwidth (`kartenplan_catalog.py`, field `mem_gbs.nominal`) enter the profile as
  `Datenblatt`, a measured SM count wins; `cards[].catalog` names the catalog card, `preset` and origin (`measured_on_rig` | `Datenblatt` |
  `borrowed-unbelegt`, per field in `origin_fields`). `GET /api/hwprofil/issue` returns the issue text "Hardware profile" as Markdown (`{ok, format, text}`);
  secrets and host paths are removed (`redact.text_for_issue`).
* **Issue text "Run report" (AP-I, profile planner 06.10.).** `POST /api/profil/issue` with `{doc, dry?, cards?, model?}` (profile, answer of the last
  dry run, the selected cards `[{card, pcie}]`, a model profile `flliper.model/1`) returns `{ok, format: "markdown", text, blocks, filename}`: a
  block to paste into a GitHub issue with the sections hardware profile (short form, `hwprofil.issue_short`, from the hardware service), model profile
  (values with source, only the folder name), operating mode (read from the flags: `--dual-layout`/`--dual-share` = dual, `--d-only` = TP only, one card =
  single card, otherwise flip), proposal and overrides (rows with `changed`, origin `nutzer`/`planer` or a deviating `planner_value`; columns
  current/profile/proposal/origin, plus `state`/`verdict` as soon as a row carries them), verdicts and force (codes with class and force state, newly read from the
  register; without a dry run that is stated), versions (tree revision, image, driver, CUDA/torch, dashboard) and the placeholder "Measurement result /
  boot log excerpt". Redacted (`redact.text_for_issue`; values of rows whose NAME names a secret, `redact.secret_name`, are dropped entirely). Also by shape: vendor prefixes, JWT, long token runs (with and without `=`/`:` before them, a whole cell value as a run), `user:pass@`; every
  absolute path outside `/app` and every `~/`/`$HOME/` path becomes `<hostpath>/<last segment>` (system roots like `/root`, `/spinning`: `<path redacted>`).
  Structural rule (fix round 5): the report shows a VALUE only for keys that are in the catalog (`catalog.json`: flags, envs, profile variables) and are no secret by name; every other key set by the user
  shows only its name and `<value hidden: unknown key>` (`redact.value_for_issue(name, value, known)`; without `known` nothing is shown). The value shapes stay the second layer
  (also for catalog keys): in addition base64 with `/` `+` `=` (AWS secret, Azure key) and dot-separated tokens (Discord). Paths are normalised before the verdict (`/app/../../root/x` is `/root/x`),
  `file://` is dropped, a path with spaces in quotation marks counts as one path; `/models-cache` (model mount of the container) stays like `/app`.
  Without the hardware service the report is still produced ("not available"). UI: section "Issue text: run report" below the export in `profil.js`.
* No background poller: only whoever operates the page asks. A running window that was not used after 180 s is returned at the next call.
* Service parameters (deploy by the lead): `--hw-tree` (staged tree with `hardware_profile.py` + `pdflip/card_identity.py`, `deploy/stage_hwprofil.sh`),
  `--hw-measure-tree` (full flliper tree for the child process), `--hw-python` (interpreter with torch + sgl_kernel; without sgl_kernel the int8/W4A16 arms stay
  empty and the run reports that as a warning), `--hw-prefix` (e.g. `systemd-run --scope -q -p MemoryMax=6G`: the service has
  `MemoryMax=1G`, torch/CUDA belongs in a cgroup frame of its own). Env: `HWPROFIL_TREE`, `HWPROFIL_MEASURE_TREE`, `HWPROFIL_PYTHON`, `HWPROFIL_PREFIX`.
* Mounting into a page: `<div id="x"></div><script src="hwprofil.js"></script><script>HwProfil.mount(document.getElementById("x"))</script>`;
  `HwProfil.render(answer)` returns only the HTML text.

## Profile planner: one page in six steps (AP-H1, plan profile planner 06.10.)

The Profile tab leads in a fixed order: **1 Hardware** (inventory: "This rig" = hardware profile with real NVML cards, or cards from the catalog,
synthetic; the three preset cards first) -> **2 Model and profile** -> **3 Operating mode** (single card, TP only, flip PP/TP, dual PP/TP, each with an explanatory
sentence; preset from the profile: `--dual-layout`/`--dual-share` = dual, `--d-only` = TP only) -> **4 Proposal** (controls "Seats at the same time" and "Context",
button "Proposal" = `POST /api/profil/propose`, "Check again" = dry run) -> **5 Adjust** -> **6 Export**.

* Data of the page: `rigdash/profil_planer.py` (`ui_info`, in `GET /api/profil/list` as `planer`): forms, sections A (split), B (KV), C (experts) with the
  names of their values, the dual ENV table with default values and source lines, control limits. If `planer` is missing (older service) or `profil_planer.js`,
  `profil.js` draws the old page. The proposal applies to the forms that `ProfilEditor.FORMS` knows (flip, tp); single card (AP-F) and dual (AP-E) show the reason.
* Display: `static/profil_planer.js` (no DOM, no network, testable with Node). One field per rank for vectors (comma lists with as many entries as cards;
  wrong length = warning), **state chip** per value (proposed / unverified / solved by the launcher / overridden by you / profile / default), **verdict chip**
  per value (works / only with --force / refused / note / not checked / unchecked since your change) with code and reason visible, **dependency chips**
  from the edge catalog. A verdict is a note, never a lock (user decision 4a): every field stays operable, force is in the export. Filter simple/expert.
* Dual ENV table (section D, plan 4c): `FLLIPER_PDFLIP_DUAL_SHARE_GREEN_TABLE` as a table (D seats up to | P share at small / large tau, rungs 0-3 = 100/75/50/25 %),
  `..._STARVE_AGE_S`, `..._STARVE_MAX_RUNG`, `FLLIPER_PDFLIP_DUAL_GRANT_RETRY_MS`; catalog entries curated, edges K109-K116 with evidence.
* Tests: `tests/test_profil_planer_aph1_1006.py`.
