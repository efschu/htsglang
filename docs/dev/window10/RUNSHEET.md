# Window 10 -- accounting for the bs=1 decode regression, and the #517 A/B

Window opened 2026-08-05 17:46Z. Cards 0,1,2 held through `/spinning/gpu-arb`
(previous holder saved to `holder.pre-window10`; production stopped by pgid,
restored verbatim from `/root/bin/start-serving-30030.sh` at the end).

## The question, and why it needed restating before it could be answered

The brief asks to account for `126.8 -> 55-64 tok/s` at bs=1. Those two numbers
do not come from the same instrument, and the record artifact says so itself.

* `126.8` is the SERVER-SIDE TICK from `s14_decode_punkt.py`, read out of the
  scheduler's own `Decode batch` log line. The very same line in
  `messen_int8_decode.log` records `client 120.5 tok/s`.
* That instrument's A-vs-A floor **in its own boot** was **33.9 %** -- the three
  identical floor draws were 104.46, 118.65 and 106.07 tok/s
  (`floor_int8_decode.log`). #424's own `RESULTS.md` writes of the +11.3 %
  barlink delta measured with it: "not established".
* The client-side instrument in that same boot is club-3090 `bench.sh`:
  narrative `decode_TPS` **86.46**, code **112.18** (`bench_int8_decode.txt`).

So `126.8` is the top draw of a 33.9 %-floor server tick, and `55-64` is a
German narrative client number. Both are real; they are not each other's
counterpart. Every arm in this window therefore runs BOTH instruments, and the
table reports each against its own record value:

| instrument | record value | what it measures |
|---|---:|---|
| `s14_decode_punkt.py` bs=1, tick | 126.8 tok/s | scheduler `Decode batch` gen throughput |
| `s14_decode_punkt.py` bs=1, client | 120.5 tok/s | tokens counted out of `meta_info` |
| `s14` ms/Verify | 30.37 ms | `accept_len * 1000 / gen_tok_s` |
| `bench.sh` narrative | 86.46 tok/s | client decode_TPS, 1000-token prose |
| `bench.sh` code | 112.18 tok/s | client decode_TPS, 800-token code |

## Two structural differences from the record, both named, neither routed around

**1. The BAR1 sub-transport is unreachable in this window.** The record ran
`SGLANG_BARLINK_TRANSPORT=bar1` inside a docker image with
`--device /dev/dmabuf_holder`. That node does not exist on this host any more
(stock NVIDIA open 595.58.03; the smallbar holder module is not loaded), so
bar1 cannot be booted without loading an out-of-tree module against the running
driver. Every arm here runs barlink's DEVICE transport instead -- which is what
production itself runs (`barlink.py:67` resolves an unset
`SGLANG_BARLINK_TRANSPORT` to `"device"`). barlink is on in every arm; NCCL is
not used anywhere.

**2. At the pin, the device transport is not in the abort gate at all.** This is
the strongest named candidate and it is visible in the source before any card
is booted:

```
git grep -n abort 84fff442e1 -- .../barlink_device.py
  -> two HIP-macro comment lines. Nothing else.
```

`barlink_abort_gate.check_aborts` opens with `if not _transports: return`, so in
a pin process running the device transport the gate registry is EMPTY and every
check -- per collective and per CUDA-graph replay boundary -- costs one truth
test on an empty list. `b001d102fa` ("[barlink][#583] device transport: a
tripped spin kernel must not kill the CUDA context") is the commit that put the
device transport into the gate via `barlink_abort_gate.register(self)`. The pin
lacks it, and lacks `0d22e1f8e5` (#517 phase 1) and `477eceac93` (#517 phase 2)
with it.

That gives a chain that can be tested one link at a time:

| state | abort-word read on the decode hot path |
|---|---|
| pin `84fff442e1` | none -- registry empty |
| `+ b001d102fa` (#583) | blocking `int(self._seq_dev[1].item())` per check |
| `+ EVERY=32` (#600 mitigation, live in production) | one blocking read per 32 checks |
| `+ 477eceac93` (#517) | none on the hot path -- watchdog thread, private stream |

## Arms

Five, each changing exactly one thing from the one before it.

| arm | tree | config | barlink env | isolates |
|---|---|---|---|---|
| `arm1_pin_record_bl` | `84fff442e1` | record | default | A-vs-A anchor |
| `arm2_int_record_bl` | `548f4cee5c` | record | default | 523 commits of TREE |
| `arm0_int_today_every32` | `548f4cee5c` | today | `ABORT_CHECK_EVERY=32` | today's CONFIG (= production as it stands) |
| `arm0_int_today_517on` | `548f4cee5c` | today | default (EVERY=1) | the #517 fix |
| `arm0_int_today_wdoff` | `548f4cee5c` | today | `PEER_WATCHDOG=0` | the #517 control (hot path reads in line) |

`record` config = the #424 `int8_decode` flags verbatim: ctx 131072, mrr 16,
reserve `auto`, `--rank-perf-tune phase-decode`, NEXTN 3/1/4, no HiCache, no
ladder, no fast-lane, no mamba pin, no `preserve_thinking`.
`today` config = `/root/bin/start-serving-30030.sh` verbatim, minus the port.

Note for arm 3 of the original brief (pin + today config): not expressible.
`--max-mamba-cache-size` and `--chat-template-default-kwargs` do not exist in
the pin's `server_args.py`, so "today's config on the old tree" is not a
configuration the pin can be asked for.

## Discipline

* A-vs-A floor FIRST in every arm: three identical `s14` bs=1 draws before any
  measured draw. No delta below its arm's own floor is reported as a delta.
* Every measured window is 12 s (`--window-seconds 12`), above the 10 s rule,
  with a 6 s ramp cut off in front of it.
* py-spy is a CENSUS, not a dump: 20 samples of the TP0 scheduler thread's leaf
  frame taken while a 30 s bs=1 load runs, reported as `n/found`. A single dump
  cannot separate a hot leaf from a lucky one.
* Power limits were restored to 320/525/320 W at 17:49Z, before any measured
  draw, and are sampled every 2 s into `vram_power_series.csv` -- the limit
  being raised is not the same as the card taking it, so the draw is evidence.
* First boot after a tree switch pays cold JIT; boot times and the floor draws
  are in the artifacts so an outlier can be seen rather than assumed.
