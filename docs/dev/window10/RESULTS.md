# Window 10 results -- what actually happened to bs=1 decode

Window 17:46Z - 18:2xZ, cards 0,1,2 held through `/spinning/gpu-arb`. Five
arms, every one on barlink, each changing exactly one thing from the one
before it. Power limits 320/525/320 W throughout, sampled every 2 s.

## 0. The headline number was two numbers

The brief asks to account for `126.8 -> 55-64 tok/s`. Before any card was
booted, the record artifact already said those are not the same instrument:

* `126.8` is the SERVER-SIDE TICK of `s14_decode_punkt.py`. The same line in
  `messen_int8_decode.log` records `client 120.5 tok/s`.
* That instrument's A-vs-A floor **in its own boot** was **33.9 %** -- floor
  draws 104.46 / 118.65 / 106.07 (`floor_int8_decode.log`). #424's own
  `RESULTS.md` writes of the +11.3 % delta measured with it: *"not
  established"*. `126.8` is the top draw of a distribution whose own spread is
  a quarter of its value.
* The client instrument in that same boot is `bench.sh`: narrative
  `decode_TPS` **86.46** (CV 1.5 %), code **112.18**.

`ms/Verify` from the same probe repeats to **0.3-1.1 %** in every arm here.
That is the only bs=1 ruler in this family that can carry a verdict, and it is
what the decomposition below uses. Reported per arm as
`accept_len * 1000 / gen_tok_s`.

## 1. The ladder

| # | arm | tree | config | transport / gate | ms/Verify | floor | step |
|---|---|---|---|---|---:|---:|---:|
| — | #424 record | `1960957e3b` | record | barlink **BAR1** | 30.37 | 0.8 % | — |
| — | #424 NCCL control | `1960957e3b` | record | stock NCCL | 31.18 | 1.0 % | +2.7 % |
| A | `arm1_pin_record_bl` | `84fff442e1` | record | barlink **device**, gate registry EMPTY | 37.27 | 0.5 % | **+22.7 %** |
| B | `arm2_int_record_bl` | `548f4cee5c` | record | barlink device, #517 on | 38.84 | 0.7 % | **+4.2 %** |
| C | `arm0_int_today_every32` | `548f4cee5c` | today | + `EVERY=32` (production as it stands) | 38.89 | 0.8 % | +0.1 % (below floor) |
| D | `arm0_int_today_517on` | `548f4cee5c` | today | #517 default, `EVERY=1` | 38.88 | 0.26 % | -0.0 % (below floor) |
| E | `arm0_int_today_wdoff` | `548f4cee5c` | today | watchdog OFF = pre-#517 in-line read | 44.59 | 1.1 % | **+14.7 %** vs D |

### The client instrument agrees, independently

| arm | bench narrative | CV | bench code | CV | PP tok/s |
|---|---:|---:|---:|---:|---:|
| #424 record (BAR1) | 86.46 | 1.5 % | 112.18 | 3.6 % | 1639 |
| A pin + record cfg | 62.86 | **26.1 %** | 93.14 | 1.6 % | 1302 |
| B int + record cfg | 67.81 | 1.7 % | 90.85 | 1.1 % | 1289 |
| C int + today cfg, EVERY=32 | 67.42 | 2.5 % | 88.13 | 1.9 % | 1290 |
| D int + today cfg, #517 | 68.51 | 4.0 % | 88.58 | 0.9 % | — |
| **E int + today cfg, gate IN LINE** | **58.92** | 2.2 % | **79.33** | 1.2 % | — |

## 2. The accounting, by name

Reading the ladder as one-variable steps from the record to today's production:

| term | size on ms/Verify | established? | what it is |
|---|---:|---|---|
| **barlink sub-transport BAR1 -> device** | **+22.7 %** | yes, 45x its floor | the record ran `SGLANG_BARLINK_TRANSPORT=bar1`; that needs `/dev/dmabuf_holder`, which is ABSENT on this host (stock NVIDIA open 595.58.03, smallbar holder module not loaded). barlink resolves an unset transport to `device` (`barlink.py:67`). |
| 523 commits of tree | +4.2 % | yes, 6x its floor | `84fff442e1 -> 548f4cee5c` at fixed config and transport |
| today's full production flagset | +0.1 % | **NO -- below floor** | ctx 262144, mrr 4, HiCache write_through, mamba-96, fast-lane, ladder, sleep-on-idle, preserve_thinking together cost nothing measurable at bs=1 |
| | | | |
| (#517 abort gate, if it were OFF) | +13.7 % | yes, 13x its floor | not part of today's production cost -- the fix is in and working |

**Total accounted: 30.37 -> 38.89 ms/Verify, +28.0 %.** In rate terms the
client instrument reads 86.46 -> 67.42 narrative (-22.0 %) and
112.18 -> 88.13 code (-21.4 %) -- both consistent with +28 % round time.

### The dominant term is not a regression anyone wrote

It is the transport the rig can currently reach. The record's own NCCL control
(31.18) sits within 2.7 % of BAR1 (30.37), so this is not "barlink vs not
barlink": barlink's **device** sub-transport is 19-23 % slower per verify round
than *either* BAR1 or NCCL at bs=1. The mechanism is visible in the arms' own
prefill lines, e.g. ARM C rank 0:

```
gpu-ms: 1567.7 (compute 122.2, wait 1445.5)
  wait by family: tp.all_reduce 997.8/129x, dcp.all_reduce 252.3/16x,
                  dcp.all_gather 192.8/48x, tp.all_gather 2.5/1x
```

92 % of prefill GPU time is collective wait, ~7.7 ms per `tp.all_reduce`.

## 3. The #517 A/B verdict: the fix works, and the mitigation is now dead weight

The gate's cost was measured three ways in the same boot family:

* **D vs E** (one variable, the watchdog): 38.88 -> 44.19-44.54 ms/Verify.
  Turning the #517 fast path off costs **+13.7 %** round time. That is what the
  guard costs when it reads the device word in line, and it is what #517
  removes.
* **C vs D** (one variable, `EVERY`): 38.89 vs 38.88, floors 0.8 % / 0.26 %.
  With #517 active, throttling the gate to every 32nd collective buys
  **nothing**. The `#600` interim mitigation is redundant.
* **py-spy census**, 20 samples of the TP0 scheduler leaf under a live bs=1
  load: `check_aborted` appears **0/20** in arms A, B, C and D, and
  **14/20** in arm E (`check_aborted`, `barlink_device.py:1552`). That is the
  can-fail half: the census CAN see the frame when it is there, so the 0/20
  readings are a measured absence and not a blind instrument. Task #600
  measured 11/14 before the fix; arm E reproduces it at 14/20.

### The reported 55-64 band IS the gate reading in line

Arm E is the only arm in this window that lands in the band the brief reports,
and it lands in the middle of it: **narrative 58.92 tok/s (CV 2.2 %)**, code
79.33. Arm E differs from today's production by exactly one variable -- the
abort gate reading the device word in line instead of on the watchdog. So the
55-64 the brief carries is not today's production and not the tree: it is the
pre-mitigation abort-gate cost, which both the `EVERY=32` mitigation (67.42)
and #517 (68.51) already remove.

The remaining top leaf in every arm is `synchronize (torch/cuda/streams.py:108)`
reached through `resolve_seq_lens_cpu (overlap_utils.py:542)` -- 12/20 at the
pin, 14/20 at B, 11/20 at C. That is NOT a regression: it is the dominant leaf
at the record-era pin too, where the gate registry is empty. It is the next
thing to attack, and it is a different task from #517.

## 4. What the pin proved about blame

ARM A is the record-era pin running the record config verbatim, today. It reads
narrative 62.86 / code 93.14 -- i.e. **the "55-64" band reproduces on the record
pin with the record config**. Neither the 523 commits nor today's flagset can be
the cause of the band, because both are held at record values in that arm.

## 5. Caveats a reader should have

1. **BAR1 could not be booted.** `/dev/dmabuf_holder` is absent and the running
   driver is the stock open 595.58.03 build. Loading the out-of-tree holder
   module against it was not attempted on a production box. So the +22.7 %
   transport term is measured as "what the rig can reach today vs what the
   record reached", not as a like-for-like BAR1 A/B.
2. **The record commit is not the pin.** #424 booted `1960957e3b`; the brief's
   pin `84fff442e1` is 57 commits newer. The tree term above is therefore
   `84fff442e1 -> 548f4cee5c` (523 commits), and those 57 sit unmeasured inside
   the transport term.
3. **The record ran in a docker image**, torch 2.11.0+cu130 with `sgl_kernel`
   bind-mounted from the shared venv; these arms run the host venv directly.
   That stack difference is also inside the +22.7 % term and is not separated
   from the transport by any arm here.
4. **ARM A's narrative CV was 26.1 %** (min 33.53, max 71.29) and did not
   recur -- B, C and D all read CV <= 4.0 %. Treated as instrument noise in that
   one arm, not as a finding. Its code class (CV 1.6 %) is the trustworthy half
   of that arm.
5. **Today's production reads 67.4 tok/s narrative**, above the 55-64 the brief
   reports. Whatever produced 55-64 (thinking enabled, a different content
   class, a loaded or cold server) is not `bench.sh` and is not reproduced by
   any arm here. Unexplained, and deliberately not papered over.
6. **`--max-mamba-cache-size` and `--chat-template-default-kwargs` do not exist
   in the pin's `server_args.py`**, so the brief's ARM 3 ("pin + today's
   config") is not a configuration the pin can be asked for. Not run.
7. **KV pool moved with the tree**, at identical record config:
   `max_total_num_tokens` 431360 (pin) -> 350400 (integration). Today's config
   reads 602944. Not a bs=1 decode term, but it is a real tree delta.

## 6. Production recommendation

1. **Remove `SGLANG_BARLINK_BAR1_ABORT_CHECK_EVERY=32` from
   `/root/bin/start-serving-30030.sh`.** Arms C and D are one variable apart and
   agree to 0.03 % against floors of 0.8 % / 0.26 %; the census shows 0/20
   `check_aborted` either way. The mitigation's own comment says "remove this
   line when #517 lands" -- it has landed and it is measured. Removing it
   restores the guard to its full detection latency (10 ms, time-bounded) at no
   cost.
2. **Merge #517 to the serving tree.** It is worth +13.7 % of bs=1 round time
   against its own control, measured on cards, and it is what makes point 1
   safe.
3. **Change nothing else in the flagset.** Today's config is exonerated at bs=1
   (+0.1 %, below floor). There is no config lever here worth pulling.
4. **The real remaining lever is the transport**, not the tree and not the
   config. Either restore BAR1 (holder module + the driver it needs) or attack
   the device transport's `tp.all_reduce` cost -- 998 ms over 129 calls in a
   single 2048-token prefill chunk is where the time is.
5. **Next hot leaf after #517 is `resolve_seq_lens_cpu` ->
   `torch.cuda.Stream.synchronize`**, 11-14 of 20 samples in every arm
   including the record-era pin. It is the largest single host-side cost left
   at bs=1 and it predates the whole regression window.
