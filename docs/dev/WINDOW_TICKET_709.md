# WINDOW TICKET — #709: uneven-TP proportional A/B in TP decode

**Owner lane:** #705/#709. **Estimate ~15 min** plus one reboot between arms.
Needs the serving lane, because `--rank-tp-ratio` is a **boot flag**.

## What it decides

#705 priced **uneven-TP proportional sharding** at **+0.780 ms/round at zero
capacity cost**, already shipped and simply not enabled (`rank_tp_ratio=None`).
This A/B confirms or declines that on metal.

## THE ACCEPTANCE RULE HAD TO CHANGE, and this is the ticket's main finding

The obvious rule — *"CONFIRM if the decode round improves beyond the A-vs-A
floor"* — **cannot return the right answer**:

| | |
|---|---:|
| #705 predicted gain | 0.780 ms |
| typical bs=1 decode round | ~30 ms |
| **gain as a fraction of the round** | **2.6 %** |
| rig A-vs-A noise floor | 14.1 % |

The predicted win is **5.4× smaller than the floor it would have to clear**. An
end-to-end rule would DECLINE a fully real win, every time, on a correct
implementation.

On the **family slice** the same delta is `0.780 / 2.506 = 31.1 %` — resolvable.
And `utils/collective_clock.py` exists for exactly this: its docstring says
*"the spread of `wait` across ranks is the shard-imbalance signal"*. Under an
equal shard the 5090 finishes early and waits; under a proportional shard that
wait should collapse.

So the acceptance is:

* **PRIMARY discriminator — per-rank WAIT SPREAD (max−min).** CONFIRM needs it
  to fall beyond the floor.
* **SECONDARY, reported but NOT the discriminator — end-to-end round.** A null
  here is *expected* and is not evidence against the lever. The report says so
  in its own output so a reader cannot mistake the silence for a refutation.
* **GATE — coherence.** The lever is lossless; a changed determined answer
  voids a speed win regardless of the numbers.

## The ratio vector is constrained — "just enable proportional" is not a thing

`--rank-tp-ratio` takes `auto`, `auto-performance`, or an integer list, and
**`sum(weights)` must divide every sharded dimension**.

* **`auto` is CAPACITY-first** — its own help says it "does NOT optimize for
  speed: it maximizes the KV pool and ignores how fast the cards are". On this
  rig that is the VRAM ratio 32:20:20 ≈ **1.6:1:1**, *not* the
  bandwidth-proportional **2.36:1:1** (1.79 vs 0.76 TB/s) that #705's +0.780
  was derived from. Arm B on `auto` would test a different lever.
* The practical integer vector is **`2,1,1`** (sum 4 divides 5120). Its ratio
  is 2.0 against the ideal 2.36 — a **~15 % shortfall**, so it cannot deliver
  the full +0.780 and must not be judged as though it should.
  `acceptance.admissible_ratios()` enumerates what this checkpoint permits.

**Record the flag verbatim.** The runner refuses an arm whose `--ratio-flag` is
unrecorded, because the three modes mean different things.

## Run it

```
# arm A: current config (rank_tp_ratio=None), server already up
python bench/709/run_709_ab.py --arm A_equal --boot-id bootA \
    --ratio-flag None --clock-json /tmp/clockA.json --out /tmp/709_A.json

# reboot with --rank-tp-ratio 2,1,1, then
python bench/709/run_709_ab.py --arm B_proportional --boot-id bootB \
    --ratio-flag 2,1,1 --clock-json /tmp/clockB.json --out /tmp/709_B.json

python bench/709/run_709_ab.py --report /tmp/709_A.json /tmp/709_B.json
```

The script **boots nothing** — it attaches to whatever is on `--port`. Each arm
measures its **own A-vs-A floor in its own boot**; the cross-boot delta must
clear the **larger** of the two. `--rank-tp-ratio` is a boot flag, so the arms
are unavoidably cross-boot; the floor is the one thing that can still be
same-boot, and importing one is refused.

## The one input the window must supply

`--clock-json`: per-rank COMPUTE/WAIT from the server's CollectiveClock. It
does not come over HTTP, and it is the only resolvable discriminator — so the
harness **refuses to judge without it** rather than returning a confident null
from the metric that cannot see the effect. Schema:

```json
[{"rank":0,"card":"5090","batch":1,"ms_round":30.1,
  "ms_compute":27.0,"ms_wait":3.1,"seconds_measured":12.0}]
```

## Desk-verified without a GPU

17 hermetic pins, including: the predicted gain is under the floor (the finding
above, asserted as arithmetic); a collapsing wait spread CONFIRMs; an unchanged
spread DECLINEs with numbers; a changed answer voids a speed win; the larger of
the two per-boot floors is used; two arms from one boot are refused; short runs
are refused; an unrecorded ratio flag is refused; and `2,1,1` is shown to fall
short of the bandwidth ideal. `--dry-run` exercises the whole acceptance path
with no server.

## Not established

No numbers measured. Nothing has run against a server, and the per-rank clock
plumbing on the decode path has not been exercised — if `--clock-json` cannot
be produced, that is the first thing the window discovers, and it is a
five-minute check before committing the reboot.
