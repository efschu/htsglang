# NOTE 616e — wedge specimen 1: the abort gate fires on time and raises late

Window: 2026-08-06, agent hunter-7. Specimen archived at
`/spinning/616e-hunter7/wedge_specimen_1/` (log, decoded flag matrix, findings).
Specimen tree: `/spinning/wt-616c-hunter5`, `fix/accept-index-616c` @ `02481079a3`.
Config: `tp_size=3`, `dcp_size=3`, `rank_gpu_id=[0,1,2]`, Qwen3.6-27B-INT8-W8A8,
`SGLANG_BARLINK_BAR1_CAP_CYCLES=3e11` (~156 s at 1.9 GHz), `watchdog_timeout=300`.

**Provenance, stated up front.** The specimen was not harvested live. The process
tree self-terminated at 20:06:21, about a minute before this window's first
probe. Everything below comes from what the instruments wrote before death, and
there are no py-spy stacks for it — see §4 for why that is not a coincidence.

## 1. Timeline

| time | event |
|---|---|
| 19:57:22 | last forward progress: `Decode batch ... cuda graph: True` |
| 19:57:22 | last `MAMBA-PIN-TRACE` on **all three** ranks; last completed request |
| 19:59:53 | health tic start |
| 20:00:13 | health check failed (no detokenizer response for 20 s) |
| 20:04:17 | TP0 collective census + 4096-entry history dumped |
| 20:06:01 | flag snapshots, `Bar1CollectiveAborted`, py-spy failure, watchdog 300 s |
| 20:06:21 | `kill_process_tree` |

freeze → abort **raised** = 519 s. Cycle cap = 156 s → expiry ≈ 19:59:58, which
sits within ~5 s of the first health tic. The cap almost certainly fired on
schedule; only the raise was late.

## 2. Reading the flag region (a wrong first answer, recorded on purpose)

A first pass assumed a flat `(block = i//R, sender = i%R)` layout and produced a
striking result: "every sender completed exactly one of its two peer writes,
perfectly cyclic, all three ranks symmetric." **That was an artifact of the wrong
layout and is retracted.**

The real layout is segmented by topology first
(`barlink_bar1_ext.py:1262-1268`, `barlink_bar1.py:986-1004`):

```
FSLOT(topo, step, sender) = FBASE[topo] + (step*R + sender)*256
FBASE = { mesh: 0, ring: 2*R*256 }
```

For R=3, one 256-byte line per `(topology, step, sender)`:

| lines | topology | steps |
|---|---|---|
| 0–5 | mesh | 2 |
| 6–17 | ring | 2(R-1) = 4 |
| 18–20 | a2a | 1 |
| 21–63 | unused | — |

which matches the dump exactly: only lines 0–20 are ever non-zero.

The cell value is the round counter (`round = *roundDev + 1`, written by
`writeU64`, waited on with `readFlag(...) != round`). A rank's own sender cell is
skipped everywhere (`if (s == r) continue;`) and so always reads 0.

**The consequence that invalidated the first reading:** in the ring topology a
rank receives *only* from its predecessor `(r-1) mod R`. The zero cells for the
non-predecessor sender are structurally never written. They are not missing
flags. Any future reader of these dumps should decode the topology segmentation
before drawing conclusions — the dump's own help text says "compare cells by
(block, sender)", which is easy to read as a flat matrix, and it is not one.

## 3. What the specimen actually shows: a round skew, rank 0 ahead

Decoded correctly, three of the four generations in flight are complete:

| region | round | state |
|---|---|---|
| mesh, both steps, all receivers | 2205518 | complete |
| a2a, all receivers | 2205521 | complete |
| ring, all 4 steps, each receiver ← predecessor | 2205528 | complete |
| **ring step 0, receiver rank 1 ← rank 0** | **2205531** | **the anomaly** |

Predecessor check confirms the ring is well formed at round 2205528: rank 0 ← 2,
rank 1 ← 0, rank 2 ← 1.

Exactly one cell disagrees. Rank 0 has published round 2205531 to its ring
successor while ranks 1 and 2 are still at 2205528, i.e. **rank 0 started a
collective the other two never started**. It then waits for its own predecessor
(rank 2) to publish 2205531 into its ring step-0 slot, that slot still reads
2205528, and the kernel spins to its deadline. This matches the abort text
verbatim: *"A peer did not arrive."*

The instrument also excludes two competing explanations by direct measurement:
*"The host abort word exists on this rank and was NOT set, which excludes it: no
peer was declared dead and no host wait gave up."*

Round spacing is +3 and +7, so the counter is shared across topologies and is
**not** a dense per-collective +1. Do not read 2205528 → 2205531 as "3
collectives".

**The lead worth pulling next:** the divergence direction is *rank 0 ran ahead*.
The question is therefore not "who stalled" but "which collective did rank 0
issue that ranks 1 and 2 did not" — a control-flow divergence, with the last
progress line placing the wedge inside a CUDA-graph decode replay.

## 4. The abort gate: armed, fired on time, raised 363 s late

The premise that this mode "runs past the cap without triggering it" does not
survive the log. The abort path *was* taken. What is broken is raise latency.

Per #517, detection is meant to be time-bounded: the abort-gate poll runs on the
barlink watchdog thread with `SGLANG_BARLINK_BAR1_ABORT_POLL_MS` (default 10 ms),
explicitly replacing the older check-count-lagged scheme.

Observed: **staging is time-bounded, raising is not.** The abort text says the
status word "was read from the STAGED copy (#517)". The watchdog thread stages
the abort promptly, but the exception is only thrown once the *main* thread
reaches a host-side check (`_after_transport` → `check(op)`). The same message
reports "4 collective(s) ran on the host path since the previous check".

*Inference, not proven by this log alone:* in this mode the main thread never
reaches such a check, because the wedge sits inside a captured graph replay where
the per-collective host code does not run. Detection then has to wait for an
out-of-band actor, and the one that arrived was the 300 s scheduler watchdog.

**Why that is itself a bug.** An abort known to the watchdog thread within 10 ms
should not wait an unbounded time for a main thread that, by construction of this
failure mode, is not going to volunteer. Once staged and still un-raised after a
bounded grace period, it should escalate out of band. That is the next cheap
hermetic fix and it is **not** implemented here.

**Confound:** 519 s is the freeze→*raise* gap. The moment the kernel set the abort
word is not logged, so "the cap fired at ~19:59:58" is computed and corroborated,
not measured. A timestamp written next to the abort word would settle it.

## 5. Instrumentation gap, fixed in this commit

The crash handler shelled out to a bare `py-spy` with `shell=True`, resolving it
against `/bin/sh`'s PATH. A server started as `<venv>/bin/python -m
sglang.launch_server` never puts the venv bin dir on PATH, so every automatic
dump in this window produced nothing:

```
[TP0] Pyspy failed (py-spy dump  --pid 20955). Error: /bin/sh: 1: py-spy: not found
[TP0] All pyspy dump attempts failed for PID 20955.
```

py-spy was installed at `<venv>/bin/py-spy` throughout. This is why the specimen
has no stacks and had to be read from flag words.

The fix resolves the binary next to `sys.executable` and runs an argv list
instead of a shell string. One trap worth recording: the first version of the fix
called `.resolve()` on `sys.executable`, which follows the venv symlink to
`/usr/bin/python3.12` — a directory holding no venv console scripts — and so
reproduced the exact bug it was meant to fix. The non-hermetic test that reads
the real venv is what exposed it, and both that test and a hermetic
symlinked-venv case are now in the suite.
