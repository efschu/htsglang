# #493 — the DSV4 indexer prefill transient: turnkey A/B for the next GPU window

Desk work is done and no GPU was taken. What is left is one measurement, and it
fits inside any boot of the DeepSeek-V4-Flash recipe — it needs **two runs of the
same prompt with one env var different**, plus two probes that already exist.

## The claim under test

The corridor breach of window 3 (2026-08-03, `corridor.csv`: both 3080 ranks fell
from 873 MiB free to 271 MiB, 214 samples under the 400 MiB floor) is the
paged-MQA-logits transient of the C4 indexer, and nothing else.

Modelled, at that run's geometry (`--chunked-prefill-size 256`, C4 span 8196,
`SGLANG_DSV4_INDEXER_LOGITS_SEQ_CHUNK=2048`, `index_n_heads=64`,
`index_head_dim=128`):

| `SGLANG_DSV4_INDEXER_QUERY_CHUNK_MIB` | chunk_rows | steps | peak | free at peak |
|---|---|---|---|---|
| `0` (pre-#449 shape) | 256 | 1 | 588.0 MiB | 285 MiB — **breach** |
| `2048` (what #449 shipped) | 256 | 1 | 588.0 MiB | 285 MiB — **breach** |
| `256` (#493 default) | 112 | 3 | 261.8 MiB | 611 MiB — OK |

The middle row is the defect: the cap existed and did not bind.

Run `python3 predict.py` to regenerate this table for any other geometry — it
calls the production formula, so it cannot drift from the code it predicts.

## Procedure

Hold `/spinning/gpu-arb/` with a heartbeat; stop the heartbeat before releasing.

1. **Predict**, and keep the number:

   ```
   CUDA_VISIBLE_DEVICES=99 python3 predict.py --rows 256 --span 8196 --seq-chunk 2048
   ```

2. **Arm both probes** in the boot script (they are off unless set):

   ```
   export SGLANG_FORWARD_PEAK_PATH=$RUN/peak_off      # arm A
   ```

3. **Arm A — budget off**, i.e. the shape that breached:

   ```
   export SGLANG_DSV4_INDEXER_QUERY_CHUNK_MIB=0
   ./sample_corridor.sh $RUN/corridor_off.csv 100 &
   # boot, send the 32K prompt, generate ~16 tokens, shut down cleanly
   touch $RUN/corridor_off.csv.stop
   ```

   The clean shutdown is load-bearing: `forward_peak` dumps at exit.

4. **Arm B — shipped default**, same boot script, same prompt, only these two
   lines different:

   ```
   export SGLANG_FORWARD_PEAK_PATH=$RUN/peak_on
   unset SGLANG_DSV4_INDEXER_QUERY_CHUNK_MIB      # take the 256 MiB default
   ./sample_corridor.sh $RUN/corridor_on.csv 100 &
   ```

5. **Verdict**:

   ```
   python3 verdict.py --off $RUN/peak_off --on $RUN/peak_on \
       --off-corridor $RUN/corridor_off.csv --on-corridor $RUN/corridor_on.csv \
       --predicted-delta-mib 326
   ```

   Exit 0 only when BOTH halves pass. Half 1 is the falsifier: if
   `peak_bytes_max` does not fall by roughly 326 MiB per rank between the arms,
   the attribution is wrong and #493 must be reported as refuted, not as fixed.

## Two traps this run must not repeat

**Sample fast enough to see a transient.** Window 3 used `sleep 1` in a shell
loop. The transient is sub-second and recurs once per prefill chunk (~12 s apart
on that recipe), so the trace caught it only occasionally and at random points on
its rise and fall. Its 602 MiB excursion is therefore a LOWER bound on the peak,
and the apparent growth of the dip over the first 700 s of prefill is
extreme-value statistics of undersampling, not a ramp. `sample_corridor.sh` uses
`nvidia-smi -lms` (100 ms, no per-sample process start); `forward_peak`'s
per-forward `nvml_free_bytes_min` remains the authoritative number.

**Do not pull large files in a RAM-near window.** Attempt 1 of window 3 was
killed by the cgroup OOM killer, not by CUDA: `memory.peak` hit the 104 GiB
swapless limit, and the ~5.4 GB of model files plus a torch install pulled
earlier in the same window were charged to it as page cache. Attempt 2 survived
only after tightening the GGUF stream trim to
`SGLANG_GGUF_STREAM_TRIM_SOFT_GIB=70` / `SGLANG_GGUF_STREAM_TRIM_TARGET_GIB=60`,
and still ran the whole prefill pinned at 100–101 GiB. Set the trim before the
boot, and do the downloads in a different window.

## Speed, if it is cheap to take

The cap trades allocation for loop iterations: 1 step becomes 3 at this geometry.
`NOTE_449` section 5 asked for the launch-count counter-check and never got a
window. If the same boot can afford it, record ms/prefill-round **per rank** in
both arms, interleaved A/B/A/B with an A-vs-A floor first. The memory reading is
the primary result either way — this is a corridor fix, not a speed one.
