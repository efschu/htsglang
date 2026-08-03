# MOCK_SMOKE — what was actually executed at desk time

Desk phase, 2026-08-03. **No GPU was touched. No server was started. No
`nvidia-smi` workload was run.** The production serving tenant (`pgid 3850257`,
INT8-W8A8 Qwen3.6-27B on port 30030) was live throughout and was not disturbed.

Everything hermetic ran under `CUDA_VISIBLE_DEVICES=99`.

Interpreter: `/spinning/htsglang-gpu/.venv/bin/python`
`PYTHONPATH=/spinning/wt-dsv4f-window/python` where an `sglang` import was needed.

---

## EXECUTED — 1. `bash -n` on every shell file

```
$ for f in *.sh; do printf '%-24s ' "$f"; bash -n "$f" && echo OK; done
boot_462_f2.sh           OK
boot_470_dspark.sh       OK
boot_478_quant.sh        OK
lib.sh                   OK
restore_serving.sh       OK
```

Re-run clean after every later edit.

## EXECUTED — 2. `ruff check`

```
$ /spinning/htsglang-gpu/.venv/bin/ruff check probes.py extract_chat_template.py
All checks passed!
```

One finding was fixed rather than suppressed: `F401 json imported but unused`
in `extract_chat_template.py` (left over after the renderer was switched to
transformers' own jinja environment).

## EXECUTED — 3. `codespell`

```
$ codespell lib.sh probes.py extract_chat_template.py boot_478_quant.sh \
            boot_470_dspark.sh boot_462_f2.sh restore_serving.sh \
            logging_break_debug.json
(no output)
```

Seven findings were fixed rather than suppressed: three misspellings of
"unparsable", two of "keep-alive", one three-letter test-fixture byte string
that collided with a dictionary entry, and one truncated regex prefix widened
to `(disabled|disabling|off)`. (The literal spellings are deliberately not
reproduced here so this file itself stays codespell-clean.)

## EXECUTED — 4. Chat template: extraction, verification, selftest

```
$ python extract_chat_template.py --write
wrote 13698 bytes -> .../dsv4f_chat_template.jinja
sha256 e643c31fcec17f342f72296e02c46d35846bf4c70f6a0271f23bad73fd4eb645

$ python extract_chat_template.py --verify
verified: .../dsv4f_chat_template.jinja == GGUF tokenizer.chat_template (e643c31fcec17f34...)
```

The GGUF headers of **both** quants were read (header KV block only, never the
tensor section — a pure file read, safe on a 120 GiB checkpoint). Both carry
`tokenizer.chat_template`, 13698 bytes, **identical sha256**.

```
$ CUDA_VISIBLE_DEVICES=99 python extract_chat_template.py --selftest
  [PASS] renders marker '<｜User｜>'
  [PASS] renders marker '<｜Assistant｜>'
  [PASS] renders marker '</think>'
  [PASS] ends with the generation prompt -- 'r｜>Second question.<｜Assistant｜></think>'
  [PASS] can-fail: a marker-less template is rejected
  [PASS] the mangling is not a no-op -- 60 literal backslash-n sequences rewritten
  [PASS] file-load mangling is neutral: plain multi-turn
  [PASS] file-load mangling is neutral: no generation prompt
  [PASS] file-load mangling is neutral: thinking on
  [PASS] file-load mangling is neutral: with tools
  [PASS] can-fail: the equivalence comparison detects a real difference
  [PASS] content format detects as 'string' -- got 'string'
SELFTEST PASSED
```

What the mangling check is about: `TemplateManager._load_jinja_template`
(`parser/template_manager.py:269-270`) does **not** load a `.jinja` file
verbatim — it applies `.strip("\n")` and then
`.replace("\\n", "\n")`, rewriting every literal backslash-n in the file into a
real newline. This template has 60 of them inside jinja string literals. The
selftest proves the rewrite is semantically neutral across four message shapes
**and** proves (can-fail arm) that the comparison can detect a real difference.

A first draft of the renderer used a hand-rolled `jinja2.Environment` with
`StrictUndefined` and **failed** on `message['tool_calls']` — a message the
server renders fine. It now borrows
`transformers.utils.chat_template_utils._compile_jinja_template`
(`chat_template_utils.py:480-494`), the exact environment sglang's chat path
compiles with, so the selftest predicts server behaviour instead of
approximating it.

## EXECUTED — 5. `probes.py --selftest` (the instrument precondition)

```
$ CUDA_VISIBLE_DEVICES=99 python probes.py --selftest
probes.py selftest (hermetic: no GPU, no server, no network)

 scorers: can-discriminate on known-different inputs
  [PASS] arith_mul: accepts the known-good answer
  [PASS] arith_mul: rejects the known-bad answer -- want 391, got 402.0
  [PASS] capital_au: accepts / rejects           (Canberra vs Sydney)
  [PASS] letter_count: accepts / rejects         (3 vs 2)
  [PASS] reverse_word: accepts / rejects         (kcats vs stcak)
  [PASS] unit_convert: accepts / rejects         (250 vs 25)
  [PASS] geom_seq: accepts / rejects             (32 64 128 vs 32 48 64)
  [PASS] json_echo: accepts / rejects            ({"a":1,...} vs {"a":2,...})
  [PASS] date_arith: accepts / rejects           (Thursday vs Wednesday)

 reasoning stripping
  [PASS] strips everything up to the last </think>
  [PASS] leaves a plain answer alone
  [PASS] a scorer sees through a reasoning prefix

 accept reader
  [PASS] reads spec_accept_length
  [PASS] reads spec_verify_ct
  [PASS] REFUSES to substitute spec_ema_accept_len for the accept length
  [PASS] still records the EMA as provenance

 A-vs-A floor gate
  [PASS] spread_pct on a known pair
  [PASS] spread_pct of identical readings is 0
  [PASS] a 1% delta under a 5% floor is REFUSED
  [PASS] a 40% delta over a 5% floor is admitted
  [PASS] the signed delta keeps its sign

 rate derivation
  [PASS] ms/round from rounds, not tokens
  [PASS] ms/token stays separate
  [PASS] tok/s is present but labelled secondary
  [PASS] zero tokens yields None, never a divide-by-zero

 SSE parsing
  [PASS] parses a data: payload
  [PASS] ignores [DONE]
  [PASS] ignores keep-alive lines
  [PASS] ignores a malformed payload instead of raising

 idempotence comparator
  [PASS] identical texts hash the same
  [PASS] different texts hash differently (can-fail arm)

 prompt construction
  [PASS] prompt length grows monotonically with the target
  [PASS] the targets match the prior window's points

SELFTEST PASSED
```

44 checks, every scorer with both a known-good and a known-bad arm, and the
floor gate proven able to **refuse** as well as admit. No server, no network.

## EXECUTED — 6. `lib.sh` guards, hermetically, with can-fail arms

Driven from a synthetic `device_order.json` matching the measured rig
(5090 = CUDA 0 / NVML 1). No `nvidia-smi` call, no CUDA context.

```
lib.sh hermetic smoke (CUDA_VISIBLE_DEVICES=99, no GPU touched)

 assert_metrics_flag
  [PASS] accepts a line carrying --enable-metrics (rc=0)
  [PASS] CAN-FAIL: refuses a line without it (rc=1)

 assert_draft_gpu_is_5090 -- the priority defect
  [PASS] accepts CUDA ordinal 0 (the 5090) (rc=0)
  [PASS] CAN-FAIL: REFUSES NVML index 1 (a 3080 in CUDA space) (rc=1)
  [PASS] CAN-FAIL: REFUSES CUDA ordinal 2 (the other 3080) (rc=1)
  [PASS] CAN-FAIL: REFUSES an ordinal that does not exist (rc=1)

 assert_rank0_is_5090
  [PASS] accepts --rank-gpu-id 0,1,2 (rank0 -> cuda:0 -> 5090) (rc=0)
  [PASS] CAN-FAIL: refuses 1,0,2 (rank0 would be a 3080) (rc=1)

 nvml_index_for_rank (the UUID bridge)
  rank 0 -> NVML index 1
  rank 1 -> NVML index 0
  rank 2 -> NVML index 2
```

The last block is the whole point: with `--rank-gpu-id 0,1,2`, rank 0 sits on
**NVML index 1**. Any per-rank `nvidia-smi` reading that used the rank number
directly would read the wrong card for ranks 0 and 1.

## EXECUTED — 7. `power_tag`, `rammon`, `stop_server` against a fake `nvidia-smi`

A stub `nvidia-smi` on `PATH` emitted the three-card CSV; no real driver call.

```
power_tag smoketest/start: 3 GPUs recorded
power_tag smoketest/end: 3 GPUs recorded

$ jq -r '[.[].phase]|@csv' powerstate_smoketest.json
"start","end"
5090 power.limit at start: 400.00 W
gpus per record: [3, 3]
```

Confirms the **deliberate deviation** documented in `lib.sh`: the briefing asks
for `powerstate_<arm>.json` written at start *and* end; writing one filename
twice would destroy the start reading, which is the value the rule exists to
preserve. Each call appends to `powerstate_<arm>.jsonl` and rewrites the
`.json` as the array of all records. Both phases survive.

```
$ head -4 ram_smoketest.log
utc                     memory_current_bytes  anon_bytes    file_bytes    mem_available_kib
2026-08-03T23:25:36Z    48057622528           19170537472   27868557312   95784762
2026-08-03T23:25:37Z    48041070592           19154837504   27868557312   95800178
2026-08-03T23:25:38Z    48044154880           19158831104   27868557312   95798201
```

`anon` and `file` recorded separately, as required — with no swap on this host,
`anon` is the structurally unreclaimable term and `file` is the reclaimable page
cache; that split is the feasibility argument.

```
$ stop_server nonexistent
[..] stop_server: no pidfile for nonexistent, nothing to stop
returned 0 as required
```

## EXECUTED — 8. `logging_break_debug.json` is a valid `dictConfig`

Loaded exactly as sglang loads it (`orjson` → `logging.config.dictConfig`,
`srt/utils/common.py:2685-2693`), including the `_comment` key:

```
dictConfig accepted _comment; module level = DEBUG root = INFO
orjson + dictConfig OK
```

Logger name verified against source: `breakable_cuda_graph.py:43`
(`logger = logging.getLogger(__name__)`) and `:230`
(`logger.debug("Break graph due to function: %s", inner.__name__)`).

## EXECUTED — 9. Static verification of every flag, string and path used

Read from source in this worktree, not from memory:

| claim | source |
|---|---|
| `--speculative-draft-gpu` takes a CUDA ordinal | `server_args.py:3581-3589` |
| `--rank-gpu-id` is CUDA-indexed | `server_args.py:8476-8477` |
| per-rank vectors zip positionally against it | `server_args.py:9111` |
| solo-draft rank resolution | `server_args.py:7086-7117` |
| `validate_breakable_boot` preconditions | `offload_capture_gate.py:311-420` |
| `SGLANG_BREAK_COST_PATH` used verbatim, no rank tag | `break_cost_clock.py:513` |
| rank tag comes from `torch.distributed` | `break_cost_clock.py:478-494` |
| `summarise.py` exists and groups by `rank_tag` | `scripts/dev/494_break_cost/summarise.py` |
| `Preparing MXFP4 experts for Marlin backend` | `mxfp4_marlin_moe.py:133` |
| `is a draft SHADOW` | `model_runner.py:544` |
| `deepseek-v4` reasoning parser registered | `parser/reasoning_parser.py:1132` |
| `deepseekv4` tool-call parser registered | `function_call/function_call_parser.py:66` |
| `/health_generate` exists | `entrypoints/http_server.py:861-862` |
| jinja file load mangles `\n` | `parser/template_manager.py:269-270` |
| `SGLANG_LOGGING_CONFIG_PATH` honoured | `srt/utils/common.py:2685-2693` |
| `compressor_v2.forward_unified` write sites | `layers/attention/dsv4/compressor_v2.py:516-596` |
| no runtime resident-fraction endpoint exists | searched `http_server.py`, `srt/managers/` — no match |
| `--cuda-graph-bs-decode` is the live flag | `server_args.py:3095-3097` |
| model shards, `dspark-head-filtered` contents, sizes | `ls` / `du`: IQ3_XXS 98 G, Q3_K_XL 120 G |
| `prompts.json` schema (`name`/`domain`/`text`/`max_new_tokens`) | `2026-08-03_447_dspark/prompts.json` |
| `/tmp/w530_boot.sh` recipe and its own bounded wait | read in full |
| swap is 0, MemAvailable 90 GiB with serving up | `/proc/meminfo` |
| translator front door listens on 30800 | `ss -ltnp` (read-only) |

---

## NOT EXECUTED — labelled DESK-WRITTEN, NEVER EXECUTED

Everything below has never run. It is written from verified sources but no line
of it has executed against hardware.

1. **`boot_478_quant.sh`** — both arms. Never launched a server.
2. **`boot_470_dspark.sh`** — all three sub-arms. The DSpark solo path has never
   booted on this rig on any placement (TICKET_470 §7 is the standing inventory
   of what that ticket is the only evidence for).
3. **`boot_462_f2.sh`** — all three arms. The breakable route has never been
   served here.
4. **`restore_serving.sh`** — deliberately not executed; the serving tenant is
   currently up and this script would tear down and re-boot it.
5. **`lib.sh::preflight`** — the `nvidia-smi`-dependent branches (compute-app
   emptiness, per-card corridor). Only the swap / MemAvailable / holder logic
   was exercised indirectly.
6. **`lib.sh::resolve_cards`** — needs a real CUDA context
   (`registry/nvml.py:588-604`: `get_device_properties` goes through
   `_lazy_init` and costs a few hundred MiB on every visible card). Its outputs
   were smoked from a synthetic `device_order.json` instead. The **torch-vs-NVML
   bridge-disagreement branches** (exit codes 4/5/6) have therefore never
   fired.
7. **`lib.sh::wait_ready` / `arb_claim` / `arb_heartbeat_stop`** — never run
   against a real server or a real holder file.
8. **`probes.py` server-facing modes** — `prefill`, `decode`, `avsa`,
   `determined`, `accept`, `chatprobe`, `idem-record`, `idem-compare`. Only
   their pure helpers were selftested. In particular `stream_bounded`'s SSE
   loop has never seen a real sglang stream: **the assumption that
   `meta_info.spec_verify_ct` and `completion_tokens` appear on streamed
   chunks is UNVERIFIED.** The code degrades named (`round_kind` reports
   `"token (no spec_verify_ct in meta_info)"`) rather than silently, but the
   first F2/DSpark arm must check that line before any ms/round is quoted.
9. **The `43` crossings assertion** — the number comes from TICKET_462 §3; it
   has not been observed here.
10. **The determined-answer scorers against the actual model.** They are proven
    to discriminate; whether DSV4F answers "Thursday" or spells `kcats` is
    unknown. Expect to re-tune the set on first contact, and record the
    baseline arm's score before treating it as a quality gate.
11. **`SGLANG_LOGGING_CONFIG_PATH` inside a real launch.** The dict is valid and
    the logger name is right, but whether `configure_logger` runs early enough
    in every worker to catch the break-graph DEBUG lines is unverified. If the
    independent count comes back 0, that is the first thing to suspect — the
    script warns rather than failing the arm on it.

---

## Contradictions found between the briefing/tickets and the code

Recorded, not smoothed over. **In all three, the code wins.**

### C1 — `--speculative-draft-gpu` is a CUDA ordinal, not an NVML index

The briefing (and TICKET_470 §3, and `rig-runbook` §4.5.4 as quoted) say "NVML
index of the 5090", with the aside "today it happens to be index 1". The code
says the opposite, verbatim at `server_args.py:3581-3589`: *"CUDA device index
(torch.cuda order, same space as `--rank-gpu-id`)"*. On this rig the 5090 is
CUDA 0 / NVML 1, so the briefed value would have designated a **3080** — and
would not have errored, because rank 1 legitimately maps to cuda:1. The MXFP4
Marlin path is SM90/SM120 only, so the arm would have measured the wrong kernel
or refused deep inside the draft build.

Fixed by `assert_draft_gpu_is_5090`, whose can-fail arms are executed above.
(The launching agent independently sent the same correction mid-task; it was
already found and fixed.)

### C2 — the GGUF checkpoints DO carry a chat template

The briefing says they carry none and instructs me to produce one. The
`tokenizer_config.json` sidecars indeed have no `chat_template`, but the GGUF
metadata KV block does — 13698 bytes, identical across both quants. Extracting
the authoritative template is strictly better than desk-writing a plausible
one, and the cross-quant identity is itself a precondition for arm 1.

### C3 — `SGLANG_BREAK_COST_PATH` is **not** expanded per rank

The briefing sets it to `"$RUN/break_cost"`; TICKET_462 §3 sets it to
`"$RUN/break_cost.jsonl"` and comments `# becomes one file per rank`. Neither is
true: `break_cost_clock.py:513` uses a supplied value verbatim, so all three TP
ranks would append to one file and the ticket's own readout glob
`"$RUN"/break_cost.rank*.jsonl` would match nothing. `boot_462_f2.sh` leaves the
variable unset, takes the documented per-rank `/tmp` default, clears stale files
first, and copies the results into `$RUN`.

### C4 (deviation, not a contradiction) — the #470 residency cut needs two boots

TICKET_470 §2 asks for the cut "same boot".
`--rank-moe-resident-fraction` is a launch-time `ServerArgs` field and no
runtime endpoint changes it. The ticket's own escape clause covers this; Boot A
is split into `a_base` and `a_cut`, each with its own A-vs-A floor.

### C5 (deviation) — `powerstate_<arm>.json` written twice

Writing the named file at start and again at end would destroy the start
reading. Each call appends to a `.jsonl` and rewrites the `.json` as the array
of all records. The named artifact still exists and nothing is lost.

---

## Places where I had to guess rather than derive

Named explicitly so they can be checked first in the window.

| guess | why it is a guess | how to settle it |
|---|---|---|
| `RESIDENT_FRACTION_CUT=0.383,0.42,0.42` | 0.485 × 0.79, from TICKET_470 §0's "~21 %". §7.6 says the ~11 GiB ask is arithmetic, not a measurement | override from the GGUF footprint analysis before the window |
| `MEM_AVAIL_FLOOR_GIB=96` | `SGLANG_GGUF_STREAM_TRIM_SOFT_GIB` (88) + 8 GiB headroom. The headroom term is mine | lower it deliberately if a q3kxl boot refuses |
| `READY_ITERS` 90 / 132 / 108 | IQ3 measured ~5.5-6 min; Q3_K_XL scaled by the 120/98 GiB size ratio and rounded up; the DSpark arm +3 min for the head | first boot's `ready_<arm>.txt` settles all three |
| the determined-answer set | chosen for determinacy, not from a prior DSV4F run | score the baseline arm first |
| prefill prompt lengths | `build_prompt` targets 240/480/940/1850 at ~4 chars/token; actual `prompt_tokens` is recorded next to the target | the recorded actuals |
| `--chat-probe-tokens 24` | short enough to dodge the known GDN prefill nondeterminism above ~109 tokens | if the probe reports `instrument void`, lengthen it |
| §3.1 check 2 (markov_w2) is a WARNING | the exact log wording is not pinned anywhere in this tree; failing an arm on an unpinned string would fail it for the wrong reason | read the log once and pin the real string |
| translator smoke = one MT chat call on 30030 + `/metrics` on 30800 | the full front-door gate is `front_door_test.py` and needs audio + playwright, which is not a bounded end-of-window step | run `front_door_test.py` manually if the light smoke warns |
| `restore_serving.sh` restores the SERVING holder line over the one `w530_boot.sh` writes | the briefing says restore the SERVING line verbatim; `w530_boot.sh` writes its own | both are saved (`holder_after_w530_boot.txt`); confirm which the user wants |
