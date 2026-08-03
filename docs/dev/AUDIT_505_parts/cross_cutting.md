## Cross-cutting — the reach of the C1 remedy itself

The C1 incident (a packed draft weight name that matched no parameter, logged as a
warning, `continue`, speculative accept rate 0) was fixed by a completeness check in
the direction the loop does not cover. That fix exists and is well argued
(`models/deepseek_v4_dspark.py:868-886`):

> Every parameter the draft DECLARES must have been written. / The loop above drops
> a checkpoint tensor it cannot match to a parameter with only a warning, and that
> direction stays a warning on purpose: a checkpoint may legitimately carry tensors
> this build has no module for … The opposite direction must not be silent. A remap
> that produces a name matching no parameter leaves that parameter at its
> uninitialised construction value; the draft still "loads", and the only symptom is
> a speculative accept rate pinned at zero.

Pinned by `test/registered/unit/models/test_dspark_draft_load_completeness.py:28`.

**Its reach, counted at this tip** (`python/sglang/srt/models/`, 186 files defining
`load_weights`):

| | count |
|---|---|
| models defining `load_weights` | 186 |
| of those, containing a `logger.warning` in the loader | 55 |
| containing any completeness comparison at all | 11 |
| where that comparison **raises** | **2** — `deepseek_v4_dspark.py:896`, `dflash.py:582` |
| where it only warns | `deepseek_v4.py:3123`, `mllama4.py:718` |
| where it logs at a computed level | `gemma4_unified.py:434` (`logger.log(level, ...)`, DEBUG bucket for non-persistent buffers) |
| where it is commented out | `gemma3_causal.py:903-907` (upstream, `9d02bb3e2a`, 2025 — not a fork defect) |

So the remedy for the defect class this audit is named after currently binds in two
model files out of 186. Both are fork files, both are speculative-draft loaders —
i.e. exactly where the incident happened, and nowhere else. That is a REACH INCLUDES
PARAMETERS result in the structural rather than the numeric sense: the mechanism
exists, is correct, is tested, and covers ~1 % of the surface where the same silence
is possible.

*Task #505-X1:* `promote the load-completeness check to a shared helper and apply it to the fork's own bring-ups (Qwen3.5/3.6, Gemma4, DSV4/GGUF), not only the two draft loaders`

Note on `gemma3_causal.py`: independently of the commented-out check, `loaded_params.add(name)`
sits at `:902`, one indent level OUT of the block that assigns `name` in the inner
loop, so the set records only the last name of each outer iteration. Upstream code,
recorded for completeness, not proposed as fork work.
