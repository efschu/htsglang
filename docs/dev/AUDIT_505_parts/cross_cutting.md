## Cross-cutting — the reach of the C1 remedy itself

**There are TWO completeness remedies in this tree, at different altitudes, and
neither covers the other's surface.** Read this section together with axis A1's
finding #505-A1-01, which audits the loader-level one in depth; this section audits
the model-level one and states the combined picture.

- **Loader-level:** `model_loader/weight_utils.raise_on_unloaded_draft_parameters`
  (`:2032`), called once, from `DefaultModelLoader` (`model_loader/loader.py:903`).
  Its own docstring states the ambition — *"Hoisting the check to the loader makes
  it a property of loading A DRAFT, not of one model class"* — and immediately below
  it, two silent-return arms bound it: `if loaded_params is None: return` (`:2058`)
  and a `if not loaded: return` for the empty-set case (`:2063-2067`). Reach and the
  GGUF / QuantizedRL gaps are quantified in **#505-A1-01**.
- **Model-level:** `models/deepseek_v4_dspark.py:868` `_assert_required_params_loaded`,
  counted below.

The two do not compose: the loader guard fires only for DRAFT loads through
`DefaultModelLoader` on self-reporting models, and the model-level check exists in
two files. Everything else in `models/` is on the warning path.

The model-level fix is well argued (`models/deepseek_v4_dspark.py:868-886`):

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

So the model-level remedy binds in two files out of 186, both fork files, both
speculative-draft loaders — i.e. exactly where the incident happened and nowhere
else — and the loader-level remedy that was written to generalise it reaches, per
#505-A1-01, three draft classes and no GGUF boot. That is a REACH INCLUDES PARAMETERS
result in the structural rather than the numeric sense: the mechanism exists, is
correct, is tested, and covers a few percent of the surface on which the same silence
is possible. The remedy was hoisted once already, for exactly this reason, and the
hoist did not travel.

*Task #505-X1:* `promote the load-completeness check to a shared helper and apply it to the TARGET bring-ups (Qwen3.5/3.6, Gemma4, DSV4/GGUF), not only the draft path` — this is the target-side twin of #505-A1-01 and #505-A1-08; the three should be scoped together rather than fixed one loader at a time.

Note on `gemma3_causal.py`: independently of the commented-out check, `loaded_params.add(name)`
sits at `:902`, one indent level OUT of the block that assigns `name` in the inner
loop, so the set records only the last name of each outer iteration. Upstream code,
recorded for completeness, not proposed as fork work.
