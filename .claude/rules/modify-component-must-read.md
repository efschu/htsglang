# Must-Read Skills Before Modifying Components

Before modifying the following components, read the listed skill first.

- **Speculative decoding code** (anything under `python/sglang/srt/speculative/`, related attention backends, scheduler accumulators, IPC fields, observability metrics, or CLI flags) → [`speculative-naming`](../skills/speculative-naming/SKILL.md)
- **`Scheduler` / `TokenizerManager` / `ModelRunner` `__init__`** (`python/sglang/srt/managers/scheduler.py`, `python/sglang/srt/managers/tokenizer_manager.py`, `python/sglang/srt/model_executor/model_runner.py`) → [`large-class-style`](../skills/large-class-style/SKILL.md)
- **Any edit to a frozen core file** (currently `python/sglang/srt/model_executor/model_runner.py`) → [`large-class-style`](../skills/large-class-style/SKILL.md)
- **Environment variables** (adding, renaming, or reviewing any `SGLANG_*` env var, migrating a legacy `SGL_*` alias, or touching `python/sglang/srt/environ.py`) → [`env-var-conventions`](../skills/env-var-conventions/SKILL.md)
- **Scripted runtime** (anything related to the scripted runtime) → [`scripted-runtime-notes`](../skills/scripted-runtime-notes/SKILL.md)
- **Dual-Layout code (27B NVFP4)** (the dual parts of `python/flliper/srt/pdflip/launcher.py` -- `resolve_dual_layout`, `start_dual_mps`, `dual_p_sm_env`, `dual_duty_env`, `refuse_dual_priority`, the P/D start -- plus `python/flliper/srt/planner/pp_cut_launch.py`, `python/flliper/srt/managers/scheduler_phase_layout_mixin.py`, `python/flliper/srt/managers/pdflip_memory_saver.py` (dual P), `python/flliper/srt/pdflip/dual_*.py`) → [`docs/dev/dual-layout.md`](../../docs/dev/dual-layout.md) (the user's explanation of the dual layout: weights in the D layout, P loads only the byte difference, no all-reduce in prefill, prefill and decode at once, GEMM share by load via MPS; plus the open questions)
