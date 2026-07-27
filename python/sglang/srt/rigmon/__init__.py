# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""``rigmon`` — host-side rig telemetry (design #97 / DESIGN_216).

The predecessor prototype (``tools/rig_dashboard/server.py``) sampled NVML and
scraped the engine **inside the HTTP request handler**, i.e. once per browser
poll. Four consequences made that unfixable by tuning:

1. **No history.** Sampling only happened while a tab was open; every gap in
   browsing is a gap in the data.
2. **No second node.** A browser on rig A cannot read the GPUs of rig B; only
   a process on rig B can.
3. **tok/s cannot be attributed from NVML alone.** Throughput lives in the
   engine, card state lives in NVML — joining them requires a process that
   sits where both are.
4. **A fixed sample rate is only guaranteeable server-side.** Client polling
   drifts with tab throttling, network jitter and page visibility.

So collection moves to the host:

    node A  ──[ Collector ]──┐                 ┌── read-only HTTP ── browser
                             ├─▶ [ Aggregator ]┤
    node B  ──[ Collector ]──┘ (push, outbound)

* :mod:`~sglang.srt.rigmon.series` — the time-series store: a cascade of
  ring buffers at configurable resolutions with automatic downsampling.
* :mod:`~sglang.srt.rigmon.sources` — NVML / engine sampling, with an
  explicit "field unavailable here, because X" for every field a given
  vendor/arch cannot supply.
* :mod:`~sglang.srt.rigmon.rates` — the honest per-rank view: group tok/s as
  the ONE throughput number, plus per-rank work share / work per watt /
  roofline position, which are attributable where tokens are not.
* :mod:`~sglang.srt.rigmon.capabilities` — the capability table, derived from
  PROBES rather than from configuration, with a reason attached to every
  "not available".
* :mod:`~sglang.srt.rigmon.collector` — the fixed-cadence node loop and the
  outbound push client (so the second node needs no inbound firewall rule).
* :mod:`~sglang.srt.rigmon.aggregator` — the multi-node store the web UI
  reads. The UI is a pure READER; it never touches hardware.
* :mod:`~sglang.srt.rigmon.bootwatch` — boot-log failure classification.
* :mod:`~sglang.srt.rigmon.kvbudget` — the measured-KV-budget cache, visible
  and resettable.
* :mod:`~sglang.srt.rigmon.provenance` — run identity (flags, commit, model
  hashes, probe used), so a comparison across time is admissible at all.

Everything in this package is import-light and CPU-only: no torch, no CUDA
context, no server boot. NVML access is read-only (``pynvml`` with an
``nvidia-smi`` fallback).
"""
