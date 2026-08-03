# DESIGN #466 — Live speech-to-speech translator, voice-preserving

Status: **Phase 1 complete (survey + architecture + selection + MVP skeleton),
desk only.** No GPU was taken; every python invocation ran under
`CUDA_VISIBLE_DEVICES=99`. Hard deadline ~2026-08-10.

The system: a phone streams microphone audio over a WireGuard tunnel to the
rig. Each closed utterance is recognized, attributed to a speaker, translated
by our own htsglang server over its OpenAI-compatible API, and re-synthesized
so that **everyone keeps their voice and speaks a different language**.

---

## 0. Dated user decisions

These are decisions, not findings. They override anything below that predates
them.

| Date | Decision |
|---|---|
| 2026-08-03 | **License gate lifted for runtime backends.** This is a private, non-commercial deployment. Restrictive licenses (CPML, CC-BY-NC, gated weights) do NOT disqualify a backend. The constraint is repo cleanliness: no restrictively-licensed source vendored into this Apache-2.0 repository, no weights redistributed. Candidates are ranked on fitness; license is annotated factually so a future public/commercial decision can swap backends through the interface. |
| 2026-08-03 | **Accent carry-over is wanted.** The speaker's original-language accent surviving into the cloned translated output is explicitly fine and "even cool", provided intelligibility holds. Quality ordering for TTS selection: **intelligibility first (hard gate), speaker similarity second (ranked), accent character neutral-to-positive (never penalised)**. OmniVoice's documented cross-lingual accent carry is therefore a feature here, not a defect. |
| 2026-08-03 | **Voice mode is a per-session option.** `clone` (each speaker in their own voice) and `preset` (each speaker gets a distinct artificial voice, class-matched to man/woman/boy/girl) are both first-class, switchable at runtime. `preset` is also the automatic degradation path when a reference buffer is too short or too noisy, and the downgrade is visible per speaker. |
| 2026-08-03 | **Preset pool sizing.** Realistic worst case 6-8 participants, usually 2-4. Pool shape **6 man / 6 woman / 3 boy / 3 girl = 18**: eight participants can plausibly skew hard to one adult class, children rarely exceed three. On exhaustion: never crash, never silently reuse an identical voice — share a base voice with a deterministic pitch offset and raise a named notice. |
| 2026-08-03 | **Crash-safety backup.** Every restrictive or irreplaceable component (weights, pinned wheels, patched libs) is backed up to a PRIVATE GitHub repo. Private forever — CPML and NC terms permit private copies, not redistribution. The public fork never points at it. |
| 2026-08-03 | **Target phone:** OnePlus 10 Pro NE2213_11, Android 16, unlocked bootloader, **not rooted**. Nothing may require root. |

---

## 1. Architecture: cascade, not end-to-end

```
 phone ──WireGuard──► WS /api/translator/stream
                       │
                       ├─ VAD + turn segmenter        (hangover closes the turn)
                       ├─ ASR + language ID           (what was said, in which language)
                       ├─ speaker embedding + cluster (who said it; reference buffer)
                       ├─ route: detected language → target(s) by elimination
                       ├─ MT  ── HTTP ──► htsglang /v1/chat/completions   ← the dogfood hop
                       └─ TTS conditioned on THAT speaker's voice (or their preset)
                       │
 phone ◄──audio frames─┘
```

**End-to-end S2ST was surveyed and rejected on evidence, not preference.** No
open-weight end-to-end model does DE↔ES with speaker preservation:

- Hibiki / Hibiki-Zero (Apache code, CC-BY-4.0 weights — permissive, and the
  best-architected streaming S2ST found): **French→English only**, stated on
  the model card.
- StreamSpeech (MIT): **X→English only**, unit HiFi-GAN with a fixed voice.
- SeamlessM4T v2 / SeamlessStreaming: cover de↔es, but synthesize with a
  **fixed synthetic vocoder voice** — no speaker preservation. Secondary
  sources claiming otherwise conflate them with SeamlessExpressive's PRETSSEL.
- SeamlessExpressive: preserves vocal style, but `gated: manual`.
- Qwen3-Omni: **three preset voices only**, no arbitrary reference cloning.
  Independently, ANALYSE_334 §3a already recorded that its Talker/Code2Wav
  audio-out is a three-stage tenant-composition problem gated behind #333, not
  a model-loading task.

Zero end-to-end models accept a reference clip. True cloning exists only on the
TTS side. The cascade is therefore the only architecture that meets
requirement 1, and it also keeps our 27B multilingual LLM — by far the
strongest translator in the stack — on the translation hop.

### 1.1 Why a separate process, not an `srt` tenant

Same escape hatch the Class-3 video tenant took (`video_enhance/tenant.py`).
Per DESIGN_333 §2.3 the scheduler class that would host ASR/TTS (Class 3)
**does not exist yet**; building it is L-effort and this feature has a deadline
in days. So the translator runs as its own process with its own CUDA context,
pinned to one card by NVML UUID, with an absolute MiB budget meaning the whole
budget (no implicit ceiling, no safety factor — leaving headroom is the
operator's job). Nothing in `sglang/srt/translator/` imports `srt` internals,
so it becomes a Class-3 citizen later without its interfaces changing.

A second, decisive reason emerged from the survey: **dependency incompatibility
is structural**. `qwen-tts` pins `transformers==4.57.3` while the venv carries
5.12.1 and sglang; vLLM-Omni pulls vLLM against sglang's torch. A separate venv
per audio tenant is not tidiness, it is the only way these coexist.

### 1.2 Transport: WebSocket over WireGuard, not WebRTC

WebRTC would give congestion control, NACK, jitter buffering and DTLS-SRTP —
genuinely better for lossy mobile audio. It also drags in ICE/STUN/TURN, SDP
negotiation and a media server. Over WireGuard the peers share one private
subnet, so ICE has nothing to discover and the tunnel already encrypts. What
WebRTC would actually buy here is uplink loss recovery, and half-duplex
turn-taking is tolerant of the jitter its machinery hides — the audio is
seconds behind the speaker by design. Against a one-week deadline, WebSocket is
one connection, one reconnect path, and a wire format that can be printed.

Codec: Opus ~24 kbps both directions when PyAV is present, PCM16 negotiated
fallback (always available, ~256 kbps at 16 kHz — fine on the LAN, too
expensive for Spanish mobile data). The client advertises, the server picks,
and a deployment without a decoder simply does not offer Opus rather than
failing on the first frame from Spain.

---

## 2. Component survey and verdicts

Full evidence and URLs live in the two survey agent reports; this is the
decision record. **Note on provenance:** the surveying agents' knowledge
cutoffs precede several of the strongest candidates (Nemotron-3.5-ASR,
Qwen3-TTS, VoxCPM2, Voxtral Realtime), which is why the picks below are not
the Whisper/XTTS pair one would have guessed six months ago.

### 2.1 ASR + language identification

| Candidate | License (code / weights) | Streaming | Languages (de/es) | VRAM | Verdict |
|---|---|---|---|---|---|
| **nvidia/nemotron-3.5-asr-streaming-0.6b** | Apache-2.0 / **OpenMDW-1.1** (permissive) | true cache-aware, 80 ms–1.12 s | 40 locales, both in the transcription-ready tier; built-in LID via `target_lang=auto` | ~2–2.5 GB | **PRIMARY** |
| **faster-whisper large-v3-turbo** | MIT / MIT | chunked only | 99, published list | 1545 MB int8 | **FALLBACK (certainty)** |
| **nvidia/parakeet-tdt-0.6b-v3** | Apache-2.0 / CC-BY-4.0 | cache-aware capable | 25 European, auto-LID | ~2–2.5 GB | second fallback |
| Voxtral-Mini-4B-Realtime | Apache-2.0 both | true streaming, 80 ms grid | 13 incl. de/es | **≥16 GB** | reject on budget |
| Qwen3-ASR-0.6B/1.7B | Apache-2.0 | via vLLM | 52 | 1.5–4 GB | fallback; streaming is vLLM-coupled and we are an sglang fork |
| Kyutai STT | MIT/Apache / CC-BY-4.0 | true streaming | **EN/FR only** | — | reject, language set fatal |
| Moonshine | MIT | true streaming, no 30 s pad | **no German** | <1 GB | reject |
| distil-large-v3.5 | MIT | chunked | **English only** | 756 MB | reject |
| Qwen3-Omni-30B-A3B | Apache-2.0 | real-time | 19 | 79–145 GB | reject on budget |

**The decisive technical argument, and it is architectural rather than about
speed:** Whisper pads every input to a fixed 30 s mel window, so a 4 s
conversational turn costs the same encoder pass as a 30 s one. Our entire
workload is short turns. Cache-aware streaming CTC/TDT models scale with the
actual utterance. Whisper remains the fallback because it is MIT, mature, one
dependency (CTranslate2) and certain to work by 08-10; NeMo is a heavy,
torch-version-sensitive dependency that must be **proven to coexist with this
venv in the GPU window** before it can be the default.

Measured anchor (RTX 2080 Ti, 13 min audio, beam 5): large-v3-turbo 19.155 s
fp16 / 19.591 s int8, 2537 / 1545 MB. A 3080 is ~1.5–1.8× that card. **Do not
divide this by length for a 5 s turn** — the 30 s pad makes RTFx actively
misleading at our utterance length. The 250–450 ms figure in §3 is an
extrapolation from architecture, explicitly labelled, and is the first thing
the GPU window measures.

**Do not build a separate language-ID stage.** Every viable candidate does LID
in-pass for free. **Do not collapse translation** into the ASR: every open
ASR+AST model is English-pivoted (Whisper's `translate` task is verified
X→English only; Canary is EN↔24; Granite is hub-and-spoke through English), and
routing DE→EN→ES through a 0.6–2B speech decoder compounds error worse than
DE→text→27B LLM→ES.

**VAD:** silero-vad (MIT, <1 ms per 30 ms chunk, CPU, zero VRAM). Fallback
ten-vad if turn-end latency binds — its license says "Apache 2.0 with
additional conditions" and that wording must be read before vendoring.
`EnergyVad` is the dependency-free default that makes the hermetic suite
runnable; it is honest about being crude.

### 2.2 Diarization — deliberately none

**Online diarization does not earn its complexity here, and the reasoning is
worth recording because it looks like a corner being cut and is not.** The goal
is a retrospective label on a *completed* segment so the right reference audio
is reused. Online diarization's entire value is emitting labels *mid-utterance*
under bounded latency — value we cannot spend in a turn-based UI.

- The segment boundary is free: the pause that finalizes ASR *is* the
  diarization boundary.
- Online clustering is measurably worse (documented DER drop vs offline
  spectral/VBx; offline can revise early guesses, online cannot peek ahead).
- diart's rolling buffer imposes 500 ms – 5 s *before translation can start*,
  to produce a label computable in ~15 ms.
- We get retroactive correction for free: every segment embedding is retained,
  so the whole conversation can be re-clustered to repair an early mislabel.
  For a voice-cloning reference buffer this matters — one mislabel poisons a
  speaker's voice for every later turn.
- Overlapped speech, online diarization's other selling point, produces
  unusable ASR under half-duplex anyway.

**Chosen:** per-segment embedding + incremental cosine clustering behind a
pluggable interface. `pyannote/wespeaker-voxceleb-resnet34-LM` (CC-BY-4.0,
**ungated**) via ONNX Runtime, which the venv already has — ~26 M params,
~100 MB, no new heavy dependency. Fallbacks: SpeechBrain ECAPA (Apache-2.0,
EER 0.80%), ReDimNet (MIT, 1–15 M).

The one real failure mode of per-segment diarization is handled cheaply: two
people speaking back-to-back inside one VAD segment. `split_points_by_dispersion`
embeds ~1.5 s windows and flags an adjacent pair below a cosine threshold. A
few dozen lines instead of a streaming diarizer.

`nvidia/diar_streaming_sortformer_4spk-v2` (CC-BY-4.0, 0.32 s, 4 speakers) is
the drop-in alternate **if** the product ever goes full-duplex. Note the
counter-intuitive licensing: the *streaming* v2 is CC-BY-4.0 while the offline
v1 is CC-BY-**NC**-4.0. Easy to get backwards.

Gated pyannote weights are legally fine for us: the gate collects contact info,
the licenses behind it (MIT / CC-BY-4.0) impose no use restriction. One-time
click, operational not legal.

### 2.3 TTS — zero-shot cross-lingual cloning

| Candidate | License (code / weights) | Streaming | Languages (de/es) | Min ref | VRAM | Verdict |
|---|---|---|---|---|---|---|
| **Qwen3-TTS-12Hz-0.6B-Base** | Apache-2.0 / Apache-2.0 | via vLLM-Omni: 97 ms first packet, PCM + WebSocket | 10, both ✓ | **3 s** | 2.5 GB weights, ~3–4 GB runtime | **PRIMARY** |
| **VoxCPM2** (2B) | Apache-2.0 / Apache-2.0 | true in-process `generate_streaming()` | 30, both ✓ | short | **~8 GB on a 4090** | **FALLBACK (quality when VRAM allows)** |
| **Fun-CosyVoice3-0.5B-2512** | Apache-2.0 / Apache-2.0 | best-in-class, 150 ms bi-streaming, named `inference_cross_lingual()` | 9, both ✓ | 3 s | 9.75 GB repo | second fallback if latency binds; no pip package, offline-only in vLLM-Omni |
| **XTTS-v2** | MPL-2.0 / **CPML non-commercial** | real `inference_stream()` | 17, both ✓ | 6 s | — | **permitted here** (private use), but 2023 quality; keep as legacy comparison arm |
| Chatterbox Multilingual V3 | MIT / MIT | none public | 23, both ✓ | ~5 s | — | **reject on this rig**: pins `torch==2.6.0`, predates sm_120 — will not run on the 5090 |
| OmniVoice | Apache-2.0 | — | 646 listed | 3–10 s | — | candidate; its documented cross-lingual accent carry is a **feature** here per the dated decision |
| F5-TTS / E2-TTS | MIT / CC-BY-NC | — | **zh/en base**; German and Spanish are *separate community checkpoints* | — | — | structural reject: de→es in one model is impossible |
| CosyVoice **2** | Apache-2.0 | yes | **zh/en/ja/ko only** | — | — | **trap**: the HF card was overwritten with the unified CosyVoice README advertising 9 languages. No German, no Spanish. Pick CosyVoice**3**. |
| IndexTTS-2/2.5 | bilibili Model Use License (not Apache, widely misreported) | — | **no German** | — | — | reject |
| Higgs Audio v2/v3 | Apache code / `license: other` (relicensed 2026-06-25) | — | en/zh/de/ko — **no Spanish** | — | — | reject |
| MegaTTS3 | Apache-2.0 | — | — | — | — | reject: WaveVAE encoder weights withheld, cloning needs a manual approval round-trip |
| Kokoro-82M | Apache-2.0 | — | — | — | — | reject: cannot clone, no German |
| Fish-Speech / OpenAudio S1 | Fish Audio Research License, gated | — | 13 | — | — | permitted but gated weights block unattended setup |

**The honest caveat, verified across every paper read: nobody publishes de↔es
cross-lingual speaker-similarity numbers.** CosyVoice3's cross-lingual eval set
covers only `to_en/to_zh/to_ja/to_ko`; Qwen3-TTS Table 7 covers the same CJK+en
square; VoxCPM2 has no cross-lingual table. Every de/es number in circulation
is **in-language**. The ranking below is mechanism + in-language quality +
deployment fit, and §7 specifies the local A/B that actually settles it.

The convergent evidence that *is* available: all four leaders implement
cross-lingual cloning the same way — **by keeping the reference transcript out
of the LM context**. CosyVoice `inference_cross_lingual()` · Qwen3-TTS
`x_vector_only_mode=True` · VoxCPM2 `reference_wav_path` alone · Chatterbox
`cfg_weight=0`. Four spellings of one mechanism.

**Why Qwen3-TTS-Base primary:** Apache-2.0 on both halves; 2.5 GB (the only top
candidate that comfortably fits beside the 27B); 3-second cloning; the best
published per-language similarity of the field (de 0.737 / es 0.731, highest of
its 10 languages against MiniMax and ElevenLabs; WER de 0.96 / es 1.13); a real
runtime language contract (`get_supported_languages()` plus
`config.json → talker_config.codec_language_id`, with `_validate_languages()`
raising on unknowns); and 2.5 M downloads/month.

**Why VoxCPM2 fallback:** beats CosyVoice3 on CV3-Eval in *both* our languages
(de 4.77 vs 6.43, es 3.80 vs 4.47) and has the cleanest reference decoupling.
It costs ~8 GB, so it is the quality option when VRAM allows, not the default.
It also has **no language parameter at all** — synthesis language is inferred
from the input text — which for a translator that *knows* its target language
is a point against it.

**Serving.** vLLM-Omni (Apache-2.0) serves TTS behind stock
`POST /v1/audio/speech` via `vllm serve <model> --omni`, and solves the schema
problem (OpenAI's `voice` field takes a preset string, not an audio upload)
with a `/v1/audio/voices` registry: upload a reference clip once, then call
`voice="alice"`. One server, swap the model, identical client code — no
per-model HTTP wrapper. That registry maps directly onto our preset pool.

### 2.4 Machine translation

Our own **Qwen3.6-27B-INT8** over `/v1/chat/completions`, called as an ordinary
HTTP client (client-compatibility principle: we do not reach into the engine).
Requirement 6 is satisfied structurally rather than by gesture. `temperature=0`,
a system prompt that forbids conversational output, and a bounded rolling
history of previous turns — context measurably helps pronoun and formality
choice (tu/usted, German gender agreement on a referent named two turns ago),
which is exactly where a context-free sentence translator embarrasses itself
live.

---

## 3. Latency budget

Per turn, measured from **the moment the speaker stops talking** (which is what
the listener experiences), for a ~4 s utterance.

| Stage | Streams? | Fallback stack (Whisper) | Target stack (Nemotron + Qwen3-TTS) |
|---|---|---|---|
| VAD hangover — turn declared closed | — | **550 ms** | **550 ms** |
| ASR | no (Whisper) / partially (Nemotron) | 250–450 ms *(extrapolated, unmeasured)* | 50–150 ms |
| Speaker embedding | no | 15–40 ms | 15–40 ms |
| MT first token | **yes** | 150–400 ms | 150–400 ms |
| MT first clause complete | yes | +200–500 ms | +200–500 ms |
| TTS first audio | **yes** | 150–300 ms | ~100–200 ms |
| Network (tunnel, mobile) | — | 50–150 ms | 50–150 ms |
| **First translated audio after the pause** | | **~1.4–2.4 s** | **~1.1–2.0 s** |

Both stacks land inside the 2–3 s target with margin, which is why the safe
fallback is genuinely acceptable and the target stack is an improvement rather
than a requirement.

**What overlaps.** MT streams tokens; `SentenceAccumulator` regroups them into
clauses; TTS starts on the first complete clause while the tail is still
generating; audio frames go to the phone as they are produced. For a short
conversational turn this saves little. For a 20-word turn it removes most of
the MT and TTS stages from the critical path.

**What does not overlap, and cannot.** The hangover. It is the single largest
*fixed* term and the one the user feels. 550 ms is the default: below ~400 ms a
German speaker's clause-internal pauses start cutting turns; above ~800 ms the
exchange feels laggy. **This is the first thing to tune on real recordings** —
and if turn-end detection becomes the bottleneck, the shape of the fix is
already known (`parakeet_realtime_eou_120m-v1` collapses ASR + end-of-utterance
into one model, 160 ms p50 — English-only today, but it validates the pattern).

Every turn records its own `Stopwatch` unconditionally, not behind a benchmark
flag: the end-to-end number is the acceptance criterion, and a number only
available when someone remembered to enable it is a number nobody has when it
matters.

---

## 4. The skeleton

`python/sglang/srt/translator/`, following the `video_enhance` precedent
exactly (own package, own process, `test/registered/<name>/`).

| Module | Responsibility |
|---|---|
| `languages.py` | `LanguageMatrix` (runtime ASR × MT × TTS intersection), `ConversationLanguages` (routing by elimination), ISO canonicalisation |
| `backends.py` | `AudioChunk`, the four backend Protocols, and the hermetic fakes the whole suite runs against |
| `segmenter.py` | VAD protocol, `EnergyVad`, the turn state machine (onset / hangover / pre-roll / forced cut / floor) |
| `speakers.py` | Speaker registry, incremental clustering, the two-slot reference buffer, intra-segment speaker-change detection |
| `voices.py` | `VoiceMode`, preset pool, F0 voice classifier, sticky class-matched assignment, exhaustion offsets |
| `mt.py` | `OpenAiMt` against our own endpoint, direction prompt, `SentenceAccumulator` |
| `session.py` | The turn pipeline, the event journal, reconnect resume, the session manager |
| `audio.py` | Codec negotiation, PCM16/Opus, rational resampling |
| `server.py` | FastAPI: WS stream, `/api/translator/languages`, health, sessions, enrollment, voice mode |
| `asr_backends.py` | `FasterWhisperAsr`, `NemoStreamingAsr`, `SileroVad`, `OnnxSpeakerEmbedder` — heavy imports inside constructors only |
| `launch.py` | The tenant entry point; pins the card by NVML UUID before any CUDA context |
| `client/index.html` | The PWA: one file, no build, no external asset |

### 4.1 The language set is derived, never declared (requirement 5)

```
speakable(system) = ASR.transcribes  ×  MT.translates  ×  TTS.synthesizes
```

Recomputed from the live backends per request, so swapping a TTS checkpoint
changes the advertised set without a code edit. `GET /api/translator/languages`
returns the intersection **and the per-stage sets**, so a missing language is
attributable ("the TTS backend does not speak es") rather than merely absent.
An MT backend that claims universal coverage declares `None` explicitly, which
is a claim rather than a silent empty set.

Routing is elimination over a conversation's *participant set*, not a
source/target pair. Two participants reduces to "the other one"; three fans
out. **No language pair appears anywhere in the deciding modules** — the
falsifier asserts this two ways: behaviourally (the same code drives `ja↔fr`)
and by AST inspection (`session.py`, `segmenter.py`, `speakers.py`, `mt.py`
contain no `de`/`es`/`german`/`spanish` string literals).

### 4.2 Reference buffer: two slots

Following X-Translator's session-level speaker-prompt manager (arXiv
2607.17544) — **reimplemented from the paper's description; their code is
CC-BY-NC-SA and was neither linked nor vendored**:

- **Fixed enrollment prompt** (6 s, trimmed from the middle of curated audio):
  anchors identity, never evicted.
- **Rolling prompt** (6 s of the best recent field audio): tracks the current
  channel — the phone moved, the room changed.

The two sources disagree on retention and the disagreement is real:
speaker-verification practice says keep the best-K segments, the prompt-manager
design says keep the most recent window. Both are right about different failure
modes — a great old slice beats a poor new one, but a slice from a different
room is worse than a mediocre one from this one. The synthesis is quality
scored with an exponential recency decay (`recency_half_life_s`, default 180 s).

Admission thresholds, from the survey: **3 s hard minimum** before cloning is
enabled at all, **10–15 s target**, one contiguous 6–12 s slice preferred over
a concatenation of short ones (splices produce prosodic discontinuities that
zero-shot cloners reproduce audibly). An *ambiguous* segment (cosine between
0.70 and 0.80) is translated but barred from the reference buffer: a voice
split degrades gracefully, a voice merge corrupts both voices audibly.

### 4.3 Voice modes

`clone` uses the speaker's reference; `preset` assigns a distinct artificial
voice, sticky per session, matched to the speaker's F0-derived class. The
classifier is labelled heuristic everywhere because it is: F0 distributions
overlap, and **boy versus girl is not recoverable from F0 before puberty**. So
the classifier returns `CHILD` and pool entries tagged `BOY` or `GIRL` both
match it; anything finer would be a guess dressed as a measurement, and the API
offers a per-speaker override instead.

A preset is expressed as **reference audio per language**, not an opaque backend
voice name — so preset mode is clone mode pointed at a curated clip, works with
any zero-shot backend without an interface change, and a preset recorded in the
target language has no accent to carry. `backend_voice_id` is the escape hatch
for vLLM-Omni's voice registry.

Allocation: unused voice of the speaker's class → unused voice of any class
(distinctness beats class match: a listener who cannot tell two speakers apart
has lost more than one whose preset is the wrong gender) → shared base voice at
the next variant index, with a deterministic ±1.5/±3.0 semitone offset applied
to the reference and a named notice surfaced in the UI.

**Downgrade preference.** An unclonable speaker gets a *preset*, not another
participant's borrowed voice: a preset is honestly artificial, while a borrowed
voice attributes words to the wrong person. Borrowing survives only as the last
resort when no pool is loaded at all, and both are marked on the turn event.

### 4.4 Reconnect

The stated top operational risk is the mobile link dropping while roaming, so
reconnect is a first-class path. The journal is append-only with monotonic
sequence numbers; the client tracks its cursor; a reconnect replays from there.
Audio payloads are evicted under a byte budget **while their events remain**, so
a replay after a long outage yields a complete transcript with samples marked
absent, and a cursor below the floor produces an explicit `resume.gap` rather
than a silently shortened conversation.

The session outlives the socket. A reconnect resets the segmenter (audio in
flight is gone) and touches nothing else — the speaker registry and reference
buffers, which are the expensive thing to rebuild, survive untouched. That
asymmetry is the reason the two are separate objects.

---

## 5. Test results (desk, hermetic)

```
CUDA_VISIBLE_DEVICES=99 PYTHONPATH=/spinning/wt-466-translator/python \
  /spinning/htsglang-gpu/.venv/bin/python -m pytest test/registered/translator/ -q
```

**119 passed** (96 + 23 voice tests), ~16 s, no GPU, no network, no model.

Per-file:

```
CUDA_VISIBLE_DEVICES=99 python -m pytest test/registered/translator/test_languages.py -v      # 17
CUDA_VISIBLE_DEVICES=99 python -m pytest test/registered/translator/test_segmenter.py -v      # 13
CUDA_VISIBLE_DEVICES=99 python -m pytest test/registered/translator/test_speakers.py -v       # 24
CUDA_VISIBLE_DEVICES=99 python -m pytest test/registered/translator/test_session.py -v        # 17
CUDA_VISIBLE_DEVICES=99 python -m pytest test/registered/translator/test_audio_and_http.py -v # 25
CUDA_VISIBLE_DEVICES=99 python -m pytest test/registered/translator/test_voices.py -v         # 23
```

Live boot smoke (fake backends, real HTTP, no GPU):

```
CUDA_VISIBLE_DEVICES=99 python -m sglang.srt.translator.launch --port 30841 --participants de,es
curl -s http://127.0.0.1:30841/api/translator/health      # status ok, budgets 7500 MiB
curl -s http://127.0.0.1:30841/api/translator/languages   # pair_count 2, default supported
curl -s http://127.0.0.1:30841/api/translator/voices      # clone default, no pool loaded
curl -s http://127.0.0.1:30841/                           # 21491 bytes of PWA
```

### 5.1 Six real bugs the suite caught before any GPU time

Recorded because each is a family this project has seen before.

1. **`journal or Journal()`** — a fresh journal has zero events and is
   therefore *falsy*, so every caller's configured bounds were silently
   replaced by defaults. Fixed to an `is not None` test.
2. **Dataclass `__eq__` over ndarray fields** — `item in list_of_chunks` raised
   "truth value of an array is ambiguous". `AudioChunk`, `SpeakerEmbedding` and
   `_ReferenceItem` are now `eq=False`; identity is the only meaningful
   equality here.
3. **`EnergyVad` deafness** — the noise floor was seeded from the first frame,
   so a stream that *opens* with speech (which push-to-talk always does) set
   the floor at speech level and no later frame could clear it. The detector
   went permanently deaf, silently: the turn simply never appeared. Found
   through the WebSocket test, because the unit tests used a scripted VAD.
   Falsifier added.
4. **Dead resampler guard** — `Fraction.limit_denominator(1000)` never fails,
   it *approximates*, so checking its output against the bound it was given was
   dead code and an unsupported rate pair passed through as a wrong pitch. Now
   compared against the exact ratio.
5. **Reconnect replayed the whole conversation** — a fresh connection's
   delivery cursor started at zero, and if the resume window happened to be
   empty nothing advanced it. Seeded from the resume point before replay.
6. **Unbounded wait in the test helper** — `ws.receive()` blocks forever;
   the suite wedged for real. Replaced with a ping/pong protocol fence plus a
   wall-clock budget, per the robustness canon.

Bug 3 is the instructive one: the unit tests injected a scripted VAD to isolate
the state machine, which was right, and that isolation is exactly what hid a
defect only the integrated path could show. **Only the integrated boot found
it** — the same lesson `integration/r2` recorded.

---

## 6. Deployment

### 6.1 Co-residency plan

| Tenant | Card | Budget | Notes |
|---|---|---|---|
| htsglang LLM (Qwen3.6-27B-INT8) | all three, uneven TP | as today | unchanged; the translator is a client |
| ASR (Nemotron-3.5 0.6B or faster-whisper) | 5090 | 3000 MiB | own process, own venv, pinned by NVML UUID |
| Speaker embedder (ResNet34-LM ONNX) | CPU or 5090 | ~500 MiB | ~26 M params; CPU is viable |
| VAD (silero) | CPU | 0 | <1 ms per chunk, one thread |
| TTS (Qwen3-TTS-0.6B via vLLM-Omni) | 5090 | 4000 MiB | own venv — `transformers` pin conflict is structural |

Total additional ~7.5 GB, taken from the 5090's allocation. **This directly
competes with the LLM's budget** and must be entered in the reserve per the
VRAM corridor rule (free ≥400 MiB absolute, waste ≤1.5 GiB net) before the GPU
window — the LLM boot line needs its per-rank budget reduced accordingly. That
arithmetic is a GPU-window task, not a desk one.

### 6.2 WireGuard

No real addresses in this repository; the `/root/rig-env.sh` convention
supplies them (`${VAR:-<PLACEHOLDER>}`). Server side, on the rig:

```bash
sudo apt install wireguard
wg genkey | tee /etc/wireguard/server.key | wg pubkey > /etc/wireguard/server.pub
wg genkey | tee /etc/wireguard/phone.key  | wg pubkey > /etc/wireguard/phone.pub
```

`/etc/wireguard/wg0.conf`:

```ini
[Interface]
Address    = ${WG_SERVER_ADDR:-<10.x.y.1/24>}
ListenPort = ${WG_PORT:-<51820>}
PrivateKey = <contents of server.key>

[Peer]
PublicKey  = <contents of phone.pub>
AllowedIPs = ${WG_PHONE_ADDR:-<10.x.y.2/32>}
```

```bash
sudo sysctl -w net.ipv4.ip_forward=1
sudo systemctl enable --now wg-quick@wg0
sudo wg show
```

Forward `${WG_PORT}/udp` on the router to the rig. Bind the translator to the
tunnel address, never to `0.0.0.0`:

```bash
python -m sglang.srt.translator.launch --host ${WG_SERVER_ADDR%%/*} --port 30800
```

Phone side (OnePlus 10 Pro, Android 16, **no root needed** — the official app
is a userspace VPN):

1. Install **WireGuard** from Play Store or F-Droid.
2. `+` → *Create from scratch*. Name `rig`.
3. Private key: *Generate*. Send the shown **public** key to the rig's
   `[Peer] PublicKey`.
4. Addresses `${WG_PHONE_ADDR}`, DNS blank.
5. Add peer: server public key, endpoint `<home-ddns>:${WG_PORT}`,
   AllowedIPs `${WG_SERVER_ADDR%%/*}/32` (split tunnel — only rig traffic goes
   through, so Spanish mobile data still routes normally), persistent
   keepalive **25**.
6. Toggle on, then open `http://${WG_SERVER_ADDR%%/*}:30800/` in Chrome.
7. Chrome menu → *Add to Home screen* for standalone PWA mode.

**Known constraint, stated rather than discovered in Spain:** `getUserMedia`
needs a secure context. `http://` over the tunnel is **not** one on Android
Chrome unless the origin is a loopback. Two options, both to be decided in the
GPU window: (a) a self-signed TLS cert on the tunnel address with the CA
installed on the phone (Android 16 requires user-CA installation via Settings →
Security → Encryption & credentials; note apps do not trust user CAs by default,
but *Chrome does* for its own navigation), or (b) `chrome://flags` →
*Insecure origins treated as secure*, adding `http://${WG_SERVER_ADDR%%/*}:30800`.
Option (b) is one setting and is the MVP path; option (a) is the durable one.
**This is on the critical path — it must be verified on the actual phone before
the flight.**

### 6.3 The PWA wall, evaluated (not silently switched)

Per the architecture directive, the native-app decision hinges on a named wall.
The wall is real and it is **screen-off / background operation**: Android
suspends a backgrounded page's `AudioContext`, and no web API overrides that.
`navigator.wakeLock` keeps the screen on while the app is in use and is wired
in; it does **not** enable background audio.

**Recommendation: stay with the PWA for the trip, screen-on as a stated MVP
constraint.** In the actual use case — a phone held between two people having a
conversation — the screen is on and in hand. A native Kotlin client buys
lock-screen operation and a foreground-service notification, which is days of
work for a constraint that does not bind in the intended use. Everything the
native client would need is already on the server side of the WebSocket
protocol, so the port stays cheap if the constraint ever does bind.

Android 16 Chrome supports `getUserMedia` with `echoCancellation` /
`noiseSuppression` / `autoGainControl` and the Wake Lock API. Those three
constraints are the reason the client is a browser at all: they dissolve the
loudspeaker-into-microphone feedback loop for half-duplex playback, for free.

---

### 6.4 Crash-safety backup (dated decision, 2026-08-03)

Components that are restrictively licensed, gated, or published by a source
that can disappear are backed up to a **private** GitHub repository. A local
copy on one machine is not a backup, and Coqui — publisher of XTTS-v2 — is
defunct, so its HF mirror is exactly the kind of source that vanishes.

`scripts/translator/vendor_backup.sh` — small files (configs, licences,
vocabularies, install recipes) go into git; large weights go into GitHub
release assets, split at 1900 MiB because a single asset caps at 2 GiB. Every
entry lands in `MANIFEST.md` with its sha256, license and source URL, plus the
`cat <name>.part-* > <name>` reassembly line, so a restore is *verified* rather
than hoped at.

`init` will not push anything until the GitHub API itself answers
`private: true`. Trusting the create request's intent instead of the API's
answer is how a non-commercial checkpoint ends up world-readable.

The repository is **private forever**: CPML and CC-BY-NC permit private copies,
not redistribution. The public fork never references it — the fork only knows
the neutral backend interface.

Backed up in this phase: **XTTS-v2** (`coqui/XTTS-v2`, Coqui Public Model
License 1.0.0, non-commercial) — `LICENSE.txt`, `config.json`, `vocab.json`,
`README.md` in git; `model.pth`, `dvae.pth`, `speakers_xtts.pth`,
`mel_stats.pth` as assets on release tag `xtts-v2`. Qwen3-TTS and VoxCPM2 are
Apache-2.0 and ungated, so they are re-downloadable and do not need this; they
get backed up anyway once pinned to a specific revision, because Higgs Audio
silently moved from `apache-2.0` to `license: other` on 2026-06-25 and that is
the failure mode this guards against. **Pin revisions, not repo names.**

## 7. Test plan before the flight

**(a) Desk, done.** 119 hermetic tests + the live boot smoke above.

**(b) GPU window — the ticket.** Ordered so a failure at step 1 does not waste
the window:

1. **NeMo coexistence spike (30 min, decides the ASR).** Install
   `nemo_toolkit[asr]` into a *separate* venv; confirm it does not disturb
   `/spinning/htsglang-gpu/.venv`. If it fights the torch/CUDA stack, fall
   straight back to faster-whisper and stop spending time here.
2. **Measure ASR latency for a 5 s utterance** — faster-whisper
   `large-v3-turbo` `int8_float16` `beam_size=1` on a 3080, and the streaming
   model if step 1 passed. This replaces the extrapolated 250–450 ms in §3 with
   a number. A-vs-A floor first, per the measurement canon.
3. **TTS head-to-head, DE↔ES cross-lingual.** One German clip → the same
   Spanish sentence through Qwen3-TTS-Base (`x_vector_only_mode=True` **vs**
   ICL — the highest-value single comparison, it isolates the mechanism all
   four leaders rely on), VoxCPM2 (`reference_wav_path` alone), and XTTS-v2 as
   the legacy arm. Repeat es→de. **Scoring per the dated accent decision:**
   ASR-round-trip WER as a **hard intelligibility gate** (transcribe the output
   with Whisper in the target language; above a WER threshold the candidate is
   out regardless of how good it sounds), then rank surviving candidates by
   ECAPA/ReDimNet speaker similarity against the source clip. **Accent is not
   scored and never penalised.**
4. **End-to-end latency** with real ASR + MT + TTS, `Stopwatch` per turn,
   against the §3 budget.
5. **VRAM corridor check at peak** (`forward_peak.py`), with the LLM's
   per-rank budget already reduced by the tenant total.

**(c) Real phone over mobile data.** User in the loop. WireGuard from a
non-home network (mobile data, not home WiFi), the secure-context question
settled, a full conversation, and a **deliberate connectivity kill** — aeroplane
mode for 20 s mid-conversation — to exercise reconnect + journal resume on the
real link rather than in a test.

**(d) Multi-speaker, two real voices.** Both voice modes. Clone mode: does each
speaker keep their own voice across turns, and does the second speaker's first
turn downgrade visibly? Preset mode: are the two presets distinct and
class-appropriate, and does the mapping survive a reconnect?

---

## 8. Open risks, ranked

The operator's proposed top three were roaming connectivity, diarization in
noise, and cross-lingual cloning quality. **Re-ranked on the evidence:**

1. **Secure-context / `getUserMedia` over the tunnel** (new, and now first).
   Not on the original list, and it is the one that stops the demo *before it
   starts* rather than degrading it. Android Chrome refuses microphone access
   on a plain-`http://` non-loopback origin. Cheap to fix (§6.2) and cheap to
   verify — but only if verified before the flight.
2. **Roaming connectivity** (confirmed, was #1). Mitigated by design —
   reconnect with backoff, journal resume, explicit gap reporting, split-tunnel
   AllowedIPs, keepalive 25 — but a CGNAT-ed Spanish carrier plus a dynamic
   home IP can still defeat it. Mitigation to arrange before departure: a
   dynamic-DNS name and a verified inbound UDP path, tested from a foreign
   network, not from home WiFi.
3. **Cross-lingual cloning quality** (confirmed, promoted above diarization).
   Promoted because the survey found **no published de↔es evidence for any
   model** — the selection rests on mechanism and in-language numbers. This is
   an unmeasured assumption at the heart of the feature, which is exactly why
   step (b)(3) exists. The dated accent decision materially de-risks it: with
   accent removed as a failure mode, the bar is intelligibility, which is the
   property most likely to hold.
4. **Diarization in noise** (demoted from #2). Demoted for two reasons: the
   design already reduces the hard case (per-segment embedding on a completed
   turn, not frame-level tracking under overlap), and both failure modes
   degrade gracefully — an ambiguous segment is translated but barred from the
   reference buffer, and preset mode sidesteps voice identity entirely. Still
   real in a Spanish street or bar, and the mitigation lever is silero-vad
   (never `EnergyVad`) plus preset mode as the noisy-environment default.
5. **NeMo dependency risk.** Contained by design: faster-whisper is the
   fallback and the spike is step 1 of the window, so this cannot consume the
   deadline.
6. **VRAM contention with the 27B.** ~7.5 GB from the 5090's allocation is real
   and must go through the corridor rule before the window.

---

## 9. What is NOT built (Phase 1 boundary)

Stated so the next phase does not rediscover it:

- **The real TTS backend adapter.** Qwen3-TTS runs behind vLLM-Omni's
  `/v1/audio/speech` in its own venv, so the adapter is an HTTP client, not an
  in-process backend. `--tts` currently offers only `fake`.
- **Opus is wired but unexercised** — PyAV is not installed in this venv, so
  `available_codecs()` correctly reports only `pcm16` and negotiation lands
  there. Needs `pip install av` and a real round-trip.
- **The preset voice clips themselves.** The pool loader, the classifier, the
  assignment and the exhaustion path are built and tested against synthetic
  tones; the 18 recorded clips (6/6/3/3, per target language) do not exist yet.
- **Full-duplex / barge-in.** Explicitly a stretch goal, not MVP.
- **Intra-segment speaker splitting is detected but not yet acted on** —
  `split_points_by_dispersion` returns the boundaries; the session does not yet
  re-cut a segment at them.
