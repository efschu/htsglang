# Video-Enhance on the fly (browser extension)

A comfort shell around one public endpoint. The page has a video, you click
the extension, the video's URL goes to an htsglang video-enhance server, and
the `<video>` element plays the enhanced stream instead of the original.

This extension is **not** a privileged path into the enhancer. It is one
client of `GET /v1/video/enhance`, the same URL VLC, mpv, ffmpeg and curl
open directly:

```
curl -o out.mp4 'http://127.0.0.1:8100/v1/video/enhance?\
source_url=https%3A%2F%2Fcdn.example%2Fclip.mp4&source_width=1920&source_height=1080\
&target=1920x1080&fps_multiplier=2&enable_sr=false'
```

Everything the extension does is build that URL, put it in `video.src`, and
send `DELETE` when the tab goes away. If it ever gets in your way, the URL
still works without it.

## What it is not

It does not intercept frames. There is no Media Source Extensions
re-encapsulation, no decryption, no YouTube support, and none is planned in
this component. The handoff is a URL, so anything the server's ffmpeg cannot
open by URL cannot be enhanced this way — see the matrix below, which the
extension enforces by refusing with a stated reason rather than swapping in a
source that will never load.

## Install (unpacked)

The repository ships the extension unpacked and unsigned. There is no build
step, no bundler and no `npm install`.

### Chrome / Chromium / Edge

1. Open `chrome://extensions`.
2. Turn on **Developer mode**.
3. **Load unpacked**, and select `clients/browser-extension/`.
4. Open the extension's **Details -> Extension options**, set the server URL,
   press **Save**, and accept the host-permission prompt for that origin.

### Firefox

Firefox reads `manifest.json`, so the Firefox-specific manifest has to be put
in place first. Copy rather than symlink — `about:debugging` follows the file:

```bash
cd clients/browser-extension
cp manifest.json manifest.chrome.json.bak     # keep the Chrome one
cp manifest.firefox.json manifest.json
```

1. Open `about:debugging#/runtime/this-firefox`.
2. **Load Temporary Add-on**, and select `manifest.json` in that directory.
3. Open the add-on's **Preferences**, set the server URL, press **Save**.

A temporary add-on is unloaded when Firefox closes; repeat after a restart.
Restore `manifest.json` from the backup before loading it in Chrome again.

The two manifests differ in exactly two things: Firefox needs a
`browser_specific_settings.gecko.id`, and it loads the background as an event
page (`background.scripts`) where Chrome uses a service worker
(`background.service_worker`). `src/background.js` handles both — it calls
`importScripts` only where that function exists. A test asserts the two
manifests otherwise agree, so they cannot drift.

## Permissions, and why each one is there

| Permission | Why |
|---|---|
| `activeTab` | Access to the page you clicked on, for that click only. |
| `scripting` | Inject `src/content.js` into that tab on the click. |
| `storage` | Remember the server URL and the chain preset. |
| `optional_host_permissions` | Requested at runtime for the server origin you configure, from the options page. |

There is no `content_scripts` block and no standing `host_permissions`.
Nothing runs on any page until you click the action, and the extension can
reach exactly one server: the one you granted.

## The three modes

A viewer can ask for three things, and the chain preset is how they ask:

| Mode | Preset | What the server runs | Geometry | Frame rate |
|---|---|---|---|---|
| Upscale | `sr_only` | x4 super-resolution, then resize to target | changed | unchanged |
| Interpolate | `rife_only` | RIFE only | unchanged | doubled |
| Both | `full_chain` | SR, resize, then RIFE | changed | doubled |

The names are the server's own `CHAIN_PRESETS`, not a client-side vocabulary,
so the string a viewer picks here is the string a measurement row is filed
under. `sr_only` carries `fps_multiplier: 1`, which the chain builder reads as
"no RIFE stage" — the preset costs no interpolation memory and needs no RIFE
engine, which is why it is its own name rather than `full_chain` with the
multiplier turned down.

The mode is chosen in the options page, not per click: the action is a toggle,
and a toggle that also had to ask "which mode?" would need a popup, and a
popup would replace the single click that is the whole point of the component.

## Settings

| Setting | Meaning |
|---|---|
| Server URL | Base URL of the video-enhance server, e.g. `http://127.0.0.1:8100`. |
| Chain preset | One of the three modes above. |
| Target resolution | Only used by presets that change geometry. `rife_only` ignores it. |
| Interpolation flow scale | RIFE's optical-flow scale. `0.5` is the 4K headroom arm, `1.0` full quality. |
| Start / Duration | Optional time range in seconds, passed as `start_s` / `duration_s`. This is the only seek there is; see below. |

**Test server** in the options page calls `GET /v1/video/capabilities` and
prints what that deployment reports: its presets, its budget, and its
measured capability frontier — or an explicit "not measured" when no probe
report has been imported. It never invents a rate.

## Eligibility matrix

The extension classifies the page's video before it does anything. This is
the honest version, tested in `test/cases.js`:

| Case | Verdict | Code |
|---|---|---|
| Direct file URL (`https://…/clip.mp4`, `.webm`, …) | **Works** | `direct` |
| Relative `src` or `<source>` on an http(s) page | **Works** (resolved against the page URL) | `direct` |
| HLS / DASH manifest (`.m3u8`, `.mpd`) | **Works only if** the server's ffmpeg can reach the manifest *and* every segment URL in it | `manifest` |
| DRM / EME (Netflix, Prime, anything with `mediaKeys`) | Refused | `drm` |
| MSE / `blob:` source (YouTube, most adaptive players) | Refused | `mse` |
| `MediaStream` (camera, screen share, WebRTC) | Refused | `stream_object` |
| `data:` URI | Refused | `data_uri` |
| `file:` URL | Refused | `local_file` |
| Element with no source yet | Refused | `no_source` |
| No `<video>` on the page | Refused | `no_video` |
| Any other scheme | Refused | `unsupported_scheme` |

A refusal shows its reason in the action's tooltip and in the page console.
It never silently does nothing.

Two caveats apply to sources that *are* eligible:

* **The server fetches the URL itself**, without your cookies, your session or
  your `Authorization` header. A source behind a login works in your browser
  and fails on the server.
* **The intrinsic size may not be known yet** if the video has not started
  loading. The request then carries the element's reported size, and the
  server refuses it if that disagrees with the real source — the chain and its
  memory reservation are planned from the source geometry.

Why `mse` cannot be worked around here: a MediaSource is assembled inside the
page from segments the player fetches. There is no single URL, and the
segments are frequently signed per-session. Handing over the stale `src`
attribute would enhance whatever that attribute happened to point at, which is
why the classifier prefers `currentSrc` and refuses on `blob:` even when a
plausible-looking `src` is present.

## What happens on a click

1. `src/content.js` is injected, reads every `<video>`, and picks the largest
   one (a playing element breaks a tie).
2. `src/shared.js` classifies it. A refusal stops here.
3. A job id is generated client-side — `ext-` plus 16 characters — because a
   URL handed to a `<video>` element never surfaces a response header, so a
   server-minted id would be unknowable and `DELETE` unreachable.
4. The element's `src` is replaced with the enhance URL, child `<source>`
   elements are removed (they would win over the attribute on `load()`), and
   playback starts.
5. The swap is watched, not assumed — see below. On the first decoded frame a
   notice appears over the video saying enhancement is on and that seeking is
   not available.
6. A second click restores the original source, seeks back to where you were,
   and sends `DELETE /v1/video/enhance/{job_id}`.
7. Closing the tab or navigating away sends the same `DELETE`.

The `DELETE` is the polite path, not the safety net. The server's client
liveness layer reclaims a stream whose consumer went quiet, but its
`video_stream` timeout is 300 s, because a paused player is a normal thing. A
client that *knows* the viewer is gone should say so, and then the card is
free in milliseconds.

## When the engine is not there

Step 4 above points a `<video>` element at a URL that may produce nothing:
the server can be down, refuse the job, or die mid-stream. The page's own
player has no idea why, and a black frame with no explanation is worse than
not enhancing at all — so the swap is provisional until a frame arrives.

`watchStream` in `src/content.js` arms two failure signals, because a dead
engine produces one or the other depending on how it is dead:

* an `error` event on the element — a refused connection, a 503, or a body
  that is not a decodable stream; and
* silence — a socket that accepts and then never writes, caught by a
  12-second first-frame timer.

`loadeddata`, the first decoded frame, is the success signal and disarms both.
On either failure the original source is put back, the position and play state
are restored, a notice naming the reason is shown over the video, and the
background is told so the badge stops claiming `ON` and the job is cancelled.

## Seeking

There is none, and the extension says so rather than letting you find out.

The enhanced response is a chunked live body with no byte index, so the
element cannot honour the scrub bar; browsers differ in whether they fire
`seeking` for an unreachable position or silently clamp it. The notice shown
on the first frame therefore states the limitation unconditionally, and a
`seeking` handler snaps the position back and repeats it where the event does
fire.

What the endpoint *does* have is a time range: `start_s` and `duration_s`,
the #338 API extension, surfaced as **Start / Duration** in the options page.
Enhancing a different stretch of the source is a new request, not a seek
within an existing stream.

## Server requirements

The extension speaks to these endpoints:

| Endpoint | Used for |
|---|---|
| `GET /v1/video/enhance` | the stream itself (`source_url`, `source_width`, `source_height`, `target`, `fps_multiplier`, `enable_sr`, `sr_scale`, `rife_scale`, `container`, `start_s`, `duration_s`, `job_id`) |
| `DELETE /v1/video/enhance/{job_id}` | abort on restore, tab close or navigation |
| `GET /v1/video/capabilities` | the options page's **Test server** button |

The response is fragmented MP4 (`+frag_keyframe+empty_moov+delay_moov`),
which is what makes it playable in a `<video>` element while it is still
being produced. A non-fragmented MP4 writes its `moov` atom last and would
not start playing until the final byte arrived.

If the page you are on sets a restrictive Content-Security-Policy with a
`media-src` directive that does not include your server's origin, the browser
blocks the swapped source. That is a page-level policy the extension cannot
override, and it appears as a media error in the page console.

## Tests

Pure logic — URL construction, eligibility, source picking — runs under plain
node, with no dependencies and no package manager:

```bash
cd clients/browser-extension
node test/run.js
```

`test/registered/video_enhance/test_browser_extension.py` runs in the normal
Python suite and covers what can rot silently across the language boundary:
the manifests parse and agree, the permission set has not widened, every
referenced file exists, the chain presets in `src/shared.js` match the
server's `CHAIN_PRESETS` exactly, the three presets are three distinguishable
modes, the query-parameter names are ones the endpoint accepts, the generated
job ids pass the server's validator, and the fallback path's two halves — the
message `content.js` sends and the handler `background.js` registers — still
name the same string.

### End-to-end, in a real browser

There is an end-to-end acceptance run, and it lives outside this repository
because it needs a browser binary and an engine that CI has neither of:

```bash
cd /spinning/video-extension/acceptance
python3 run.py            # ~20 s, writes evidence/run-<stamp>/report.json
```

It loads *this* extension unpacked into Chrome, serves a generated clip over
HTTP, points the extension at the shared pseudo-engine, and asserts through
CDP: the action is present, each of the three modes posts the request its
label promises, the enhanced stream decodes at the promised geometry and
plays, a saved sample's ffprobe numbers match, a second click and a tab close
tear a running engine job down (read back out of the engine's request log),
the time range is honoured, and an unreachable or refusing engine leaves the
original video playing with a notice.

Two things it does not prove, stated because they are easy to imply:

* **The toolbar click itself.** The toolbar is browser chrome and no
  automation protocol can click it. The harness calls the function the click
  listener is registered with (`self.enhanceToggleTab`), inside the real
  service worker, against a real tab. Everything downstream of that call is
  the production path; Chrome's delivery of the click event to the listener is
  not exercised.
* **The permission gestures.** `activeTab` comes from that click and the
  server origin from the options page's permission prompt; neither gesture
  exists under CDP, so the harness loads a copy of the extension with
  `host_permissions` for `127.0.0.1` added, and records the difference in its
  report. The shipped manifest is unchanged, and the registered Python test
  fails if it ever grows a standing host permission.

Firefox is not covered by the harness. The manual steps below are the Firefox
story.

### Manual test steps (Chrome and Firefox)

Run through these after any change to `content.js`, `background.js` or the
manifests. None of it is automated.

1. **Direct file, happy path.** Serve a short mp4 over http and open a page
   with `<video controls src="…mp4">`. Click the action. Expect: the badge
   reads `ON`, playback restarts from the enhanced stream, and
   `GET /v1/video/enhance/{job_id}` on the server reports frames moving.
2. **Restore.** Click again. Expect: the original source is back, the position
   is roughly preserved, the badge clears, and the server reports the job
   cancelled.
3. **Tab close.** Enhance, then close the tab. Expect: a `DELETE` in the
   server log within a second, not a 300 s liveness reclamation.
4. **Navigation.** Enhance, then navigate the tab elsewhere. Same expectation.
5. **DRM refusal.** Open any EME-protected player. Expect: badge `!`, tooltip
   naming DRM, and the video untouched.
6. **MSE refusal.** Open a YouTube video. Expect: badge `!`, tooltip naming
   Media Source Extensions, and the video untouched.
7. **No video.** Click on a page with no `<video>`. Expect: badge `!` and the
   `no_video` reason.
8. **Wrong server.** Point the options page at a port with nothing on it and
   press **Test server**. Expect: a stated failure, not a silent one.
9. **Time range.** Set start 30 and duration 10, then enhance a clip longer
   than 40 s. Expect: playback begins 30 s into the source and stops after
   about 10 s of content.
10. **Permission decline.** Save a server URL and decline the permission
    prompt. Expect: the options page says access was declined and the
    extension cannot reach that server.

### Known gaps

* Firefox has no automated end-to-end proof; the harness drives Chrome only.
  The manual steps above are the Firefox coverage.
* The manual steps have not been run by CI and are not claimed as passing;
  what *is* claimed as passing is the acceptance run above, against the
  pseudo-engine. The real engine on the rig is a change of the server URL and
  nothing else, and has not been exercised from this harness.
* mpv and VLC integrations are a separate task. They need nothing from this
  extension — they open the same URL.
* MSE re-encapsulation, which is the only route to YouTube-class sources, is
  explicitly out of scope.
