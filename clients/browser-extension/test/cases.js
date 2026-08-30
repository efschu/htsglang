// Copyright 2025 SGLang Team
// Licensed under the Apache License, Version 2.0
//
// The assertions themselves, kept apart from the runner so the same list can
// be driven by node (test/run.js) or by any other JS engine. Everything here
// is a call into src/shared.js with plain objects -- no DOM, no fetch, no
// extension API -- which is exactly the part of the extension that can be
// tested without a browser. The rest is manual; see ../README.md.

(function (root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory(require("../src/shared.js"));
  } else {
    root.EnhanceCases = factory(root.EnhanceShared);
  }
})(typeof self !== "undefined" ? self : this, function (shared) {
  "use strict";

  var PAGE = "https://example.test/watch/1";

  function video(overrides) {
    var base = {
      currentSrc: "",
      src: "",
      sourceUrls: [],
      hasMediaKeys: false,
      hasSrcObject: false,
      videoWidth: 1920,
      videoHeight: 1080,
      clientWidth: 960,
      clientHeight: 540,
      playing: false,
      pageUrl: PAGE,
    };
    return Object.assign(base, overrides || {});
  }

  function params(url) {
    var out = {};
    var query = url.indexOf("?") === -1 ? "" : url.slice(url.indexOf("?") + 1);
    query.split("&").forEach(function (pair) {
      if (!pair) {
        return;
      }
      var bits = pair.split("=");
      out[decodeURIComponent(bits[0])] = decodeURIComponent(bits.slice(1).join("="));
    });
    return out;
  }

  // -- eligibility ---------------------------------------------------------

  var cases = [];

  function test(name, fn) {
    cases.push({ name: name, fn: fn });
  }

  test("a direct mp4 source is eligible", function (t) {
    var verdict = shared.classify(
      video({ currentSrc: "https://cdn.test/a.mp4", videoWidth: 1280, videoHeight: 720 })
    );
    t.ok(verdict.eligible, "should be eligible");
    t.equal(verdict.kind, "direct");
    t.equal(verdict.sourceUrl, "https://cdn.test/a.mp4");
    t.equal(verdict.width, 1280);
    t.equal(verdict.height, 720);
  });

  test("a relative source is resolved against the page URL", function (t) {
    var verdict = shared.classify(video({ src: "../media/clip.mp4" }));
    t.ok(verdict.eligible, "should be eligible");
    t.equal(verdict.sourceUrl, "https://example.test/media/clip.mp4");
  });

  test("a <source> child is used when the element has no src", function (t) {
    var verdict = shared.classify(
      video({ sourceUrls: ["https://cdn.test/b.webm"] })
    );
    t.ok(verdict.eligible, "should be eligible");
    t.equal(verdict.sourceUrl, "https://cdn.test/b.webm");
  });

  test("DRM is refused even when the src looks ordinary", function (t) {
    var verdict = shared.classify(
      video({ currentSrc: "https://cdn.test/a.mp4", hasMediaKeys: true })
    );
    t.ok(!verdict.eligible, "must be refused");
    t.equal(verdict.kind, "drm");
    t.ok(verdict.reason.indexOf("DRM") !== -1, "reason names DRM");
  });

  test("an MSE blob source is refused with the MSE reason", function (t) {
    var verdict = shared.classify(video({ currentSrc: "blob:https://example.test/xyz" }));
    t.ok(!verdict.eligible, "must be refused");
    t.equal(verdict.kind, "mse");
  });

  test("a stale src attribute does not override a blob currentSrc", function (t) {
    // The YouTube shape: the page's markup says mp4, the player swapped in a
    // MediaSource. Handing over the attribute would enhance the wrong thing.
    var verdict = shared.classify(
      video({ currentSrc: "blob:https://example.test/xyz", src: "https://cdn.test/a.mp4" })
    );
    t.ok(!verdict.eligible, "must be refused");
    t.equal(verdict.kind, "mse");
  });

  test("a MediaStream object is refused", function (t) {
    var verdict = shared.classify(video({ hasSrcObject: true }));
    t.ok(!verdict.eligible, "must be refused");
    t.equal(verdict.kind, "stream_object");
  });

  test("a data: URI is refused", function (t) {
    var verdict = shared.classify(video({ currentSrc: "data:video/mp4;base64,AAAA" }));
    t.ok(!verdict.eligible, "must be refused");
    t.equal(verdict.kind, "data_uri");
  });

  test("a file: URL is refused with the server-host reason", function (t) {
    var verdict = shared.classify(
      video({ currentSrc: "file:///home/user/a.mp4", pageUrl: "file:///home/user/x.html" })
    );
    t.ok(!verdict.eligible, "must be refused");
    t.equal(verdict.kind, "local_file");
  });

  test("an element with no source at all is refused", function (t) {
    var verdict = shared.classify(video({}));
    t.ok(!verdict.eligible, "must be refused");
    t.equal(verdict.kind, "no_source");
  });

  test("an HLS manifest is eligible but carries the reachability caveat", function (t) {
    var verdict = shared.classify(video({ currentSrc: "https://cdn.test/x.m3u8?t=1" }));
    t.ok(verdict.eligible, "should be eligible");
    t.equal(verdict.kind, "manifest");
    t.ok(verdict.caveats.join(" ").indexOf("HLS/DASH") !== -1, "caveat names HLS/DASH");
  });

  test("a DASH manifest is recognised the same way", function (t) {
    t.ok(shared.isManifestUrl("https://cdn.test/x.mpd"), "mpd is a manifest");
    t.ok(!shared.isManifestUrl("https://cdn.test/x.mp4"), "mp4 is not");
  });

  test("every http source warns that the server has no cookies", function (t) {
    var verdict = shared.classify(video({ currentSrc: "https://cdn.test/a.mp4" }));
    t.ok(verdict.caveats.join(" ").indexOf("cookies") !== -1, "cookie caveat present");
  });

  test("an unknown intrinsic size is flagged rather than hidden", function (t) {
    var verdict = shared.classify(
      video({ currentSrc: "https://cdn.test/a.mp4", videoWidth: 0, videoHeight: 0 })
    );
    t.ok(verdict.eligible, "still eligible");
    t.ok(verdict.caveats.join(" ").indexOf("intrinsic size") !== -1, "size caveat present");
  });

  // -- picking one video out of several ------------------------------------

  test("the largest video wins", function (t) {
    var small = video({ currentSrc: "https://cdn.test/small.mp4", clientWidth: 320, clientHeight: 180 });
    var big = video({ currentSrc: "https://cdn.test/big.mp4", clientWidth: 1280, clientHeight: 720 });
    t.equal(shared.pickBest([small, big]).currentSrc, "https://cdn.test/big.mp4");
    t.equal(shared.pickBest([big, small]).currentSrc, "https://cdn.test/big.mp4");
  });

  test("a playing video breaks a tie against a paused one", function (t) {
    var paused = video({ currentSrc: "https://cdn.test/p.mp4", playing: false });
    var playing = video({ currentSrc: "https://cdn.test/q.mp4", playing: true });
    t.equal(shared.pickBest([paused, playing]).currentSrc, "https://cdn.test/q.mp4");
  });

  test("no videos means no pick", function (t) {
    t.equal(shared.pickBest([]), null);
  });

  // -- URL construction ----------------------------------------------------

  test("rife_only sends the source resolution as the target", function (t) {
    // The server refuses a chain that cannot reach the named target, and
    // rife_only changes no geometry -- so sending the settings' 4K target
    // with it would turn every request into a 422.
    var url = shared.buildEnhanceUrl(
      { serverUrl: "http://127.0.0.1:8100", preset: "rife_only", target: "3840x2160" },
      { sourceUrl: "https://cdn.test/a.mp4", width: 1280, height: 720 },
      "ext-abc"
    );
    var q = params(url);
    t.equal(q.target, "1280x720");
    t.equal(q.enable_sr, "false");
    t.equal(q.fps_multiplier, "2");
    t.equal(q.source_url, "https://cdn.test/a.mp4");
    t.equal(q.job_id, "ext-abc");
  });

  test("full_chain sends the configured target", function (t) {
    var url = shared.buildEnhanceUrl(
      { serverUrl: "http://127.0.0.1:8100", preset: "full_chain", target: "3840x2160" },
      { sourceUrl: "https://cdn.test/a.mp4", width: 1920, height: 1080 },
      "ext-abc"
    );
    var q = params(url);
    t.equal(q.target, "3840x2160");
    t.equal(q.enable_sr, "true");
    t.equal(q.sr_scale, "4");
  });

  test("the endpoint path is the enhance-by-URL GET", function (t) {
    var url = shared.buildEnhanceUrl(
      { serverUrl: "http://127.0.0.1:8100/", preset: "rife_only" },
      { sourceUrl: "https://cdn.test/a.mp4", width: 1280, height: 720 },
      null
    );
    t.equal(url.indexOf("http://127.0.0.1:8100/v1/video/enhance?"), 0);
    t.equal(params(url).job_id, undefined);
  });

  test("the source URL is percent-encoded, query string and all", function (t) {
    var source = "https://cdn.test/a.mp4?token=a&b=c";
    var url = shared.buildEnhanceUrl(
      { serverUrl: "http://127.0.0.1:8100", preset: "rife_only" },
      { sourceUrl: source, width: 1280, height: 720 },
      null
    );
    t.ok(url.indexOf("source_url=https%3A%2F%2Fcdn.test%2Fa.mp4%3Ftoken%3Da%26b%3Dc") !== -1,
      "the source's own & must not become a parameter separator");
    t.equal(params(url).source_url, source);
  });

  test("a time range is sent only when set", function (t) {
    var none = params(
      shared.buildEnhanceUrl(
        { serverUrl: "http://s", preset: "rife_only" },
        { sourceUrl: "https://cdn.test/a.mp4", width: 8, height: 8 },
        null
      )
    );
    t.equal(none.start_s, undefined);
    t.equal(none.duration_s, undefined);

    var ranged = params(
      shared.buildEnhanceUrl(
        { serverUrl: "http://s", preset: "rife_only", startS: 12.5, durationS: 30 },
        { sourceUrl: "https://cdn.test/a.mp4", width: 8, height: 8 },
        null
      )
    );
    t.equal(ranged.start_s, "12.5");
    t.equal(ranged.duration_s, "30");
  });

  test("an unknown preset is an error, not a silent default", function (t) {
    t.throws(function () {
      shared.buildEnhanceUrl(
        { serverUrl: "http://s", preset: "nope" },
        { sourceUrl: "https://cdn.test/a.mp4", width: 8, height: 8 },
        null
      );
    }, "unknown chain preset");
  });

  test("a missing source resolution is an error", function (t) {
    t.throws(function () {
      shared.buildEnhanceUrl(
        { serverUrl: "http://s", preset: "rife_only" },
        { sourceUrl: "https://cdn.test/a.mp4", width: 0, height: 0 },
        null
      );
    }, "source resolution is required");
  });

  test("the cancel URL is the job path under the same server", function (t) {
    t.equal(
      shared.buildCancelUrl("http://127.0.0.1:8100/", "ext-abc"),
      "http://127.0.0.1:8100/v1/video/enhance/ext-abc"
    );
  });

  test("the capability URL drops empty query values", function (t) {
    t.equal(
      shared.buildCapabilitiesUrl("http://s", { source: "", target_fps: 48 }),
      "http://s/v1/video/capabilities?target_fps=48"
    );
    t.equal(shared.buildCapabilitiesUrl("http://s", {}), "http://s/v1/video/capabilities");
  });

  test("an enhance stream is recognised as ours, for the restore click", function (t) {
    var url = shared.buildEnhanceUrl(
      { serverUrl: "http://127.0.0.1:8100", preset: "rife_only" },
      { sourceUrl: "https://cdn.test/a.mp4", width: 8, height: 8 },
      "ext-abc"
    );
    t.ok(shared.isEnhanceUrl(url, "http://127.0.0.1:8100"), "own stream");
    t.ok(shared.isEnhanceUrl(url, "http://127.0.0.1:8100/"), "trailing slash tolerated");
    t.ok(!shared.isEnhanceUrl("https://cdn.test/a.mp4", "http://127.0.0.1:8100"), "not ours");
    t.ok(!shared.isEnhanceUrl(url, "http://other:8100"), "different server is not ours");
  });

  test("sr_only upscales without asking for interpolation", function (t) {
    // The mode a viewer picks as "upscale": geometry changes, the frame rate
    // does not. fps_multiplier=1 is what the chain builder reads as "no RIFE
    // stage", so the request costs no interpolation memory.
    var url = shared.buildEnhanceUrl(
      { serverUrl: "http://127.0.0.1:8100", preset: "sr_only", target: "3840x2160" },
      { sourceUrl: "https://cdn.test/a.mp4", width: 1920, height: 1080 },
      "ext-abc"
    );
    var q = params(url);
    t.equal(q.target, "3840x2160");
    t.equal(q.enable_sr, "true");
    t.equal(q.sr_scale, "4");
    t.equal(q.fps_multiplier, "1");
  });

  test("the three presets are the three modes, and they differ", function (t) {
    // upscale / interpolate / both, as three distinct requests. A preset that
    // silently produced the same URL as another would be a menu entry that
    // lies about what it does.
    var source = { sourceUrl: "https://cdn.test/a.mp4", width: 1920, height: 1080 };
    var seen = {};
    ["sr_only", "rife_only", "full_chain"].forEach(function (preset) {
      var q = params(
        shared.buildEnhanceUrl(
          { serverUrl: "http://s", preset: preset, target: "3840x2160" },
          source,
          null
        )
      );
      seen[preset] = q.target + "|" + q.enable_sr + "|" + q.fps_multiplier;
    });
    t.equal(seen.sr_only, "3840x2160|true|1");
    t.equal(seen.rife_only, "1920x1080|false|2");
    t.equal(seen.full_chain, "3840x2160|true|2");
  });

  test("a generated job id is within the server's accepted alphabet", function (t) {
    for (var i = 0; i < 50; i++) {
      var id = shared.makeJobId();
      t.ok(/^[A-Za-z0-9_-]{1,64}$/.test(id), "id " + id + " is acceptable");
    }
  });

  test("the origin pattern covers exactly the configured server", function (t) {
    t.equal(shared.originPattern("http://127.0.0.1:8100/v1/x"), "http://127.0.0.1:8100/*");
    t.equal(shared.originPattern("not a url"), null);
  });

  return cases;
});
